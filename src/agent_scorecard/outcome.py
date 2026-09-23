"""Outcome classification: did the run deliver?

The checks run in a fixed order, because the first applicable rule decides:
a recorded failure beats a failure pattern, a failure pattern beats the
tests, and the tests are only consulted for runs that edited files — a
read-only run investigating a failure is expected to see red tests.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from agent_scorecard.lifecycle import Lifecycle, LifecycleEvent, final_report_for
from agent_scorecard.models import Outcome
from agent_scorecard.runs import AgentRun, RunFacts


def classify_outcome(
    *,
    lifecycle: Lifecycle,
    lifecycle_reason: str | None,
    final_report: str | None,
    edited_files: bool,
    test_runs: int,
    last_test_passed: bool | None,
    failure_pattern: re.Pattern[str] | None,
    success_pattern: re.Pattern[str] | None,
) -> tuple[Outcome, str | None, bool]:
    """One run's outcome, its reason, and whether it was inferred.

    Returns `(outcome, reason, inferred)`. `inferred` is true only when the
    success pattern marked an otherwise-unknown run a success.
    """
    if lifecycle in (Lifecycle.FAILED, Lifecycle.KILLED, Lifecycle.STOPPED):
        return Outcome.FAILED, lifecycle_reason, False
    if (
        failure_pattern is not None
        and final_report is not None
        and failure_pattern.search(final_report)
    ):
        return Outcome.FAILED, "agent reported failure", False
    if edited_files and test_runs > 0 and last_test_passed is False:
        return Outcome.FAILED_TESTS, "left tests failing", False
    if lifecycle is Lifecycle.COMPLETED:
        return Outcome.SUCCEEDED, None, False
    if (
        lifecycle is Lifecycle.UNKNOWN
        and success_pattern is not None
        and final_report is not None
        and success_pattern.search(final_report)
    ):
        return Outcome.SUCCEEDED, "agent reported success", True
    return Outcome.UNKNOWN, None, False


def attach_outcomes(
    runs: Sequence[AgentRun],
    facts: Sequence[RunFacts],
    events: Sequence[LifecycleEvent],
    *,
    success_pattern: re.Pattern[str] | None = None,
    failure_pattern: re.Pattern[str] | None = None,
) -> tuple[AgentRun, ...]:
    """Resolve each run's final report and classify its outcome, returning
    the runs with outcome fields filled in. Events keyed to agent ids
    matching no run are ignored."""
    facts_by_id = {item.agent_id: item for item in facts}
    events_by_id: dict[str, list[LifecycleEvent]] = {}
    for event in events:
        if event.agent_id in facts_by_id:
            events_by_id.setdefault(event.agent_id, []).append(event)
    resolved: list[AgentRun] = []
    for run in runs:
        item = facts_by_id.get(run.agent_id)
        report = (
            final_report_for(item, events_by_id.get(run.agent_id, ())) if item is not None else None
        )
        outcome, reason, inferred = classify_outcome(
            lifecycle=run.lifecycle,
            lifecycle_reason=run.reason,
            final_report=report,
            edited_files=run.edited_files,
            test_runs=run.test_runs,
            last_test_passed=run.last_test_passed,
            failure_pattern=failure_pattern,
            success_pattern=success_pattern,
        )
        resolved.append(
            run.model_copy(
                update={
                    "outcome": outcome,
                    "outcome_reason": reason,
                    "outcome_inferred": inferred,
                }
            )
        )
    return tuple(resolved)
