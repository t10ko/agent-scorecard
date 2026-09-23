"""Grouping runs, computing their metrics, and deciding verdicts.

The verdict is a starting point for a human decision, not an automatic
kill switch: every threshold below is a judgement call, and every verdict
carries the reasons and notes a reader needs to overrule it.
"""

from __future__ import annotations

import statistics
from collections.abc import Sequence
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from agent_scorecard.models import Outcome, Verdict
from agent_scorecard.runs import AgentRun


class GroupBy(StrEnum):
    """How runs are grouped into scorecard rows."""

    AGENT_TYPE = "agent-type"
    MODEL = "model"
    AGENT_TYPE_MODEL = "agent-type+model"


class Thresholds(BaseModel):
    """The verdict cutoffs; every one is a judgement call."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    min_runs: int = Field(default=5, ge=0)
    remove_below: float = Field(default=0.5, ge=0.0, le=1.0)
    fix_below: float = Field(default=0.8, ge=0.0, le=1.0)
    max_tool_error_rate: float = Field(default=0.2, ge=0.0, le=1.0)


class GroupStats(BaseModel):
    """One group's metrics, verdict, reasons, and notes."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    key: str
    runs: int = Field(ge=0)
    decided_runs: int = Field(ge=0)
    succeeded: int = Field(ge=0)
    failed: int = Field(ge=0)
    failed_tests: int = Field(ge=0)
    unknown: int = Field(ge=0)
    inferred_outcomes: int = Field(ge=0)
    success_rate: float | None = None
    cost_microusd: int | None = None
    unpriced_runs: int = Field(ge=0)
    cost_per_run_usd: float | None = None
    cost_per_success_usd: float | None = None
    median_turns: float | None = None
    median_duration_seconds: float | None = None
    tool_calls: int = Field(ge=0)
    tool_errors: int = Field(ge=0)
    tool_error_rate: float = 0.0
    tool_denials: int = Field(ge=0)
    runs_with_tests: int = Field(ge=0)
    runs_ending_green: int = Field(ge=0)
    commits: int = Field(ge=0)
    cache_read_share: float = 0.0
    full_rebuild_turns: int = Field(ge=0)
    models: dict[str, int] = Field(default_factory=dict)
    verdict: Verdict
    reasons: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()


def group_key_of(run: AgentRun, group_by: GroupBy) -> str:
    """The group a run belongs to. `model` is the run's primary model; a
    run that billed nothing has none, and lands under `unknown`."""
    if group_by is GroupBy.AGENT_TYPE:
        return run.agent_type
    if group_by is GroupBy.MODEL:
        return run.primary_model or "unknown"
    model = run.primary_model or "unknown"
    return f"{run.agent_type} ({model})"


def _decide_verdict(
    decided: int,
    succeeded: int,
    tool_calls: int,
    tool_errors: int,
    thresholds: Thresholds,
) -> tuple[Verdict, tuple[str, ...]]:
    """The verdict, in the brief's fixed order, with its exact reason
    shapes (tests assert them)."""
    if decided < thresholds.min_runs:
        return Verdict.NOT_ENOUGH_DATA, (
            f"not enough data: {decided} decided runs, need {thresholds.min_runs}",
        )
    rate = succeeded / decided if decided else None
    success_reason = (
        f"{succeeded} of {decided} decided runs succeeded ({rate:.0%})"
        if rate is not None
        else None
    )
    if rate is not None and rate < thresholds.remove_below:
        return Verdict.REMOVE, (f"remove: {success_reason}, below {thresholds.remove_below:.0%}",)
    reasons: list[str] = []
    if rate is not None and rate < thresholds.fix_below:
        reasons.append(f"fix: {success_reason}, below {thresholds.fix_below:.0%}")
    tool_rate = tool_errors / tool_calls if tool_calls else 0.0
    if tool_rate > thresholds.max_tool_error_rate:
        reasons.append(
            f"fix: {tool_rate:.0%} of tool calls failed, above {thresholds.max_tool_error_rate:.0%}"
        )
    if reasons:
        return Verdict.FIX, tuple(reasons)
    return Verdict.KEEP, (f"keep: {success_reason}",)


def build_scorecard(
    runs: Sequence[AgentRun],
    group_by: GroupBy,
    thresholds: Thresholds,
) -> list[GroupStats]:
    """One `GroupStats` per group, sorted by verdict severity, then by cost
    (most expensive first, unknown-cost groups first)."""
    grouped: dict[str, list[AgentRun]] = {}
    for run in runs:
        grouped.setdefault(group_key_of(run, group_by), []).append(run)

    groups: list[GroupStats] = [
        _group_stats(key, members, thresholds) for key, members in grouped.items()
    ]

    # The cost-per-success note is cross-group: a group only counts as
    # expensive against the median group that has a cost per success.
    per_success = [
        group.cost_per_success_usd for group in groups if group.cost_per_success_usd is not None
    ]
    median_cost = statistics.median(per_success) if per_success else None
    noted: list[GroupStats] = []
    for group in groups:
        notes = list(group.notes)
        if median_cost is not None and group.cost_per_success_usd is not None and median_cost > 0:
            ratio = group.cost_per_success_usd / median_cost
            if ratio > 3:
                notes.append(f"costs {ratio:.1f}× the median cost per successful run")
        noted.append(group.model_copy(update={"notes": tuple(notes)}))

    def sort_key(group: GroupStats) -> tuple[int, int, float, str]:
        severity = [
            Verdict.REMOVE,
            Verdict.FIX,
            Verdict.KEEP,
            Verdict.NOT_ENOUGH_DATA,
        ].index(group.verdict)
        return (
            severity,
            0 if group.cost_microusd is None else 1,
            -(group.cost_microusd or 0),
            group.key,
        )

    return sorted(noted, key=sort_key)


def _group_stats(key: str, runs: list[AgentRun], thresholds: Thresholds) -> GroupStats:
    succeeded = sum(1 for run in runs if run.outcome is Outcome.SUCCEEDED)
    failed = sum(1 for run in runs if run.outcome is Outcome.FAILED)
    failed_tests = sum(1 for run in runs if run.outcome is Outcome.FAILED_TESTS)
    unknown = sum(1 for run in runs if run.outcome is Outcome.UNKNOWN)
    decided = succeeded + failed + failed_tests
    inferred = sum(1 for run in runs if run.outcome_inferred)
    unpriced_runs = sum(1 for run in runs if run.cost_microusd is None)
    priced_runs = len(runs) - unpriced_runs
    cost_microusd = (
        sum(run.cost_microusd for run in runs if run.cost_microusd is not None)
        if priced_runs
        else None
    )
    durations = [run.duration_seconds for run in runs if run.duration_seconds is not None]
    tool_calls = sum(run.tool_calls for run in runs)
    tool_errors = sum(run.tool_errors for run in runs)
    primary_models: dict[str, int] = {}
    for run in runs:
        model = run.primary_model or "unknown"
        primary_models[model] = primary_models.get(model, 0) + 1
    cache_read = sum(run.cache_read_tokens for run in runs)
    billed_input = sum(run.billed_input_tokens for run in runs)

    verdict, reasons = _decide_verdict(decided, succeeded, tool_calls, tool_errors, thresholds)

    notes: list[str] = []
    if unknown:
        notes.append(f"{unknown} runs with unknown outcome")
    if inferred:
        notes.append(f"outcome inferred for {inferred} runs")
    if unpriced_runs:
        unpriced_models = sorted({model for run in runs for model in run.unpriced_models})
        notes.append(f"{unpriced_runs} runs used unpriced models: {', '.join(unpriced_models)}")

    return GroupStats(
        key=key,
        runs=len(runs),
        decided_runs=decided,
        succeeded=succeeded,
        failed=failed,
        failed_tests=failed_tests,
        unknown=unknown,
        inferred_outcomes=inferred,
        success_rate=(succeeded / decided) if decided else None,
        cost_microusd=cost_microusd,
        unpriced_runs=unpriced_runs,
        cost_per_run_usd=(
            round(cost_microusd / priced_runs / 1_000_000, 4)
            if cost_microusd is not None and priced_runs
            else None
        ),
        cost_per_success_usd=(
            round(cost_microusd / succeeded / 1_000_000, 4)
            if cost_microusd is not None and succeeded
            else None
        ),
        median_turns=statistics.median([run.turns for run in runs]) if runs else None,
        median_duration_seconds=(statistics.median(durations) if durations else None),
        tool_calls=tool_calls,
        tool_errors=tool_errors,
        tool_error_rate=(tool_errors / tool_calls) if tool_calls else 0.0,
        tool_denials=sum(run.tool_denials for run in runs),
        runs_with_tests=sum(1 for run in runs if run.test_runs > 0),
        runs_ending_green=sum(1 for run in runs if run.last_test_passed is True),
        commits=sum(run.commits for run in runs),
        cache_read_share=(cache_read / billed_input) if billed_input else 0.0,
        full_rebuild_turns=sum(run.full_rebuild_turns for run in runs),
        models=primary_models,
        verdict=verdict,
        reasons=reasons,
        notes=tuple(notes),
    )
