"""Outcome classification: the six rules of §5.5, applied strictly in
order."""

from __future__ import annotations

import re

from agent_scorecard.models import Lifecycle, Outcome
from agent_scorecard.outcome import classify_outcome


def classify(
    lifecycle: Lifecycle = Lifecycle.COMPLETED,
    report: str | None = "All done, tests pass",
    edited: bool = False,
    test_runs: int = 0,
    last_test_passed: bool | None = None,
    failure_pattern: str | None = None,
    success_pattern: str | None = None,
):
    return classify_outcome(
        lifecycle=lifecycle,
        lifecycle_reason=(
            "failed: stalled"
            if lifecycle in (Lifecycle.FAILED, Lifecycle.KILLED, Lifecycle.STOPPED)
            else None
        ),
        final_report=report,
        edited_files=edited,
        test_runs=test_runs,
        last_test_passed=last_test_passed,
        failure_pattern=re.compile(failure_pattern) if failure_pattern else None,
        success_pattern=re.compile(success_pattern) if success_pattern else None,
    )


def test_a_failed_killed_or_stopped_lifecycle_fails() -> None:
    for lifecycle in (Lifecycle.FAILED, Lifecycle.KILLED, Lifecycle.STOPPED):
        outcome, reason, inferred = classify(lifecycle)
        assert outcome is Outcome.FAILED
        assert reason == "failed: stalled"
        assert inferred is False


def test_a_failure_pattern_overrides_a_completed_lifecycle() -> None:
    # The agent said it finished, but its report admits the job failed.
    outcome, reason, inferred = classify(
        report="steps done. TASK FAILED: exit 3",
        failure_pattern=r"TASK FAILED",
    )
    assert outcome is Outcome.FAILED
    assert reason == "agent reported failure"
    assert inferred is False


def test_failing_tests_fail_only_runs_that_edited_files() -> None:
    # An editing run that left the tests red failed to deliver.
    outcome, reason, _ = classify(
        lifecycle=Lifecycle.COMPLETED,
        edited=True,
        test_runs=2,
        last_test_passed=False,
    )
    assert outcome is Outcome.FAILED_TESTS
    assert reason == "left tests failing"


def test_failing_tests_do_not_fail_read_only_runs() -> None:
    # A read-only agent investigating a failure is expected to see failing
    # tests; its outcome comes from the lifecycle.
    outcome, _, _ = classify(
        lifecycle=Lifecycle.COMPLETED,
        edited=False,
        test_runs=2,
        last_test_passed=False,
    )
    assert outcome is Outcome.SUCCEEDED


def test_a_completed_lifecycle_succeeds() -> None:
    outcome, reason, inferred = classify()

    assert outcome is Outcome.SUCCEEDED
    assert reason is None
    assert inferred is False


def test_a_success_pattern_infers_a_success_from_an_unknown_lifecycle() -> None:
    outcome, reason, inferred = classify(
        lifecycle=Lifecycle.UNKNOWN,
        report="wrapped up. ALL TESTS PASS",
        success_pattern=r"ALL TESTS PASS",
    )

    assert outcome is Outcome.SUCCEEDED
    assert reason == "agent reported success"
    assert inferred is True


def test_a_success_pattern_never_overrides_a_recorded_failure() -> None:
    # Rule order: the failure rule fires long before the success pattern.
    outcome, _, _ = classify(
        lifecycle=Lifecycle.FAILED,
        success_pattern=r"ALL TESTS PASS",
    )

    assert outcome is Outcome.FAILED


def test_an_unknown_lifecycle_with_no_signal_is_unknown() -> None:
    outcome, reason, inferred = classify(lifecycle=Lifecycle.UNKNOWN, report=None)

    assert outcome is Outcome.UNKNOWN
    assert reason is None
    assert inferred is False
