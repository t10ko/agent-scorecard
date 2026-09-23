"""Scorecard grouping, verdict boundaries, reason and note shapes, and
sorting."""

from __future__ import annotations

import datetime

from agent_scorecard.models import Lifecycle, LifecycleSource, Outcome
from agent_scorecard.runs import AgentRun
from agent_scorecard.scorecard import GroupBy, GroupStats, Thresholds, build_scorecard

_SEQ = [iter(range(1000))]


def run(
    outcome: Outcome = Outcome.SUCCEEDED,
    *,
    agent_type: str = "worker",
    model: str | None = "claude-sonnet-5",
    cost_microusd: int | None = 1_000_000,
    unpriced_models: tuple[str, ...] = (),
    turns: int = 3,
    duration: float | None = 60.0,
    tool_calls: int = 4,
    tool_errors: int = 0,
    test_runs: int = 0,
    last_test_passed: bool | None = None,
    commits: int = 0,
    inferred: bool = False,
    agent_id: str | None = None,
) -> AgentRun:
    if agent_id is None:
        agent_id = f"agent-{next(next_gen())}"
    return AgentRun(
        agent_id=agent_id,
        agent_type=agent_type,
        spawn_depth=1,
        session_id="session-1",
        started_at=datetime.datetime(2026, 9, 20, tzinfo=datetime.UTC),
        ended_at=datetime.datetime(2026, 9, 20, 0, 5, tzinfo=datetime.UTC),
        duration_seconds=duration,
        turns=turns,
        cost_microusd=cost_microusd,
        unpriced_records=len(unpriced_models),
        unpriced_models=unpriced_models,
        primary_model=model,
        models=(model,) if model else (),
        tool_calls=tool_calls,
        tool_errors=tool_errors,
        test_runs=test_runs,
        last_test_passed=last_test_passed,
        commits=commits,
        lifecycle=Lifecycle.COMPLETED if outcome is Outcome.SUCCEEDED else Lifecycle.FAILED,
        lifecycle_source=LifecycleSource.FOREGROUND,
        outcome=outcome,
        outcome_inferred=inferred,
    )


def next_gen():
    # A deterministic agent-id counter shared by every test in this module.
    while True:
        yield _SEQ[0]


def by_key(groups: list[GroupStats], key: str) -> GroupStats:
    return next(group for group in groups if group.key == key)


def test_the_remove_boundary_is_exclusive() -> None:
    # A rate of exactly 0.5 is fix, not remove.
    runs = [run() for _ in range(5)] + [run(Outcome.FAILED) for _ in range(5)]

    (group,) = build_scorecard(runs, GroupBy.AGENT_TYPE, Thresholds())

    assert group.verdict.value == "fix"
    assert group.reasons == ("fix: 5 of 10 decided runs succeeded (50%), below 80%",)


def test_the_keep_boundary_is_inclusive() -> None:
    # A rate of exactly 0.8 is keep.
    runs = [run() for _ in range(8)] + [run(Outcome.FAILED) for _ in range(2)]

    (group,) = build_scorecard(runs, GroupBy.AGENT_TYPE, Thresholds())

    assert group.verdict.value == "keep"
    assert group.reasons == ("keep: 8 of 10 decided runs succeeded (80%)",)


def test_the_remove_reason_shape() -> None:
    runs = [run() for _ in range(2)] + [run(Outcome.FAILED) for _ in range(4)]

    (group,) = build_scorecard(runs, GroupBy.AGENT_TYPE, Thresholds())

    assert group.reasons == ("remove: 2 of 6 decided runs succeeded (33%), below 50%",)


def test_the_fix_reason_shape_prints_62_percent_for_five_of_eight() -> None:
    # .0% rounds half to even: 62.5% prints as 62%.
    runs = [run() for _ in range(5)] + [run(Outcome.FAILED) for _ in range(3)]

    (group,) = build_scorecard(runs, GroupBy.AGENT_TYPE, Thresholds())

    assert group.reasons == ("fix: 5 of 8 decided runs succeeded (62%), below 80%",)


def test_a_high_tool_error_rate_forces_at_least_fix() -> None:
    runs = [run(tool_calls=100, tool_errors=27) for _ in range(10)]

    (group,) = build_scorecard(runs, GroupBy.AGENT_TYPE, Thresholds())

    assert group.verdict.value == "fix"
    assert group.reasons == ("fix: 27% of tool calls failed, above 20%",)


def test_not_enough_data_boundary_and_reason() -> None:
    runs = [run() for _ in range(3)]

    (group,) = build_scorecard(runs, GroupBy.AGENT_TYPE, Thresholds())

    assert group.verdict.value == "not_enough_data"
    assert group.reasons == ("not enough data: 3 decided runs, need 5",)


def test_decided_runs_exactly_at_min_runs_get_a_verdict() -> None:
    runs = [run() for _ in range(5)]

    (group,) = build_scorecard(runs, GroupBy.AGENT_TYPE, Thresholds())

    assert group.verdict.value == "keep"
    assert group.decided_runs == 5


def test_unknown_runs_are_left_out_of_rates_but_counted() -> None:
    runs = [run() for _ in range(5)] + [run(Outcome.UNKNOWN, inferred=False) for _ in range(4)]

    (group,) = build_scorecard(runs, GroupBy.AGENT_TYPE, Thresholds())

    assert group.runs == 9
    assert group.decided_runs == 5
    assert group.unknown == 4
    assert group.success_rate == 1.0
    assert group.notes == ("4 runs with unknown outcome",)


def test_inferred_outcomes_are_noted() -> None:
    runs = [run() for _ in range(5)] + [run(Outcome.SUCCEEDED, inferred=True) for _ in range(7)]

    (group,) = build_scorecard(runs, GroupBy.AGENT_TYPE, Thresholds())

    assert group.inferred_outcomes == 7
    assert "outcome inferred for 7 runs" in group.notes


def test_unpriced_runs_are_noted_with_their_models() -> None:
    runs = [run(unpriced_models=("claude-opus-4-8",), cost_microusd=None) for _ in range(2)] + [
        run() for _ in range(5)
    ]

    (group,) = build_scorecard(runs, GroupBy.AGENT_TYPE, Thresholds())

    assert group.unpriced_runs == 2
    assert "2 runs used unpriced models: claude-opus-4-8" in group.notes
    # Only the priced runs' cost is in the total.
    assert group.cost_microusd == 5_000_000


def test_a_group_where_every_run_is_unpriced_has_unknown_cost() -> None:
    runs = [run(unpriced_models=("claude-opus-4-8",), cost_microusd=None) for _ in range(5)]

    (group,) = build_scorecard(runs, GroupBy.AGENT_TYPE, Thresholds())

    assert group.cost_microusd is None
    assert group.cost_per_run_usd is None


def test_cost_per_success_divides_by_successes_and_is_null_without_them() -> None:
    runs = [run(cost_microusd=3_000_000) for _ in range(2)] + [
        run(Outcome.FAILED, cost_microusd=2_000_000) for _ in range(3)
    ]

    (group,) = build_scorecard(runs, GroupBy.AGENT_TYPE, Thresholds())

    # Failed runs are part of what each success costs: $12 total / 2.
    assert group.cost_microusd == 2 * 3_000_000 + 3 * 2_000_000
    assert group.cost_per_success_usd == 6.0

    no_successes = [run(Outcome.FAILED) for _ in range(5)]
    (group,) = build_scorecard(no_successes, GroupBy.AGENT_TYPE, Thresholds())
    assert group.cost_per_success_usd is None
    assert group.verdict.value == "remove"


def test_group_metrics_summarize_their_runs() -> None:
    runs = [
        run(turns=2, duration=30.0, test_runs=1, last_test_passed=True, commits=1),
        run(turns=4, duration=90.0, test_runs=1, last_test_passed=True),
    ]

    (group,) = build_scorecard(runs, GroupBy.AGENT_TYPE, Thresholds())

    assert group.median_turns == 3
    assert group.median_duration_seconds == 60.0
    assert group.runs_with_tests == 2
    assert group.runs_ending_green == 2
    assert group.commits == 1
    assert group.tool_calls == 8
    assert group.models == {"claude-sonnet-5": 2}


def test_sorting_puts_remove_first_then_cost_desc_then_unknown_first() -> None:
    runs = [
        # fix group, $10 total: a 0.6 success rate is fix, not remove.
        *[run(agent_type="cheap-fix", cost_microusd=1_000_000) for _ in range(3)]
        + [run(Outcome.FAILED, agent_type="cheap-fix", cost_microusd=1_000_000) for _ in range(2)],
        # remove group, cheap: still leads on severity.
        *[run(Outcome.FAILED, agent_type="bad", cost_microusd=1_000_000) for _ in range(6)]
        + [run(agent_type="bad") for _ in range(0)],
        # keep group, expensive.
        *[run(agent_type="expensive", cost_microusd=10_000_000) for _ in range(5)],
        # keep group with unknown cost leads the keep block.
        *[
            run(
                agent_type="unpriced",
                cost_microusd=None,
                unpriced_models=("m",),
            )
            for _ in range(5)
        ],
    ]

    groups = build_scorecard(runs, GroupBy.AGENT_TYPE, Thresholds())

    assert [group.key for group in groups] == [
        "bad",
        "cheap-fix",
        "unpriced",
        "expensive",
    ]
    assert [group.verdict.value for group in groups] == [
        "remove",
        "fix",
        "keep",
        "keep",
    ]


def test_the_cost_per_success_note_flags_an_outlier() -> None:
    runs = [
        # $100 per success.
        *[run(agent_type="pricey", cost_microusd=100_000_000) for _ in range(5)],
        # $1 and $2 per success: the median across groups is $2.
        *[run(agent_type="normal", cost_microusd=1_000_000) for _ in range(5)],
        *[run(agent_type="middle", cost_microusd=2_000_000) for _ in range(5)],
    ]

    groups = build_scorecard(runs, GroupBy.AGENT_TYPE, Thresholds())

    pricey = by_key(groups, "pricey")
    assert "costs 50.0× the median cost per successful run" in pricey.notes
    assert by_key(groups, "normal").notes == ()


def test_each_group_by_value_groups_differently() -> None:
    runs = [
        run(agent_type="worker", model="claude-sonnet-5"),
        run(agent_type="worker", model="claude-opus-5"),
        run(agent_type="solo", model=None, cost_microusd=None, unpriced_models=("m",)),
    ]

    by_type = build_scorecard(runs, GroupBy.AGENT_TYPE, Thresholds())
    assert [group.key for group in by_type] == ["solo", "worker"]

    by_model = build_scorecard(runs, GroupBy.MODEL, Thresholds())
    # The unknown-cost group leads its severity block, then cost desc with
    # an alphabetical tie-break.
    assert [group.key for group in by_model] == [
        "unknown",
        "claude-opus-5",
        "claude-sonnet-5",
    ]

    by_both = build_scorecard(runs, GroupBy.AGENT_TYPE_MODEL, Thresholds())
    assert {group.key for group in by_both} == {
        "worker (claude-sonnet-5)",
        "worker (claude-opus-5)",
        "solo (unknown)",
    }
