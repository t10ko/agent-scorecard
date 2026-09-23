"""Cost aggregation: rebuild classification, per-scope summaries, and the
breakdowns the `cost` command reports."""

from __future__ import annotations

import datetime

from agent_scorecard.costs import (
    classify_prefix_rebuild,
    records_in_scope,
    records_in_window,
    summarize,
    summarize_by_agent_type,
    summarize_by_day,
    summarize_by_spawn_depth,
)
from agent_scorecard.models import (
    MainThreadOrigin,
    RebuildKind,
    SubagentOrigin,
    TranscriptScope,
    UsageRecord,
)
from agent_scorecard.pricing import load_prices


def rec(
    request_id: str,
    *,
    input_tokens: int = 0,
    cache_read_tokens: int = 0,
    cache_write_1h_tokens: int = 0,
    cache_write_5m_tokens: int = 0,
    output_tokens: int = 0,
    origin: MainThreadOrigin | SubagentOrigin | None = None,
    timestamp: str = "2026-09-20T12:00:00Z",
) -> UsageRecord:
    return UsageRecord(
        request_id=request_id,
        timestamp=datetime.datetime.fromisoformat(timestamp.replace("Z", "+00:00")),
        model="claude-sonnet-5",
        input_tokens=input_tokens,
        cache_read_tokens=cache_read_tokens,
        cache_write_1h_tokens=cache_write_1h_tokens,
        cache_write_5m_tokens=cache_write_5m_tokens,
        output_tokens=output_tokens,
        origin=origin if origin is not None else MainThreadOrigin(),
    )


def sub(agent_id: str, agent_type: str = "Explore", spawn_depth: int = 1) -> SubagentOrigin:
    return SubagentOrigin(agent_id=agent_id, agent_type=agent_type, spawn_depth=spawn_depth)


def test_each_rebuild_boundary_is_classified() -> None:
    # Exactly the 0.9 ratio is a rebuild; below it is growth; a record with
    # no input at all has nothing to rebuild.
    assert classify_prefix_rebuild(rec("a", input_tokens=10)) is RebuildKind.NORMAL_GROWTH
    assert (
        classify_prefix_rebuild(rec("b", input_tokens=11, cache_write_5m_tokens=89))
        is RebuildKind.NORMAL_GROWTH
    )
    assert (
        classify_prefix_rebuild(rec("c", input_tokens=10, cache_write_5m_tokens=90))
        is RebuildKind.FULL_REBUILD
    )
    # The ratio must use the sum of both TTL buckets.
    assert (
        classify_prefix_rebuild(
            rec("d", cache_read_tokens=20, cache_write_1h_tokens=40, cache_write_5m_tokens=40)
        )
        is RebuildKind.NORMAL_GROWTH
    )
    assert classify_prefix_rebuild(rec("e")) is RebuildKind.NORMAL_GROWTH


def test_summarize_totals_both_ttls_and_isolates_unpriced_records() -> None:
    table = load_prices()
    records = [
        rec(
            "req_1",
            cache_write_1h_tokens=600_000,
            cache_write_5m_tokens=400_000,
            output_tokens=1_000_000,
        ),
        rec("req_2", input_tokens=1_000_000, cache_read_tokens=1_000_000),
        UsageRecord(
            request_id="req_3",
            timestamp=datetime.datetime(2026, 9, 20, 12, 0, tzinfo=datetime.UTC),
            model="model-with-no-row",
            input_tokens=500,
            cache_read_tokens=0,
            cache_write_1h_tokens=0,
            cache_write_5m_tokens=0,
            output_tokens=7,
        ),
    ]

    summary = summarize(records, table)

    assert summary.record_count == 3
    assert summary.full_rebuild_count == 1
    assert summary.normal_growth_count == 2
    # Tokens count for every record, priced or not.
    assert summary.total_input_tokens == 1_000_500
    assert summary.total_cache_read_tokens == 1_000_000
    assert summary.total_cache_write_tokens == 1_000_000
    assert summary.total_output_tokens == 1_000_007
    # Cost covers only the two priced records:
    # (600k*2.0 + 400k*1.25) * $2/MTok = 3_400_000, plus 1M output *
    # $10/MTok = 10_000_000, plus (1M + 1M*0.1) * $2/MTok = 2_200_000.
    assert summary.total_cost_microusd == 3_400_000 + 10_000_000 + 2_200_000
    assert summary.unpriced_record_count == 1
    assert summary.unpriced_models == ("model-with-no-row",)


def test_records_in_scope_splits_main_thread_from_agent_spend() -> None:
    records = [
        rec("req_main", input_tokens=1_000_000),
        rec("req_sub", input_tokens=3_000_000, origin=sub("aaa")),
    ]

    main_only = records_in_scope(records, TranscriptScope.MAIN_THREAD)
    subagent_only = records_in_scope(records, TranscriptScope.SUBAGENT)

    assert [r.request_id for r in main_only] == ["req_main"]
    assert [r.request_id for r in subagent_only] == ["req_sub"]
    # Combined stays the sum of the two, so no reader has to choose.
    table = load_prices()
    assert (
        summarize(main_only, table).total_cost_microusd
        + summarize(subagent_only, table).total_cost_microusd
        == summarize(records, table).total_cost_microusd
    )


def test_records_in_window_bounds_are_inclusive() -> None:
    records = [
        rec("a", timestamp="2026-09-01T00:00:00Z"),
        rec("b", timestamp="2026-09-22T23:59:59Z"),
        rec("c", timestamp="2026-08-31T12:00:00Z"),
        rec("d", timestamp="2026-09-23T00:00:00Z"),
    ]

    windowed = records_in_window(records, datetime.date(2026, 9, 1), datetime.date(2026, 9, 22))

    assert [r.request_id for r in windowed] == ["a", "b"]


def test_by_agent_type_and_depth_exclude_main_thread_records() -> None:
    # A main-session record has no agent type and no spawn depth. Bucketing
    # it under a placeholder would attribute real spend to an agent that
    # never ran.
    table = load_prices()
    records = [
        rec("req_main", input_tokens=9_000_000),
        rec("req_a", input_tokens=1_000_000, origin=sub("a")),
        rec("req_b", input_tokens=2_000_000, origin=sub("b", spawn_depth=2)),
        rec(
            "req_c",
            input_tokens=4_000_000,
            origin=sub("c", agent_type="general-purpose"),
        ),
    ]

    by_type = summarize_by_agent_type(records, table)
    by_depth = summarize_by_spawn_depth(records, table)

    # claude-sonnet-5 input is $2/MTok, so cost is 2x the token count.
    assert set(by_type) == {"Explore", "general-purpose"}
    assert by_type["Explore"].record_count == 2
    assert by_type["Explore"].total_cost_microusd == 3_000_000 * 2
    assert by_type["general-purpose"].total_cost_microusd == 4_000_000 * 2
    assert set(by_depth) == {1, 2}
    assert by_depth[1].total_cost_microusd == 5_000_000 * 2
    assert by_depth[2].total_cost_microusd == 2_000_000 * 2


def test_by_day_splits_main_thread_from_agent_spend() -> None:
    table = load_prices()
    records = [
        rec("req_main", input_tokens=1_000_000),
        rec("req_sub", input_tokens=3_000_000, origin=sub("a")),
    ]

    by_day = summarize_by_day(records, table)

    assert set(by_day) == {"2026-09-20"}
    bucket = by_day["2026-09-20"]
    assert bucket.main_thread_requests == 1
    assert bucket.main_thread_cost_microusd == 2_000_000
    assert bucket.subagent_requests == 1
    assert bucket.subagent_cost_microusd == 6_000_000
