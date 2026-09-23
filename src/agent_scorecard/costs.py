"""Cost aggregation: rebuild classification, token/cost summaries, and
breakdowns by scope, agent type, spawn depth, and day.

Everything here is a pure function over the records the scan produced —
nothing after the scan touches the disk.
"""

from __future__ import annotations

import datetime
from collections import defaultdict
from collections.abc import Callable, Sequence

from pydantic import BaseModel, ConfigDict

from agent_scorecard.models import (
    RebuildKind,
    SubagentOrigin,
    TranscriptScope,
    UsageRecord,
    UsageSummary,
)
from agent_scorecard.pricing import PriceTable, estimate_cost

# A turn whose cache-write tokens are at least this fraction of its billed
# input tokens paid to rebuild the whole prefix from scratch rather than
# extending an existing cache. A judgement call, not a published boundary:
# a turn extending a warm cache reads far more than it writes, so the ratio
# separates the two populations well away from either extreme.
_FULL_REBUILD_CACHE_WRITE_RATIO = 0.9


def classify_prefix_rebuild(record: UsageRecord) -> RebuildKind:
    """Whether a request's prompt-cache write looks like a full
    context-prefix rebuild or normal growth on an existing cache. A request
    with zero billed input tokens has nothing to rebuild."""
    if record.billed_input_tokens == 0:
        return RebuildKind.NORMAL_GROWTH
    if record.cache_write_tokens / record.billed_input_tokens >= (_FULL_REBUILD_CACHE_WRITE_RATIO):
        return RebuildKind.FULL_REBUILD
    return RebuildKind.NORMAL_GROWTH


def summarize(records: Sequence[UsageRecord], table: PriceTable) -> UsageSummary:
    """Token totals, estimated cost, and rebuild-kind counts across records.

    Records whose model has no price row contribute their tokens and their
    rebuild classification, but no cost — they are counted in
    `unpriced_record_count` so the dollar figure is never mistaken for
    covering the whole corpus.
    """
    total_cost_microusd = 0
    full_rebuild_count = 0
    normal_growth_count = 0
    total_input_tokens = 0
    total_cache_read_tokens = 0
    total_cache_write_tokens = 0
    total_output_tokens = 0
    unpriced_record_count = 0
    unpriced_models: set[str] = set()

    for record in records:
        cost_microusd = estimate_cost(record, table)
        if cost_microusd is None:
            unpriced_record_count += 1
            unpriced_models.add(record.model)
        else:
            total_cost_microusd += cost_microusd
        if classify_prefix_rebuild(record) is RebuildKind.FULL_REBUILD:
            full_rebuild_count += 1
        else:
            normal_growth_count += 1
        total_input_tokens += record.input_tokens
        total_cache_read_tokens += record.cache_read_tokens
        total_cache_write_tokens += record.cache_write_tokens
        total_output_tokens += record.output_tokens

    return UsageSummary(
        record_count=len(records),
        total_input_tokens=total_input_tokens,
        total_cache_read_tokens=total_cache_read_tokens,
        total_cache_write_tokens=total_cache_write_tokens,
        total_output_tokens=total_output_tokens,
        total_cost_microusd=total_cost_microusd,
        full_rebuild_count=full_rebuild_count,
        normal_growth_count=normal_growth_count,
        unpriced_record_count=unpriced_record_count,
        unpriced_models=tuple(sorted(unpriced_models)),
    )


def records_in_scope(
    records: Sequence[UsageRecord], scope: TranscriptScope
) -> tuple[UsageRecord, ...]:
    """The records read from one kind of file, so main-session and agent
    spend stay separately reportable."""
    return tuple(record for record in records if record.origin.scope is scope)


def records_in_window(
    records: Sequence[UsageRecord],
    since: datetime.date | None,
    until: datetime.date | None,
) -> tuple[UsageRecord, ...]:
    """The records whose own timestamp falls in the inclusive date window."""
    kept: list[UsageRecord] = []
    for record in records:
        day = record.timestamp.date()
        if since is not None and day < since:
            continue
        if until is not None and day > until:
            continue
        kept.append(record)
    return tuple(kept)


def _summarize_subagents_by[GroupKey](
    records: Sequence[UsageRecord],
    table: PriceTable,
    key_of: Callable[[SubagentOrigin], GroupKey],
) -> dict[GroupKey, UsageSummary]:
    """Group subagent records by one axis of their origin and summarize each.

    Main-session records are skipped rather than bucketed under a
    placeholder: they have no agent type and no spawn depth, and inventing
    one would put spend in a row that names an agent that never ran.
    """
    grouped: dict[GroupKey, list[UsageRecord]] = defaultdict(list)
    for record in records:
        origin = record.origin
        if isinstance(origin, SubagentOrigin):
            grouped[key_of(origin)].append(record)
    return {key: summarize(group, table) for key, group in grouped.items()}


def summarize_by_agent_type(
    records: Sequence[UsageRecord], table: PriceTable
) -> dict[str, UsageSummary]:
    """Spend and token totals per agent type, the breakdown that says which
    kind of agent a fan-out's cost actually went to."""
    return _summarize_subagents_by(records, table, lambda origin: origin.agent_type)


def summarize_by_spawn_depth(
    records: Sequence[UsageRecord], table: PriceTable
) -> dict[int, UsageSummary]:
    """Spend and token totals per spawn depth: 1 for an agent the main
    session started, 2 for one that agent started, and so on."""
    return _summarize_subagents_by(records, table, lambda origin: origin.spawn_depth)


class DayBucket(BaseModel):
    """One calendar day's spend, split by scope and marked with the unpriced
    records each side excludes."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    main_thread_cost_microusd: int = 0
    main_thread_requests: int = 0
    main_thread_unpriced: int = 0
    subagent_cost_microusd: int = 0
    subagent_requests: int = 0
    subagent_unpriced: int = 0

    @property
    def total_cost_microusd(self) -> int:
        return self.main_thread_cost_microusd + self.subagent_cost_microusd

    @property
    def total_requests(self) -> int:
        return self.main_thread_requests + self.subagent_requests


def summarize_by_day(records: Sequence[UsageRecord], table: PriceTable) -> dict[str, DayBucket]:
    """Per-day spend by the records' own timestamps, split by scope."""
    totals: dict[str, _DayTotals] = defaultdict(_DayTotals)
    for record in records:
        day = record.timestamp.date().isoformat()
        cost = estimate_cost(record, table)
        bucket = totals[day]
        if isinstance(record.origin, SubagentOrigin):
            bucket.subagent_requests += 1
            if cost is None:
                bucket.subagent_unpriced += 1
            else:
                bucket.subagent_cost_microusd += cost
        else:
            bucket.main_thread_requests += 1
            if cost is None:
                bucket.main_thread_unpriced += 1
            else:
                bucket.main_thread_cost_microusd += cost
    return {
        day: DayBucket(
            main_thread_requests=totals[day].main_thread_requests,
            main_thread_cost_microusd=totals[day].main_thread_cost_microusd,
            main_thread_unpriced=totals[day].main_thread_unpriced,
            subagent_requests=totals[day].subagent_requests,
            subagent_cost_microusd=totals[day].subagent_cost_microusd,
            subagent_unpriced=totals[day].subagent_unpriced,
        )
        for day in sorted(totals)
    }


class _DayTotals:
    """Mutable accumulator for one day, folded into a `DayBucket` at the end."""

    def __init__(self) -> None:
        self.main_thread_cost_microusd = 0
        self.main_thread_requests = 0
        self.main_thread_unpriced = 0
        self.subagent_cost_microusd = 0
        self.subagent_requests = 0
        self.subagent_unpriced = 0
