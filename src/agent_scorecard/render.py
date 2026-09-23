"""Output rendering: `table` (rich), `markdown`, and `json`.

Only this module and `cli` write to stdout. The json shapes are part of the
public contract: keys are sorted, money goes out as both integer
`*_microusd` and float `*_usd` rounded to 4 decimals, and output is
deterministic so golden-file tests work.
"""

from __future__ import annotations

import datetime
import io
import json
from collections.abc import Sequence

from rich.console import Console
from rich.table import Table as RichTable
from rich.text import Text

from agent_scorecard.costs import (
    DayBucket,
    summarize,
    summarize_by_agent_type,
    summarize_by_day,
    summarize_by_spawn_depth,
)
from agent_scorecard.models import TranscriptScope, UsageRecord, UsageSummary
from agent_scorecard.pricing import PriceTable
from agent_scorecard.runs import AgentRun
from agent_scorecard.scan import ScanResult
from agent_scorecard.scorecard import GroupStats, Thresholds

SCHEMA_VERSION = 1


def usd(microusd: int) -> float:
    return microusd / 1_000_000


def fmt_usd(microusd: int) -> str:
    return f"${usd(microusd):,.2f}"


def cost_label(summary: UsageSummary) -> str:
    """One group's cost, marked when unpriced records are excluded from it.

    A group whose every record is unpriced has an unknown cost, not a zero
    one: rendered as $0.00 it would read as the cheapest row in the one
    table an operator scans to decide which agent to stop spawning.
    """
    if summary.record_count and summary.unpriced_record_count == summary.record_count:
        return f"unknown (all {summary.record_count} records unpriced)"
    if summary.unpriced_record_count:
        return (
            f"{fmt_usd(summary.total_cost_microusd)} "
            f"(excluding {summary.unpriced_record_count} unpriced)"
        )
    return fmt_usd(summary.total_cost_microusd)


def _breakdown_sort_key(item: tuple[str, UsageSummary]) -> tuple[bool, int, int, str]:
    """Groups with unknown cost lead (most-unknown first), then fully
    priced groups by descending cost, then by name for determinism."""
    name, summary = item
    return (
        summary.unpriced_record_count == 0,
        -summary.unpriced_record_count,
        -summary.total_cost_microusd,
        name,
    )


def _summary_json(summary: UsageSummary) -> dict[str, object]:
    return {
        "record_count": summary.record_count,
        "total_input_tokens": summary.total_input_tokens,
        "total_cache_read_tokens": summary.total_cache_read_tokens,
        "total_cache_write_tokens": summary.total_cache_write_tokens,
        "total_output_tokens": summary.total_output_tokens,
        "total_cost_microusd": summary.total_cost_microusd,
        "total_cost_usd": round(usd(summary.total_cost_microusd), 4),
        "full_rebuild_count": summary.full_rebuild_count,
        "normal_growth_count": summary.normal_growth_count,
        "unpriced_record_count": summary.unpriced_record_count,
        "unpriced_models": list(summary.unpriced_models),
    }


def _window_text(window: tuple[datetime.date | None, datetime.date | None]) -> str:
    since, until = window
    start = since.isoformat() if since else "..."
    end = until.isoformat() if until else "..."
    return f"{start} → {end}"


def _accounting_json(result: ScanResult) -> dict[str, object]:
    tally = result.tally
    parsed = result.parsed
    files = {
        "globbed": tally.globbed,
        "skipped_by_since": tally.skipped_by_since,
        "read_fully": tally.read_fully,
        "failed": tally.failed,
        "missing_agent_metadata": tally.missing_agent_metadata,
        "journal_files_found": tally.journal_files_found,
        "journal_files_read": tally.journal_files_read,
        "journal_files_failed": tally.journal_files_failed,
        "journal_lines": tally.journal_lines,
    }
    lines = {
        "lines_read": parsed.lines_read,
        "blank_lines": parsed.blank_lines,
        "non_assistant_lines": parsed.non_assistant_lines,
        "non_object_json_lines": parsed.non_object_json_lines,
        "lines_without_type_discriminator": parsed.lines_without_type_discriminator,
        "malformed_json_lines": parsed.malformed_json_lines,
        "undecodable_assistant_lines": parsed.undecodable_assistant_lines,
        "synthetic_lines": parsed.synthetic_lines,
        "assistant_lines_without_usage": parsed.assistant_lines_without_usage,
        "assistant_lines_without_request_id": parsed.assistant_lines_without_request_id,
        "tokens_dropped_without_request_id": parsed.tokens_dropped_without_request_id,
        "duplicate_lines_collapsed": parsed.duplicate_lines_collapsed,
        "conflict_lines_discarded": parsed.conflict_lines_discarded,
        "conflicting_request_ids": parsed.conflicting_request_ids,
        "unresolved_conflict_request_ids": parsed.unresolved_conflict_request_ids,
    }
    return {"files": files, "lines": lines}


def _cost_json(
    result: ScanResult,
    records: Sequence[UsageRecord],
    table: PriceTable,
    *,
    window: tuple[datetime.date | None, datetime.date | None],
    sources: list[str],
    by_day: bool,
    now: datetime.datetime,
) -> dict[str, object]:
    totals = summarize(records, table)
    main_thread = summarize(
        [r for r in records if r.origin.scope is TranscriptScope.MAIN_THREAD],
        table,
    )
    agents = summarize([r for r in records if r.origin.scope is TranscriptScope.SUBAGENT], table)
    by_type = summarize_by_agent_type(records, table)
    by_depth = summarize_by_spawn_depth(records, table)
    since, until = window
    payload: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "command": "cost",
        "generated_at": now.isoformat(),
        "window": {
            "since": since.isoformat() if since else None,
            "until": until.isoformat() if until else None,
        },
        "sources": sources,
        "prices": {
            "checked_on": table.checked_on.isoformat(),
            "source": table.source,
        },
        "totals": _summary_json(totals),
        "by_scope": {
            "main_thread": _summary_json(main_thread),
            "subagent": _summary_json(agents),
        },
        "by_agent_type": {
            name: _summary_json(summary)
            for name, summary in sorted(by_type.items(), key=_breakdown_sort_key)
        },
        "by_spawn_depth": {
            str(depth): _summary_json(summary) for depth, summary in sorted(by_depth.items())
        },
        "accounting": _accounting_json(result),
    }
    if by_day:
        payload["by_day"] = [
            {
                "date": day,
                "main_thread_cost_microusd": bucket.main_thread_cost_microusd,
                "main_thread_cost_usd": round(usd(bucket.main_thread_cost_microusd), 4),
                "main_thread_requests": bucket.main_thread_requests,
                "subagent_cost_microusd": bucket.subagent_cost_microusd,
                "subagent_cost_usd": round(usd(bucket.subagent_cost_microusd), 4),
                "subagent_requests": bucket.subagent_requests,
            }
            for day, bucket in summarize_by_day(records, table).items()
        ]
    return payload


def _cost_table_rows(
    records: Sequence[UsageRecord], table: PriceTable
) -> list[tuple[str, UsageSummary]]:
    main_thread = summarize(
        [r for r in records if r.origin.scope is TranscriptScope.MAIN_THREAD],
        table,
    )
    agents = summarize([r for r in records if r.origin.scope is TranscriptScope.SUBAGENT], table)
    return [
        ("Main sessions", main_thread),
        ("Agents", agents),
        ("Combined", summarize(records, table)),
    ]


def _add_summary_table(console: Console, rows: list[tuple[str, UsageSummary]]) -> None:
    table = RichTable(box=None, show_header=True, pad_edge=False)
    table.add_column("Scope", ratio=1)
    table.add_column("Records", justify="right")
    table.add_column("Billed tokens", justify="right")
    table.add_column("Cost", justify="right")
    for label, summary in rows:
        billed = (
            summary.total_input_tokens
            + summary.total_cache_read_tokens
            + summary.total_cache_write_tokens
            + summary.total_output_tokens
        )
        table.add_row(
            label,
            str(summary.record_count),
            f"{billed:,}",
            cost_label(summary),
        )
    console.print(table)


def _add_breakdown_table(
    console: Console,
    title: str,
    column: str,
    groups: dict[str, UsageSummary],
) -> None:
    console.print()
    console.print(Text(title, style="bold"))
    table = RichTable(box=None, show_header=True, pad_edge=False)
    table.add_column(column, ratio=1)
    table.add_column("Records", justify="right")
    table.add_column("Cost", justify="right")
    for name, summary in sorted(groups.items(), key=_breakdown_sort_key):
        table.add_row(name, str(summary.record_count), cost_label(summary))
    console.print(table)


def _add_day_table(console: Console, buckets: dict[str, DayBucket]) -> None:
    table = RichTable(box=None, show_header=True, pad_edge=False)
    table.add_column("Date")
    table.add_column("Main sessions", justify="right")
    table.add_column("Agents", justify="right")
    table.add_column("Total", justify="right")
    for day, bucket in buckets.items():
        table.add_row(
            day,
            _day_cost_label(
                bucket.main_thread_cost_microusd,
                bucket.main_thread_requests,
                bucket.main_thread_unpriced,
            ),
            _day_cost_label(
                bucket.subagent_cost_microusd,
                bucket.subagent_requests,
                bucket.subagent_unpriced,
            ),
            _day_cost_label(
                bucket.total_cost_microusd,
                bucket.total_requests,
                bucket.main_thread_unpriced + bucket.subagent_unpriced,
            ),
        )
    console.print(table)


def _day_cost_label(cost_microusd: int, requests: int, unpriced: int) -> str:
    if requests and unpriced == requests:
        return f"unknown ({requests} unpriced)"
    if unpriced:
        return f"{fmt_usd(cost_microusd)} (excluding {unpriced})"
    return fmt_usd(cost_microusd)


def render_cost(
    result: ScanResult,
    records: Sequence[UsageRecord],
    table: PriceTable,
    *,
    window: tuple[datetime.date | None, datetime.date | None],
    sources: list[str],
    by_day: bool,
    fmt: str,
    now: datetime.datetime,
) -> str:
    """The `cost` report in one of the three output formats."""
    if fmt == "json":
        payload = _cost_json(
            result,
            records,
            table,
            window=window,
            sources=sources,
            by_day=by_day,
            now=now,
        )
        return json.dumps(payload, indent=2, sort_keys=True)

    totals = summarize(records, table)
    if fmt == "markdown":
        return _cost_markdown(records, table, totals, window, by_day)

    out = io.StringIO()
    console = Console(file=out, width=100, highlight=False)
    console.print(Text(f"Cost report · {_window_text(window)} · {totals.record_count:,} records"))
    console.print(Text(f"Sources: {', '.join(sources)}", style="dim"))
    console.print()
    _add_summary_table(console, _cost_table_rows(records, table))
    by_type = summarize_by_agent_type(records, table)
    if by_type:
        _add_breakdown_table(console, "By agent type", "Agent type", by_type)
    by_depth = summarize_by_spawn_depth(records, table)
    if by_depth:
        _add_breakdown_table(
            console,
            "By spawn depth",
            "Depth",
            {str(depth): summary for depth, summary in by_depth.items()},
        )
    if by_day:
        console.print()
        console.print(Text("By day", style="bold"))
        _add_day_table(console, summarize_by_day(records, table))
    console.print()
    if totals.unpriced_record_count:
        console.print(
            Text(
                f"Cost excludes {totals.unpriced_record_count} of "
                f"{totals.record_count} records on "
                f"{len(totals.unpriced_models)} unpriced model(s): "
                f"{', '.join(totals.unpriced_models)}.",
                style="yellow",
            )
        )
    rebuild_share = (
        f" ({totals.full_rebuild_count / totals.record_count:.1%})" if totals.record_count else ""
    )
    console.print(
        Text(
            f"Full-rebuild turns: {totals.full_rebuild_count:,} of "
            f"{totals.record_count:,}{rebuild_share}."
        )
    )
    console.print(
        Text(
            f"Prices checked {table.checked_on.isoformat()} against "
            f"{table.source}. Costs are API list prices.",
            style="dim",
        )
    )
    return out.getvalue()


def _cost_markdown(
    records: Sequence[UsageRecord],
    table: PriceTable,
    totals: UsageSummary,
    window: tuple[datetime.date | None, datetime.date | None],
    by_day: bool,
) -> str:
    lines: list[str] = [
        f"Cost report · {_window_text(window)} · {totals.record_count:,} records",
        "",
        "| Scope | Records | Cost |",
        "|---|---:|---:|",
    ]
    for label, summary in _cost_table_rows(records, table):
        lines.append(f"| {label} | {summary.record_count:,} | {cost_label(summary)} |")
    by_type = summarize_by_agent_type(records, table)
    if by_type:
        lines += ["", "By agent type", "", "| Agent type | Records | Cost |", "|---|---:|---:|"]
        for name, summary in sorted(by_type.items(), key=_breakdown_sort_key):
            lines.append(f"| {name} | {summary.record_count:,} | {cost_label(summary)} |")
    if by_day:
        lines += [
            "",
            "By day",
            "",
            "| Date | Main sessions | Agents | Total |",
            "|---|---:|---:|---:|",
        ]
        for day, bucket in summarize_by_day(records, table).items():
            main = _day_cost_label(
                bucket.main_thread_cost_microusd,
                bucket.main_thread_requests,
                bucket.main_thread_unpriced,
            )
            agents = _day_cost_label(
                bucket.subagent_cost_microusd,
                bucket.subagent_requests,
                bucket.subagent_unpriced,
            )
            unpriced = bucket.main_thread_unpriced + bucket.subagent_unpriced
            total = _day_cost_label(bucket.total_cost_microusd, bucket.total_requests, unpriced)
            lines.append(f"| {day} | {main} | {agents} | {total} |")
    lines += [
        "",
        f"Full-rebuild turns: {totals.full_rebuild_count:,} of {totals.record_count:,}.",
        f"Prices checked {table.checked_on.isoformat()} against {table.source}. "
        f"Costs are API list prices.",
    ]
    return "\n".join(lines) + "\n"


__all__ = [
    "cost_label",
    "fmt_usd",
    "render_cost",
    "render_report",
    "render_runs",
    "usd",
]


# ---- report ---------------------------------------------------------------


def _run_money(microusd: int | None) -> float | None:
    return None if microusd is None else round(usd(microusd), 4)


def _success_cell(group: GroupStats) -> str:
    if group.success_rate is None:
        return "-"
    return f"{group.succeeded}/{group.decided_runs} {group.success_rate:.0%}"


def _verdict_text(verdict: str) -> str:
    return verdict.replace("_", " ")


def _group_json(group: GroupStats) -> dict[str, object]:
    payload = group.model_dump()
    payload["verdict"] = group.verdict.value
    payload["reasons"] = list(group.reasons)
    payload["notes"] = list(group.notes)
    payload["cost_usd"] = _run_money(group.cost_microusd)
    return payload


def _report_json(
    groups: list[GroupStats],
    result: ScanResult,
    totals: UsageSummary,
    table: PriceTable,
    *,
    window: tuple[datetime.date | None, datetime.date | None],
    sources: list[str],
    thresholds: Thresholds,
    total_runs: int,
    agents_cost_microusd: int | None,
    main_cost_microusd: int | None,
    now: datetime.datetime,
) -> dict[str, object]:
    since, until = window
    return {
        "schema_version": SCHEMA_VERSION,
        "command": "report",
        "generated_at": now.isoformat(),
        "window": {
            "since": since.isoformat() if since else None,
            "until": until.isoformat() if until else None,
        },
        "sources": sources,
        "prices": {
            "checked_on": table.checked_on.isoformat(),
            "source": table.source,
        },
        "totals": {
            "agent_runs": total_runs,
            "agents_cost_microusd": agents_cost_microusd,
            "agents_cost_usd": _run_money(agents_cost_microusd),
            "main_sessions_cost_microusd": main_cost_microusd,
            "main_sessions_cost_usd": _run_money(main_cost_microusd),
            "records": totals.record_count,
            "total_input_tokens": totals.total_input_tokens,
            "total_cache_read_tokens": totals.total_cache_read_tokens,
            "total_cache_write_tokens": totals.total_cache_write_tokens,
            "total_output_tokens": totals.total_output_tokens,
            "full_rebuild_count": totals.full_rebuild_count,
            "normal_growth_count": totals.normal_growth_count,
            "unpriced_record_count": totals.unpriced_record_count,
            "unpriced_models": list(totals.unpriced_models),
        },
        "accounting": _accounting_json(result),
        "thresholds": {
            "min_runs": thresholds.min_runs,
            "remove_below": thresholds.remove_below,
            "fix_below": thresholds.fix_below,
            "max_tool_error_rate": thresholds.max_tool_error_rate,
        },
        "groups": [_group_json(group) for group in groups],
    }


def _report_table_lines(
    groups: list[GroupStats],
) -> list[tuple[str, str, str, str, str, str, str]]:
    return [
        (
            group.key,
            str(group.runs),
            _success_cell(group),
            fmt_usd(group.cost_microusd) if group.cost_microusd is not None else "unknown",
            (fmt_usd_maybe(group.cost_per_success_usd)),
            f"{group.tool_error_rate:.0%}",
            _verdict_text(group.verdict.value),
        )
        for group in groups
    ]


def fmt_usd_maybe(value: float | None) -> str:
    return "-" if value is None else f"${value:,.2f}"


def render_report(
    groups: list[GroupStats],
    result: ScanResult,
    totals: UsageSummary,
    table: PriceTable,
    *,
    window: tuple[datetime.date | None, datetime.date | None],
    sources: list[str],
    thresholds: Thresholds,
    total_runs: int,
    agents_cost_microusd: int | None,
    main_cost_microusd: int | None,
    fmt: str,
    now: datetime.datetime,
) -> str:
    """The `report` command: one row per group, with verdicts."""
    if fmt == "json":
        payload = _report_json(
            groups,
            result,
            totals,
            table,
            window=window,
            sources=sources,
            thresholds=thresholds,
            total_runs=total_runs,
            agents_cost_microusd=agents_cost_microusd,
            main_cost_microusd=main_cost_microusd,
            now=now,
        )
        return json.dumps(payload, indent=2, sort_keys=True)

    header = (
        f"Agent scorecard · {_window_text(window)} · {total_runs:,} agent runs · "
        f"agents {fmt_usd(agents_cost_microusd) if agents_cost_microusd is not None else 'unknown'}"
        f" · main sessions "
        f"{fmt_usd(main_cost_microusd) if main_cost_microusd is not None else 'unknown'}"
    )

    if fmt == "markdown":
        lines = [
            header,
            "",
            "| Group | Runs | Success | Cost | $/success | Tool errors | "
            "Verdict | Reasons and notes |",
            "|---|---:|---:|---:|---:|---:|---|---|",
        ]
        for group in groups:
            cells = _report_table_lines([group])[0]
            reason_text = "; ".join((*group.reasons, *group.notes))
            lines.append(
                f"| {cells[0]} | {cells[1]} | {cells[2]} | {cells[3]} | "
                f"{cells[4]} | {cells[5]} | {cells[6]} | {reason_text} |"
            )
        lines += [
            "",
            f"Prices checked {table.checked_on.isoformat()} against "
            f"{table.source}. Costs are API list prices. The verdict is a "
            f"starting point for a human decision, not an automatic kill "
            f"switch.",
        ]
        return "\n".join(lines) + "\n"

    out = io.StringIO()
    console = Console(file=out, width=110, highlight=False)
    console.print(Text(header))
    console.print()
    rich_table = RichTable(box=None, show_header=True, pad_edge=False)
    rich_table.add_column("Group", ratio=1)
    rich_table.add_column("Runs", justify="right")
    rich_table.add_column("Success", justify="right")
    rich_table.add_column("Cost", justify="right")
    rich_table.add_column("$/success", justify="right")
    rich_table.add_column("Tool errors", justify="right")
    rich_table.add_column("Verdict", justify="right")
    for cells in _report_table_lines(groups):
        rich_table.add_row(*cells)
    console.print(rich_table)
    for group in groups:
        for line in (*group.reasons, *group.notes):
            console.print(Text(f"  └ {line}", style="dim"))
    console.print()
    console.print(
        Text(
            f"Prices checked {table.checked_on.isoformat()} against "
            f"{table.source}. Costs are API list prices.",
            style="dim",
        )
    )
    return out.getvalue()


# ---- runs -----------------------------------------------------------------


def _run_json(run: AgentRun, *, show_descriptions: bool) -> dict[str, object]:
    payload: dict[str, object] = {
        "agent_id": run.agent_id,
        "agent_type": run.agent_type,
        "spawn_depth": run.spawn_depth,
        "session_id": run.session_id,
        "parent_agent_id": run.parent_agent_id,
        "started_at": run.started_at.isoformat() if run.started_at else None,
        "ended_at": run.ended_at.isoformat() if run.ended_at else None,
        "duration_seconds": run.duration_seconds,
        "turns": run.turns,
        "cost_microusd": run.cost_microusd,
        "cost_usd": _run_money(run.cost_microusd),
        "unpriced_records": run.unpriced_records,
        "unpriced_models": list(run.unpriced_models),
        "primary_model": run.primary_model,
        "models": list(run.models),
        "tool_calls": run.tool_calls,
        "tool_errors": run.tool_errors,
        "tool_denials": run.tool_denials,
        "edited_files": run.edited_files,
        "test_runs": run.test_runs,
        "last_test_passed": run.last_test_passed,
        "commits": run.commits,
        "cache_read_share": round(run.cache_read_share, 4),
        "full_rebuild_turns": run.full_rebuild_turns,
        "lifecycle": run.lifecycle.value,
        "lifecycle_source": run.lifecycle_source.value,
        "reason": run.reason,
        "outcome": run.outcome.value,
        "outcome_reason": run.outcome_reason,
    }
    if show_descriptions:
        payload["description"] = run.description
    return payload


def render_runs(
    runs: Sequence[AgentRun],
    table: PriceTable,
    *,
    window: tuple[datetime.date | None, datetime.date | None],
    sources: list[str],
    fmt: str,
    now: datetime.datetime,
    show_descriptions: bool,
) -> str:
    """The `runs` command: one row per agent run, newest first."""
    if fmt == "json":
        since, until = window
        payload = {
            "schema_version": SCHEMA_VERSION,
            "command": "runs",
            "generated_at": now.isoformat(),
            "window": {
                "since": since.isoformat() if since else None,
                "until": until.isoformat() if until else None,
            },
            "sources": sources,
            "prices": {
                "checked_on": table.checked_on.isoformat(),
                "source": table.source,
            },
            "runs": [_run_json(run, show_descriptions=show_descriptions) for run in runs],
        }
        return json.dumps(payload, indent=2, sort_keys=True)

    header = f"Agent runs · {_window_text(window)} · {len(runs):,} runs" + (
        " · descriptions shown" if show_descriptions else ""
    )

    if fmt == "markdown":
        lines = [
            header,
            "",
            "| Agent | Started | Turns | Cost | Outcome |",
            "|---|---|---:|---:|---|",
        ]
        for run in runs:
            lines.append(
                f"| {run.agent_type} ({run.agent_id}) | "
                f"{run.started_at.date().isoformat() if run.started_at else '-'} | "
                f"{run.turns} | "
                f"{fmt_usd(run.cost_microusd) if run.cost_microusd is not None else 'unknown'} | "
                f"{_verdict_text(run.outcome.value)} |"
            )
        return "\n".join(lines) + "\n"

    out = io.StringIO()
    console = Console(file=out, width=110, highlight=False)
    console.print(Text(header))
    console.print()
    rich_table = RichTable(box=None, show_header=True, pad_edge=False)
    rich_table.add_column("Agent", ratio=1)
    rich_table.add_column("Started", justify="right")
    rich_table.add_column("Turns", justify="right")
    rich_table.add_column("Cost", justify="right")
    rich_table.add_column("Outcome", justify="right")
    for run in runs:
        cells = [
            f"{run.agent_type} ({run.agent_id})",
            run.started_at.date().isoformat() if run.started_at else "-",
            str(run.turns),
            fmt_usd(run.cost_microusd) if run.cost_microusd is not None else "unknown",
            _verdict_text(run.outcome.value),
        ]
        rich_table.add_row(*cells)
        if show_descriptions and run.description:
            console.print(Text(f"  └ {run.description}", style="dim"))
    console.print(rich_table)
    return out.getvalue()
