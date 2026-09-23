"""The argparse entry point: commands, exit codes, refusal handling. No
business logic lives here.

Exit codes: 0 for success (including "nothing found in this window", which
is printed as a clear message, never a table of zeros); 1 when a refusal
gate fires or an input path or price file is unusable; 2 for usage errors,
which is argparse's default.
"""

from __future__ import annotations

import argparse
import datetime
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

from agent_scorecard import render
from agent_scorecard.costs import records_in_window
from agent_scorecard.locate import transcript_dir_for_project
from agent_scorecard.pricing import PriceFileError, load_prices
from agent_scorecard.scan import accounting_lines, refusal_reason, scan

logger = logging.getLogger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="agent-scorecard",
        description=(
            "Measure what each Claude Code agent costs and delivers, then keep, fix, or remove it."
        ),
    )
    parser.add_argument(
        "command",
        nargs="?",
        default="report",
        choices=["report", "runs", "cost"],
        help="report (default): scorecard per agent type; runs: one row per "
        "agent run; cost: the cost report",
    )
    parser.add_argument(
        "--project",
        type=Path,
        default=Path.cwd(),
        help="project whose logs to read (default: the current directory)",
    )
    parser.add_argument(
        "--transcripts",
        type=Path,
        action="append",
        default=None,
        help="read this log folder directly; can be repeated; overrides --project",
    )
    parser.add_argument(
        "--since",
        type=datetime.date.fromisoformat,
        default=None,
        help="inclusive start date (YYYY-MM-DD, UTC)",
    )
    parser.add_argument(
        "--until",
        type=datetime.date.fromisoformat,
        default=None,
        help="inclusive end date (YYYY-MM-DD, UTC)",
    )
    parser.add_argument(
        "--prices",
        type=Path,
        default=None,
        help="TOML price file merged over the bundled default prices",
    )
    parser.add_argument(
        "--format",
        choices=["table", "markdown", "json"],
        default="table",
        help="output format (default: table)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="print file and line accounting to stderr",
    )
    report = parser.add_argument_group("report options")
    report.add_argument(
        "--group-by",
        choices=["agent-type", "model", "agent-type+model"],
        default="agent-type",
        help="how runs are grouped (default: agent-type)",
    )
    report.add_argument(
        "--min-runs",
        type=int,
        default=5,
        help="fewest decided runs a group needs before it gets a verdict",
    )
    report.add_argument(
        "--remove-below",
        type=float,
        default=0.5,
        help="success rate below which the verdict is remove",
    )
    report.add_argument(
        "--fix-below",
        type=float,
        default=0.8,
        help="success rate below which the verdict is fix",
    )
    report.add_argument(
        "--max-tool-error-rate",
        type=float,
        default=0.2,
        help="tool error rate above which the verdict is at least fix",
    )
    report.add_argument(
        "--success-pattern",
        default=None,
        help="regex matching an agent's final report to mark the run a success",
    )
    report.add_argument(
        "--failure-pattern",
        default=None,
        help="regex matching an agent's final report to mark the run a failure",
    )
    report.add_argument(
        "--test-command",
        action="append",
        default=None,
        help="regex treating matching shell commands as test runs; can be "
        "repeated; replaces the built-in list",
    )
    runs = parser.add_argument_group("runs options")
    runs.add_argument("--type", dest="run_type", default=None, help="filter by agent type")
    runs.add_argument(
        "--outcome",
        choices=["succeeded", "failed", "failed_tests", "unknown"],
        default=None,
        help="filter by outcome",
    )
    runs.add_argument(
        "--limit",
        type=int,
        default=50,
        help="newest N runs to show (default: 50)",
    )
    runs.add_argument(
        "--show-descriptions",
        action="store_true",
        help="print each run's short task description (off by default for privacy)",
    )
    cost = parser.add_argument_group("cost options")
    cost.add_argument(
        "--by-day",
        action="store_true",
        help="also show per-day spend, split into main sessions and agents",
    )
    return parser


def _resolve_roots(args: argparse.Namespace) -> list[Path] | str:
    """The log folders to read, or an error message when the input is
    unusable."""
    if args.transcripts:
        roots = list(args.transcripts)
    else:
        roots = [transcript_dir_for_project(args.project)]
    for root in roots:
        if not root.is_dir():
            return (
                f"Log folder does not exist: {root}. Pass --project with the "
                f"project's path, or --transcripts with the log folder itself."
            )
    return roots


def _fail(message: str, accounting: list[str] | None = None) -> int:
    print(message, file=sys.stderr)
    if accounting is not None:
        for line in accounting:
            print(line, file=sys.stderr)
    return 1


def main(argv: Sequence[str] | None = None, *, now: datetime.datetime | None = None) -> int:
    """CLI entry point; returns the process exit code."""
    args = _build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s: %(message)s",
        stream=sys.stderr,
    )
    clock = now if now is not None else datetime.datetime.now(datetime.UTC)

    roots = _resolve_roots(args)
    if isinstance(roots, str):
        return _fail(roots)
    try:
        table = load_prices(args.prices)
    except PriceFileError as exc:
        return _fail(str(exc))

    result = scan(roots, since=args.since)
    refusal = refusal_reason(result)
    if refusal is not None:
        return _fail(refusal, accounting_lines(result))
    if args.verbose:
        for line in accounting_lines(result):
            print(line, file=sys.stderr)

    if args.command == "cost":
        records = records_in_window(result.parsed.records, args.since, args.until)
        if not records:
            print("No usage records fall in this window.")
            return 0
        print(
            render.render_cost(
                result,
                records,
                table,
                window=(args.since, args.until),
                sources=[str(root) for root in roots],
                by_day=args.by_day,
                fmt=args.format,
                now=clock,
            )
        )
        return 0

    print(
        f"{args.command} is not implemented yet in this build.",
        file=sys.stderr,
    )
    return 1
