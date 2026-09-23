"""agent-scorecard: measure what each Claude Code agent costs and delivers.

Reads the session logs Claude Code already saves on this machine and answers
three questions per kind of agent: what does it cost, does it deliver, and
should the team keep it, fix it, or remove it. The exports below are a
curated public surface for library use; the CLI (`agent_scorecard.cli`) is
the primary interface.
"""

from __future__ import annotations

from agent_scorecard.costs import (
    classify_prefix_rebuild,
    records_in_scope,
    records_in_window,
    summarize,
    summarize_by_agent_type,
    summarize_by_day,
    summarize_by_spawn_depth,
)
from agent_scorecard.lifecycle import attach_lifecycles, resolve_lifecycle
from agent_scorecard.models import (
    FileTally,
    Lifecycle,
    LifecycleSource,
    MainThreadOrigin,
    Outcome,
    ParsedTranscripts,
    RebuildKind,
    SubagentOrigin,
    TranscriptScope,
    UsageRecord,
    UsageSummary,
    Verdict,
)
from agent_scorecard.outcome import attach_outcomes, classify_outcome
from agent_scorecard.pricing import PriceTable, estimate_cost, load_prices
from agent_scorecard.runs import AgentRun, RunFacts, build_runs
from agent_scorecard.scan import ScanResult, scan
from agent_scorecard.scorecard import GroupBy, GroupStats, Thresholds, build_scorecard

__version__ = "0.1.0"

__all__ = [
    "AgentRun",
    "FileTally",
    "GroupBy",
    "GroupStats",
    "Lifecycle",
    "LifecycleSource",
    "MainThreadOrigin",
    "Outcome",
    "ParsedTranscripts",
    "PriceTable",
    "RebuildKind",
    "RunFacts",
    "ScanResult",
    "SubagentOrigin",
    "Thresholds",
    "TranscriptScope",
    "UsageRecord",
    "UsageSummary",
    "Verdict",
    "__version__",
    "attach_lifecycles",
    "attach_outcomes",
    "build_runs",
    "build_scorecard",
    "classify_outcome",
    "classify_prefix_rebuild",
    "estimate_cost",
    "load_prices",
    "records_in_scope",
    "records_in_window",
    "resolve_lifecycle",
    "scan",
    "summarize",
    "summarize_by_agent_type",
    "summarize_by_day",
    "summarize_by_spawn_depth",
]
