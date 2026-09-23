"""Per-run facts: what one agent run did, collected from its own log file.

A run is one agent started once — one `agent-<id>.jsonl` file. The
collector reads the run's tool calls, test runs, commits, edits, and final
report; `build_runs` then joins the de-duplicated records the run's origin
produced to add cost, primary model, and cache share. The final report
text stays in memory only: it feeds the success and failure patterns and is
never printed or saved.
"""

from __future__ import annotations

import datetime
import re
from collections.abc import Iterable

from pydantic import BaseModel, ConfigDict, Field

from agent_scorecard.costs import classify_prefix_rebuild
from agent_scorecard.logfile import DecodedLine, LineKind
from agent_scorecard.models import (
    ParsedTranscripts,
    RawTranscriptLine,
    RawUserLine,
    RebuildKind,
    SubagentOrigin,
    ToolUseBlock,
    UsageRecord,
    as_dict,
    as_str,
    text_blocks,
    tool_result_blocks,
    tool_use_blocks,
)
from agent_scorecard.pricing import PriceTable, estimate_cost

# Tools that change files. A run that never touched a file is expected to
# see failing tests while investigating; only an editing run is judged by
# the state it left the tests in.
EDIT_TOOL_NAMES = frozenset({"Edit", "MultiEdit", "Write", "NotebookEdit"})

# A successful `git commit` in a Bash call counts as a commit made by the
# run; `git -C <path> commit` counts too.
COMMIT_PATTERN = re.compile(r"\bgit\s+(?:-C\s+\S+\s+)?commit\b")

# Shell commands treated as test runs. This is a heuristic — see the README
# caveat — and `--test-command` replaces the whole list.
DEFAULT_TEST_PATTERNS: tuple[str, ...] = (
    r"\bpytest\b",
    r"\bgo\s+test\b",
    r"\b(?:npm|pnpm|yarn|bun)\s+(?:run\s+)?test\b",
    r"\bvitest\b",
    r"\bjest\b",
    r"\bcargo\s+test\b",
    r"\bmake\s+(?:test|check|verify)\b",
    r"\b(?:mvn|gradlew?)\s+test\b",
    r"\bdotnet\s+test\b",
    r"\brspec\b",
    r"\bphpunit\b",
    r"\bmix\s+test\b",
)


def compile_test_patterns(patterns: Iterable[str]) -> tuple[re.Pattern[str], ...]:
    return tuple(re.compile(pattern) for pattern in patterns)


class RunFacts(BaseModel):
    """What the collector read out of one run's file, before pricing."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    agent_id: str
    agent_type: str
    spawn_depth: int
    session_id: str
    parent_agent_id: str | None = None
    started_at: datetime.datetime | None = None
    ended_at: datetime.datetime | None = None
    request_ids: tuple[str, ...] = ()
    models: tuple[str, ...] = ()
    tool_call_ids: tuple[str, ...] = ()
    tool_errors: int = Field(default=0, ge=0)
    tool_denials: int = Field(default=0, ge=0)
    edited_files: bool = False
    test_call_ids: tuple[str, ...] = ()
    test_results: tuple[tuple[str, bool], ...] = ()
    commits: int = Field(default=0, ge=0)
    final_report: str | None = None
    last_stop_reason: str | None = None
    last_stop_is_api_error: bool = False
    last_assistant_is_api_error: bool = False

    @property
    def duration_seconds(self) -> float | None:
        if self.started_at is None or self.ended_at is None:
            return None
        return (self.ended_at - self.started_at).total_seconds()

    @property
    def test_runs(self) -> int:
        """How many test runs were decided: calls that matched a test
        pattern, were not denied, and got a result."""
        return len(self.test_results)


class RunCollector:
    """Folds one run's decoded lines into its facts."""

    def __init__(
        self,
        *,
        agent_id: str,
        agent_type: str,
        spawn_depth: int,
        session_id: str,
        parent_agent_id: str | None = None,
        test_patterns: tuple[re.Pattern[str], ...] = (),
    ) -> None:
        self._agent_id = agent_id
        self._agent_type = agent_type
        self._spawn_depth = spawn_depth
        self._session_id = session_id
        self._parent_agent_id = parent_agent_id
        self._test_patterns = test_patterns
        self._started_at: datetime.datetime | None = None
        self._ended_at: datetime.datetime | None = None
        self._request_ids: list[str] = []
        self._models: set[str] = set()
        self._tool_call_ids: list[str] = []
        self._tool_errors = 0
        self._tool_denials = 0
        self._edited_files = False
        self._test_calls: dict[str, None] = {}
        self._commit_calls: dict[str, None] = {}
        self._test_results: dict[str, bool] = {}
        self._commits = 0
        self._final_report: str | None = None
        self._last_stop_reason: str | None = None
        self._last_stop_is_api_error = False
        self._last_assistant_is_api_error = False

    def feed(self, line: DecodedLine) -> None:
        if line.timestamp is not None:
            if self._started_at is None:
                self._started_at = line.timestamp
            self._ended_at = line.timestamp
        if line.kind is LineKind.ASSISTANT and line.assistant is not None:
            self._feed_assistant(line.assistant)
        elif line.kind is LineKind.NON_ASSISTANT and line.user is not None:
            self._feed_user(line.user)

    def finish(self) -> RunFacts:
        # `test_results` keyed by call id; the ordered `test_calls` dict
        # decides which result was the run's last.
        decided = [
            (call_id, self._test_results[call_id])
            for call_id in self._test_calls
            if call_id in self._test_results
        ]
        return RunFacts(
            agent_id=self._agent_id,
            agent_type=self._agent_type,
            spawn_depth=self._spawn_depth,
            session_id=self._session_id,
            parent_agent_id=self._parent_agent_id,
            started_at=self._started_at,
            ended_at=self._ended_at,
            request_ids=tuple(self._request_ids),
            models=tuple(sorted(self._models)),
            tool_call_ids=tuple(self._tool_call_ids),
            tool_errors=self._tool_errors,
            tool_denials=self._tool_denials,
            edited_files=self._edited_files,
            test_call_ids=tuple(self._test_calls),
            test_results=tuple(decided),
            commits=self._commits,
            final_report=self._final_report,
            last_stop_reason=self._last_stop_reason,
            last_stop_is_api_error=self._last_stop_is_api_error,
            last_assistant_is_api_error=self._last_assistant_is_api_error,
        )

    def _feed_assistant(self, entry: RawTranscriptLine) -> None:
        if entry.message.model != "<synthetic>":
            self._models.add(entry.message.model)
        if entry.request_id and entry.request_id not in self._request_ids:
            self._request_ids.append(entry.request_id)
        self._last_assistant_is_api_error = entry.is_api_error_message
        if entry.message.stop_reason is not None:
            self._last_stop_reason = entry.message.stop_reason
            self._last_stop_is_api_error = entry.is_api_error_message
        for block in tool_use_blocks(entry.message.content):
            self._feed_tool_use(block)
        texts = text_blocks(entry.message.content)
        if texts:
            self._final_report = texts[-1]

    def _feed_tool_use(self, block: ToolUseBlock) -> None:
        if block.id not in self._tool_call_ids:
            self._tool_call_ids.append(block.id)
        if block.name in EDIT_TOOL_NAMES:
            self._edited_files = True
        if block.name != "Bash":
            return
        input_dict = as_dict(block.input)
        command = as_str(input_dict.get("command")) if input_dict else None
        if command is None:
            return
        if any(pattern.search(command) for pattern in self._test_patterns):
            self._test_calls.setdefault(block.id)
        if COMMIT_PATTERN.search(command):
            self._commit_calls.setdefault(block.id)

    def _feed_user(self, user: RawUserLine) -> None:
        denial = user.tool_denial_kind is not None
        for result in tool_result_blocks(user.message.content):
            if denial:
                self._tool_denials += 1
                # A denied call never ran, so it cannot count as a test run
                # or a commit.
                self._test_calls.pop(result.tool_use_id, None)
                self._commit_calls.pop(result.tool_use_id, None)
                self._test_results.pop(result.tool_use_id, None)
                continue
            if result.is_error:
                self._tool_errors += 1
            elif result.tool_use_id in self._commit_calls:
                self._commits += 1
            if result.tool_use_id in self._test_calls:
                # A failing test run is still a run: last_test_passed reads
                # `not is_error` off the result either way.
                self._test_results[result.tool_use_id] = not result.is_error


class AgentRun(BaseModel):
    """One agent run: the collected facts joined with the records priced
    from the run's own file. `final_report` is deliberately excluded — it
    is consumed at classify time and never leaves memory."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    agent_id: str
    agent_type: str
    spawn_depth: int
    session_id: str
    parent_agent_id: str | None = None
    started_at: datetime.datetime | None = None
    ended_at: datetime.datetime | None = None
    duration_seconds: float | None = None
    turns: int = Field(default=0, ge=0)
    cost_microusd: int | None = None
    unpriced_records: int = Field(default=0, ge=0)
    unpriced_models: tuple[str, ...] = ()
    primary_model: str | None = None
    models: tuple[str, ...] = ()
    tool_calls: int = Field(default=0, ge=0)
    tool_errors: int = Field(default=0, ge=0)
    tool_denials: int = Field(default=0, ge=0)
    edited_files: bool = False
    test_runs: int = Field(default=0, ge=0)
    last_test_passed: bool | None = None
    commits: int = Field(default=0, ge=0)
    cache_read_share: float = Field(default=0.0, ge=0.0)
    full_rebuild_turns: int = Field(default=0, ge=0)


def build_runs(
    facts: Iterable[RunFacts], parsed: ParsedTranscripts, table: PriceTable
) -> tuple[AgentRun, ...]:
    """Join each run's facts with its own records and price them.

    A run's cost is None when any of its records is unpriced: a total that
    silently dropped a record would understate exactly the runs worth
    looking at. The primary model is the one with the most billed tokens,
    ties broken alphabetically.
    """
    records_by_agent: dict[str, list[UsageRecord]] = {}
    for record in parsed.records:
        origin = record.origin
        if isinstance(origin, SubagentOrigin):
            records_by_agent.setdefault(origin.agent_id, []).append(record)

    runs: list[AgentRun] = []
    for facts_item in facts:
        records = records_by_agent.get(facts_item.agent_id, [])
        cost_microusd: int | None = 0
        unpriced_records = 0
        unpriced_models: set[str] = set()
        billed_by_model: dict[str, int] = {}
        cache_read = 0
        billed_input = 0
        full_rebuilds = 0
        for record in records:
            cost = estimate_cost(record, table)
            if cost is None:
                unpriced_records += 1
                unpriced_models.add(record.model)
            else:
                cost_microusd += cost
            billed_by_model[record.model] = (
                billed_by_model.get(record.model, 0) + record.billed_tokens
            )
            cache_read += record.cache_read_tokens
            billed_input += record.billed_input_tokens
            if classify_prefix_rebuild(record) is RebuildKind.FULL_REBUILD:
                full_rebuilds += 1
        if unpriced_records:
            cost_microusd = None
        primary_model = (
            min(billed_by_model, key=lambda m: (-billed_by_model[m], m))
            if billed_by_model
            else None
        )
        runs.append(
            AgentRun(
                agent_id=facts_item.agent_id,
                agent_type=facts_item.agent_type,
                spawn_depth=facts_item.spawn_depth,
                session_id=facts_item.session_id,
                parent_agent_id=facts_item.parent_agent_id,
                started_at=facts_item.started_at,
                ended_at=facts_item.ended_at,
                duration_seconds=facts_item.duration_seconds,
                turns=len(facts_item.request_ids),
                cost_microusd=cost_microusd,
                unpriced_records=unpriced_records,
                unpriced_models=tuple(sorted(unpriced_models)),
                primary_model=primary_model,
                models=facts_item.models,
                tool_calls=len(facts_item.tool_call_ids),
                tool_errors=facts_item.tool_errors,
                tool_denials=facts_item.tool_denials,
                edited_files=facts_item.edited_files,
                test_runs=facts_item.test_runs,
                last_test_passed=_last_test_passed(facts_item),
                commits=facts_item.commits,
                cache_read_share=(cache_read / billed_input) if billed_input else 0.0,
                full_rebuild_turns=full_rebuilds,
            )
        )
    return tuple(runs)


def _last_test_passed(facts: RunFacts) -> bool | None:
    """`not is_error` for the run's last decided test run, or None when
    there were none."""
    if not facts.test_results:
        return None
    return facts.test_results[-1][1]
