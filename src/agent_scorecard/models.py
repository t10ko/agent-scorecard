"""Pydantic models: the raw Claude Code log-line shapes and the domain types
derived from them.

The `Raw*` models mirror the log's own field names. Everything else is a
domain type this tool owns. The accounting types (`FileTally`,
`ParsedTranscripts`) make the scan report its own denominator, so a reader
can reconcile every discarded line against the total read.
"""

from __future__ import annotations

import datetime
from enum import StrEnum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, ValidationError, model_validator


class TranscriptScope(StrEnum):
    """Which kind of file a line came from: the session's own log, or one
    of the subagent logs under `<session-id>/subagents/`."""

    MAIN_THREAD = "main_thread"
    SUBAGENT = "subagent"


class MainThreadOrigin(BaseModel):
    """A line from a session's own top-level log file.

    Deliberately field-free rather than a nullable agent block on one shared
    model: a main-session line has no agent type, spawn depth, or parent, and
    making that structural means no consumer can read one off a record that
    never had it.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    scope: Literal[TranscriptScope.MAIN_THREAD] = TranscriptScope.MAIN_THREAD


class SubagentOrigin(BaseModel):
    """A line from one subagent's log file, attributed through the
    `agent-<id>.meta.json` sidecar written beside it.

    `agent_type` and `spawn_depth` are the two axes spend breaks down by.
    `parent_agent_id` is absent on most sidecars, so it is optional and
    reconstructs only the part of the tree Claude Code actually recorded.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    scope: Literal[TranscriptScope.SUBAGENT] = TranscriptScope.SUBAGENT
    agent_id: str = Field(min_length=1)
    agent_type: str = Field(min_length=1)
    spawn_depth: int = Field(ge=1)
    parent_agent_id: str | None = None


RecordOrigin = Annotated[MainThreadOrigin | SubagentOrigin, Field(discriminator="scope")]


class Lifecycle(StrEnum):
    """How a run ended."""

    COMPLETED = "completed"
    FAILED = "failed"
    KILLED = "killed"
    STOPPED = "stopped"
    UNKNOWN = "unknown"


class LifecycleSource(StrEnum):
    """Where the lifecycle answer came from; `inferred` marks a guess from
    the run's own file rather than a recorded outcome."""

    FOREGROUND = "foreground"
    NOTIFICATION = "notification"
    WORKFLOW_JOURNAL = "workflow_journal"
    SIDECAR = "sidecar"
    INFERRED = "inferred"
    NONE = "none"


class Outcome(StrEnum):
    """Whether a run delivered: `failed_tests` means it edited files and
    left the tests red; `unknown` runs are left out of every rate."""

    SUCCEEDED = "succeeded"
    FAILED = "failed"
    FAILED_TESTS = "failed_tests"
    UNKNOWN = "unknown"


class Verdict(StrEnum):
    """The scorecard's recommendation for one group of runs. A starting
    point for a human decision, never an automatic kill switch."""

    REMOVE = "remove"
    FIX = "fix"
    KEEP = "keep"
    NOT_ENOUGH_DATA = "not_enough_data"


class RebuildKind(StrEnum):
    """Whether a request's prompt-cache write looks like a full context-prefix
    rebuild or normal incremental growth on an existing cache."""

    FULL_REBUILD = "full_rebuild"
    NORMAL_GROWTH = "normal_growth"


class UsageRecord(BaseModel):
    """One de-duplicated API request's token usage: one row per `requestId`.

    Claude Code writes one log line per content block of a single API turn,
    and every one of those lines repeats the same `requestId`; this is the
    single record that survives de-duplication. `timestamp` is the first-seen
    line's own timestamp; duplicate removal never compares it, because the
    lines of one turn can carry slightly different wall-clock times.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    request_id: str = Field(min_length=1)
    timestamp: datetime.datetime
    model: str
    input_tokens: int = Field(ge=0)
    cache_read_tokens: int = Field(ge=0)
    cache_write_1h_tokens: int = Field(ge=0)
    cache_write_5m_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    origin: RecordOrigin = MainThreadOrigin()

    @property
    def cache_write_tokens(self) -> int:
        """Cache-write tokens across both TTL buckets."""
        return self.cache_write_1h_tokens + self.cache_write_5m_tokens

    @property
    def billed_input_tokens(self) -> int:
        """Every token billed as input: plain, cache-read, and cache-write."""
        return self.input_tokens + self.cache_read_tokens + self.cache_write_tokens

    @property
    def billed_tokens(self) -> int:
        """Every billed token, input and output."""
        return self.billed_input_tokens + self.output_tokens

    def same_usage(self, other: UsageRecord) -> bool:
        """True when the two copies of one request report identical billing.

        Timestamp is deliberately excluded: the lines of one turn can carry
        slightly different times, and comparing them would turn harmless
        duplicates into conflicts.
        """
        return (
            self.origin == other.origin
            and self.model == other.model
            and self.input_tokens == other.input_tokens
            and self.cache_read_tokens == other.cache_read_tokens
            and self.cache_write_1h_tokens == other.cache_write_1h_tokens
            and self.cache_write_5m_tokens == other.cache_write_5m_tokens
            and self.output_tokens == other.output_tokens
        )


class UsageSummary(BaseModel):
    """Aggregate token, cost, and rebuild-kind counts across a set of
    `UsageRecord`s.

    `total_cost_microusd` covers only records whose model has a price row.
    `unpriced_record_count` and `unpriced_models` report what it excludes,
    so a reader can tell a genuinely cheap scan from one that silently
    priced half its records at nothing.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    record_count: int = Field(ge=0)
    total_input_tokens: int = Field(ge=0)
    total_cache_read_tokens: int = Field(ge=0)
    total_cache_write_tokens: int = Field(ge=0)
    total_output_tokens: int = Field(ge=0)
    total_cost_microusd: int = Field(ge=0)
    full_rebuild_count: int = Field(ge=0)
    normal_growth_count: int = Field(ge=0)
    unpriced_record_count: int = Field(ge=0)
    unpriced_models: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _every_record_classifies_exactly_once(self) -> UsageSummary:
        classified = self.full_rebuild_count + self.normal_growth_count
        if classified != self.record_count:
            raise ValueError(
                f"rebuild-kind counts must partition the records: "
                f"{self.full_rebuild_count} full_rebuild + "
                f"{self.normal_growth_count} normal_growth != "
                f"{self.record_count} records"
            )
        if self.unpriced_record_count > self.record_count:
            raise ValueError(
                f"unpriced_record_count {self.unpriced_record_count} exceeds "
                f"record_count {self.record_count}"
            )
        return self


class RawCacheCreation(BaseModel):
    """Raw `cache_creation` object inside a log line's `usage`: the split of
    cache-write tokens by cache lifetime (TTL).

    `extra="forbid"` where every sibling raw model ignores extras, because
    this is the one closed object: every key it may carry is a bucket of
    billable cache-write tokens, so a key this parser does not know is by
    definition tokens it would price at zero. Real logs (about 3,600 files,
    August 2026) carry exactly these two keys on every line.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    ephemeral_1h_input_tokens: int = Field(default=0, ge=0)
    ephemeral_5m_input_tokens: int = Field(default=0, ge=0)

    @property
    def total(self) -> int:
        """Cache-write tokens across both TTL buckets."""
        return self.ephemeral_1h_input_tokens + self.ephemeral_5m_input_tokens


class RawUsage(BaseModel):
    """Raw `usage` object inside one log line's `message`.

    All four token counts are required: an absent one is a schema change
    worth surfacing as a skipped line, never a silent zero that would report
    a confident $0.0000. `cache_creation_input_tokens` is the flat cache-write
    total; `cache_creation` carries the authoritative split by TTL — on real
    lines the split is sometimes positive while the flat total reads 0, so
    the two are not interchangeable.
    """

    model_config = ConfigDict(frozen=True, extra="ignore")

    input_tokens: int = Field(ge=0)
    cache_creation_input_tokens: int = Field(ge=0)
    cache_read_input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    cache_creation: RawCacheCreation | None = None

    @model_validator(mode="after")
    def _ttl_split_does_not_lose_tokens(self) -> RawUsage:
        """Reject a line whose TTL split reports FEWER tokens than the flat
        total beside it.

        Both buckets default to 0, so a `cache_creation` object with no
        recognizable bucket beside a POSITIVE flat total would otherwise
        price that write at zero. An unknown bucket beside a flat total of 0
        slips past this guard; `RawCacheCreation`'s `extra="forbid"` catches
        it — neither check subsumes the other.
        """
        if self.cache_creation is None:
            return self
        if self.cache_creation.total < self.cache_creation_input_tokens:
            raise ValueError(
                f"cache_creation TTL split totals "
                f"{self.cache_creation.total} but cache_creation_input_tokens "
                f"is {self.cache_creation_input_tokens}; a TTL bucket this "
                f"parser does not know about would price at zero"
            )
        return self


class RawMessage(BaseModel):
    """Raw `message` object inside one log line."""

    model_config = ConfigDict(frozen=True, extra="ignore")

    model: str
    id: str | None = None
    usage: RawUsage | None = None
    stop_reason: str | None = None
    content: JsonValue | None = None


class RawTranscriptLine(BaseModel):
    """Raw shape of one `assistant` log line, validated before any
    `UsageRecord` is derived from it.

    Only lines already known to be `type: "assistant"` are validated against
    this model, so a validation failure means "an assistant line we could not
    decode" rather than the expected noise of every user line. The
    `timestamp` is required: every real line carries one, and a line without
    it cannot be placed in a time window.
    """

    model_config = ConfigDict(frozen=True, extra="ignore")

    type: Literal["assistant"]
    timestamp: datetime.datetime
    request_id: str | None = Field(default=None, alias="requestId")
    is_api_error_message: bool = Field(default=False, alias="isApiErrorMessage")
    message: RawMessage


class RawAgentMeta(BaseModel):
    """Raw shape of the `agent-<id>.meta.json` sidecar written beside every
    subagent log file.

    `agentType` and `spawnDepth` are required because a run whose type and
    depth cannot be read cannot be attributed, and an unattributable file is
    counted and skipped rather than folded anonymously into a total.
    `toolUseId` links the run to the `Agent` tool call that started it;
    `description` and `stoppedByUser` feed lifecycle reporting.
    """

    model_config = ConfigDict(frozen=True, extra="ignore")

    agent_type: str = Field(min_length=1, alias="agentType")
    spawn_depth: int = Field(ge=1, alias="spawnDepth")
    parent_agent_id: str | None = Field(default=None, alias="parentAgentId")
    description: str | None = None
    tool_use_id: str | None = Field(default=None, alias="toolUseId")
    stopped_by_user: bool = Field(default=False, alias="stoppedByUser")


class RawOrigin(BaseModel):
    """Raw `origin` object on a log line; only its `kind` matters here."""

    model_config = ConfigDict(frozen=True, extra="ignore")

    kind: str | None = None


class RawUserMessage(BaseModel):
    """Raw `message` object inside one `user` log line."""

    model_config = ConfigDict(frozen=True, extra="ignore")

    role: str | None = None
    content: JsonValue | None = None


class RawUserLine(BaseModel):
    """Raw shape of one `user` log line: tool results, background-task
    notifications, and the `toolUseResult` object a parent writes when an
    agent it waited for came back."""

    model_config = ConfigDict(frozen=True, extra="ignore")

    type: Literal["user"]
    timestamp: datetime.datetime
    message: RawUserMessage
    tool_use_result: JsonValue | None = Field(default=None, alias="toolUseResult")
    tool_denial_kind: str | None = Field(default=None, alias="toolDenialKind")
    origin: RawOrigin | None = None


class ToolUseBlock(BaseModel):
    """One `tool_use` content block: a tool call made by the model."""

    model_config = ConfigDict(frozen=True, extra="ignore")

    id: str
    name: str
    input: JsonValue = Field(default_factory=dict)


class ToolResultBlock(BaseModel):
    """One `tool_result` content block: the result of a tool call."""

    model_config = ConfigDict(frozen=True, extra="ignore")

    tool_use_id: str
    is_error: bool = False
    content: JsonValue | None = None


def content_blocks(content: JsonValue | None) -> tuple[dict[str, JsonValue], ...]:
    """A line's `message.content` as a tuple of content blocks. A string
    content — how tool results and notifications sometimes arrive — has no
    blocks."""
    if not isinstance(content, list):
        return ()
    return tuple(block for block in content if isinstance(block, dict))


def tool_use_blocks(content: JsonValue | None) -> tuple[ToolUseBlock, ...]:
    """The `tool_use` blocks inside one line's content. A streamed partial
    can repeat a block, so callers deduplicate by block id."""
    blocks: list[ToolUseBlock] = []
    for raw in content_blocks(content):
        if raw.get("type") == "tool_use":
            try:
                blocks.append(ToolUseBlock.model_validate(raw))
            except ValidationError:
                continue
    return tuple(blocks)


def tool_result_blocks(content: JsonValue | None) -> tuple[ToolResultBlock, ...]:
    """The `tool_result` blocks inside one line's content."""
    blocks: list[ToolResultBlock] = []
    for raw in content_blocks(content):
        if raw.get("type") == "tool_result":
            try:
                blocks.append(ToolResultBlock.model_validate(raw))
            except ValidationError:
                continue
    return tuple(blocks)


def text_blocks(content: JsonValue | None) -> tuple[str, ...]:
    """The text of every text block inside one line's content. A bare
    string content counts as one text block."""
    if isinstance(content, str):
        return (content,)
    texts: list[str] = []
    for raw in content_blocks(content):
        if raw.get("type") == "text" and isinstance(raw.get("text"), str):
            texts.append(str(raw["text"]))
    return tuple(texts)


def as_dict(value: JsonValue | None) -> dict[str, JsonValue] | None:
    if isinstance(value, dict):
        return value
    return None


def as_str(value: JsonValue | None) -> str | None:
    if isinstance(value, str):
        return value
    return None


class FileTally(BaseModel):
    """How many log files a scan found, skipped, and read.

    Mutable by design: the file iterator is a generator, so it cannot return
    this alongside the lines it yields. The caller owns the tally and reads
    it once the generator is exhausted. Journals are counted separately and
    sit outside the `globbed` reconciliation, because they are read for
    lifecycle only and are not agent logs.
    """

    model_config = ConfigDict(extra="forbid")

    globbed: int = Field(default=0, ge=0)
    skipped_by_since: int = Field(default=0, ge=0)
    read_fully: int = Field(default=0, ge=0)
    failed: int = Field(default=0, ge=0)
    missing_agent_metadata: int = Field(default=0, ge=0)
    journal_files_found: int = Field(default=0, ge=0)
    journal_files_read: int = Field(default=0, ge=0)
    journal_files_failed: int = Field(default=0, ge=0)
    journal_lines: int = Field(default=0, ge=0)


class ParsedTranscripts(BaseModel):
    """De-duplicated records plus everything discarded reaching them.

    `lines_read` plus the per-reason counters let a reader reconstruct the
    whole input: every line read falls into exactly one counter, is a kept
    record, or was collapsed into one as an identical duplicate. The
    validator fails if they do not add up. `synthetic_lines` counts turns
    Claude Code generated locally (`message.model` == `<synthetic>`); they
    were never billed API calls, so they are counted apart, not as unpriced.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    records: tuple[UsageRecord, ...] = ()
    lines_read: int = Field(default=0, ge=0)
    blank_lines: int = Field(default=0, ge=0)
    non_assistant_lines: int = Field(default=0, ge=0)
    non_object_json_lines: int = Field(default=0, ge=0)
    lines_without_type_discriminator: int = Field(default=0, ge=0)
    malformed_json_lines: int = Field(default=0, ge=0)
    undecodable_assistant_lines: int = Field(default=0, ge=0)
    synthetic_lines: int = Field(default=0, ge=0)
    assistant_lines_without_usage: int = Field(default=0, ge=0)
    assistant_lines_without_request_id: int = Field(default=0, ge=0)
    tokens_dropped_without_request_id: int = Field(default=0, ge=0)
    duplicate_lines_collapsed: int = Field(default=0, ge=0)
    conflict_lines_discarded: int = Field(default=0, ge=0)
    conflicting_request_ids: int = Field(default=0, ge=0)
    unresolved_conflict_request_ids: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def _counters_account_for_every_line(self) -> ParsedTranscripts:
        accounted = (
            self.blank_lines
            + self.non_assistant_lines
            + self.non_object_json_lines
            + self.lines_without_type_discriminator
            + self.malformed_json_lines
            + self.undecodable_assistant_lines
            + self.synthetic_lines
            + self.assistant_lines_without_usage
            + self.assistant_lines_without_request_id
            + self.duplicate_lines_collapsed
            + self.conflict_lines_discarded
            + len(self.records)
        )
        if accounted != self.lines_read:
            raise ValueError(
                f"counters account for {accounted} line(s) but {self.lines_read} "
                f"were read; the reported denominator would be wrong"
            )
        return self
