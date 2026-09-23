"""Decodes validated assistant lines into de-duplicated `UsageRecord`s.

Claude Code writes one log line per content block of a single API turn, and
every one of those lines repeats the same `requestId`, so naive summing
overcounts every multi-block turn. De-duplication keeps, per `requestId`,
the copy reporting the most billed tokens: in the author's logs (about 3,600
files, August 2026) subagent turns are written twice — a streamed partial
and then the final count, where the final is always the larger — and
session replays are strict subsets of the original.

Two disagreeing copies that tie on billed tokens cannot be separated that
way; the tie is counted as unresolved and logged rather than letting file
order decide silently.

Every discarded line is counted by reason, and the accounting closes so the
counters can be reconciled against the input.
"""

from __future__ import annotations

import logging

from pydantic import ValidationError

from agent_scorecard.logfile import DecodedLine, LineKind
from agent_scorecard.models import (
    ParsedTranscripts,
    RawTranscriptLine,
    RawUsage,
    RecordOrigin,
    UsageRecord,
)

logger = logging.getLogger(__name__)

# Per-kind warning budget: under the schema changes these warnings exist to
# surface, nearly every line takes that branch, and one warning per line
# buries the counters that report the same fact once. Budgeted per DISTINCT
# failure shape, so a second, unrelated break stays visible.
_MAX_WARNINGS_PER_KIND = 3

# Claude Code's placeholder model for turns it generated locally, such as an
# API error notice: never a billed API call, so never a record and never
# "unpriced".
_SYNTHETIC_MODEL = "<synthetic>"


def _cache_write_split(usage: RawUsage) -> tuple[int, int]:
    """Split a line's cache-write total into (1h, 5m) TTL buckets.

    The `cache_creation` breakdown wins over the flat
    `cache_creation_input_tokens` beside it whenever both are present: real
    lines sometimes report a flat total of 0 alongside an intact breakdown.
    When the breakdown is missing entirely, the flat total goes to the
    5-minute bucket, the API's default TTL.
    """
    if usage.cache_creation is not None:
        return (
            usage.cache_creation.ephemeral_1h_input_tokens,
            usage.cache_creation.ephemeral_5m_input_tokens,
        )
    return 0, usage.cache_creation_input_tokens


def _validation_error_kind(exc: ValidationError) -> str:
    """A stable label for one validation failure's shape, so warnings can be
    budgeted per shape."""
    first = exc.errors()[0] if exc.errors() else None
    if first is None:
        return "unknown"
    location = ".".join(str(part) for part in first["loc"])
    return f"{first['type']} at {location or '(root)'}"


class UsageParser:
    """Folds decoded lines into one `UsageRecord` per `requestId`, counting
    every discarded line by reason."""

    def __init__(self) -> None:
        self._records: dict[str, UsageRecord] = {}
        self._conflicting_ids: set[str] = set()
        self._unresolved_ids: set[str] = set()
        self._dropped_message_ids: set[str] = set()
        self._warned_decode_kinds: dict[str, int] = {}
        self._counts: dict[str, int] = dict.fromkeys(
            (
                "lines_read",
                "blank_lines",
                "non_assistant_lines",
                "non_object_json_lines",
                "lines_without_type_discriminator",
                "malformed_json_lines",
                "undecodable_assistant_lines",
                "synthetic_lines",
                "assistant_lines_without_usage",
                "assistant_lines_without_request_id",
                "tokens_dropped_without_request_id",
                "duplicate_lines_collapsed",
                "conflict_lines_discarded",
            ),
            0,
        )

    def feed(self, line: DecodedLine, origin: RecordOrigin) -> None:
        """Account for one decoded line, keeping it when it is a billable
        assistant turn."""
        self._counts["lines_read"] += 1
        kind = line.kind
        if kind is LineKind.BLANK:
            self._counts["blank_lines"] += 1
            return
        if kind is LineKind.MALFORMED_JSON:
            self._counts["malformed_json_lines"] += 1
            return
        if kind is LineKind.NON_OBJECT_JSON:
            self._counts["non_object_json_lines"] += 1
            return
        if kind is LineKind.NO_TYPE:
            self._counts["lines_without_type_discriminator"] += 1
            return
        if kind is LineKind.NON_ASSISTANT:
            self._counts["non_assistant_lines"] += 1
            return
        if kind is LineKind.UNDECODABLE_ASSISTANT:
            self._counts["undecodable_assistant_lines"] += 1
            if line.error is not None:
                self._warn_decode_failure(line.error)
            return

        entry = line.assistant
        if entry is None:  # pragma: no cover - kind guarantees this
            raise ValueError("assistant line without a validated payload")
        if entry.message.model == _SYNTHETIC_MODEL:
            self._counts["synthetic_lines"] += 1
            return
        usage = entry.message.usage
        if usage is None:
            self._counts["assistant_lines_without_usage"] += 1
            return
        cache_write_1h, cache_write_5m = _cache_write_split(usage)
        if not entry.request_id:
            self._count_dropped_without_request_id(entry, usage, cache_write_1h, cache_write_5m)
            return

        candidate = UsageRecord(
            request_id=entry.request_id,
            timestamp=entry.timestamp,
            model=entry.message.model,
            input_tokens=usage.input_tokens,
            cache_read_tokens=usage.cache_read_input_tokens,
            cache_write_1h_tokens=cache_write_1h,
            cache_write_5m_tokens=cache_write_5m,
            output_tokens=usage.output_tokens,
            origin=origin,
        )
        existing = self._records.get(entry.request_id)
        if existing is not None:
            if existing.same_usage(candidate):
                self._counts["duplicate_lines_collapsed"] += 1
                return
            self._conflicting_ids.add(entry.request_id)
            # Either the candidate loses now, or it replaces the incumbent —
            # either way exactly one record is discarded, and the accounting
            # has to see it.
            self._counts["conflict_lines_discarded"] += 1
            if not self._candidate_wins(existing, candidate):
                return
        self._records[entry.request_id] = candidate

    def finish(self) -> ParsedTranscripts:
        """The kept records plus the full line accounting."""
        return ParsedTranscripts(
            records=tuple(sorted(self._records.values(), key=lambda record: record.request_id)),
            conflicting_request_ids=len(self._conflicting_ids),
            unresolved_conflict_request_ids=len(self._unresolved_ids),
            **self._counts,
        )

    def _candidate_wins(self, existing: UsageRecord, candidate: UsageRecord) -> bool:
        """Decide which of two disagreeing copies of one `requestId` to keep:
        the copy reporting more billed tokens. A tie is recorded as
        unresolved and logged once per `requestId`."""
        if existing.billed_tokens == candidate.billed_tokens:
            already_known = existing.request_id in self._unresolved_ids
            self._unresolved_ids.add(existing.request_id)
            if not already_known:
                logger.warning(
                    "requestId %r has disagreeing copies with identical billed "
                    "token totals (%d); keeping the first seen, so this one "
                    "record's attribution is decided by file sort order and is "
                    "arbitrary.",
                    existing.request_id,
                    existing.billed_tokens,
                )
            return False
        if existing.billed_tokens > candidate.billed_tokens:
            return False
        self._unresolved_ids.discard(existing.request_id)
        return True

    def _count_dropped_without_request_id(
        self,
        entry: RawTranscriptLine,
        usage: RawUsage,
        cache_write_1h: int,
        cache_write_5m: int,
    ) -> None:
        """Count an assistant turn that carries tokens but no `requestId`.

        With no `requestId` to deduplicate on, the `message.id` is the only
        identity left: one turn writes one line per content block, so the
        dropped tokens are counted once per message id, never once per line.
        """
        self._counts["assistant_lines_without_request_id"] += 1
        message_id = entry.message.id
        if message_id is not None and message_id in self._dropped_message_ids:
            return
        if message_id is not None:
            self._dropped_message_ids.add(message_id)
        self._counts["tokens_dropped_without_request_id"] += (
            usage.input_tokens
            + usage.cache_read_input_tokens
            + cache_write_1h
            + cache_write_5m
            + usage.output_tokens
        )

    def _warn_decode_failure(self, exc: ValidationError) -> None:
        """Log a decode failure, budgeted per distinct failure shape.

        The log message uses `errors(include_input=False)`: pydantic's
        default embeds the offending input, and the input here is a line of
        the operator's own log — private prompts and file contents. The
        failing field's location and type are the diagnostic value; the
        payload echo is a leak.
        """
        kind = _validation_error_kind(exc)
        seen = self._warned_decode_kinds.get(kind, 0)
        if seen >= _MAX_WARNINGS_PER_KIND:
            return
        self._warned_decode_kinds[kind] = seen + 1
        logger.warning(
            "Assistant line failed validation (%s), skipped: %s",
            kind,
            exc.errors(include_url=False, include_input=False),
        )
