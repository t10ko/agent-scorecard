"""Streams log files line by line and decodes each line exactly once.

Owns per-file failure handling (a torn file is counted and skipped, never
fatal) and the mtime shortcut for `--since`. The decoded result is shared
with every collector, so no line is parsed or validated twice: the largest
real log file is 35 MB and the corpus runs to gigabytes.
"""

from __future__ import annotations

import datetime
import json
import logging
from collections.abc import Iterator
from enum import StrEnum
from pathlib import Path
from typing import NamedTuple

from pydantic import JsonValue, ValidationError

from agent_scorecard.models import FileTally, RawTranscriptLine

logger = logging.getLogger(__name__)


class LineKind(StrEnum):
    """What a line turned out to be; every line read lands in exactly one."""

    BLANK = "blank"
    MALFORMED_JSON = "malformed_json"
    NON_OBJECT_JSON = "non_object_json"
    NO_TYPE = "no_type"
    NON_ASSISTANT = "non_assistant"
    UNDECODABLE_ASSISTANT = "undecodable_assistant"
    ASSISTANT = "assistant"


class DecodedLine(NamedTuple):
    """One decoded line, shared by every collector.

    Which fields are set depends on `kind`: `raw` and `timestamp` are set for
    every valid JSON object, `assistant` only for `ASSISTANT`, and `error`
    only for `UNDECODABLE_ASSISTANT`.
    """

    kind: LineKind
    line_type: str = ""
    raw: dict[str, JsonValue] | None = None
    timestamp: datetime.datetime | None = None
    assistant: RawTranscriptLine | None = None
    error: ValidationError | None = None


def _parse_timestamp(value: JsonValue | None) -> datetime.datetime | None:
    """Parse a line's ISO 8601 UTC timestamp, or None when absent or
    malformed."""
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.datetime.fromisoformat(value)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=datetime.UTC)
    return parsed.astimezone(datetime.UTC)


def decode_line(text: str) -> DecodedLine:
    """Decode one raw line into a `DecodedLine`.

    A blank line, malformed JSON, a non-object, and a missing `type` are
    expected noise counted apart from each other, so a renamed field can
    never hide inside six figures of user lines. An assistant line that
    fails validation is its own counted kind; the error travels with the
    line so the warning can name the field without echoing the line's
    content (a log line can carry private prompts and file contents).
    """
    stripped = text.strip()
    if not stripped:
        return DecodedLine(kind=LineKind.BLANK)
    try:
        raw: JsonValue = json.loads(stripped)
    except (ValueError, RecursionError):
        return DecodedLine(kind=LineKind.MALFORMED_JSON)
    if not isinstance(raw, dict):
        return DecodedLine(kind=LineKind.NON_OBJECT_JSON)
    line_type = raw.get("type")
    if line_type is None:
        return DecodedLine(kind=LineKind.NO_TYPE)
    timestamp = _parse_timestamp(raw.get("timestamp"))
    if line_type != "assistant":
        return DecodedLine(
            kind=LineKind.NON_ASSISTANT,
            line_type=str(line_type),
            raw=raw,
            timestamp=timestamp,
        )
    try:
        entry = RawTranscriptLine.model_validate(raw)
    except ValidationError as exc:
        return DecodedLine(
            kind=LineKind.UNDECODABLE_ASSISTANT,
            line_type="assistant",
            raw=raw,
            timestamp=timestamp,
            error=exc,
        )
    return DecodedLine(
        kind=LineKind.ASSISTANT,
        line_type="assistant",
        raw=raw,
        timestamp=entry.timestamp,
        assistant=entry,
    )


def iter_file_lines(
    path: Path, since: datetime.date | None, tally: FileTally
) -> Iterator[DecodedLine]:
    """Stream one log file's decoded lines, recording its outcome in
    `tally`.

    Logs are appended to by live sessions, so a file can be truncated or
    removed between listing and reading. One unreadable file is counted,
    logged, and skipped rather than aborting the scan of every other file —
    but a failure part-way through means the lines already yielded are kept
    while the remainder is lost, and the log says so.

    One cheap shortcut: when the file's modification date is entirely before
    `--since`, every line in it is too, so it is skipped unread.
    """
    lines_yielded = 0
    try:
        if since is not None:
            modified_date = datetime.datetime.fromtimestamp(
                path.stat().st_mtime, tz=datetime.UTC
            ).date()
            if modified_date < since:
                tally.skipped_by_since += 1
                return
        with path.open(encoding="utf-8") as handle:
            for text in handle:
                lines_yielded += 1
                yield decode_line(text)
        tally.read_fully += 1
    except (OSError, UnicodeDecodeError) as exc:
        tally.failed += 1
        logger.error(
            "Log file %r failed after %d line(s); its remaining records are "
            "MISSING from this scan: %s",
            path.name,
            lines_yielded,
            exc,
        )


def iter_journal_lines(path: Path, tally: FileTally) -> Iterator[dict[str, JsonValue]]:
    """Stream one workflow journal's JSON objects, counting its lines.

    A journal is lifecycle bookkeeping, not an agent log: its lines never
    touch the usage accounting. One unreadable journal is counted and
    skipped; the affected runs fall back to inferred or unknown lifecycles.
    """
    try:
        with path.open(encoding="utf-8") as handle:
            for text in handle:
                stripped = text.strip()
                if not stripped:
                    continue
                try:
                    raw: JsonValue = json.loads(stripped)
                except (ValueError, RecursionError):
                    continue
                if isinstance(raw, dict):
                    tally.journal_lines += 1
                    yield raw
        tally.journal_files_read += 1
    except (OSError, UnicodeDecodeError) as exc:
        tally.journal_files_failed += 1
        logger.warning("Journal %r could not be read: %s", path.name, exc)
