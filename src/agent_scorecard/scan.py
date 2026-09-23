"""The single pass over the log folders: files, then decoded lines, then
collectors, then a `ScanResult`.

Every file is opened once and every line decoded once, so the scan's cost
is one read of the corpus while memory stays bounded by the records and
runs, never by file size. The refusal gates live here, next to the counters
they read: a scan whose accounting does not reconcile, or that lost more
than half the lines that should have carried usage, is not a measurement,
and the tool refuses to report one rather than printing wrong numbers.
"""

from __future__ import annotations

import datetime
from collections.abc import Sequence
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from agent_scorecard.locate import TranscriptFiles, list_transcript_files
from agent_scorecard.logfile import iter_file_lines, iter_journal_lines
from agent_scorecard.models import (
    FileTally,
    MainThreadOrigin,
    ParsedTranscripts,
)
from agent_scorecard.runs import (
    DEFAULT_TEST_PATTERNS,
    RunCollector,
    RunFacts,
    compile_test_patterns,
)
from agent_scorecard.usage import UsageParser

# A scan that lost more than this fraction of the lines that should have
# carried usage is not a measurement, whatever survived. Real logs lose a
# small fraction of a percent, so anything approaching half signals a
# changed log format rather than noise.
_MAX_CORPUS_LOSS_FRACTION = 0.5


class ScanResult(BaseModel):
    """Everything one scan read: the files, their per-file outcomes, and the
    de-duplicated records with the full line accounting."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    files: TranscriptFiles
    tally: FileTally
    parsed: ParsedTranscripts
    run_facts: tuple[RunFacts, ...] = ()


def scan(
    roots: list[Path],
    since: datetime.date | None = None,
    test_patterns: Sequence[str] | None = None,
) -> ScanResult:
    """Read every log file under each root once, feeding each decoded line
    to the usage parser.

    `since` skips whole files whose modification date is entirely before it
    — every line in such a file is older too. Journals are read for
    lifecycle only; their lines stay out of the usage accounting.
    """
    patterns = compile_test_patterns(
        DEFAULT_TEST_PATTERNS if test_patterns is None else test_patterns
    )
    tally = FileTally()
    files = list_transcript_files(roots, tally)
    usage = UsageParser()
    for path in files.main:
        for line in iter_file_lines(path, since, tally):
            usage.feed(line, MainThreadOrigin())
    run_facts: list[RunFacts] = []
    for source in files.subagents:
        origin = source.origin
        collector = RunCollector(
            agent_id=origin.agent_id,
            agent_type=origin.agent_type,
            spawn_depth=origin.spawn_depth,
            session_id=source.session_id,
            parent_agent_id=origin.parent_agent_id,
            test_patterns=patterns,
        )
        for line in iter_file_lines(source.path, since, tally):
            usage.feed(line, origin)
            collector.feed(line)
        run_facts.append(collector.finish())
    for journal in files.journals:
        for _raw in iter_journal_lines(journal, tally):
            pass  # lifecycle collection joins in a later milestone
    return ScanResult(files=files, tally=tally, parsed=usage.finish(), run_facts=tuple(run_facts))


def loss_lines(parsed: ParsedTranscripts) -> int:
    """Every line that should have produced a billable record and did not,
    for a structural reason.

    This is the single definition of corpus loss: both the warning and the
    refusal gate read it, so a break can never be described in one place
    while the gate stays blind to it. It spans every structural failure
    rather than the most common one — counting only undecodable lines would
    let a renamed `type` discriminator drop 60% of a corpus and still report
    a confident dollar figure, because a line with no `type` key produces no
    undecodable line at all.
    """
    return (
        parsed.malformed_json_lines
        + parsed.non_object_json_lines
        + parsed.lines_without_type_discriminator
        + parsed.undecodable_assistant_lines
        + parsed.assistant_lines_without_request_id
        + parsed.assistant_lines_without_usage
    )


def attempted_lines(parsed: ParsedTranscripts) -> int:
    """Every line that should have carried usage: all lines read except the
    ones legitimately expected to produce nothing (blank lines, genuine
    non-assistant lines, and synthetic turns that were never API calls)."""
    return (
        parsed.lines_read - parsed.blank_lines - parsed.non_assistant_lines - parsed.synthetic_lines
    )


def refusal_reason(result: ScanResult) -> str | None:
    """The first refusal gate the scan trips, or None when the scan is
    reportable. Every message names the counter that tripped it."""
    tally = result.tally
    parsed = result.parsed
    if tally.globbed == 0:
        return (
            f"No log files found under "
            f"{', '.join(str(root) for root in result.files.roots)}: 0 files "
            f"globbed. There is nothing to measure."
        )
    if tally.globbed != (
        tally.skipped_by_since + tally.read_fully + tally.failed + tally.missing_agent_metadata
    ):
        return (
            f"File accounting does not reconcile: {tally.globbed} globbed vs "
            f"{tally.skipped_by_since} skipped + {tally.read_fully} read + "
            f"{tally.failed} failed + {tally.missing_agent_metadata} "
            f"unattributable. Refusing to report a total built on it."
        )
    if parsed.lines_read == 0:
        return (
            f"Read 0 lines from {tally.globbed} file(s) "
            f"({tally.skipped_by_since} skipped by --since, {tally.failed} "
            f"failed). There is nothing to measure."
        )
    if not parsed.records:
        return (
            f"Read {parsed.lines_read} lines but decoded 0 usage records. "
            f"The accounting below says where every line went; a zero here "
            f"is not a measurement."
        )
    attempted = attempted_lines(parsed)
    lost = loss_lines(parsed)
    if attempted and lost / attempted > _MAX_CORPUS_LOSS_FRACTION:
        return (
            f"Lost {lost} of {attempted} lines that should have carried usage "
            f"({100 * lost / attempted:.1f}%). Whatever survived is too small "
            f"a share of the corpus to report as a measurement; the log format "
            f"has most likely changed."
        )
    return None


def accounting_lines(result: ScanResult) -> list[str]:
    """Human-readable file and line accounting, printed on `-v` and before
    every refusal."""
    tally = result.tally
    parsed = result.parsed
    return [
        f"Files: {tally.globbed} globbed, {tally.skipped_by_since} skipped by "
        f"--since, {tally.read_fully} read fully, {tally.failed} failed, "
        f"{tally.missing_agent_metadata} skipped for a missing sidecar",
        f"Journals: {tally.journal_files_found} found, "
        f"{tally.journal_files_read} read, {tally.journal_lines} lines, "
        f"{tally.journal_files_failed} failed",
        f"Lines: {parsed.lines_read} read: {parsed.blank_lines} blank, "
        f"{parsed.non_assistant_lines} non-assistant, "
        f"{parsed.non_object_json_lines} non-object JSON, "
        f"{parsed.lines_without_type_discriminator} without a type field, "
        f"{parsed.malformed_json_lines} malformed JSON, "
        f"{parsed.undecodable_assistant_lines} undecodable assistant, "
        f"{parsed.synthetic_lines} synthetic, "
        f"{parsed.assistant_lines_without_usage} without usage, "
        f"{parsed.assistant_lines_without_request_id} without requestId "
        f"({parsed.tokens_dropped_without_request_id} tokens dropped), "
        f"{parsed.duplicate_lines_collapsed} duplicates collapsed, "
        f"{parsed.conflict_lines_discarded} conflicting copies discarded",
        f"De-duplication: {parsed.conflicting_request_ids} requestId(s) had "
        f"disagreeing copies, "
        f"{parsed.unresolved_conflict_request_ids} of which tied and were "
        f"resolved arbitrarily",
    ]
