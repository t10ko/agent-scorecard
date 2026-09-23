"""Finding and reading log files: folder naming, file listing, sidecar
attribution, the `--since` mtime shortcut, and per-file failure handling."""

from __future__ import annotations

import datetime
import json
import os
from pathlib import Path

import pytest
from conftest import assistant_line, sidecar, write_journal, write_run, write_session

from agent_scorecard.locate import (
    agent_id_of,
    list_transcript_files,
    logs_root,
    transcript_dir_for_project,
)
from agent_scorecard.logfile import iter_file_lines
from agent_scorecard.models import FileTally, MainThreadOrigin, SubagentOrigin
from agent_scorecard.usage import UsageParser


def parse_folder(dir: Path, since: datetime.date | None = None) -> tuple[FileTally, list[str]]:
    """Read every file `list_transcript_files` finds and return the tally
    plus the kept request ids, mirroring what a scan feeds its parser."""
    tally = FileTally()
    files = list_transcript_files([dir], tally)
    parser = UsageParser()
    request_ids: list[str] = []
    for path in files.main:
        for line in iter_file_lines(path, since, tally):
            parser.feed(line, MainThreadOrigin())
    for source in files.subagents:
        for line in iter_file_lines(source.path, since, tally):
            parser.feed(line, source.origin)
    for record in parser.finish().records:
        request_ids.append(record.request_id)
    return tally, request_ids


def test_folder_name_is_the_project_path_with_every_non_alphanumeric_flattened(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path))
    project = Path("/Users/me/dev/my-app")

    folder = transcript_dir_for_project(project)

    assert folder == logs_root() / "-Users-me-dev-my-app"


def test_logs_root_honors_claude_config_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path))

    assert logs_root() == tmp_path / "projects"


def test_agent_id_is_the_file_name_without_the_prefix(tmp_path: Path) -> None:
    assert agent_id_of(tmp_path / "agent-a1b2c3d4e5f6a7b8c.jsonl") == "a1b2c3d4e5f6a7b8c"


def test_an_unreadable_file_is_skipped_and_the_rest_are_read(tmp_path: Path) -> None:
    # Logs are appended to by live sessions, so one torn file must not
    # discard every other file's records.
    write_session(tmp_path, [assistant_line()])
    (tmp_path / "b_torn.jsonl").write_bytes(b"\xff\xfe not utf-8")

    tally, request_ids = parse_folder(tmp_path)

    assert request_ids == ["req_1"]
    assert tally.read_fully == 1
    assert tally.failed == 1


def test_files_modified_before_since_are_skipped(tmp_path: Path) -> None:
    write_session(tmp_path, [assistant_line(request_id="req_old")], name="old")
    write_session(tmp_path, [assistant_line(request_id="req_recent")], name="recent")
    stale = datetime.datetime(2020, 1, 1, tzinfo=datetime.UTC).timestamp()
    os.utime(tmp_path / "old.jsonl", (stale, stale))

    tally, request_ids = parse_folder(tmp_path, since=datetime.date(2021, 1, 1))

    assert request_ids == ["req_recent"]
    assert tally.skipped_by_since == 1
    assert tally.globbed == 2


def test_a_file_modified_exactly_on_since_is_included(tmp_path: Path) -> None:
    # --since skips files whose date PRECEDES it, so the boundary date
    # itself must be included.
    path = write_session(tmp_path, [assistant_line()], name="boundary")
    stamp = datetime.datetime(2021, 6, 1, 12, 0, tzinfo=datetime.UTC).timestamp()
    os.utime(path, (stamp, stamp))

    tally = FileTally()
    files = list_transcript_files([tmp_path], tally)

    assert len(files.main) == 1
    assert tally.skipped_by_since == 0


def test_per_file_outcomes_are_all_recorded(tmp_path: Path) -> None:
    write_session(tmp_path, [assistant_line()], name="a_good")
    (tmp_path / "b_torn.jsonl").write_bytes(b"\xff\xfe not utf-8")
    old = write_session(tmp_path, [assistant_line(request_id="req_old")], name="c_old")
    stale = datetime.datetime(2020, 1, 1, tzinfo=datetime.UTC).timestamp()
    os.utime(old, (stale, stale))

    tally = FileTally()
    list_transcript_files([tmp_path], tally)
    list(iter_file_lines(tmp_path / "a_good.jsonl", datetime.date(2021, 1, 1), tally))
    list(iter_file_lines(tmp_path / "b_torn.jsonl", datetime.date(2021, 1, 1), tally))
    list(iter_file_lines(tmp_path / "c_old.jsonl", datetime.date(2021, 1, 1), tally))

    assert tally.globbed == 3
    assert tally.skipped_by_since == 1
    assert tally.read_fully == 1
    assert tally.failed == 1


def test_main_session_files_are_read_from_the_top_level_only(tmp_path: Path) -> None:
    # Claude Code stores logs one folder per project; recursing from a
    # parent path would silently fold unrelated projects into the total.
    write_session(tmp_path, [assistant_line()])
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "deep.jsonl").write_text(assistant_line(request_id="req_deep"), encoding="utf-8")

    _tally, request_ids = parse_folder(tmp_path)

    assert request_ids == ["req_1"]


def test_subagent_files_are_attributed_through_their_sidecar(tmp_path: Path) -> None:
    write_session(tmp_path, [assistant_line(request_id="req_main")])
    write_run(
        tmp_path,
        "session-1",
        "aaa",
        sidecar("Explore", 1, tool_use_id=None),
        [assistant_line(request_id="req_explore", agent_id="aaa")],
    )
    write_run(
        tmp_path,
        "session-1",
        "bbb",
        sidecar("general-purpose", 2, parent_agent_id="aaa"),
        [assistant_line(request_id="req_general", agent_id="bbb")],
    )

    files = list_transcript_files([tmp_path], FileTally())
    assert [source.origin.agent_id for source in files.subagents] == ["aaa", "bbb"]
    assert files.subagents[0].origin == SubagentOrigin(
        agent_id="aaa", agent_type="Explore", spawn_depth=1
    )
    assert files.subagents[0].session_id == "session-1"
    assert files.subagents[1].origin.parent_agent_id == "aaa"


def test_subagents_nested_under_a_workflow_are_found(tmp_path: Path) -> None:
    # Agents spawned inside a Workflow sit one directory deeper, under
    # `subagents/workflows/<wf-id>/`.
    write_session(tmp_path, [assistant_line()])
    write_run(
        tmp_path,
        "session-1",
        "deep",
        sidecar("analyst", 3),
        [assistant_line(request_id="req_deep", agent_id="deep")],
        workflow="wf_1234",
    )

    tally, request_ids = parse_folder(tmp_path)

    assert set(request_ids) == {"req_1", "req_deep"}
    assert tally.missing_agent_metadata == 0


def test_a_subagent_without_a_readable_sidecar_is_counted_and_skipped(
    tmp_path: Path,
) -> None:
    # An unattributable file is counted, never read anonymously: an
    # unattributed run inflates the total while contributing nothing to the
    # breakdown that total is for.
    write_session(tmp_path, [assistant_line()])
    orphan_dir = tmp_path / "session-1" / "subagents"
    orphan_dir.mkdir(parents=True)
    (orphan_dir / "agent-orphan.jsonl").write_text(
        assistant_line(request_id="req_orphan"), encoding="utf-8"
    )

    tally, request_ids = parse_folder(tmp_path)

    assert tally.missing_agent_metadata == 1
    assert request_ids == ["req_1"]


def test_a_sidecar_with_missing_required_keys_is_unattributable(
    tmp_path: Path,
) -> None:
    # `agentType` and `spawnDepth` are required; without them the run cannot
    # be attributed.
    write_session(tmp_path, [assistant_line()])
    write_run(tmp_path, "session-1", "incomplete", {}, [assistant_line()])

    tally, request_ids = parse_folder(tmp_path)

    assert tally.missing_agent_metadata == 1
    assert request_ids == ["req_1"]


def test_a_journal_is_not_counted_as_a_missing_sidecar(tmp_path: Path) -> None:
    # A journal is Workflow bookkeeping, not an agent log: it is collected
    # for lifecycle only and never counted as a subagent file missing its
    # sidecar.
    write_session(tmp_path, [assistant_line()])
    write_journal(tmp_path, "session-1", "wf_1", [journal_started_line()])

    tally = FileTally()
    files = list_transcript_files([tmp_path], tally)

    assert files.journals == (
        tmp_path / "session-1" / "subagents" / "workflows" / "wf_1" / "journal.jsonl",
    )
    assert tally.journal_files_found == 1
    assert tally.missing_agent_metadata == 0


def journal_started_line() -> str:
    return json.dumps({"type": "started", "agentId": "wf1", "key": "k"})
