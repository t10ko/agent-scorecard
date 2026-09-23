"""Finds the Claude Code log folders for a project and lists the files in
them: main-session logs, subagent logs with their sidecars, and journals.

Claude Code writes every project's logs under `~/.claude/projects/` (or
`$CLAUDE_CONFIG_DIR/projects/`), one folder per project, named after the
project's absolute path with every character that is not a letter or digit
replaced by `-`.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

from pydantic import BaseModel, ConfigDict, JsonValue, ValidationError

from agent_scorecard.models import FileTally, RawAgentMeta, SubagentOrigin

# Journals are Workflow bookkeeping, not agent logs: they carry no sidecar,
# so they are collected separately for lifecycle only and never counted as a
# subagent file missing its sidecar.
_JOURNAL_FILE_NAME = "journal.jsonl"
_AGENT_FILE_PREFIX = "agent-"


def logs_root() -> Path:
    """The folder that holds one subfolder of logs per project."""
    config_dir = Path(os.environ.get("CLAUDE_CONFIG_DIR", Path.home() / ".claude"))
    return config_dir / "projects"


def transcript_dir_for_project(project: Path) -> Path:
    """The log folder Claude Code keeps for `project`.

    The folder name is the project's absolute path with every character that
    is not a letter or digit replaced by `-`:
    `/Users/me/dev/my-app` becomes `-Users-me-dev-my-app`.
    """
    slug = re.sub(r"[^A-Za-z0-9]", "-", str(project.absolute()))
    return logs_root() / slug


def agent_id_of(path: Path) -> str:
    """The agent id of a subagent log file: its name without the `agent-`
    prefix. Every line in the file carries the same `agentId` field."""
    stem = path.stem
    if stem.startswith(_AGENT_FILE_PREFIX):
        return stem[len(_AGENT_FILE_PREFIX) :]
    return stem


class SubagentSource(BaseModel):
    """One subagent log file, its sidecar-derived identity, and the session
    folder it belongs to. The extra sidecar fields feed lifecycle
    attribution: `tool_use_id` links the run to the parent's tool call, and
    `description` is what gets stripped out of notification reasons."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    path: Path
    origin: SubagentOrigin
    session_id: str
    description: str | None = None
    tool_use_id: str | None = None
    stopped_by_user: bool = False


class TranscriptFiles(BaseModel):
    """Every log file a scan will read, grouped by kind and sorted."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    roots: tuple[Path, ...]
    main: tuple[Path, ...]
    subagents: tuple[SubagentSource, ...]
    journals: tuple[Path, ...]


def _subagent_source(path: Path, root: Path) -> SubagentSource | None:
    """Read one subagent log's identity from the `agent-<id>.meta.json`
    sidecar beside it, or None when there is no readable sidecar.

    Reading every `*.jsonl` under the `subagents/` subtree and demanding a
    sidecar — rather than filtering by file name — makes a format change a
    counted, reported miss instead of a silent drop.
    """
    meta_path = path.with_suffix(".meta.json")
    try:
        raw: JsonValue = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, ValueError):
        return None
    try:
        meta = RawAgentMeta.model_validate(raw)
    except ValidationError:
        return None
    origin = SubagentOrigin(
        agent_id=agent_id_of(path),
        agent_type=meta.agent_type,
        spawn_depth=meta.spawn_depth,
        parent_agent_id=meta.parent_agent_id,
    )
    session_id = path.relative_to(root).parts[0]
    return SubagentSource(
        path=path,
        origin=origin,
        session_id=session_id,
        description=meta.description,
        tool_use_id=meta.tool_use_id,
        stopped_by_user=meta.stopped_by_user,
    )


def list_transcript_files(roots: list[Path], tally: FileTally) -> TranscriptFiles:
    """List the files to read under each log folder, recording per-file
    outcomes in `tally`.

    Main-session files are read from the top level of the folder only, never
    recursively: Claude Code stores one folder per project, so recursing from
    a parent path would fold unrelated projects' logs into the total. A
    subagent log with no readable sidecar cannot be attributed: it is counted
    in `tally.missing_agent_metadata` and skipped, never read anonymously.
    """
    main: list[Path] = []
    subagents: list[SubagentSource] = []
    journals: list[Path] = []
    for root in roots:
        for path in sorted(root.glob("*.jsonl")):
            tally.globbed += 1
            main.append(path)
        for path in sorted(root.glob("*/subagents/**/*.jsonl")):
            if path.name == _JOURNAL_FILE_NAME:
                journals.append(path)
                tally.journal_files_found += 1
                continue
            tally.globbed += 1
            source = _subagent_source(path, root)
            if source is None:
                tally.missing_agent_metadata += 1
                continue
            subagents.append(source)
    return TranscriptFiles(
        roots=tuple(roots),
        main=tuple(main),
        subagents=tuple(subagents),
        journals=tuple(journals),
    )
