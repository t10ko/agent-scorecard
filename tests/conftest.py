"""Builders that emit valid Claude Code log lines and folders with the exact
shapes the logs use, so tests read like a story of what happened in a
session. All data is synthetic."""

from __future__ import annotations

import datetime
import json
from pathlib import Path
from typing import Any

_BASE = datetime.datetime(2026, 9, 20, 10, 15, 0, tzinfo=datetime.UTC)


def ts(offset_seconds: int = 0) -> str:
    """One synthetic ISO 8601 UTC timestamp, `offset_seconds` after the base."""
    moment = _BASE + datetime.timedelta(seconds=offset_seconds)
    return moment.isoformat(timespec="milliseconds").replace("+00:00", "Z")


def text_block(text: str) -> dict[str, Any]:
    return {"type": "text", "text": text}


def tool_use(
    id: str, name: str, input: dict[str, Any], *, raw_input: dict[str, Any] | None = None
) -> dict[str, Any]:
    return {"type": "tool_use", "id": id, "name": name, "input": raw_input or input}


def tool_result(
    tool_use_id: str, *, content: str | list[dict[str, Any]] = "ok", is_error: bool = False
) -> dict[str, Any]:
    block: dict[str, Any] = {"type": "tool_result", "tool_use_id": tool_use_id}
    if is_error:
        block["is_error"] = True
    block["content"] = content
    return block


def assistant_line(
    request_id: str = "req_1",
    *,
    timestamp: str | None = None,
    model: str = "claude-sonnet-5",
    message_id: str | None = "msg_1",
    content: list[dict[str, Any]] | None = None,
    stop_reason: str | None = "end_turn",
    is_api_error: bool = False,
    input_tokens: int = 0,
    cache_read: int = 0,
    cache_write_1h: int = 0,
    cache_write_5m: int = 0,
    output_tokens: int = 0,
    split_cache: bool = True,
    agent_id: str | None = None,
) -> str:
    """One `assistant` line: an API turn with its token usage."""
    message: dict[str, Any] = {
        "id": message_id,
        "model": model,
        "stop_reason": stop_reason,
        "content": content if content is not None else [text_block("done")],
        "usage": {
            "input_tokens": input_tokens,
            "cache_creation_input_tokens": cache_write_1h + cache_write_5m,
            "cache_read_input_tokens": cache_read,
            "output_tokens": output_tokens,
        },
    }
    if split_cache:
        message["usage"]["cache_creation"] = {
            "ephemeral_1h_input_tokens": cache_write_1h,
            "ephemeral_5m_input_tokens": cache_write_5m,
        }
    line: dict[str, Any] = {
        "type": "assistant",
        "timestamp": timestamp if timestamp is not None else ts(),
        "requestId": request_id,
        "isApiErrorMessage": is_api_error,
        "message": message,
    }
    if agent_id is not None:
        line["agentId"] = agent_id
        line["isSidechain"] = True
    return json.dumps(line)


def agent_call(
    tool_use_id: str = "toolu_01PARENT",
    *,
    description: str = "Fix the flaky login test",
    subagent_type: str = "code-writer",
    run_in_background: bool = True,
    timestamp: str | None = None,
    tool_name: str = "Agent",
) -> str:
    """An assistant line whose tool call starts one agent run."""
    return assistant_line(
        request_id=f"req_call_{tool_use_id}",
        timestamp=timestamp,
        content=[
            tool_use(
                tool_use_id,
                tool_name,
                {
                    "description": description,
                    "prompt": "synthetic prompt",
                    "subagent_type": subagent_type,
                    "run_in_background": run_in_background,
                },
            )
        ],
    )


def tool_result_line(
    tool_use_id: str,
    *,
    timestamp: str | None = None,
    content: str | list[dict[str, Any]] = "ok",
    is_error: bool = False,
    denial_kind: str | None = None,
    tool_use_result: dict[str, Any] | None = None,
    origin_kind: str | None = None,
    agent_id: str | None = None,
) -> str:
    """One `user` line carrying a tool result, plus any top-level extras
    (`toolDenialKind`, `toolUseResult`, `origin.kind`)."""
    line: dict[str, Any] = {
        "type": "user",
        "timestamp": timestamp if timestamp is not None else ts(),
        "message": {
            "role": "user",
            "content": [tool_result(tool_use_id, content=content, is_error=is_error)],
        },
    }
    if denial_kind is not None:
        line["toolDenialKind"] = denial_kind
    if tool_use_result is not None:
        line["toolUseResult"] = tool_use_result
    if origin_kind is not None:
        line["origin"] = {"kind": origin_kind}
    if agent_id is not None:
        line["agentId"] = agent_id
    return json.dumps(line)


def foreground_result(
    tool_use_id: str,
    agent_id: str,
    *,
    status: str = "completed",
    report_text: str = "<final report>",
    is_error: bool = False,
    timestamp: str | None = None,
) -> str:
    """The parent's record of a foreground agent it waited for."""
    return tool_result_line(
        tool_use_id,
        timestamp=timestamp,
        content=[text_block(report_text)],
        is_error=is_error,
        tool_use_result={
            "status": status,
            "agentId": agent_id,
            "content": [text_block(report_text)],
            "totalDurationMs": 81_234,
            "totalTokens": 152_300,
            "totalToolUseCount": 23,
        },
    )


def async_launch(tool_use_id: str, agent_id: str, *, timestamp: str | None = None) -> str:
    """The immediate tool result for a background agent launch."""
    return tool_result_line(
        tool_use_id,
        timestamp=timestamp,
        tool_use_result={
            "status": "async_launched",
            "agentId": agent_id,
            "isAsync": True,
            "outputFile": f"/tmp/demo-out/{agent_id}.output",
        },
    )


def notification(
    task_id: str,
    tool_use_id: str,
    *,
    status: str = "completed",
    description: str = "Fix the flaky login test",
    summary: str | None = None,
    result_text: str = "<final report>",
    timestamp: str | None = None,
) -> str:
    """A `user` line with `origin.kind == "task-notification"`: the parent's
    record of a background agent's outcome."""
    if summary is None:
        if status == "completed":
            summary = f'Agent "{description}" finished'
        elif status == "killed" or status == "stopped":
            summary = f'Agent "{description}" was stopped by user'
        else:
            summary = f'Agent "{description}" failed: Agent stalled: no progress for 600s'
    body = (
        "<task-notification>\n"
        f"<task-id>{task_id}</task-id>\n"
        f"<tool-use-id>{tool_use_id}</tool-use-id>\n"
        f"<output-file>/tmp/demo-out/{task_id}.output</output-file>\n"
        f"<status>{status}</status>\n"
        f"<summary>{summary}</summary>\n"
        f"<result>{result_text}</result>\n"
        "</task-notification>"
    )
    line: dict[str, Any] = {
        "type": "user",
        "origin": {"kind": "task-notification"},
        "timestamp": timestamp if timestamp is not None else ts(),
        "message": {"role": "user", "content": body},
    }
    return json.dumps(line)


def journal_line(kind: str, agent_id: str, *, key: str = "wf-key-1", result: Any = None) -> str:
    """One `journal.jsonl` line: an agent's lifecycle inside a workflow."""
    entry: dict[str, Any] = {"type": kind, "agentId": agent_id, "key": key}
    if kind == "result":
        entry["result"] = result if result is not None else "ok"
    return json.dumps(entry)


def sidecar(
    agent_type: str = "code-writer",
    spawn_depth: int = 1,
    *,
    description: str | None = "Fix the flaky login test",
    tool_use_id: str | None = "toolu_01PARENT",
    parent_agent_id: str | None = None,
    stopped_by_user: bool = False,
) -> dict[str, Any]:
    meta: dict[str, Any] = {"agentType": agent_type, "spawnDepth": spawn_depth}
    if description is not None:
        meta["description"] = description
    if tool_use_id is not None:
        meta["toolUseId"] = tool_use_id
    if parent_agent_id is not None:
        meta["parentAgentId"] = parent_agent_id
    if stopped_by_user:
        meta["stoppedByUser"] = True
    return meta


def write_session(dir: Path, lines: list[str], name: str = "session-1") -> Path:
    """Write one main-session log at the top level of the log folder."""
    dir.mkdir(parents=True, exist_ok=True)
    path = dir / f"{name}.jsonl"
    path.write_text("".join(line + "\n" for line in lines), encoding="utf-8")
    return path


def write_run(
    dir: Path,
    session: str,
    agent_id: str,
    meta: dict[str, Any],
    lines: list[str],
    workflow: str | None = None,
) -> Path:
    """Write one agent run's log and its `agent-<id>.meta.json` sidecar."""
    directory = dir / session / "subagents"
    if workflow is not None:
        directory = directory / "workflows" / workflow
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"agent-{agent_id}.jsonl"
    path.write_text("".join(line + "\n" for line in lines), encoding="utf-8")
    path.with_suffix(".meta.json").write_text(json.dumps(meta), encoding="utf-8")
    return path


def write_journal(dir: Path, session: str, workflow_id: str, lines: list[str]) -> Path:
    """Write one workflow's `journal.jsonl`."""
    directory = dir / session / "subagents" / "workflows" / workflow_id
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "journal.jsonl"
    path.write_text("".join(line + "\n" for line in lines), encoding="utf-8")
    return path


def parse_lines(*lines: str):
    """Parse raw line texts through the same path a scan uses, with
    main-session origin, for tests that exercise decoding and pricing."""
    from agent_scorecard.logfile import decode_line
    from agent_scorecard.models import MainThreadOrigin
    from agent_scorecard.usage import UsageParser

    parser = UsageParser()
    for text in lines:
        parser.feed(decode_line(text), MainThreadOrigin())
    return parser.finish()
