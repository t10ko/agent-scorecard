"""Generates the committed demo log folder at `examples/demo/`.

The data is synthetic: made-up ids, made-up task descriptions, made-up
paths. The line shapes are exactly what Claude Code writes, so the demo
exercises the same parsing paths as real logs, including nested agents, a
workflow journal, a foreground run, a run with no recorded outcome, and a
tool call denied by a hook.

The scenario is built to show every verdict:

- code-writer: 12 runs — 9 succeed, one leaves tests failing, one dies on
  an API error, one never finishes              -> keep
- test-fixer: 8 runs — 4 succeed, 3 leave tests failing, one is stopped by
  the user                                      -> fix
- web-researcher: 6 runs — 2 succeed, 4 stall  -> remove
- reviewer: 3 runs, all succeed                -> not enough data
- Explore: 10 cheap background runs, all succeed -> keep

Run `uv run python scripts/make_demo.py [DEST]` to (re)generate it. The
output is deterministic: the same script always writes the same bytes.
"""

from __future__ import annotations

import datetime
import json
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Line builders with the exact Claude Code shapes.


def ts(day: int, minute: int) -> str:
    """One deterministic timestamp: 2026-09-{day}, 09:00 + minute minutes."""
    moment = datetime.datetime(2026, 9, day, 9, 0, tzinfo=datetime.UTC)
    moment += datetime.timedelta(minutes=minute)
    return moment.isoformat(timespec="milliseconds").replace("+00:00", "Z")


def text_block(text: str) -> dict[str, str]:
    return {"type": "text", "text": text}


def tool_use(id: str, name: str, input: dict[str, str]) -> dict[str, str]:
    return {"type": "tool_use", "id": id, "name": name, "input": input}


def assistant_line(
    request_id: str,
    *,
    timestamp: str,
    model: str,
    agent_id: str | None,
    content: list[dict[str, str]] | None = None,
    stop_reason: str | None = "end_turn",
    is_api_error: bool = False,
    input_tokens: int = 0,
    cache_read: int = 0,
    cache_write_5m: int = 0,
    output_tokens: int = 0,
) -> str:
    line: dict[str, object] = {
        "type": "assistant",
        "timestamp": timestamp,
        "requestId": request_id,
        "isApiErrorMessage": is_api_error,
        "message": {
            "id": f"msg_{request_id}",
            "model": model,
            "stop_reason": stop_reason,
            "content": content if content is not None else [text_block("done")],
            "usage": {
                "input_tokens": input_tokens,
                "cache_creation_input_tokens": cache_write_5m,
                "cache_read_input_tokens": cache_read,
                "output_tokens": output_tokens,
                "cache_creation": {
                    "ephemeral_1h_input_tokens": 0,
                    "ephemeral_5m_input_tokens": cache_write_5m,
                },
            },
        },
    }
    if agent_id is not None:
        line["agentId"] = agent_id
        line["isSidechain"] = True
    return json.dumps(line)


def tool_result_line(
    tool_use_id: str,
    *,
    timestamp: str,
    agent_id: str,
    content: str,
    is_error: bool = False,
    denial_kind: str | None = None,
) -> str:
    block: dict[str, object] = {
        "type": "tool_result",
        "tool_use_id": tool_use_id,
        "content": content,
    }
    if is_error:
        block["is_error"] = True
    line: dict[str, object] = {
        "type": "user",
        "timestamp": timestamp,
        "agentId": agent_id,
        "message": {"role": "user", "content": [block]},
    }
    if denial_kind:
        line["toolDenialKind"] = denial_kind
    return json.dumps(line)


def notification_line(
    task_id: str,
    *,
    timestamp: str,
    status: str,
    summary: str,
    result_text: str,
) -> str:
    body = (
        "<task-notification>\n"
        f"<task-id>{task_id}</task-id>\n"
        f"<tool-use-id>toolu_call_{task_id}</tool-use-id>\n"
        f"<output-file>/tmp/demo-out/{task_id}.output</output-file>\n"
        f"<status>{status}</status>\n"
        f"<summary>{summary}</summary>\n"
        f"<result>{result_text}</result>\n"
        "</task-notification>"
    )
    return json.dumps(
        {
            "type": "user",
            "origin": {"kind": "task-notification"},
            "timestamp": timestamp,
            "message": {"role": "user", "content": body},
        }
    )


def foreground_line(
    tool_use_id: str,
    agent_id: str,
    *,
    timestamp: str,
    report_text: str,
) -> str:
    return json.dumps(
        {
            "type": "user",
            "timestamp": timestamp,
            "message": {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": tool_use_id,
                        "content": [text_block(report_text)],
                    }
                ],
            },
            "toolUseResult": {
                "status": "completed",
                "agentId": agent_id,
                "content": [text_block(report_text)],
            },
        }
    )


def sidecar(agent_type: str, spawn_depth: int, description: str) -> dict[str, object]:
    return {
        "agentType": agent_type,
        "spawnDepth": spawn_depth,
        "description": description,
        "toolUseId": f"toolu_call_{description[:16]}",
    }


def journal_line(kind: str, agent_id: str) -> str:
    entry: dict[str, str] = {"type": kind, "agentId": agent_id, "key": "demo-key"}
    if kind == "result":
        entry["result"] = "ok"
    return json.dumps(entry)


# ---------------------------------------------------------------------------
# The demo itself.


class Demo:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.main_lines: dict[str, list[str]] = {}

    def write_run(
        self,
        session: str,
        agent_id: str,
        meta: dict[str, object],
        lines: list[str],
        *,
        workflow: str | None = None,
    ) -> None:
        directory = self.root / session / "subagents"
        if workflow:
            directory = directory / "workflows" / workflow
        directory.mkdir(parents=True, exist_ok=True)
        (directory / f"agent-{agent_id}.jsonl").write_text(
            "".join(line + "\n" for line in lines), encoding="utf-8"
        )
        (directory / f"agent-{agent_id}.meta.json").write_text(json.dumps(meta), encoding="utf-8")

    def write_journal(self, session: str, workflow: str, lines: list[str]) -> None:
        directory = self.root / session / "subagents" / "workflows" / workflow
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "journal.jsonl").write_text(
            "".join(line + "\n" for line in lines), encoding="utf-8"
        )

    def notify(self, session: str, line: str) -> None:
        self.main_lines.setdefault(session, []).append(line)

    def finish(self) -> None:
        """Write every main-session file: one main-thread turn plus the
        parent's records of its background agents."""
        for session, lines in sorted(self.main_lines.items()):
            main_turn = assistant_line(
                f"req_main_{session}",
                timestamp=ts(15, 0),
                model="claude-sonnet-5",
                agent_id=None,
                input_tokens=30_000,
                cache_read=900_000,
                output_tokens=2_500,
            )
            path = self.root / f"{session}.jsonl"
            path.write_text("".join(line + "\n" for line in [main_turn, *lines]), encoding="utf-8")


def run_lines(
    agent_id: str,
    *,
    day: int,
    minute: int,
    model: str,
    description: str,
    turns: int = 3,
    tokens: int = 400_000,
    edits: bool = False,
    tests: bool = True,
    last_test_fails: bool = False,
    api_error: bool = False,
    still_running: bool = False,
    denied_call: bool = False,
) -> list[str]:
    """One agent run's own file: a few turns of tool use and a finish."""
    lines: list[str] = []
    step = 0

    def request(index: int) -> str:
        return f"req_{agent_id}_{index}"

    pending_results: list[str] = []

    for turn in range(turns):
        content: list[dict[str, str]] = []
        if edits and turn == 0:
            content.append(
                tool_use(
                    f"toolu_{agent_id}_edit",
                    "Edit",
                    {"file_path": "/tmp/demo/src/app.py", "old_string": "x", "new_string": "y"},
                )
            )
            pending_results.append(
                tool_result_line(
                    f"toolu_{agent_id}_edit",
                    timestamp=ts(day, minute + step),
                    agent_id=agent_id,
                    content="edited",
                )
            )
            step += 1
        if tests:
            content.append(
                tool_use(
                    f"toolu_{agent_id}_test{turn}",
                    "Bash",
                    {"command": "uv run pytest -q"},
                )
            )
            failed = last_test_fails and turn == turns - 1
            pending_results.append(
                tool_result_line(
                    f"toolu_{agent_id}_test{turn}",
                    timestamp=ts(day, minute + step),
                    agent_id=agent_id,
                    content="Exit code 1\n2 failed, 18 passed" if failed else "18 passed",
                    is_error=failed,
                )
            )
            step += 1
        if denied_call and turn == 0:
            content.append(
                tool_use(
                    f"toolu_{agent_id}_denied",
                    "Bash",
                    {"command": "uv run pytest -q"},
                )
            )
            pending_results.append(
                tool_result_line(
                    f"toolu_{agent_id}_denied",
                    timestamp=ts(day, minute + step),
                    agent_id=agent_id,
                    content="Permission denied by hook",
                    is_error=True,
                    denial_kind="permission-rule",
                )
            )
            step += 1
        content.append(text_block(f"{description}: turn {turn + 1}"))
        # The calls go out first; their results arrive after.
        lines.append(
            assistant_line(
                request(turn),
                timestamp=ts(day, minute + step),
                model=model,
                agent_id=agent_id,
                content=content,
                input_tokens=tokens // 10,
                cache_read=tokens,
                cache_write_5m=tokens // 20,
                output_tokens=900 + 40 * turn,
            )
        )
        lines.extend(pending_results)
        pending_results = []
        step += 1

    if api_error:
        lines.append(
            assistant_line(
                request(turns),
                timestamp=ts(day, minute + step),
                model=model,
                agent_id=agent_id,
                is_api_error=True,
                stop_reason=None,
            )
        )
    elif still_running:
        lines.append(
            assistant_line(
                request(turns),
                timestamp=ts(day, minute + step),
                model=model,
                agent_id=agent_id,
                stop_reason="tool_use",
            )
        )
    else:
        lines.append(
            assistant_line(
                request(turns),
                timestamp=ts(day, minute + step),
                model=model,
                agent_id=agent_id,
                content=[text_block(f"{description}: all tests pass")],
            )
        )
    return lines


def build(root: Path) -> None:
    demo = Demo(root)
    # One scheduling slot per run; the clock spreads runs across Sep 15-17.
    slot = iter((15 + i // 14, 1 + (i % 14) * 6) for i in range(200))

    def run(
        session: str,
        agent_id: str,
        agent_type: str,
        description: str,
        *,
        model: str = "claude-sonnet-5",
        tokens: int = 400_000,
        spawn_depth: int = 1,
        workflow: str | None = None,
        outcome: str = "completed",
        notify: bool = True,
        **line_kwargs: object,
    ) -> None:
        day, minute = next(slot)
        demo.write_run(
            session,
            agent_id,
            sidecar(agent_type, spawn_depth, description),
            run_lines(
                agent_id,
                day=day,
                minute=minute,
                model=model,
                description=description,
                tokens=tokens,
                **line_kwargs,  # type: ignore[arg-type]
            ),
            workflow=workflow,
        )
        if not notify:
            return
        summary = {
            "completed": f'Agent "{description}" finished',
            "failed": f'Agent "{description}" failed: Agent stalled: no progress for 600s',
            "stopped": f'Agent "{description}" was stopped by user',
        }[outcome]
        demo.notify(
            session,
            notification_line(
                agent_id,
                timestamp=ts(day, minute + 8),
                status=outcome,
                summary=summary,
                result_text=(f"{description}: all tests pass" if outcome == "completed" else ""),
            ),
        )

    # --- code-writer: keep ------------------------------------------------
    run("demo-cw-1", "cw01", "code-writer", "Add pagination to the user list")
    run("demo-cw-1", "cw02", "code-writer", "Fix the date parsing bug", model="claude-opus-5")
    run("demo-cw-1", "cw03", "code-writer", "Extract the config loader")
    run("demo-cw-1", "cw04", "code-writer", "Write the migration script")
    run("demo-cw-2", "cw05", "code-writer", "Add retries to the uploader")

    # The nested pair: the parent's own file carries the Agent call that
    # started the child and the child's foreground result.
    child_call_id = "toolu_call_Update the fixtures"
    day, minute = next(slot)
    demo.write_run(
        "demo-cw-1",
        "cw1c",
        {
            "agentType": "code-writer",
            "spawnDepth": 2,
            "description": "Update the fixtures",
            "toolUseId": child_call_id,
            "parentAgentId": "cw06",
        },
        run_lines(
            "cw1c",
            day=day,
            minute=minute,
            model="claude-sonnet-5",
            description="Update the fixtures",
            turns=2,
        ),
    )
    parent_path = root / "demo-cw-1" / "subagents" / "agent-cw06.jsonl"
    run("demo-cw-1", "cw06", "code-writer", "Tidy the fixtures folder")
    parent_lines = parent_path.read_text(encoding="utf-8").rstrip("\n").split("\n")
    parent_lines.insert(
        0,
        assistant_line(
            "req_cw06_call",
            timestamp=ts(day, minute + 6),
            model="claude-sonnet-5",
            agent_id="cw06",
            content=[
                tool_use(
                    child_call_id,
                    "Agent",
                    {
                        "description": "Update the fixtures",
                        "prompt": "synthetic prompt",
                        "subagent_type": "code-writer",
                    },
                )
            ],
            input_tokens=20_000,
            output_tokens=300,
        ),
    )
    parent_lines.append(
        foreground_line(
            child_call_id,
            "cw1c",
            timestamp=ts(day, minute + 7),
            report_text="Update the fixtures: all tests pass",
        )
    )
    parent_path.write_text("".join(line + "\n" for line in parent_lines), encoding="utf-8")

    # Foreground run: the parent waited and recorded the result itself.
    day, minute = next(slot)
    demo.write_run(
        "demo-cw-2",
        "cw07",
        sidecar("code-writer", 1, "Delete the dead code"),
        run_lines(
            "cw07",
            day=day,
            minute=minute,
            model="claude-opus-5",
            description="Delete the dead code",
        ),
    )
    demo.notify(
        "demo-cw-2",
        foreground_line(
            "toolu_call_Delete the dead co",
            "cw07",
            timestamp=ts(day, minute + 8),
            report_text="Delete the dead code: all tests pass",
        ),
    )

    # Workflow run: its outcome lives in the workflow's journal.
    day, minute = next(slot)
    demo.write_run(
        "demo-cw-2",
        "cw08",
        sidecar("code-writer", 1, "Split the test helpers"),
        run_lines(
            "cw08",
            day=day,
            minute=minute,
            model="claude-sonnet-5",
            description="Split the test helpers",
        ),
        workflow="wf-demo",
    )
    demo.write_journal(
        "demo-cw-2", "wf-demo", [journal_line("started", "cw08"), journal_line("result", "cw08")]
    )

    run(
        "demo-cw-3",
        "cw09",
        "code-writer",
        "Rename the settings keys",
        edits=True,
        last_test_fails=True,
    )
    # No parent record for these two: the API-error run is inferred failed,
    # the still-running run stays unknown.
    run(
        "demo-cw-3",
        "cw10",
        "code-writer",
        "Port the CLI to argparse",
        model="claude-opus-5",
        api_error=True,
        notify=False,
    )
    run(
        "demo-cw-3",
        "cw11",
        "code-writer",
        "Refactor the job queue",
        still_running=True,
        notify=False,
    )
    run("demo-cw-3", "cw12", "code-writer", "Cache the rendered pages")

    # --- test-fixer: fix ----------------------------------------------------
    for index in range(4):
        run(
            "demo-tf-1",
            f"tf{index:02d}",
            "test-fixer",
            f"Fix the flaky login test {index}",
            edits=True,
            tokens=300_000,
        )
    for index in range(3):
        run(
            "demo-tf-1",
            f"tf1{index}",
            "test-fixer",
            f"Stabilize the checkout test {index}",
            edits=True,
            last_test_fails=True,
            tokens=300_000,
        )
    run("demo-tf-1", "tf08", "test-fixer", "Fix the timing test", tokens=300_000, outcome="stopped")

    # --- web-researcher: remove --------------------------------------------
    run(
        "demo-wr-1",
        "wr00",
        "web-researcher",
        "Compare the caching docs",
        tokens=200_000,
        tests=False,
    )
    run("demo-wr-1", "wr01", "web-researcher", "Find the rate limits", tokens=200_000, tests=False)
    for index in range(2, 6):
        run(
            "demo-wr-1",
            f"wr0{index}",
            "web-researcher",
            f"Research framework option {index}",
            tokens=200_000,
            tests=False,
            outcome="failed",
            denied_call=(index == 2),
        )

    # --- reviewer: not enough data -----------------------------------------
    for index in range(3):
        run(
            "demo-rv-1",
            f"rv{index:02d}",
            "reviewer",
            f"Review the auth changes {index}",
            tokens=150_000,
            tests=False,
        )

    # --- Explore: keep -------------------------------------------------------
    for index in range(10):
        run(
            "demo-ex-1" if index < 5 else "demo-ex-2",
            f"ex{index:02d}",
            "Explore",
            f"Map the storage module {index}",
            tokens=60_000,
            tests=False,
        )

    demo.finish()


def main() -> int:
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("examples/demo")
    build(root)
    files = sum(1 for _ in root.rglob("*.jsonl"))
    print(f"Wrote {root} ({files} log files)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
