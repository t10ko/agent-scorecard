"""Lifecycle resolution: foreground results, background notifications,
workflow journals, the sidecar's stoppedByUser, and inference from the
run's own file."""

from __future__ import annotations

from pathlib import Path

import pytest
from conftest import (
    agent_call,
    async_launch,
    finished_runs,
    foreground_result,
    journal_line,
    notification,
    sidecar,
    text_block,
    tool_result_line,
    write_journal,
    write_run,
    write_session,
)

from agent_scorecard.lifecycle import Lifecycle, LifecycleSource, reason_from_summary
from agent_scorecard.runs import AgentRun


def by_id(runs: tuple[AgentRun, ...], agent_id: str) -> AgentRun:
    return next(run for run in runs if run.agent_id == agent_id)


def one_run(tmp_path: Path, agent_id: str = "aaa", **meta) -> AgentRun:
    runs = finished_runs(tmp_path)
    assert len(runs) == 1, f"expected one run, got {len(runs)}"
    return runs[0]


def started_run(
    tmp_path: Path,
    agent_id: str = "aaa",
    agent_type: str = "code-writer",
    spawn_depth: int = 1,
    description: str | None = "Fix the flaky login test",
    stopped_by_user: bool = False,
) -> None:
    """Write one run's log and sidecar, plus the parent's Agent call."""
    write_run(
        tmp_path,
        "session-1",
        agent_id,
        sidecar(
            agent_type,
            spawn_depth,
            description=description,
            stopped_by_user=stopped_by_user,
        ),
        [agent_call("toolu_01PARENT", tool_name="Agent")],
    )


def test_a_foreground_completed_result_records_the_lifecycle(tmp_path: Path) -> None:
    started_run(tmp_path)
    write_session(
        tmp_path,
        [foreground_result("toolu_01PARENT", "aaa", report_text="<final report>")],
    )

    run = one_run(tmp_path)

    assert run.lifecycle is Lifecycle.COMPLETED
    assert run.lifecycle_source is LifecycleSource.FOREGROUND
    assert run.reason is None


def test_a_foreground_errored_result_is_failed_with_the_error_reason(
    tmp_path: Path,
) -> None:
    started_run(tmp_path)
    write_session(
        tmp_path,
        [
            tool_result_line(
                "toolu_01PARENT",
                content=[text_block("Agent hit an API error\ndropped connection")],
                is_error=True,
                tool_use_result={"status": "error"},
            )
        ],
    )

    run = one_run(tmp_path)

    assert run.lifecycle is Lifecycle.FAILED
    assert run.lifecycle_source is LifecycleSource.FOREGROUND
    assert run.reason == "Agent hit an API error"


def test_launched_then_completed_notification(tmp_path: Path) -> None:
    # The background launch itself is not an outcome; the notification that
    # arrives later is.
    started_run(tmp_path)
    write_session(
        tmp_path,
        [
            async_launch("toolu_01PARENT", "aaa"),
            notification("aaa", "toolu_01PARENT", status="completed"),
        ],
    )

    run = one_run(tmp_path)

    assert run.lifecycle is Lifecycle.COMPLETED
    assert run.lifecycle_source is LifecycleSource.NOTIFICATION


def test_a_failed_notification_reason_never_contains_the_description(
    tmp_path: Path,
) -> None:
    started_run(tmp_path, description="Fix the flaky login test")
    write_session(
        tmp_path,
        [
            notification(
                "aaa",
                "toolu_01PARENT",
                status="failed",
                description="Fix the flaky login test",
                summary='Agent "Fix the flaky login test" failed: Agent stalled: '
                "no progress for 600s",
            )
        ],
    )

    run = one_run(tmp_path)

    assert run.lifecycle is Lifecycle.FAILED
    assert "Fix the flaky login test" not in (run.reason or "")
    assert run.reason == "failed: Agent stalled: no progress for 600s"


def test_the_reason_prefix_is_stripped_without_a_sidecar_description(
    tmp_path: Path,
) -> None:
    # The sidecar describes nothing; the summary still leads with a quoted
    # description, so everything up to the last quote before the marker goes.
    started_run(tmp_path, description=None)
    write_session(
        tmp_path,
        [
            notification(
                "aaa",
                "toolu_01PARENT",
                status="failed",
                summary='Agent "A very long task description" failed: unreachable server',
            )
        ],
    )

    run = one_run(tmp_path)

    assert run.reason == "failed: unreachable server"
    assert "A very long task description" not in run.reason


def test_a_killed_notification(tmp_path: Path) -> None:
    started_run(tmp_path)
    write_session(
        tmp_path,
        [
            notification(
                "aaa",
                "toolu_01PARENT",
                status="killed",
                summary='Agent "Fix the flaky login test" was stopped by user',
            )
        ],
    )

    run = one_run(tmp_path)

    assert run.lifecycle is Lifecycle.KILLED
    assert run.reason == "was stopped by user"


def test_a_stopped_notification(tmp_path: Path) -> None:
    started_run(tmp_path)
    write_session(
        tmp_path,
        [notification("aaa", "toolu_01PARENT", status="stopped")],
    )

    run = one_run(tmp_path)

    assert run.lifecycle is Lifecycle.STOPPED


def test_the_latest_notification_wins(tmp_path: Path) -> None:
    # A stopped agent can be resumed, so the same task id can notify more
    # than once; the newest record is the truth.
    started_run(tmp_path)
    write_session(
        tmp_path,
        [
            notification(
                "aaa",
                "toolu_01PARENT",
                status="failed",
                summary='Agent "task" failed: Agent stalled: no progress for 600s',
                timestamp="2026-09-20T10:00:00.000Z",
            ),
            notification(
                "aaa",
                "toolu_01PARENT",
                status="completed",
                timestamp="2026-09-20T10:05:00.000Z",
            ),
        ],
    )

    run = one_run(tmp_path)

    assert run.lifecycle is Lifecycle.COMPLETED


def test_copies_in_queue_and_attachment_lines_are_ignored(tmp_path: Path) -> None:
    # The same notification text is copied into `queue-operation` and
    # `attachment` lines; reading those would count every outcome twice.
    import json

    started_run(tmp_path)
    session_lines = [notification("aaa", "toolu_01PARENT", status="completed")]
    for kind in ("queue-operation", "attachment"):
        line = json.loads(session_lines[0])
        line["type"] = kind
        session_lines.append(json.dumps(line))
    write_session(tmp_path, session_lines)

    runs = finished_runs(tmp_path)

    assert len(runs) == 1
    assert runs[0].lifecycle is Lifecycle.COMPLETED


def test_a_background_command_notification_is_ignored(tmp_path: Path) -> None:
    # Not every notification is about an agent run; ones whose task-id
    # matches no run say nothing this tool should report.
    started_run(tmp_path)
    write_session(
        tmp_path,
        [
            notification(
                "bash_1",
                "toolu_other",
                status="failed",
                summary='Background command "deploy.sh" failed with exit code 2',
            ),
            notification("aaa", "toolu_01PARENT", status="completed"),
        ],
    )

    run = one_run(tmp_path)

    assert run.lifecycle is Lifecycle.COMPLETED


def test_a_journal_result_records_completion(tmp_path: Path) -> None:
    write_journal(tmp_path, "session-1", "wf_1", [journal_line("result", "aaa")])
    write_run(
        tmp_path,
        "session-1",
        "aaa",
        sidecar("code-writer", 1),
        [],
        workflow="wf_1",
    )

    run = one_run(tmp_path)

    assert run.lifecycle is Lifecycle.COMPLETED
    assert run.lifecycle_source is LifecycleSource.WORKFLOW_JOURNAL


def test_a_journal_failure_records_failure(tmp_path: Path) -> None:
    write_journal(tmp_path, "session-1", "wf_1", [journal_line("failed", "aaa")])
    write_run(
        tmp_path,
        "session-1",
        "aaa",
        sidecar("code-writer", 1),
        [],
        workflow="wf_1",
    )

    run = one_run(tmp_path)

    assert run.lifecycle is Lifecycle.FAILED
    assert run.lifecycle_source is LifecycleSource.WORKFLOW_JOURNAL


def test_a_journal_started_only_leaves_the_lifecycle_inferred(tmp_path: Path) -> None:
    # "started" is not an outcome; with nothing else recorded, the run's
    # own file decides.
    write_journal(tmp_path, "session-1", "wf_1", [journal_line("started", "aaa")])
    write_run(
        tmp_path,
        "session-1",
        "aaa",
        sidecar("code-writer", 1),
        [__import__("conftest").assistant_line("req_1", agent_id="aaa")],
        workflow="wf_1",
    )

    run = one_run(tmp_path)

    assert run.lifecycle is Lifecycle.COMPLETED
    assert run.lifecycle_source is LifecycleSource.INFERRED


def test_a_stopped_by_user_sidecar_records_killed(tmp_path: Path) -> None:
    started_run(tmp_path, stopped_by_user=True)

    run = one_run(tmp_path)

    assert run.lifecycle is Lifecycle.KILLED
    assert run.lifecycle_source is LifecycleSource.SIDECAR
    assert run.reason == "stopped by user"


def test_a_nested_agents_lifecycle_is_found_in_its_parents_file(
    tmp_path: Path,
) -> None:
    # A nested agent reports to its parent agent's log, not the session's.
    write_run(
        tmp_path,
        "session-1",
        "parent",
        sidecar("orchestrator", 1),
        [
            agent_call("toolu_nested", description="child work"),
            foreground_result("toolu_nested", "child", report_text="child done"),
        ],
    )
    write_run(
        tmp_path,
        "session-1",
        "child",
        sidecar("code-writer", 2, parent_agent_id="parent", tool_use_id="toolu_nested"),
        [],
    )

    runs = finished_runs(tmp_path)

    child = by_id(runs, "child")
    assert child.lifecycle is Lifecycle.COMPLETED
    assert child.lifecycle_source is LifecycleSource.FOREGROUND


def test_the_legacy_task_tool_name_still_resolves(tmp_path: Path) -> None:
    # Attribution goes through the sidecar's toolUseId, never the tool's
    # name, so the older `Task` name changes nothing.
    write_run(
        tmp_path,
        "session-1",
        "parent",
        sidecar("orchestrator", 1),
        [
            agent_call("toolu_nested", tool_name="Task", description="child work"),
            foreground_result("toolu_nested", "child", report_text="child done"),
        ],
    )
    write_run(
        tmp_path,
        "session-1",
        "child",
        sidecar("code-writer", 2, parent_agent_id="parent", tool_use_id="toolu_nested"),
        [],
    )

    runs = finished_runs(tmp_path)

    assert by_id(runs, "child").lifecycle is Lifecycle.COMPLETED


def test_a_foreground_result_without_an_agent_id_maps_through_the_sidecar(
    tmp_path: Path,
) -> None:
    # Where only the tool-call id is known, the sidecar's toolUseId maps it
    # to the agent.
    write_session(
        tmp_path,
        [foreground_result("toolu_01PARENT", "", report_text="done")],
    )
    write_run(
        tmp_path,
        "session-1",
        "aaa",
        sidecar("code-writer", 1, tool_use_id="toolu_01PARENT"),
        [],
    )

    run = one_run(tmp_path)

    assert run.lifecycle is Lifecycle.COMPLETED
    assert run.lifecycle_source is LifecycleSource.FOREGROUND


def test_an_end_turn_final_turn_infers_completion(tmp_path: Path) -> None:
    started_run(tmp_path)

    run = one_run(tmp_path)

    assert run.lifecycle is Lifecycle.COMPLETED
    assert run.lifecycle_source is LifecycleSource.INFERRED


def test_an_api_error_on_the_last_assistant_line_infers_failure(
    tmp_path: Path,
) -> None:
    write_run(
        tmp_path,
        "session-1",
        "aaa",
        sidecar("code-writer", 1),
        [
            __import__("conftest").assistant_line(
                "req_1",
                agent_id="aaa",
                is_api_error=True,
                stop_reason="end_turn",
            )
        ],
    )

    run = one_run(tmp_path)

    assert run.lifecycle is Lifecycle.FAILED
    assert run.lifecycle_source is LifecycleSource.INFERRED
    assert run.reason == "API error"


def test_a_run_with_no_record_and_no_signal_is_unknown(tmp_path: Path) -> None:
    started_run(tmp_path)
    # Overwrite the file with a single line that carries no stop_reason.
    from conftest import assistant_line

    write_run(
        tmp_path,
        "session-1",
        "aaa",
        sidecar("code-writer", 1),
        [assistant_line("req_1", agent_id="aaa", stop_reason=None)],
    )

    run = one_run(tmp_path)

    assert run.lifecycle is Lifecycle.UNKNOWN
    assert run.lifecycle_source is LifecycleSource.NONE


def test_a_timestamped_event_beats_a_journal_event(tmp_path: Path) -> None:
    # Journal lines carry no timestamp, so they rank below any timestamped
    # record for the same run.
    write_journal(tmp_path, "session-1", "wf_1", [journal_line("failed", "aaa")])
    write_run(
        tmp_path,
        "session-1",
        "aaa",
        sidecar("code-writer", 1),
        [],
        workflow="wf_1",
    )
    write_session(
        tmp_path,
        [
            notification("aaa", "toolu_01PARENT", status="completed"),
        ],
    )

    run = one_run(tmp_path)

    assert run.lifecycle is Lifecycle.COMPLETED
    assert run.lifecycle_source is LifecycleSource.NOTIFICATION


def test_reason_from_summary_helpers() -> None:
    assert reason_from_summary('Agent "task one" finished', description="task one") == "finished"
    assert (
        reason_from_summary('Agent "task one" was stopped by user', description="task one")
        == "was stopped by user"
    )
    assert reason_from_summary("no prefix here", description=None) == "no prefix here"


@pytest.mark.parametrize(
    ("summary", "expected_fragment"),
    [
        ('Agent "d" failed: ' + "x" * 300, "failed:"),
    ],
)
def test_a_reason_is_capped_at_120_characters(summary: str, expected_fragment: str) -> None:
    reason = reason_from_summary(summary, description="d")
    assert len(reason) <= 120
    assert reason.startswith(expected_fragment)
