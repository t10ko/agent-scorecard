"""The `report` and `runs` commands: output shapes, filters, and the
"nothing found" path that must never print a table of zeros."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
from conftest import (
    assistant_line,
    notification,
    sidecar,
    write_run,
    write_session,
)

from agent_scorecard.cli import main

_NOW = 1_700_000_000_000  # not used; a fixed clock is passed below


def completed_run(
    tmp_path: Path,
    agent_id: str,
    *,
    agent_type: str = "code-writer",
    description: str | None = "Fix the flaky login test",
    input_tokens: int = 1_000_000,
    report_text: str = "All tests pass",
    status: str = "completed",
    started: str | None = None,
) -> None:
    """One run that the parent recorded as completed via a notification."""
    run_lines = [
        assistant_line(
            f"req_{agent_id}",
            agent_id=agent_id,
            input_tokens=input_tokens,
            timestamp=started,
            content=[
                {"type": "text", "text": "working"},
                {"type": "text", "text": report_text},
            ],
        )
    ]
    meta = sidecar(agent_type, 1, description=description)
    write_run(tmp_path, "session-1", agent_id, meta, run_lines)
    write_session(
        tmp_path,
        [
            notification(
                agent_id,
                f"toolu_{agent_id}",
                status=status,
                result_text=report_text,
                timestamp=started,
            )
        ],
        name=f"session-{agent_id}",
    )


def run_cli(capsys: pytest.CaptureFixture[str], *argv: str) -> tuple[int, str, str]:
    from datetime import UTC, datetime

    code = main(list(argv), now=datetime(2026, 9, 23, tzinfo=UTC))
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def test_report_shows_groups_and_verdicts(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    for i in range(5):
        completed_run(tmp_path, f"aaa{i}", agent_type="code-writer")

    code, out, _err = run_cli(capsys, "report", "--transcripts", str(tmp_path), "--min-runs", "5")

    assert code == 0
    assert "Agent scorecard" in out
    assert "code-writer" in out
    assert "keep: 5 of 5 decided runs succeeded (100%)" in out
    assert "agents" in out and "main sessions" in out
    assert "Prices checked 2026-09-23" in out
    assert "starting point for a human decision" not in out  # table keeps it lean


def test_report_with_no_runs_prints_a_message_not_a_table(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    write_session(tmp_path, [assistant_line()])

    code, out, _ = run_cli(capsys, "report", "--transcripts", str(tmp_path))

    assert code == 0
    assert "No agent runs found in this window." in out
    assert "Verdict" not in out


def test_report_json_is_deterministic_and_typed(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    for i in range(6):
        completed_run(tmp_path, f"aaa{i}", agent_type="code-writer")
    argv = [
        "report",
        "--transcripts",
        str(tmp_path),
        "--min-runs",
        "5",
        "--format",
        "json",
    ]

    code, first, _ = run_cli(capsys, *argv)
    assert code == 0
    _, again, _ = run_cli(capsys, *argv)
    assert again == first

    payload = json.loads(first)
    assert payload["schema_version"] == 1
    assert payload["command"] == "report"
    assert payload["prices"]["checked_on"] == "2026-09-23"
    assert payload["thresholds"]["min_runs"] == 5
    (group,) = payload["groups"]
    assert group["key"] == "code-writer"
    assert group["verdict"] == "keep"
    assert group["succeeded"] == 6
    assert group["cost_microusd"] == 6 * 2_000_000  # 6 runs at 1M input, $2/MTok
    assert group["cost_usd"] == 12.0
    assert "accounting" in payload


def test_report_markdown_is_a_github_table(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    for i in range(5):
        completed_run(tmp_path, f"aaa{i}", agent_type="code-writer")

    code, out, _ = run_cli(
        capsys,
        "report",
        "--transcripts",
        str(tmp_path),
        "--min-runs",
        "5",
        "--format",
        "markdown",
    )

    assert code == 0
    assert "| Group | Runs | Success | Cost | $/success | Tool errors | " in out
    assert "| code-writer | 5 | 5/5 100% |" in out
    assert "not an automatic kill switch" in out


def test_a_failure_pattern_marks_completed_runs_failed(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    for i in range(3):
        completed_run(tmp_path, f"good{i}", agent_type="code-writer")
    for i in range(3):
        completed_run(
            tmp_path,
            f"bad{i}",
            agent_type="code-writer",
            report_text="gave up. TASK FAILED after 3 attempts",
        )

    code, out, _ = run_cli(
        capsys,
        "report",
        "--transcripts",
        str(tmp_path),
        "--min-runs",
        "5",
        "--failure-pattern",
        "TASK FAILED",
    )

    assert code == 0
    assert "3/6 50%" in out
    assert "fix: 3 of 6 decided runs succeeded (50%), below 80%" in out


def test_runs_lists_runs_newest_first(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    from datetime import UTC, datetime

    for i, minute in enumerate((1, 3, 2)):
        stamp = datetime(2026, 9, 20, 10, minute, tzinfo=UTC).isoformat().replace("+00:00", "Z")
        completed_run(tmp_path, f"aaa{i}", started=stamp)

    code, out, _ = run_cli(capsys, "runs", "--transcripts", str(tmp_path))

    assert code == 0
    assert out.index("(aaa1)") < out.index("(aaa2)") < out.index("(aaa0)")


def test_runs_filters_by_outcome_and_type_and_limit(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    for i in range(4):
        completed_run(tmp_path, f"w{i}", agent_type="code-writer")
    for i in range(3):
        completed_run(
            tmp_path,
            f"r{i}",
            agent_type="reviewer",
            status="failed",
            report_text="nope",
        )

    code, out, _ = run_cli(
        capsys, "runs", "--transcripts", str(tmp_path), "--type", "reviewer", "--limit", "2"
    )
    assert code == 0
    assert out.count("(r") == 2
    assert "(w" not in out

    code, out, _ = run_cli(capsys, "runs", "--transcripts", str(tmp_path), "--outcome", "failed")
    assert code == 0
    assert out.count("(r") == 3
    assert out.count("(w") == 0


def test_descriptions_are_hidden_unless_asked_for(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    completed_run(tmp_path, "aaa0", description="Fix the flaky login test")
    completed_run(tmp_path, "aaa1", description="Another task")

    code, out, _ = run_cli(capsys, "runs", "--transcripts", str(tmp_path))
    assert code == 0
    assert "Fix the flaky login test" not in out
    assert "Another task" not in out

    code, out, _ = run_cli(capsys, "runs", "--transcripts", str(tmp_path), "--show-descriptions")
    assert code == 0
    assert "Fix the flaky login test" in out
    assert "Another task" in out


def test_runs_json_includes_lifecycle_and_outcome(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    completed_run(tmp_path, "aaa0")

    code, out, _ = run_cli(capsys, "runs", "--transcripts", str(tmp_path), "--format", "json")

    assert code == 0
    payload = json.loads(out)
    (run,) = payload["runs"]
    assert run["lifecycle"] == "completed"
    assert run["lifecycle_source"] == "notification"
    assert run["outcome"] == "succeeded"
    # Descriptions never appear unless asked for, even in json.
    assert "description" not in run


def test_usage_errors_exit_2(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["report", "--nope"])
    assert exc.value.code == 2


def test_python_dash_m_smoke(tmp_path: Path) -> None:
    proc = subprocess.run(
        [sys.executable, "-m", "agent_scorecard", "--help"],
        capture_output=True,
        text=True,
        timeout=60,
        env={"PATH": "/usr/bin:/bin", "PYTHONPATH": "src"},
    )
    assert proc.returncode == 0
    assert "agent-scorecard" in proc.stdout
