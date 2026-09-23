"""The `cost` command and the refusal gates: a scan that cannot reconcile,
or that lost too much of the corpus, refuses to report instead of printing
a confident wrong number."""

from __future__ import annotations

import datetime
import json
from pathlib import Path

import pytest
from conftest import (
    assistant_line,
    sidecar,
    write_run,
    write_session,
)

from agent_scorecard.cli import main

_NOW = datetime.datetime(2026, 9, 23, 12, 0, tzinfo=datetime.UTC)


def run_cli(capsys: pytest.CaptureFixture[str], *argv: str) -> tuple[int, str, str]:
    """Run the CLI once and return (exit code, stdout, stderr); the capture
    is drained exactly once here, so tests must read the returned text."""
    code = main(list(argv), now=_NOW)
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def populated_dir(tmp_path: Path) -> None:
    write_session(
        tmp_path,
        [
            assistant_line(
                request_id="req_main",
                model="claude-sonnet-5",
                input_tokens=1_000_000,
                output_tokens=0,
            )
        ],
    )
    write_run(
        tmp_path,
        "session-1",
        "aaa",
        sidecar("Explore", 1),
        [assistant_line(request_id="req_sub", input_tokens=3_000_000)],
    )


def test_fails_fast_on_a_missing_or_empty_folder(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # A typo'd path reported as "$0.00, 0 records" is indistinguishable from
    # a genuinely idle period.
    code, _, err = run_cli(capsys, "cost", "--transcripts", str(tmp_path / "nope"))
    assert code == 1
    assert "does not exist" in err

    empty = tmp_path / "empty"
    empty.mkdir()
    code, _, err = run_cli(capsys, "cost", "--transcripts", str(empty))
    assert code == 1
    assert "No log files found" in err


def test_reports_a_scan_it_could_complete(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    populated_dir(tmp_path)

    code, out, _ = run_cli(capsys, "cost", "--transcripts", str(tmp_path))

    assert code == 0
    # 1M input at $2/MTok for the main session, 3M at $2/MTok for the agent.
    assert "$2.00" in out
    assert "$6.00" in out


def test_fails_when_every_file_is_unreadable(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # Reporting "$0.00" for a scan that read nothing is the worst possible
    # output a measurement tool can produce.
    (tmp_path / "torn.jsonl").write_bytes(b"\xff\xfe not utf-8")

    code, _, err = run_cli(capsys, "cost", "--transcripts", str(tmp_path))
    assert code == 1
    # The lines-read gate fires first: nothing was read at all.
    assert "Read 0 lines" in err


def test_fails_when_lines_decode_to_no_records(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # A `type` discriminator rename would route every billable line into the
    # non-assistant bucket; that is a schema break, not an idle period.
    write_session(tmp_path, ['{"kind":"assistant","requestId":"req_1","message":{}}'])

    code, _, err = run_cli(capsys, "cost", "--transcripts", str(tmp_path))
    assert code == 1
    assert "decoded 0 usage records" in err


def test_fails_when_since_filters_out_every_file(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = write_session(tmp_path, [assistant_line()])
    stale = datetime.datetime(2020, 1, 1, tzinfo=datetime.UTC).timestamp()
    import os

    os.utime(path, (stale, stale))

    code, _, err = run_cli(capsys, "cost", "--transcripts", str(tmp_path), "--since", "2030-01-01")
    assert code == 1
    assert "Read 0 lines" in err


def test_refuses_when_most_lines_lose_their_type_discriminator(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # The break the refusal gate exists to catch: a renamed `type` key
    # produces ZERO undecodable lines, so a numerator counting only those
    # sees a clean scan.
    lines = [f'{{"kind":"assistant","requestId":"req_r{i}","message":{{}}}}' for i in range(6)] + [
        assistant_line(request_id=f"req_ok{i}", input_tokens=1000) for i in range(4)
    ]
    write_session(tmp_path, lines)

    code, _, err = run_cli(capsys, "cost", "--transcripts", str(tmp_path))
    assert code == 1
    assert "Lost" in err


def test_refuses_when_most_assistant_lines_will_not_decode(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    lines = [
        '{"type":"assistant","requestId":"req_x%d","timestamp":"'
        + "2026-09-20T10:15:00.000Z"
        + '","message":{}}'
    ]
    broken = [lines[0] % i for i in range(6)] + [
        assistant_line(request_id=f"req_ok{i}", input_tokens=1000) for i in range(4)
    ]
    write_session(tmp_path, broken)

    code, _, _ = run_cli(capsys, "cost", "--transcripts", str(tmp_path))
    assert code == 1


def test_reports_a_corpus_lost_exactly_at_the_refusal_threshold(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # The comparison is strictly greater-than: half a corpus lost is the
    # last share still reported.
    broken = [
        (
            '{"type":"assistant","requestId":"req_x%d","timestamp":"'
            + "2026-09-20T10:15:00.000Z"
            + '","message":{}}'
        )
        % i
        for i in range(5)
    ] + [assistant_line(request_id=f"req_ok{i}", input_tokens=1000) for i in range(5)]
    write_session(tmp_path, broken)

    code, _, _ = run_cli(capsys, "cost", "--transcripts", str(tmp_path))
    assert code == 0


def test_a_bad_price_file_fails_with_one_clear_sentence(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    populated_dir(tmp_path)
    prices = tmp_path / "bad.toml"
    prices.write_text("[models\n", encoding="utf-8")

    code, _, err = run_cli(capsys, "cost", "--transcripts", str(tmp_path), "--prices", str(prices))

    assert code == 1
    assert "not valid TOML" in err
    assert "Traceback" not in err


def test_verbose_prints_the_accounting(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    populated_dir(tmp_path)

    code, _, err = run_cli(capsys, "cost", "--transcripts", str(tmp_path), "-v")

    assert code == 0
    assert "Files: 2 globbed" in err
    assert "Lines: 2 read" in err


def test_json_output_is_deterministic_and_typed(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    populated_dir(tmp_path)
    argv = ["cost", "--transcripts", str(tmp_path), "--format", "json"]

    code, first, _ = run_cli(capsys, *argv)
    assert code == 0
    code, again, _ = run_cli(capsys, *argv)
    assert code == 0
    assert again == first

    payload = json.loads(first)
    assert payload["schema_version"] == 1
    assert payload["command"] == "cost"
    assert payload["generated_at"] == _NOW.isoformat()
    assert payload["prices"]["checked_on"] == "2026-09-23"
    # Money goes out as both integer microusd and float usd.
    assert payload["by_scope"]["main_thread"]["total_cost_microusd"] == 2_000_000
    assert payload["by_scope"]["main_thread"]["total_cost_usd"] == 2.0
    assert payload["by_scope"]["subagent"]["total_cost_microusd"] == 6_000_000
    assert payload["totals"]["record_count"] == 2
    assert payload["by_agent_type"]["Explore"]["record_count"] == 1
    assert payload["accounting"]["lines"]["lines_read"] == 2


def test_the_window_filters_records_by_their_own_timestamp(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    populated_dir(tmp_path)

    code, out, _ = run_cli(
        capsys,
        "cost",
        "--transcripts",
        str(tmp_path),
        "--since",
        "2026-09-21",
        "--until",
        "2026-09-30",
    )

    assert code == 0
    assert "No usage records fall in this window." in out


def test_markdown_output_is_a_github_table(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    populated_dir(tmp_path)

    code, out, _ = run_cli(capsys, "cost", "--transcripts", str(tmp_path), "--format", "markdown")

    assert code == 0
    assert "| Scope | Records | Cost |" in out
    assert "| Main sessions | 1 | $2.00 |" in out
    assert "| Agents | 1 | $6.00 |" in out


def test_by_day_splits_the_table(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    populated_dir(tmp_path)

    code, out, _ = run_cli(capsys, "cost", "--transcripts", str(tmp_path), "--by-day")

    assert code == 0
    assert "By day" in out
    assert "2026-09-20" in out


def test_unpriced_models_are_reported_not_zeroed(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    write_session(tmp_path, [assistant_line(request_id="req_a", input_tokens=1000)])
    write_run(
        tmp_path,
        "session-1",
        "bbb",
        sidecar("Unpriced", 1),
        [assistant_line(request_id="req_dark", model="model-no-row", input_tokens=1)],
    )

    code, out, _ = run_cli(capsys, "cost", "--transcripts", str(tmp_path), "--format", "json")

    assert code == 0
    payload = json.loads(out)
    # Only the unpriced group exists in this scan.
    unpriced = payload["by_agent_type"]
    assert "Explore" not in unpriced
    assert unpriced["Unpriced"]["total_cost_microusd"] == 0
    assert unpriced["Unpriced"]["unpriced_record_count"] == 1
    assert unpriced["Unpriced"]["unpriced_models"] == ["model-no-row"]
