"""Golden-file tests: the CLI's output on the committed demo data, plus a
guard that `make_demo.py` still reproduces `examples/demo/` exactly.

Set `AGENT_SCORECARD_UPDATE_GOLDEN=1` to rewrite the golden files after an
intentional output change."""

from __future__ import annotations

import datetime
import os
import subprocess
import sys
from pathlib import Path

import pytest

from agent_scorecard.cli import main

ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "examples" / "demo"
# The CLI prints sources exactly as given, so goldens must be generated from
# the project root with a relative path — never an absolute one.
DEMO_ARG = "examples/demo"
GOLDEN = Path(__file__).parent / "golden"
CLOCK = datetime.datetime(2026, 9, 23, 12, 0, tzinfo=datetime.UTC)

GOLDEN_CASES = [
    ("report_json", ("report", "--format", "json"), "json"),
    ("report_markdown", ("report", "--format", "markdown"), "md"),
    ("runs_json", ("runs", "--format", "json"), "json"),
    ("cost_json", ("cost", "--by-day", "--format", "json"), "json"),
    ("cost_markdown", ("cost", "--by-day", "--format", "markdown"), "md"),
]


def _run_cli(*argv: str) -> str:
    import io
    from contextlib import redirect_stdout

    os.chdir(ROOT)
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        code = main(list(argv), now=CLOCK)
    assert code == 0, f"CLI exited {code} for {argv}"
    return buffer.getvalue()


def _golden_path(name: str, extension: str) -> Path:
    return GOLDEN / f"{name}.{extension}"


@pytest.mark.parametrize(
    ("name", "argv", "extension"), GOLDEN_CASES, ids=[case[0] for case in GOLDEN_CASES]
)
def test_cli_output_matches_the_golden_files(
    name: str, argv: str, extension: str, tmp_path: Path
) -> None:
    output = _run_cli(*argv, "--transcripts", DEMO_ARG)
    golden = _golden_path(name, extension)
    if os.environ.get("AGENT_SCORECARD_UPDATE_GOLDEN") == "1":
        golden.parent.mkdir(parents=True, exist_ok=True)
        golden.write_text(output, encoding="utf-8")
    assert golden.exists(), f"missing golden file {golden}"
    assert output == golden.read_text(encoding="utf-8"), (
        f"output for {name} drifted from {golden}; if the change is "
        f"intentional, regenerate with AGENT_SCORECARD_UPDATE_GOLDEN=1"
    )


def test_the_report_table_smoke() -> None:
    output = _run_cli("report", "--transcripts", DEMO_ARG, "--min-runs", "5")

    assert "Agent scorecard" in output
    for verdict in ("remove", "fix", "keep", "not enough data"):
        assert verdict in output


def test_make_demo_reproduces_the_committed_demo(tmp_path: Path) -> None:
    """`examples/demo/` must stay in lockstep with its generator."""
    destination = tmp_path / "demo"
    subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "make_demo.py"), str(destination)],
        check=True,
        capture_output=True,
        timeout=120,
    )

    committed = sorted(path.relative_to(DEMO) for path in DEMO.rglob("*") if path.is_file())
    regenerated = sorted(
        path.relative_to(destination) for path in destination.rglob("*") if path.is_file()
    )
    assert committed == regenerated, "the committed demo does not match make_demo.py"
    for relative in committed:
        assert (DEMO / relative).read_bytes() == (destination / relative).read_bytes(), (
            f"{relative} drifted; regenerate examples/demo with "
            f"`uv run python scripts/make_demo.py`"
        )
