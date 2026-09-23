"""Per-run facts: tool calls, errors vs denials, test detection, commits,
edits, models, and the final report."""

from __future__ import annotations

from pathlib import Path

import pytest
from conftest import (
    agent_call,
    assistant_line,
    sidecar,
    tool_result_line,
    write_run,
    write_session,
)

from agent_scorecard.pricing import load_prices
from agent_scorecard.runs import (
    DEFAULT_TEST_PATTERNS,
    AgentRun,
    build_runs,
    compile_test_patterns,
)
from agent_scorecard.scan import scan


def run_scan(tmp_path: Path, test_patterns: list[str] | None = None) -> tuple[AgentRun, ...]:
    result = scan([tmp_path], test_patterns=test_patterns)
    return build_runs(result.run_facts, result.parsed, load_prices())


def bash_line(request_id: str, command: str, *, agent_id: str | None = None) -> str:
    from conftest import tool_use

    return assistant_line(
        request_id,
        agent_id=agent_id,
        content=[tool_use(f"toolu_{request_id}", "Bash", {"command": command})],
    )


def a_run(tmp_path: Path, agent_id: str, lines: list[str], **meta) -> None:
    defaults = {"agent_type": "code-writer", "spawn_depth": 1}
    defaults.update(meta)
    write_run(tmp_path, "session-1", agent_id, sidecar(**defaults), lines)


def test_tool_calls_are_deduplicated_across_streamed_partials(tmp_path: Path) -> None:
    # A streamed partial can repeat a tool_use block; the block id is one
    # call, not two.
    from conftest import tool_use

    a_run(
        tmp_path,
        "aaa",
        [
            assistant_line(
                "req_1",
                agent_id="aaa",
                content=[tool_use("toolu_1", "Read", {"file_path": "a.py"})],
            ),
            assistant_line(
                "req_1",
                agent_id="aaa",
                content=[tool_use("toolu_1", "Read", {"file_path": "a.py"})],
            ),
            tool_result_line("toolu_1", agent_id="aaa"),
        ],
    )

    (run,) = run_scan(tmp_path)

    assert run.tool_calls == 1
    assert run.tool_errors == 0


def test_errors_and_denials_are_counted_apart(tmp_path: Path) -> None:
    from conftest import tool_use

    a_run(
        tmp_path,
        "aaa",
        [
            assistant_line(
                "req_1",
                agent_id="aaa",
                content=[
                    tool_use("toolu_err", "Bash", {"command": "false"}),
                    tool_use("toolu_denied", "Bash", {"command": "false"}),
                    tool_use("toolu_ok", "Read", {"file_path": "a.py"}),
                ],
            ),
            tool_result_line(
                "toolu_err",
                content="Exit code 1\n2 failed, 40 passed",
                is_error=True,
                agent_id="aaa",
            ),
            tool_result_line(
                "toolu_denied",
                is_error=True,
                denial_kind="permission-rule",
                agent_id="aaa",
            ),
            tool_result_line("toolu_ok", agent_id="aaa"),
        ],
    )

    (run,) = run_scan(tmp_path)

    assert run.tool_calls == 3
    assert run.tool_errors == 1
    assert run.tool_denials == 1


@pytest.mark.parametrize(
    "command",
    [
        "uv run pytest -q",
        "go test ./...",
        "npm test",
        "npm run test",
        "pnpm test",
        "yarn run test",
        "bun test",
        "uvx vitest run",
        "npx jest",
        "cargo test --all",
        "make check",
        "mvn test",
        "./gradlew test",
        "dotnet test",
        "bundle exec rspec",
        "phpunit --testsuite",
        "mix test",
    ],
)
def test_each_builtin_test_pattern_matches(tmp_path: Path, command: str) -> None:
    a_run(
        tmp_path,
        "aaa",
        [
            bash_line("req_1", command, agent_id="aaa"),
            tool_result_line("toolu_req_1", agent_id="aaa"),
        ],
    )

    (run,) = run_scan(tmp_path)

    assert run.test_runs == 1
    assert run.last_test_passed is True


def test_a_non_test_command_is_not_a_test_run(tmp_path: Path) -> None:
    a_run(
        tmp_path,
        "aaa",
        [
            bash_line("req_1", "npm run lint", agent_id="aaa"),
            tool_result_line("toolu_req_1", agent_id="aaa"),
        ],
    )

    (run,) = run_scan(tmp_path)

    assert run.test_runs == 0
    assert run.last_test_passed is None


def test_a_custom_test_command_list_replaces_the_defaults(tmp_path: Path) -> None:
    a_run(
        tmp_path,
        "aaa",
        [
            bash_line("req_1", "mocha test/", agent_id="aaa"),
            tool_result_line("toolu_req_1", agent_id="aaa"),
            bash_line("req_2", "pytest -q", agent_id="aaa"),
            tool_result_line("toolu_req_2", agent_id="aaa"),
        ],
    )

    (run,) = run_scan(tmp_path, test_patterns=[r"\bmocha\b"])

    assert run.test_runs == 1
    assert run.last_test_passed is True


def test_the_last_test_result_wins(tmp_path: Path) -> None:
    a_run(
        tmp_path,
        "aaa",
        [
            bash_line("req_1", "pytest -q", agent_id="aaa"),
            tool_result_line("toolu_req_1", content="fail", is_error=True, agent_id="aaa"),
            bash_line("req_2", "pytest -q", agent_id="aaa"),
            tool_result_line("toolu_req_2", agent_id="aaa"),
        ],
    )

    (run,) = run_scan(tmp_path)

    assert run.test_runs == 2
    assert run.last_test_passed is True


def test_a_failing_last_test_run_is_reported(tmp_path: Path) -> None:
    a_run(
        tmp_path,
        "aaa",
        [
            bash_line("req_1", "pytest -q", agent_id="aaa"),
            tool_result_line("toolu_req_1", content="ok", agent_id="aaa"),
            bash_line("req_2", "pytest -q", agent_id="aaa"),
            tool_result_line("toolu_req_2", content="Exit code 1", is_error=True, agent_id="aaa"),
        ],
    )

    (run,) = run_scan(tmp_path)

    assert run.test_runs == 2
    assert run.last_test_passed is False


def test_a_denied_test_call_does_not_count_as_a_test_run(tmp_path: Path) -> None:
    a_run(
        tmp_path,
        "aaa",
        [
            bash_line("req_1", "pytest -q", agent_id="aaa"),
            tool_result_line(
                "toolu_req_1", is_error=True, denial_kind="permission-rule", agent_id="aaa"
            ),
        ],
    )

    (run,) = run_scan(tmp_path)

    assert run.test_runs == 0
    assert run.last_test_passed is None
    assert run.tool_denials == 1
    assert run.tool_errors == 0


def test_commits_count_successful_git_commit_calls(tmp_path: Path) -> None:
    a_run(
        tmp_path,
        "aaa",
        [
            bash_line("req_1", 'git commit -m "fix"', agent_id="aaa"),
            tool_result_line("toolu_req_1", agent_id="aaa"),
            bash_line("req_2", "git -C /work commit -m 'x'", agent_id="aaa"),
            tool_result_line("toolu_req_2", agent_id="aaa"),
            bash_line("req_3", 'git commit -m "y"', agent_id="aaa"),
            tool_result_line(
                "toolu_req_3", content="nothing to commit", is_error=True, agent_id="aaa"
            ),
            bash_line("req_4", "git push", agent_id="aaa"),
            tool_result_line("toolu_req_4", agent_id="aaa"),
        ],
    )

    (run,) = run_scan(tmp_path)

    assert run.commits == 2


@pytest.mark.parametrize("tool", ["Edit", "MultiEdit", "Write", "NotebookEdit"])
def test_editing_tools_mark_a_run_as_editing(tmp_path: Path, tool: str) -> None:
    from conftest import tool_use

    a_run(
        tmp_path,
        "aaa",
        [
            assistant_line(
                "req_1",
                agent_id="aaa",
                content=[tool_use("toolu_1", tool, {"file_path": "a.py"})],
            ),
            tool_result_line("toolu_1", agent_id="aaa"),
        ],
    )

    (run,) = run_scan(tmp_path)

    assert run.edited_files is True


def test_a_read_only_run_is_not_marked_as_editing(tmp_path: Path) -> None:
    from conftest import tool_use

    a_run(
        tmp_path,
        "aaa",
        [
            assistant_line(
                "req_1",
                agent_id="aaa",
                content=[tool_use("toolu_1", "Read", {"file_path": "a.py"})],
            ),
            tool_result_line("toolu_1", agent_id="aaa"),
        ],
    )

    (run,) = run_scan(tmp_path)

    assert run.edited_files is False


def test_turns_count_distinct_request_ids(tmp_path: Path) -> None:
    a_run(
        tmp_path,
        "aaa",
        [
            assistant_line("req_1", agent_id="aaa"),
            assistant_line("req_1", agent_id="aaa"),
            assistant_line("req_2", agent_id="aaa"),
        ],
    )

    (run,) = run_scan(tmp_path)

    assert run.turns == 2


def test_duration_spans_the_first_and_last_line(tmp_path: Path) -> None:
    a_run(
        tmp_path,
        "aaa",
        [
            assistant_line("req_1", agent_id="aaa", timestamp="2026-09-20T10:00:00.000Z"),
            assistant_line("req_2", agent_id="aaa", timestamp="2026-09-20T10:01:30.000Z"),
        ],
    )

    (run,) = run_scan(tmp_path)

    assert run.duration_seconds == 90.0
    assert run.started_at is not None
    assert run.ended_at is not None


def test_the_final_report_is_the_last_text_block_of_the_last_assistant_line(
    tmp_path: Path,
) -> None:
    from conftest import text_block, tool_use

    a_run(
        tmp_path,
        "aaa",
        [
            assistant_line(
                "req_1",
                agent_id="aaa",
                content=[text_block("interim thought")],
            ),
            assistant_line(
                "req_2",
                agent_id="aaa",
                content=[
                    tool_use("toolu_1", "Read", {"file_path": "a.py"}),
                    text_block("Final report: all tests pass"),
                ],
            ),
        ],
    )

    result = scan([tmp_path])
    facts = result.run_facts

    assert facts[0].final_report == "Final report: all tests pass"


def test_primary_model_is_the_most_billed_with_alphabetical_ties(
    tmp_path: Path,
) -> None:
    a_run(
        tmp_path,
        "aaa",
        [
            # sonnet-5: 1M billed; opus-5: 2M billed -> opus-5 primary.
            assistant_line(
                "req_1", model="claude-sonnet-5", input_tokens=1_000_000, agent_id="aaa"
            ),
            assistant_line("req_2", model="claude-opus-5", input_tokens=2_000_000, agent_id="aaa"),
        ],
    )

    (run,) = run_scan(tmp_path)

    assert run.primary_model == "claude-opus-5"
    assert run.models == ("claude-opus-5", "claude-sonnet-5")

    tie = tmp_path / "tie"
    a_run(
        tie,
        "bbb",
        [
            assistant_line(
                "req_1", model="claude-sonnet-5", input_tokens=1_000_000, agent_id="bbb"
            ),
            assistant_line("req_2", model="claude-opus-5", input_tokens=1_000_000, agent_id="bbb"),
        ],
    )

    (run,) = run_scan(tie)

    assert run.primary_model == "claude-opus-5"  # tie broken alphabetically


def test_synthetic_models_are_excluded_from_the_models_list(tmp_path: Path) -> None:
    a_run(
        tmp_path,
        "aaa",
        [
            assistant_line("req_1", agent_id="aaa"),
            assistant_line("req_2", model="<synthetic>", agent_id="aaa"),
        ],
    )

    (run,) = run_scan(tmp_path)

    assert run.models == ("claude-sonnet-5",)


def test_an_unpriced_record_makes_the_run_cost_unknown(tmp_path: Path) -> None:
    a_run(
        tmp_path,
        "aaa",
        [
            assistant_line("req_1", input_tokens=1_000_000, agent_id="aaa"),
            assistant_line("req_2", model="model-no-row", input_tokens=5, agent_id="aaa"),
        ],
    )

    (run,) = run_scan(tmp_path)

    assert run.cost_microusd is None
    assert run.unpriced_records == 1
    assert run.unpriced_models == ("model-no-row",)


def test_full_rebuild_turns_and_cache_share_join_from_records(
    tmp_path: Path,
) -> None:
    a_run(
        tmp_path,
        "aaa",
        [
            # A full rebuild: the whole prefix written fresh.
            assistant_line("req_1", input_tokens=0, cache_write_1h=1_000_000, agent_id="aaa"),
            # Warm growth: mostly cache reads.
            assistant_line("req_2", input_tokens=10, cache_read=1_000_000, agent_id="aaa"),
        ],
    )

    (run,) = run_scan(tmp_path)

    assert run.full_rebuild_turns == 1
    assert 0.4 < run.cache_read_share < 0.6


def test_the_built_in_patterns_compile() -> None:
    # A typo in the shipped list would silently disable test detection.
    patterns = compile_test_patterns(DEFAULT_TEST_PATTERNS)

    assert len(patterns) == len(DEFAULT_TEST_PATTERNS)


def test_an_agent_call_in_the_parents_log_is_not_the_runs_own_tool(tmp_path: Path) -> None:
    # The parent's Agent tool call lives in the parent's file; it must not
    # inflate the run's own tool-call count.
    write_session(
        tmp_path,
        [agent_call("toolu_01PARENT", subagent_type="code-writer")],
    )
    a_run(tmp_path, "aaa", [assistant_line("req_1", agent_id="aaa")])

    (run,) = run_scan(tmp_path)

    assert run.tool_calls == 0
