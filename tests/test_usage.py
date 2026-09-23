"""De-duplication and line accounting: every line a scan reads is either a
kept record, collapsed into one, or counted by the reason it was dropped."""

from __future__ import annotations

import datetime
import json
import logging

import pytest
from conftest import assistant_line, parse_lines, ts
from pydantic import ValidationError

from agent_scorecard.logfile import decode_line
from agent_scorecard.models import SubagentOrigin, UsageSummary
from agent_scorecard.usage import UsageParser


def test_repeated_content_blocks_collapse_into_one_record() -> None:
    # A real log writes the same assistant turn once per content block it
    # emits, and every one of those lines repeats the same requestId.
    line = assistant_line(
        input_tokens=2, cache_write_5m=52_526, cache_read=29_955, output_tokens=338
    )

    parsed = parse_lines(line, line, line)

    assert len(parsed.records) == 1
    assert parsed.conflicting_request_ids == 0
    assert parsed.duplicate_lines_collapsed == 2
    assert parsed.lines_read == 3
    record = parsed.records[0]
    assert record.request_id == "req_1"
    assert record.model == "claude-sonnet-5"
    assert record.input_tokens == 2
    assert record.cache_write_1h_tokens == 0
    assert record.cache_write_5m_tokens == 52_526
    assert record.cache_read_tokens == 29_955
    assert record.output_tokens == 338


def test_a_flat_cache_write_total_lands_in_the_5m_bucket() -> None:
    # No `cache_creation` breakdown on this line: the whole flat total is
    # the API's default TTL (5 minutes).
    line = json.dumps(
        {
            "type": "assistant",
            "timestamp": ts(),
            "requestId": "req_1",
            "message": {
                "model": "claude-sonnet-5",
                "usage": {
                    "input_tokens": 1,
                    "cache_creation_input_tokens": 500,
                    "cache_read_input_tokens": 0,
                    "output_tokens": 2,
                },
            },
        }
    )

    parsed = parse_lines(line)

    assert parsed.records[0].cache_write_1h_tokens == 0
    assert parsed.records[0].cache_write_5m_tokens == 500


def test_the_largest_of_conflicting_duplicates_wins() -> None:
    # Resuming or forking a session replays a prior turn into a new file,
    # and the replayed copy can report zeroed totals. Whichever file sorts
    # first must not decide which copy is believed.
    zeroed_copy = assistant_line(cache_write_5m=261)
    real_copy = assistant_line(
        input_tokens=2, cache_write_5m=261, cache_read=361_863, output_tokens=871
    )

    parsed = parse_lines(zeroed_copy, real_copy)

    assert len(parsed.records) == 1
    assert parsed.conflicting_request_ids == 1
    assert parsed.unresolved_conflict_request_ids == 0
    assert parsed.records[0].cache_read_tokens == 361_863
    assert parsed.records[0].output_tokens == 871
    assert parse_lines(real_copy, zeroed_copy).records == parsed.records


def test_a_tie_between_disagreeing_copies_is_unresolved() -> None:
    # Two disagreeing copies with identical billed totals cannot be
    # separated by "keep the largest", so the tie is reported rather than
    # settled silently by whichever file sorted first.
    read_heavy = assistant_line(cache_read=1000)
    write_heavy = assistant_line(cache_write_5m=1000)

    parsed = parse_lines(read_heavy, write_heavy)

    assert parsed.conflicting_request_ids == 1
    assert parsed.unresolved_conflict_request_ids == 1


def test_the_final_copy_of_a_streamed_subagent_turn_wins() -> None:
    # Within one subagent file a turn is often written twice: a streamed
    # partial and then the final count, which is always the larger.
    origin = SubagentOrigin(agent_id="aaa", agent_type="Explore", spawn_depth=1)
    parser = UsageParser()
    parser.feed(decode_line(assistant_line(request_id="req_s", output_tokens=12)), origin)
    parser.feed(decode_line(assistant_line(request_id="req_s", output_tokens=338)), origin)

    parsed = parser.finish()

    assert len(parsed.records) == 1
    assert parsed.records[0].output_tokens == 338
    assert parsed.records[0].origin == origin
    assert parsed.conflicting_request_ids == 1
    assert parsed.unresolved_conflict_request_ids == 0
    reverse = UsageParser()
    reverse.feed(decode_line(assistant_line(request_id="req_s", output_tokens=338)), origin)
    reverse.feed(decode_line(assistant_line(request_id="req_s", output_tokens=12)), origin)
    assert reverse.finish().records == parsed.records


def test_records_carry_the_first_seen_timestamp_and_dedup_ignores_it() -> None:
    # The lines of one turn can carry slightly different wall-clock times;
    # comparing them would turn harmless duplicates into conflicts.
    line_a = assistant_line(input_tokens=5, timestamp=ts(0))
    line_b = assistant_line(input_tokens=5, timestamp=ts(1))

    parsed = parse_lines(line_a, line_b)

    assert len(parsed.records) == 1
    assert parsed.duplicate_lines_collapsed == 1
    assert parsed.records[0].timestamp == datetime.datetime.fromisoformat(
        ts(0).replace("Z", "+00:00")
    )


def test_each_skip_reason_is_counted_separately() -> None:
    # A user line is expected noise; an assistant line that will not decode
    # is a real signal. Collapsing both into one silent `continue` buries
    # the second under six figures of the first.
    user_line = '{"type":"user","message":{"role":"user","content":"hi"}}'
    malformed_line = '{"type":"assistant","requestId":'
    undecodable_assistant = (
        '{"type":"assistant","requestId":"req_x","timestamp":"' + ts() + '","message":{}}'
    )
    no_usage = (
        '{"type":"assistant","requestId":"req_y","timestamp":"' + ts() + '",'
        '"message":{"model":"m"}}'
    )

    parsed = parse_lines(
        user_line, malformed_line, undecodable_assistant, no_usage, assistant_line()
    )

    assert len(parsed.records) == 1
    assert parsed.non_assistant_lines == 1
    assert parsed.malformed_json_lines == 1
    assert parsed.undecodable_assistant_lines == 1
    assert parsed.assistant_lines_without_usage == 1


def test_drifted_usage_keys_are_rejected_not_zeroed() -> None:
    # An upstream rename must surface as a counted decode failure, never as
    # a silently zero-cost record that reports a confident $0.0000.
    drifted = (
        '{"type":"assistant","requestId":"req_1","timestamp":"' + ts() + '",'
        '"message":{"model":"claude-sonnet-5",'
        '"usage":{"inputTokens":50000,"cacheReadInputTokens":800000,'
        '"outputTokens":9000}}}'
    )

    parsed = parse_lines(drifted)

    assert parsed.records == ()
    assert parsed.undecodable_assistant_lines == 1


def test_a_line_without_a_timestamp_is_undecodable() -> None:
    # Every real line carries a timestamp; without one it cannot be placed
    # in a time window, so its absence is a schema change.
    line = (
        '{"type":"assistant","requestId":"req_1",'
        '"message":{"model":"claude-sonnet-5","usage":'
        '{"input_tokens":1,"cache_creation_input_tokens":0,'
        '"cache_read_input_tokens":0,"output_tokens":1}}}'
    )

    parsed = parse_lines(line)

    assert parsed.records == ()
    assert parsed.undecodable_assistant_lines == 1


def test_synthetic_lines_are_counted_apart_not_as_unpriced() -> None:
    # `<synthetic>` marks a turn Claude Code generated locally, such as an
    # API error notice. It was never a billed API call, so it is its own
    # counter rather than an unpriced record.
    synthetic = assistant_line(request_id="req_syn", model="<synthetic>")

    parsed = parse_lines(synthetic, assistant_line(request_id="req_real"))

    assert len(parsed.records) == 1
    assert parsed.synthetic_lines == 1
    assert parsed.records[0].model == "claude-sonnet-5"


def test_counters_account_for_every_line_read() -> None:
    # The denominator guarantee: a reader must be able to reconcile the
    # counters against the input, or a schema break that rejects everything
    # looks the same as a genuinely idle period.
    lines = [
        "",
        '{"type":"user","message":{"role":"user","content":"hi"}}',
        '{"type":"assistant","requestId":',
        '{"type":"assistant","requestId":"req_x","timestamp":"' + ts() + '","message":{}}',
        '{"type":"assistant","requestId":"req_y","timestamp":"'
        + ts()
        + '","message":{"model":"m"}}',
        assistant_line(request_id="req_syn", model="<synthetic>"),
        assistant_line(),
        assistant_line(),
    ]

    parsed = parse_lines(*lines)

    accounted = (
        parsed.blank_lines
        + parsed.non_assistant_lines
        + parsed.non_object_json_lines
        + parsed.lines_without_type_discriminator
        + parsed.malformed_json_lines
        + parsed.undecodable_assistant_lines
        + parsed.synthetic_lines
        + parsed.assistant_lines_without_usage
        + parsed.assistant_lines_without_request_id
        + parsed.duplicate_lines_collapsed
        + parsed.conflict_lines_discarded
        + len(parsed.records)
    )
    assert parsed.lines_read == len(lines)
    assert accounted == parsed.lines_read


def _no_request_id_line(message_id: str | None) -> str:
    line = json.loads(assistant_line(request_id="", input_tokens=5, output_tokens=7))
    assert isinstance(line["message"], dict)
    line["message"]["id"] = message_id
    line["message"]["usage"]["cache_creation"] = {
        "ephemeral_1h_input_tokens": 11,
        "ephemeral_5m_input_tokens": 0,
    }
    return json.dumps(line)


def test_dropped_spend_without_a_request_id_is_counted() -> None:
    # An assistant line carrying tokens but no requestId cannot be
    # deduplicated, so its spend is dropped -- and must stay visible.
    parsed = parse_lines(_no_request_id_line(None))

    assert parsed.records == ()
    assert parsed.assistant_lines_without_request_id == 1
    assert parsed.tokens_dropped_without_request_id == 5 + 7 + 11


def test_dropped_spend_is_counted_once_per_message_id() -> None:
    # One turn is written once per content block, each block repeating the
    # same usage. With no requestId, the message id is the only identity
    # left: without it this turn's 23 dropped tokens would be reported as 69.
    one_turn = [_no_request_id_line("msg_1")] * 3

    parsed = parse_lines(*one_turn)

    assert parsed.assistant_lines_without_request_id == 3
    assert parsed.tokens_dropped_without_request_id == 5 + 7 + 11

    three_turns = [_no_request_id_line(f"msg_{i}") for i in range(3)]
    three_turn_parsed = parse_lines(*three_turns)

    assert three_turn_parsed.tokens_dropped_without_request_id == 3 * (5 + 7 + 11)


def test_a_renamed_ttl_bucket_is_rejected() -> None:
    # If a TTL bucket were renamed upstream the split would read zero while
    # the flat total stayed positive, silently erasing cache-write spend and
    # every full-rebuild verdict with it.
    collapsed = (
        '{"type":"assistant","requestId":"req_1","timestamp":"' + ts() + '",'
        '"message":{"model":"claude-sonnet-5",'
        '"usage":{"input_tokens":0,"cache_creation_input_tokens":50000,'
        '"cache_read_input_tokens":0,"output_tokens":0,'
        '"cache_creation":{"ephemeral_9d_input_tokens":50000}}}}'
    )

    parsed = parse_lines(collapsed)

    assert parsed.records == ()
    assert parsed.undecodable_assistant_lines == 1


def test_an_unknown_ttl_bucket_beside_a_zero_flat_total_is_rejected(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # The split-vs-flat guard cannot see this line at all: real lines report
    # a flat total of 0 alongside a positive split, and 0 < 0 is false.
    # Without a check on the bucket NAMES, half a million cache-write tokens
    # price at zero and move no counter whatsoever.
    unknown_bucket = (
        '{"type":"assistant","requestId":"req_1","timestamp":"' + ts() + '",'
        '"message":{"model":"claude-sonnet-5",'
        '"usage":{"input_tokens":0,"cache_creation_input_tokens":0,'
        '"cache_read_input_tokens":0,"output_tokens":0,'
        '"cache_creation":{"ephemeral_2h_input_tokens":500000}}}}'
    )

    with caplog.at_level(logging.WARNING):
        parsed = parse_lines(unknown_bucket)

    assert parsed.records == ()
    assert parsed.undecodable_assistant_lines == 1
    assert "ephemeral_2h_input_tokens" in caplog.text


def test_a_negative_token_count_is_rejected() -> None:
    # Bad input at the boundary becomes a counted skip rather than an
    # uncaught error aborting a scan that already read gigabytes.
    parsed = parse_lines(assistant_line(output_tokens=-1))

    assert parsed.records == ()
    assert parsed.undecodable_assistant_lines == 1


def test_an_empty_request_id_is_not_an_identity() -> None:
    # Treating it as one would collapse unrelated turns into a single record.
    parsed = parse_lines(
        assistant_line(request_id="", input_tokens=10),
        assistant_line(request_id="", input_tokens=20),
    )

    assert parsed.records == ()
    assert parsed.assistant_lines_without_request_id == 2


def test_summary_rejects_counts_that_do_not_partition_the_records() -> None:
    with pytest.raises(ValidationError):
        UsageSummary(
            record_count=3,
            total_input_tokens=0,
            total_cache_read_tokens=0,
            total_cache_write_tokens=0,
            total_output_tokens=0,
            total_cost_microusd=0,
            full_rebuild_count=1,
            normal_growth_count=1,
            unpriced_record_count=0,
        )


def test_a_decode_warning_never_echoes_the_line(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # pydantic's default `errors()` embeds the offending input, and the
    # input here is a line of the operator's own log -- private prompts and
    # file contents. The failing field's location is the diagnostic value;
    # the payload echo is a leak.
    private_marker = "PRIVATE-sk-ant-api03-DO-NOT-LOG-THIS"
    line = json.dumps(
        {
            "type": "assistant",
            "timestamp": ts(),
            "requestId": "req_leak",
            "message": {"id": "msg_1", "content": private_marker, "usage": {}},
        }
    )

    with caplog.at_level(logging.WARNING):
        parsed = parse_lines(line)

    assert parsed.undecodable_assistant_lines == 1
    assert private_marker not in caplog.text
    assert "model" in caplog.text
