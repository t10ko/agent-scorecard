"""Pricing: exact and aliased model lookups, per-TTL cache multipliers,
merging an override file, and refusing to guess a rate."""

from __future__ import annotations

import datetime
from pathlib import Path

import pytest
from conftest import parse_lines

from agent_scorecard.models import UsageRecord
from agent_scorecard.pricing import (
    PriceFileError,
    estimate_cost,
    load_default_prices,
    load_prices,
)


def record(model: str, **tokens: int) -> UsageRecord:
    defaults: dict[str, int] = dict.fromkeys(
        (
            "input_tokens",
            "cache_read",
            "cache_write_1h",
            "cache_write_5m",
            "output_tokens",
        ),
        0,
    )
    defaults.update(tokens)
    return UsageRecord(
        request_id="req_1",
        timestamp=datetime.datetime(2026, 9, 20, tzinfo=datetime.UTC),
        model=model,
        input_tokens=defaults["input_tokens"],
        cache_read_tokens=defaults["cache_read"],
        cache_write_1h_tokens=defaults["cache_write_1h"],
        cache_write_5m_tokens=defaults["cache_write_5m"],
        output_tokens=defaults["output_tokens"],
    )


@pytest.mark.parametrize(
    ("kwargs", "expected_microusd"),
    [
        # claude-sonnet-5: $2/MTok input, $10/MTok output.
        ({"input_tokens": 1_000_000}, 2_000_000),  # plain input costs 1x
        ({"cache_read": 1_000_000}, 200_000),  # cache-read costs 0.1x
        ({"cache_write_5m": 1_000_000}, 2_500_000),  # 5m cache-write costs 1.25x
        ({"cache_write_1h": 1_000_000}, 4_000_000),  # 1h cache-write costs 2x
        ({"output_tokens": 1_000_000}, 10_000_000),  # output costs 1x
        # A record mixing both TTLs prices each portion at its own
        # multiplier rather than collapsing to a single flat rate.
        (
            {"cache_write_1h": 600_000, "cache_write_5m": 400_000},
            3_400_000,
        ),
    ],
)
def test_each_token_kind_is_weighted_by_its_own_multiplier(
    kwargs: dict[str, int], expected_microusd: int
) -> None:
    assert estimate_cost(record("claude-sonnet-5", **kwargs), load_prices()) == (expected_microusd)


def test_a_flat_cache_write_total_prices_at_the_default_5m_ttl() -> None:
    # A line with no `cache_creation` split lands its flat total in the
    # 5-minute bucket (see test_usage); pricing follows the same split.
    parsed = parse_lines(
        '{"type":"assistant","requestId":"req_1","timestamp":"'
        + "2026-09-20T10:15:00.000Z"
        + '","message":{"model":"claude-sonnet-5","usage":'
        '{"input_tokens":0,"cache_creation_input_tokens":1000000,'
        '"cache_read_input_tokens":0,"output_tokens":0}}}'
    )

    assert estimate_cost(parsed.records[0], load_prices()) == 2_500_000


def test_a_dated_variant_prices_through_an_explicit_alias() -> None:
    # Date suffixes are never stripped automatically; only an explicit
    # [aliases] entry bridges a dated snapshot to its model.
    dated = record("claude-haiku-4-5-20251001", input_tokens=1_000_000)

    assert estimate_cost(dated, load_prices()) == 1_000_000


def test_an_unknown_model_is_unpriced_and_never_zero() -> None:
    assert estimate_cost(record("claude-opus-4-7"), load_prices()) is None


def test_the_bundled_table_is_complete_and_dated() -> None:
    table = load_default_prices()

    assert table.checked_on == datetime.date(2026, 9, 23)
    assert table.source.startswith("https://")
    assert table.models, "the bundled table prices no model"
    # Every model that appears in real logs is priced, each with both rates.
    for name in (
        "claude-opus-5",
        "claude-opus-5-5",
        "claude-opus-4-8",
        "claude-sonnet-5",
        "claude-haiku-4-5",
    ):
        assert name in table.models, name
    for prices in table.models.values():
        assert prices.input > 0
        assert prices.output > 0


def test_an_override_file_adds_and_replaces(tmp_path: Path) -> None:
    override = tmp_path / "prices.toml"
    override.write_text(
        'checked_on = "2026-10-01"\n'
        'source = "https://example.com/pricing"\n'
        "[cache]\n"
        "read_multiplier = 0.2\n"
        "write_5m_multiplier = 1.25\n"
        "write_1h_multiplier = 2.0\n"
        "\n"
        '[models."claude-opus-7"]\n'
        "input = 9.00\n"
        "output = 45.00\n"
        '[models."claude-sonnet-5"]\n'
        "input = 3.00\n"
        "output = 15.00\n",
        encoding="utf-8",
    )

    table = load_prices(override)

    # The override replaced sonnet-5's rates and metadata, added opus-7,
    # and kept the bundled models it did not mention.
    assert table.models["claude-sonnet-5"].input == 3.0
    assert table.models["claude-opus-7"].output == 45.0
    assert table.models["claude-haiku-4-5"].input == 1.0
    assert table.checked_on == datetime.date(2026, 10, 1)
    assert table.cache.read_multiplier == 0.2


def test_an_override_file_may_state_only_rates(tmp_path: Path) -> None:
    # checked_on, source, and [cache] replace the bundled ones only when
    # present, so a minimal override inherits them.
    override = tmp_path / "prices.toml"
    override.write_text(
        '[models."claude-opus-7"]\ninput = 9.00\noutput = 45.00\n', encoding="utf-8"
    )

    table = load_prices(override)

    assert table.checked_on == datetime.date(2026, 9, 23)
    assert table.cache.read_multiplier == 0.1
    assert table.models["claude-opus-7"].input == 9.0


def test_a_missing_price_file_gives_a_clean_error(tmp_path: Path) -> None:
    with pytest.raises(PriceFileError, match="could not read"):
        load_prices(tmp_path / "nope.toml")


def test_an_invalid_price_file_gives_a_clean_error(tmp_path: Path) -> None:
    override = tmp_path / "broken.toml"
    override.write_text("[models\n", encoding="utf-8")

    with pytest.raises(PriceFileError, match="not valid TOML"):
        load_prices(override)


def test_a_price_file_with_bad_rates_gives_a_clean_error(tmp_path: Path) -> None:
    override = tmp_path / "wrong.toml"
    override.write_text('[models."claude-x"]\ninput = 0\noutput = 1\n', encoding="utf-8")

    with pytest.raises(PriceFileError, match="invalid entry"):
        load_prices(override)
