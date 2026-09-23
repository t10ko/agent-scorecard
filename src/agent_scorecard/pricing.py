"""The price table: loading, merging, and turning usage into micro-dollars.

Prices live in a bundled TOML file (USD per million tokens) and can be
extended or replaced per run with `--prices`. Money is kept as integer
micro-dollars and rounded once per record. A model with no row — and no
explicit alias — is unpriced: its cost is reported as unknown, never
guessed and never a silent zero.
"""

from __future__ import annotations

import datetime
import tomllib
from importlib import resources
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from agent_scorecard.models import UsageRecord

DEFAULT_PRICES_RESOURCE = "default_prices.toml"


class PriceFileError(ValueError):
    """A price file could not be read or has an invalid shape."""


class CacheMultipliers(BaseModel):
    """Prompt-caching prices as multipliers on a model's base input rate.

    Fixed by the API contract, not per model: a cache read costs 0.1x the
    input rate, a 5-minute cache write 1.25x, a 1-hour cache write 2x.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    read_multiplier: float = 0.1
    write_5m_multiplier: float = 1.25
    write_1h_multiplier: float = 2.0


class ModelPrices(BaseModel):
    """One model's input and output rates, USD per million tokens."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    input: float = Field(gt=0)
    output: float = Field(gt=0)


class PriceTable(BaseModel):
    """Every rate the tool prices with, plus what it was checked against."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    checked_on: datetime.date
    source: str
    cache: CacheMultipliers
    models: dict[str, ModelPrices]
    aliases: dict[str, str] = Field(default_factory=dict)

    def prices_for(self, model: str) -> ModelPrices | None:
        """The rates for a model's exact name, or through an explicit alias.

        Date suffixes are never stripped automatically, and a missing model
        is never guessed: None means unpriced.
        """
        name = self.aliases.get(model, model)
        return self.models.get(name)


class _PriceFile(BaseModel):
    """What one price file directly states; everything but `[models]` is
    optional, because an override file may only add or replace rates."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    checked_on: datetime.date | None = None
    source: str | None = None
    cache: CacheMultipliers | None = None
    models: dict[str, ModelPrices] = Field(default_factory=dict)
    aliases: dict[str, str] = Field(default_factory=dict)


def _parse_price_file(data: object, origin: str) -> _PriceFile:
    """Validate one parsed TOML document, raising `PriceFileError` with a
    single clear sentence when the shape is wrong."""
    if not isinstance(data, dict):
        raise PriceFileError(f"{origin} is not a TOML table")
    checked_on = data.get("checked_on")
    if isinstance(checked_on, datetime.date):
        parsed_date: datetime.date | None = checked_on
    elif isinstance(checked_on, str):
        try:
            parsed_date = datetime.date.fromisoformat(checked_on)
        except ValueError as exc:
            raise PriceFileError(
                f"{origin} has a checked_on that is not a date: {checked_on!r}"
            ) from exc
    elif checked_on is None:
        parsed_date = None
    else:
        raise PriceFileError(f"{origin} has a checked_on that is not a date")
    try:
        return _PriceFile(
            checked_on=parsed_date,
            source=data.get("source"),
            cache=CacheMultipliers.model_validate(data.get("cache", {}))
            if isinstance(data.get("cache"), dict)
            else None,
            models={
                str(name): ModelPrices.model_validate(rates)
                for name, rates in data.get("models", {}).items()
            }
            if isinstance(data.get("models"), dict)
            else {},
            aliases={
                str(dated): str(canonical) for dated, canonical in data.get("aliases", {}).items()
            }
            if isinstance(data.get("aliases"), dict)
            else {},
        )
    except ValueError as exc:
        raise PriceFileError(f"{origin} has an invalid entry: {exc}") from exc


def _read_toml(path: Path) -> object:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise PriceFileError(f"could not read the price file {path}: {exc}") from exc
    try:
        return tomllib.loads(text)
    except (tomllib.TOMLDecodeError, RecursionError) as exc:
        raise PriceFileError(f"the price file {path} is not valid TOML: {exc}") from exc


def load_default_prices() -> PriceTable:
    """The price table bundled inside the package."""
    text = resources.files("agent_scorecard").joinpath(DEFAULT_PRICES_RESOURCE).read_text()
    try:
        data: object = tomllib.loads(text)
    except tomllib.TOMLDecodeError as exc:  # pragma: no cover - bundled file
        raise PriceFileError(f"the bundled price table is not valid TOML: {exc}") from exc
    parsed = _parse_price_file(data, "the bundled price table")
    if parsed.checked_on is None or parsed.source is None or parsed.cache is None:
        raise PriceFileError(  # pragma: no cover - guarded by tests on the file
            "the bundled price table must state checked_on, source, and [cache]"
        )
    return PriceTable(
        checked_on=parsed.checked_on,
        source=parsed.source,
        cache=parsed.cache,
        models=parsed.models,
        aliases=parsed.aliases,
    )


def load_prices(override: Path | None = None) -> PriceTable:
    """The bundled price table, merged with `--prices` when given: the
    override's models add to or replace the bundled ones, and its
    checked_on, source, and cache multipliers replace the bundled ones when
    present."""
    table = load_default_prices()
    if override is None:
        return table
    parsed = _parse_price_file(_read_toml(override), str(override))
    return PriceTable(
        checked_on=parsed.checked_on or table.checked_on,
        source=parsed.source or table.source,
        cache=parsed.cache or table.cache,
        models={**table.models, **parsed.models},
        aliases={**table.aliases, **parsed.aliases},
    )


def estimate_cost(record: UsageRecord, table: PriceTable) -> int | None:
    """One request's estimated cost in micro-dollars, or None when the
    record's model is unpriced.

    Cache-read and cache-write tokens are weighted by their multipliers on
    the input rate; plain input and output tokens are priced at their own
    rates. Returning None rather than zero keeps one unpriced model from
    discarding every priced record alongside it.
    """
    prices = table.prices_for(record.model)
    if prices is None:
        return None
    input_rate_microusd = prices.input * 1_000_000
    output_rate_microusd = prices.output * 1_000_000
    weighted_input_tokens = (
        record.input_tokens
        + record.cache_read_tokens * table.cache.read_multiplier
        + record.cache_write_1h_tokens * table.cache.write_1h_multiplier
        + record.cache_write_5m_tokens * table.cache.write_5m_multiplier
    )
    input_cost = round(input_rate_microusd * weighted_input_tokens / 1_000_000)
    output_cost = round(output_rate_microusd * record.output_tokens / 1_000_000)
    return input_cost + output_cost
