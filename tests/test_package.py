"""The package imports, reports its version, and nothing more (for now)."""

from __future__ import annotations

import agent_scorecard


def test_version() -> None:
    assert agent_scorecard.__version__ == "0.1.0"
