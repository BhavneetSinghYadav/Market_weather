import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))
from mw.adapters import polygon_rest  # noqa: E402
from mw.features.scaling import minmax_causal  # noqa: E402
from mw.features.smoothing import ema  # noqa: E402
from mw.io.canonicalizer import canonicalize  # noqa: E402
from mw.scoring.tradability import (score_tradability,  # noqa: E402
                                    state_machine)


def test_pipeline_integration(monkeypatch, tmp_path):
    """End-to-end pipeline from adapter to state machine."""

    def fake_fetch(symbol, start, end, limit=50_000):
        ts = pd.date_range("2024-01-01", periods=8, freq="1min", tz="UTC")
        close = pd.Series([1, 1, 10, 11, 12, 1, 1, 1], index=ts)
        df = pd.DataFrame(
            {
                "timestamp": ts,
                "open": close,
                "high": close,
                "low": close,
                "close": close,
                "volume": [0] * len(close),
            }
        )
        return df

    monkeypatch.setattr(polygon_rest, "fetch_fx_agg_minute", fake_fetch)
    raw = polygon_rest.fetch_fx_agg_minute(
        "EURUSD",
        "2024-01-01",
        "2024-01-02",
    )

    canon = canonicalize(raw, str(tmp_path / "out.parquet"))

    e_hat = minmax_causal(canon["close"], win=3)
    l_hat = ema(e_hat, span=2)
    scores = score_tradability(e_hat, l_hat)
    states = state_machine(scores)

    expected = pd.Series(
        [
            "YELLOW",
            "GREEN",
            "GREEN",
            "GREEN",
            "RED",
            "RED",
            "RED",
            "GREEN",
        ],
        index=canon.index,
    )

    pd.testing.assert_series_equal(states, expected)
