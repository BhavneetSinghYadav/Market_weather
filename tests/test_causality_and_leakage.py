"""Test for causality and leakage protections.

This test simulates a live pipeline that processes data incrementally and
compares its feature and score outputs against an offline batch computation on
the same data.  Equality of the two paths ensures that calculations are causal
and no future information leaks into the live computations.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))
from mw.features.scaling import minmax_causal  # noqa: E402
from mw.features.smoothing import ema  # noqa: E402
from mw.scoring.tradability import score_tradability  # noqa: E402


def _run_batch(df: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Compute features and scores in a single offline batch."""

    e_hat = ema(df["error"], span=3)
    l_hat = minmax_causal(df["latency"], win=3)
    score = score_tradability(e_hat, l_hat)
    score.name = "score"
    return e_hat, l_hat, score


def _run_live(df: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Simulate a live pipeline processing observations sequentially."""

    e_vals = []
    l_vals = []
    s_vals = []
    for i in range(len(df)):
        current = df.iloc[: i + 1]
        e_current = ema(current["error"], span=3).iloc[-1]
        l_current = minmax_causal(current["latency"], win=3).iloc[-1]
        s_current = score_tradability(
            ema(current["error"], span=3),
            minmax_causal(current["latency"], win=3),
        ).iloc[-1]
        e_vals.append(e_current)
        l_vals.append(l_current)
        s_vals.append(s_current)

    e_series = pd.Series(e_vals, index=df.index, name="error")
    l_series = pd.Series(l_vals, index=df.index, name="latency")
    s_series = pd.Series(s_vals, index=df.index, name="score")
    return e_series, l_series, s_series


def test_live_pipeline_matches_batch() -> None:
    """Live incremental processing should match offline batch results."""

    rng = np.random.default_rng(0)
    idx = pd.date_range("2024-01-01", periods=50, freq="min")
    df = pd.DataFrame(
        {
            "error": rng.random(len(idx)),
            "latency": rng.random(len(idx)),
        },
        index=idx,
    )

    batch_e, batch_l, batch_score = _run_batch(df)
    live_e, live_l, live_score = _run_live(df)

    pd.testing.assert_series_equal(live_e, batch_e)
    pd.testing.assert_series_equal(live_l, batch_l)
    pd.testing.assert_series_equal(live_score, batch_score)
