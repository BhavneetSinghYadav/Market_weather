import csv
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))
from mw.adapters import polygon_rest  # noqa: E402
from mw.features.scaling import minmax_causal  # noqa: E402
from mw.features.smoothing import ema  # noqa: E402
from mw.io.canonicalizer import canonicalize  # noqa: E402
from mw.live import minute_loop  # noqa: E402
from mw.live.logger import SessionLogger  # noqa: E402
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


def test_live_pipeline_persists_outputs(monkeypatch, tmp_path):
    """Ensure compute results and decision logs are written to disk."""

    monkeypatch.chdir(tmp_path)

    # Time control
    def gen():
        t = datetime(2024, 1, 1, tzinfo=timezone.utc)
        while True:
            yield t
            t += timedelta(seconds=1)

    times = gen()
    monkeypatch.setattr(minute_loop, "now_utc", lambda: next(times))
    monkeypatch.setattr("time.sleep", lambda x: None)

    df1 = pd.DataFrame({"x": [1]})
    df2 = pd.DataFrame({"x": [2]})
    data_iter = iter([df1, df2])

    def poll():
        pass

    def compute():
        return {"feat": next(data_iter)}

    def persist():
        pass

    logger = SessionLogger(Path("decisions.csv"), Path("summary.json"))

    def log():
        logger.log_minute(minute_loop.now_utc(), 0.1, "GREEN", {})

    def plot():
        pass

    def health():
        pass

    params = {"minute_loop_offsets": {}}

    for _ in range(2):
        minute_loop.run_minute_loop(
            poll, compute, persist, log, plot, health, params
        )
    logger.close()

    assert (tmp_path / "data" / "feat.parquet").exists()
    csv_path = tmp_path / "decisions.csv"
    assert csv_path.exists()
    with csv_path.open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2
    assert (tmp_path / "summary.json").exists()
