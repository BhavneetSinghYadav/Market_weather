import pandas as pd
import numpy as np

from mw.io.canonicalizer import canonicalize


def test_rest_vs_ws_microbatch_equality(tmp_path):
    ts = pd.date_range("2024-01-01 09:30", periods=30, freq="1min", tz="America/New_York")
    base = np.linspace(1.0, 2.9, num=30)
    rest_df = pd.DataFrame(
        {
            "timestamp": ts.tz_localize(None),
            "open": base,
            "high": base + 0.1,
            "low": base - 0.1,
            "close": base + 0.05,
        }
    )
    rest_df.attrs["source_time_basis"] = "America/New_York"
    rest_canon = canonicalize(rest_df, str(tmp_path / "rest.parquet"))

    batches = [rest_df.iloc[i * 10 : (i + 1) * 10] for i in range(3)]
    ws_df = pd.concat(batches)
    ws_df.attrs["source_time_basis"] = "America/New_York"
    ws_canon = canonicalize(ws_df, str(tmp_path / "ws.parquet"))

    pd.testing.assert_frame_equal(rest_canon, ws_canon, check_exact=False, atol=1e-9)
