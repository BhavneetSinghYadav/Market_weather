import sys
from pathlib import Path

import pandas as pd
import pandas.testing as pdt

sys.path.append(str(Path(__file__).resolve().parents[1]))

from mw.features.second_order import curvature, tension, velocity  # noqa: E402  # isort: skip


def test_velocity_and_curvature_compute_causal_differences():
    idx = pd.date_range("2024-01-01", periods=5, freq="D")
    x = pd.Series([0.0, 1.0, 4.0, 9.0, 16.0], index=idx)

    v = velocity(x)
    c = curvature(x)

    expected_v = pd.Series([float("nan"), 1.0, 3.0, 5.0, 7.0], index=idx)
    expected_c = pd.Series([float("nan"), float("nan"), 2.0, 2.0, 2.0], index=idx)

    pdt.assert_series_equal(v, expected_v)
    pdt.assert_series_equal(c, expected_c)


def test_tension_combines_scaled_features():
    idx = pd.date_range("2024-01-01", periods=3, freq="D")
    e_hat = pd.Series([0.2, 0.5, 0.8], index=idx)
    l_hat = pd.Series([0.1, 0.2, 0.3], index=idx)

    result = tension(e_hat, l_hat, alpha=0.6, beta=0.4)
    expected = pd.Series(0.6 * (1 - e_hat) - 0.4 * l_hat, index=idx)

    pdt.assert_series_equal(result, expected)
