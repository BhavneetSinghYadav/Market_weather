import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from mw.features.ftle import ftle_rosenstein  # noqa: E402
from mw.features.ftle import rolling_ftle_rosenstein  # noqa: E402


def logistic_map(r: float, x0: float, n: int) -> np.ndarray:
    x = np.empty(n)
    x[0] = x0
    for i in range(1, n):
        x[i] = r * x[i - 1] * (1 - x[i - 1])
    return x


def test_ftle_rosenstein_logistic_map():
    data = logistic_map(4.0, 0.2, 500)
    series = pd.Series(data)
    val = ftle_rosenstein(series, window=200, m=2, tau=1, horizon=5, theiler=2)
    assert val == pytest.approx(np.log(2), abs=0.2)


def test_rolling_matches_direct():
    data = logistic_map(4.0, 0.2, 300)
    series = pd.Series(data)
    window = 100
    rolling = rolling_ftle_rosenstein(
        series, window=window, m=2, tau=1, horizon=5, theiler=2
    )
    direct = ftle_rosenstein(
        series.iloc[-window:], window=window, m=2, tau=1, horizon=5, theiler=2
    )
    assert rolling.iloc[-1] == pytest.approx(direct)
