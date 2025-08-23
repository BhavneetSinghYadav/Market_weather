import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))
from mw.features.entropy import permutation_entropy  # noqa: E402


def test_monotonic_series_has_zero_entropy():
    series = pd.Series(range(10))
    assert permutation_entropy(series, m=3, tau=1) == 0.0


def test_constant_series_has_zero_entropy():
    series = pd.Series([5] * 10)
    assert permutation_entropy(series, m=3, tau=1) == 0.0


def test_random_series_has_high_entropy():
    rng = np.random.default_rng(0)
    series = pd.Series(rng.normal(size=1000))
    h = permutation_entropy(series, m=3, tau=1)
    assert h > 0.95
