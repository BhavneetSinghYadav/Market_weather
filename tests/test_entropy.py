import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))
from mw.features.entropy import (  # noqa: E402
    permutation_entropy,
    rolling_permutation_entropy,
    sample_entropy,
)


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


def test_rolling_permutation_entropy_alignment():
    series = pd.Series([1, 3, 2, 4, 5, 0])
    window = 4
    result = rolling_permutation_entropy(series, window=window)

    expected = pd.Series([np.nan] * len(series))
    for i in range(window - 1, len(series)):
        expected.iloc[i] = permutation_entropy(
            series.iloc[i - window + 1 : i + 1]
        )

    pd.testing.assert_series_equal(result, expected)


def test_sample_entropy_constant_zero():
    series = pd.Series([5] * 50)
    assert sample_entropy(series, m=2, r=0.2) == 0.0


def test_sample_entropy_random_greater_than_deterministic():
    rng = np.random.default_rng(0)
    random_series = pd.Series(rng.normal(size=1000))
    deterministic = pd.Series(np.sin(np.linspace(0, 10 * np.pi, 1000)))
    h_rand = sample_entropy(random_series, m=2, r=0.2)
    h_det = sample_entropy(deterministic, m=2, r=0.2)
    assert h_rand > h_det


def test_permutation_entropy_with_repeated_values():
    import mw.features.entropy as entropy

    entropy._rng = np.random.default_rng(0)
    series = pd.Series([1, 1, 2, 2, 3, 3] * 5)
    h1 = permutation_entropy(series, m=3, tau=1)
    entropy._rng = np.random.default_rng(0)
    h2 = permutation_entropy(series, m=3, tau=1)
    assert h1 == h2 and h1 > 0
