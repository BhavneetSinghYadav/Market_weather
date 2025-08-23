"""
Entropy metrics (stubs).

Exports (to implement):
- permutation_entropy(series: pd.Series, m=3, tau=1) -> float
- rolling_permutation_entropy(
    series: pd.Series, window: int, m=3, tau=1
  ) -> pd.Series
- sample_entropy(series: pd.Series, m=2, r=0.2) -> float
Notes:
- Use closes or returns; causal windows only.
- Handle ties by tiny jitter or averaged ranks.
"""

from collections import Counter
from math import factorial, log
from typing import List, Tuple

import numpy as np
import pandas as pd


def _ordinal_patterns(
    values: np.ndarray,
    m: int,
    tau: int,
) -> List[Tuple[int, ...]]:
    """Return ordinal patterns for ``values``."""
    n = len(values)
    if n < (m - 1) * tau + 1:
        return []

    patterns: List[Tuple[int, ...]] = []
    for i in range(n - (m - 1) * tau):
        window = values[i : i + (m - 1) * tau + 1 : tau]  # noqa: E203
        inner = np.argsort(window, kind="mergesort")
        ranks = np.argsort(inner, kind="mergesort")
        patterns.append(tuple(ranks))
    return patterns


def permutation_entropy(series: pd.Series, m: int = 3, tau: int = 1) -> float:
    """Return normalized permutation entropy in [0,1] for the last window of
    ``series``.
    """
    values = series.dropna().to_numpy()
    patterns = _ordinal_patterns(values, m, tau)
    if not patterns:
        return float("nan")

    counts = Counter(patterns)
    probs = np.fromiter(counts.values(), dtype=float)
    probs /= probs.sum()
    entropy = -np.sum(probs * np.log(probs))
    return float(entropy / log(factorial(m)))


def rolling_permutation_entropy(
    series: pd.Series, window: int, m: int = 3, tau: int = 1
) -> pd.Series:
    """Return permutation entropy over sliding windows.

    The result is *causal* and aligned to the window end, i.e. the value at
    ``series.index[i]`` uses the data from ``series[i - window + 1 : i + 1]``.
    Positions that do not have a full window of observations are filled with
    ``NaN`` so that the returned :class:`~pandas.Series` always matches the
    length and index of ``series``.
    """

    values = np.full(len(series), np.nan, dtype=float)
    if window <= 0:
        raise ValueError("window must be positive")

    for i in range(window - 1, len(series)):
        window_slice = series.iloc[i - window + 1 : i + 1]
        # Require a full window of observations; mimic pandas' ``min_periods``.
        if window_slice.isna().any():
            continue
        values[i] = permutation_entropy(window_slice, m=m, tau=tau)

    return pd.Series(values, index=series.index)


def sample_entropy(series: pd.Series, m: int = 2, r: float = 0.2) -> float:
    """Return Sample Entropy for the last window of `series`.

    Uses robust sigma for r*sigma.
    """
    # TODO: implement (with Theiler exclusion)
    raise NotImplementedError
