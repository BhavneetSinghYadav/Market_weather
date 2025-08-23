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
    """Causal rolling PE; aligns result to window end."""

    def _pe(x: np.ndarray) -> float:
        return permutation_entropy(pd.Series(x), m=m, tau=tau)

    return series.rolling(window, min_periods=window).apply(
        _pe,
        raw=True,
    )


def sample_entropy(series: pd.Series, m: int = 2, r: float = 0.2) -> float:
    """Return Sample Entropy for the last window of `series`.

    Uses robust sigma for r*sigma.
    """
    # TODO: implement (with Theiler exclusion)
    raise NotImplementedError
