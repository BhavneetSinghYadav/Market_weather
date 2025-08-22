"""
Entropy metrics (stubs).

Exports (to implement):
- permutation_entropy(series: pd.Series, m=3, tau=1) -> float
- rolling_permutation_entropy(series: pd.Series, window: int, m=3, tau=1) -> pd.Series
- sample_entropy(series: pd.Series, m=2, r=0.2) -> float
Notes:
- Use closes or returns; causal windows only.
- Handle ties by tiny jitter or averaged ranks.
"""
from typing import Optional
import pandas as pd

def permutation_entropy(series: pd.Series, m: int = 3, tau: int = 1) -> float:
    """Return normalized permutation entropy in [0,1] for the last window of `series`."""
    # TODO: implement (ordinal patterns -> frequencies -> normalized entropy)
    raise NotImplementedError

def rolling_permutation_entropy(series: pd.Series, window: int, m: int = 3, tau: int = 1) -> pd.Series:
    """Causal rolling PE; aligns result to window end."""
    # TODO: implement
    raise NotImplementedError

def sample_entropy(series: pd.Series, m: int = 2, r: float = 0.2) -> float:
    """Return Sample Entropy for the last window of `series` (use robust sigma for r*sigma)."""
    # TODO: implement (with Theiler exclusion)
    raise NotImplementedError
