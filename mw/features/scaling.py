"""
Scaling & normalization (stubs).

Exports:
- minmax_causal(x: pd.Series, win: int) -> pd.Series in [0,1]
- tod_percentile_fit(x: pd.Series) -> dict      # minute-of-day profiles (offline)
- tod_percentile_transform(x: pd.Series, model: dict) -> pd.Series in [0,1]
"""
import pandas as pd

def minmax_causal(x: pd.Series, win: int) -> pd.Series:
    """Causal min-max scaling over trailing window `win`."""
    # TODO: implement (rolling min/max; protect divide-by-zero)
    raise NotImplementedError

def tod_percentile_fit(x: pd.Series) -> dict:
    """Fit time-of-day percentile references (offline)."""
    # TODO: implement
    raise NotImplementedError

def tod_percentile_transform(x: pd.Series, model: dict) -> pd.Series:
    """Map values to [0,1] by minute-of-day percentiles."""
    # TODO: implement
    raise NotImplementedError
