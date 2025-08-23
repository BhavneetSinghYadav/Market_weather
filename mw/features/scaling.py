"""
Scaling & normalization (stubs).

Exports:
- minmax_causal(x: pd.Series, win: int) -> pd.Series in [0,1]
- tod_percentile_fit(x: pd.Series) -> dict
  # minute-of-day profiles (offline)
- tod_percentile_transform(x: pd.Series, model: dict) -> pd.Series in [0,1]
"""

import pandas as pd


def minmax_causal(x: pd.Series, win: int, eps: float = 1e-9) -> pd.Series:
    """Causal min-max scaling over a trailing window.

    Parameters
    ----------
    x
        Input series to scale.
    win
        Size of the trailing window used for the min/max calculation.
    eps
        Small constant added to the denominator to avoid division by zero when
        the windowed range is constant.

    Returns
    -------
    pd.Series
        Series normalised to the ``[0, 1]`` range.
    """

    if win <= 0:
        raise ValueError("win must be positive")

    roll = x.rolling(win, min_periods=1)
    x_min = roll.min()
    x_max = roll.max()
    scaled = (x - x_min) / (x_max - x_min + eps)
    return scaled.clip(0.0, 1.0)


def tod_percentile_fit(x: pd.Series) -> dict:
    """Fit time-of-day percentile references (offline)."""
    # TODO: implement
    raise NotImplementedError


def tod_percentile_transform(x: pd.Series, model: dict) -> pd.Series:
    """Map values to [0,1] by minute-of-day percentiles."""
    # TODO: implement
    raise NotImplementedError
