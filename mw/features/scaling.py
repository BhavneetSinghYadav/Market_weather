"""Scaling & normalisation utilities.

Exports
-------
- ``minmax_causal(x: pd.Series, win: int) -> pd.Series`` in ``[0, 1]``
- ``tod_percentile_fit(x: pd.Series) -> dict``
  minute-of-day profiles (offline)
- ``tod_percentile_transform(x: pd.Series, model: dict) -> pd.Series`` in
  ``[0, 1]``
"""

from __future__ import annotations

from typing import Dict

import numpy as np
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


def tod_percentile_fit(x: pd.Series) -> Dict[int, np.ndarray]:
    """Fit time-of-day percentile references for offline use.

    Parameters
    ----------
    x:
        Series indexed by ``pd.DatetimeIndex`` containing the observations.

    Returns
    -------
    dict
        Dictionary mapping ``minute_of_day`` to a sorted array of past values
        observed at that minute.  The arrays represent empirical
        distributions which can later be used to compute percentiles.
    """

    if not isinstance(x.index, pd.DatetimeIndex):
        raise TypeError("x must be indexed by a DatetimeIndex")

    minute_of_day = x.index.hour * 60 + x.index.minute
    groups = x.groupby(minute_of_day)
    model = {int(m): g.sort_values().to_numpy() for m, g in groups}
    return model


def tod_percentile_transform(
    x: pd.Series,
    model: Dict[int, np.ndarray],
) -> pd.Series:
    """Map values to ``[0, 1]`` by minute-of-day percentiles.

    Parameters
    ----------
    x:
        Series indexed by ``pd.DatetimeIndex`` to transform.
    model:
        Output of :func:`tod_percentile_fit`.

    Returns
    -------
    pd.Series
        Series of percentile scores in ``[0, 1]``. Minutes not present in the
        ``model`` yield ``NaN`` values instead of raising ``KeyError``.
    """

    if not isinstance(x.index, pd.DatetimeIndex):
        raise TypeError("x must be indexed by a DatetimeIndex")

    minute_of_day = x.index.hour * 60 + x.index.minute
    result = []
    for val, mod in zip(x.to_numpy(), minute_of_day):
        arr = model.get(int(mod))
        if arr is None or len(arr) == 0:
            result.append(np.nan)
            continue
        rank = np.searchsorted(arr, val, side="right")
        result.append(rank / len(arr))

    return pd.Series(result, index=x.index).clip(0.0, 1.0)
