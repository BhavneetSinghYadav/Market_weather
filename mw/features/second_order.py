"""Second-order time series features using causal differences.

Exports
-------
velocity
    First discrete derivative (velocity) of a series.
curvature
    Second discrete derivative (curvature) of a series.
tension
    Weighted combination ``alpha*(1 - e_hat) - beta*l_hat``.
"""

from __future__ import annotations

import pandas as pd


def velocity(series: pd.Series) -> pd.Series:
    """Return the causal first difference of ``series``.

    Parameters
    ----------
    series : pd.Series
        Input series.

    Returns
    -------
    pd.Series
        Velocity aligned with the input index where ``result[i] = series[i] - series[i-1]``.
    """

    return series.diff()


def curvature(series: pd.Series) -> pd.Series:
    """Return the causal second difference of ``series``.

    This is simply the discrete difference of the velocity, equivalent to
    ``series.diff().diff()``.
    """

    return series.diff().diff()


def tension(e_hat: pd.Series, l_hat: pd.Series, *, alpha: float = 0.6, beta: float = 0.4) -> pd.Series:
    """Return tension from normalized entropy and lambda features.

    Parameters
    ----------
    e_hat, l_hat : pd.Series
        Normalized entropy and lambda features in ``[0, 1]``.
    alpha, beta : float, default 0.6 and 0.4
        Weights applied to the two components.

    Returns
    -------
    pd.Series
        ``alpha * (1 - e_hat) - beta * l_hat`` aligned to the input index.
    """

    return alpha * (1 - e_hat) - beta * l_hat
