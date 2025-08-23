"""Smoothing helpers for time series.

Exports:
- ema(series: pd.Series, span: int = 3) -> pd.Series
"""
import pandas as pd


def ema(series: pd.Series, span: int = 3) -> pd.Series:
    """Simple exponential moving average (causal).

    Parameters
    ----------
    series : pd.Series
        Input data.
    span : int, default 3
        Span parameter for the EMA.

    Returns
    -------
    pd.Series
        Exponentially weighted moving average aligned with the input index.
    
    Raises
    ------
    ValueError
        If ``span`` is less than or equal to zero.
    """
    if span <= 0:
        raise ValueError("span must be positive")

    result = series.ewm(span=span, adjust=False).mean()
    return result.loc[series.index]
