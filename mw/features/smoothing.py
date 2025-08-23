"""
Smoothing / hysteresis helpers (stubs).

Exports:
- ema(series: pd.Series, span: int=3) -> pd.Series
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
    """

    result = series.ewm(span=span, adjust=False).mean()
    return result.loc[series.index]
