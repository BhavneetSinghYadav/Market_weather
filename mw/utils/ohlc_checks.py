"""
OHLC integrity checks (stub).
"""
import pandas as pd


def validate_ohlc(df: pd.DataFrame) -> pd.Series:
    """Return boolean mask for ``low <= min(open, close) <= max(open, close) <= high``.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing at least ``open``, ``high``, ``low`` and ``close``
        columns.

    Returns
    -------
    pd.Series
        Boolean Series indexed like ``df`` where ``True`` indicates the row
        satisfies the OHLC relationship.
    """

    # Compute min/max between open and close for each row
    oc = df[["open", "close"]]
    min_val = oc.min(axis=1)
    max_val = oc.max(axis=1)

    # Validate the min/max values lie within the low/high bounds
    return (df["low"] <= min_val) & (min_val <= max_val) & (max_val <= df["high"])
