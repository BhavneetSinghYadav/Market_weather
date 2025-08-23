"""Integrity checks for OHLC price data.

Exports:
- validate_ohlc(df: pd.DataFrame, *, return_flags: bool = False,
  return_clipped: bool = False)
"""
from typing import Tuple, Union

import pandas as pd


def validate_ohlc(
    df: pd.DataFrame,
    *,
    return_flags: bool = False,
    return_clipped: bool = False,
) -> Union[pd.Series, Tuple[pd.Series, pd.DataFrame], Tuple[pd.Series, pd.DataFrame, pd.DataFrame], Tuple[pd.Series, pd.DataFrame]]:
    """Validate OHLC relationships for each row.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing at least ``open``, ``high``, ``low`` and ``close``
        columns.
    return_flags : bool, default ``False``
        If ``True``, also return a DataFrame of booleans indicating where
        values were adjusted.
    return_clipped : bool, default ``False``
        If ``True``, also return a DataFrame with the clipped OHLC values.

    Returns
    -------
    pd.Series or tuple
        Boolean Series indexed like ``df`` where ``True`` indicates the row
        satisfies the OHLC relationship.  Additional return values depend on
        ``return_flags`` and ``return_clipped``.
    """

    # Compute min/max between open and close for each row
    oc = df[["open", "close"]]
    min_val = oc.min(axis=1)
    max_val = oc.max(axis=1)

    # Validate the min/max values lie within the low/high bounds
    mask = (df["low"] <= min_val) & (min_val <= max_val) & (max_val <= df["high"])

    if not return_flags and not return_clipped:
        return mask

    # Prepare clipped values ensuring ``low <= high``
    clipped = df.copy()
    clipped_high = df[["high", "low"]].max(axis=1)
    clipped_low = df[["high", "low"]].min(axis=1)
    clipped["high"] = clipped_high
    clipped["low"] = clipped_low
    clipped["open"] = clipped["open"].clip(lower=clipped_low, upper=clipped_high)
    clipped["close"] = clipped["close"].clip(lower=clipped_low, upper=clipped_high)

    if return_flags and return_clipped:
        flags = df != clipped
        return mask, clipped, flags
    if return_clipped:
        return mask, clipped
    if return_flags:
        flags = df != clipped
        return mask, flags

    return mask
