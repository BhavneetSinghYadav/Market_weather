"""
OHLC integrity checks (stub).
"""
import pandas as pd

def validate_ohlc(df: pd.DataFrame) -> pd.Series:
    """Return boolean mask of rows passing L <= min(O,C) <= max(O,C) <= H."""
    # TODO: implement
    raise NotImplementedError
