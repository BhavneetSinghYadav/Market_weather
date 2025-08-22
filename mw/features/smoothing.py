"""
Smoothing / hysteresis helpers (stubs).

Exports:
- ema(series: pd.Series, span: int=3) -> pd.Series
"""
import pandas as pd

def ema(series: pd.Series, span: int = 3) -> pd.Series:
    """Simple exponential moving average (causal)."""
    # TODO: implement (use pandas ewm with adjust=False)
    raise NotImplementedError
