"""
Finite-Time Lyapunov Exponent (Rosenstein) (stubs).

Exports:
- ftle_rosenstein(series: pd.Series, window: int, m: int=3, tau: int=1,
                  horizon: int=10, theiler: int=2) -> float
- rolling_ftle_rosenstein(series: pd.Series, window: int, m: int=3, tau: int=1,
                          horizon: int=10, theiler: int=2) -> pd.Series
Notes:
- Delay embedding, nearest neighbor excluding Theiler window,
  robust slope fit of log distance vs k, median across anchors.
- Clamp tiny distances with epsilon to avoid -inf.
"""
import pandas as pd

def ftle_rosenstein(series: pd.Series, window: int, m: int = 3, tau: int = 1,
                    horizon: int = 10, theiler: int = 2) -> float:
    # TODO: implement
    raise NotImplementedError

def rolling_ftle_rosenstein(series: pd.Series, window: int, m: int = 3, tau: int = 1,
                            horizon: int = 10, theiler: int = 2) -> pd.Series:
    # TODO: implement
    raise NotImplementedError
