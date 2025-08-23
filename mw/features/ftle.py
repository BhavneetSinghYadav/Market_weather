"""Finite-Time Lyapunov Exponent via the Rosenstein method.

Exports
-------
ftle_rosenstein
    Estimate the largest Lyapunov exponent for the last ``window`` points of a
    series using delay embedding and nearest neighbours.
rolling_ftle_rosenstein
    Causal rolling version of :func:`ftle_rosenstein` aligned to window end.

Notes
-----
- Delay embedding, nearest neighbour excluding Theiler window, robust slope fit
  of log distance vs k, median across anchors.
- Distances are clamped with an ``epsilon`` to avoid ``-inf`` when taking the
  logarithm.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


def _delay_embed(x: np.ndarray, m: int, tau: int) -> np.ndarray:
    """Return m-dimensional delay embedding of ``x`` with delay ``tau``."""
    n = x.size - (m - 1) * tau
    if n <= 0:
        return np.empty((0, m))
    cols = [x[i : i + n] for i in range(0, m * tau, tau)]  # noqa: E203
    return np.column_stack(cols)


def _slope(y: Iterable[float]) -> float:
    """Return slope of least-squares fit of ``y`` vs ``1..len(y)``."""
    x = np.arange(1, len(y) + 1, dtype=float)
    y = np.asarray(list(y), dtype=float)
    x_mean = x.mean()
    y_mean = y.mean()
    denom = ((x - x_mean) ** 2).sum()
    if denom == 0:
        return float("nan")
    return float(((x - x_mean) * (y - y_mean)).sum() / denom)


def ftle_rosenstein(
    series: pd.Series,
    window: int,
    m: int = 3,
    tau: int = 1,
    horizon: int = 10,
    theiler: int = 2,
) -> float:
    """Estimate the finite-time Lyapunov exponent using Rosenstein's method.

    Parameters
    ----------
    series
        Input time series. Only the last ``window`` observations are used.
    window
        Number of trailing observations to analyse.
    m, tau
        Embedding dimension and delay.
    horizon
        Forecast horizon ``k`` over which to track divergence.
    theiler
        Exclusion window to avoid temporally adjacent points when searching for
        nearest neighbours.
    """

    x = series.dropna().to_numpy()[-window:]
    embed = _delay_embed(x, m, tau)
    n_vectors = embed.shape[0]
    if n_vectors <= horizon + 1:
        return float("nan")

    valid = n_vectors - horizon
    anchors = embed[:valid]
    n = anchors.shape[0]
    slopes = []
    eps = 1e-8

    for i in range(n):
        dists = np.linalg.norm(anchors[i] - anchors, axis=1)
        mask = np.abs(np.arange(n) - i) <= theiler
        dists[mask] = np.inf
        j = int(np.argmin(dists))
        logs = []
        for k in range(1, horizon + 1):
            dist = np.linalg.norm(embed[i + k] - embed[j + k])
            logs.append(np.log(max(dist, eps)))
        slopes.append(_slope(logs))

    return float(np.nanmedian(slopes))


def rolling_ftle_rosenstein(
    series: pd.Series,
    window: int,
    m: int = 3,
    tau: int = 1,
    horizon: int = 10,
    theiler: int = 2,
) -> pd.Series:
    """Causal rolling FTLE using Rosenstein's method."""

    def _apply(x: pd.Series) -> float:
        return ftle_rosenstein(
            x, window=len(x), m=m, tau=tau, horizon=horizon, theiler=theiler
        )

    return series.rolling(window, min_periods=window).apply(_apply, raw=False)
