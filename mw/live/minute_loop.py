"""Live minute loop orchestration.

This module coordinates the high level steps executed every minute:

``poll -> compute -> persist -> log -> plot -> health``.

The function waits for configured offsets from the start of the current
minute before invoking each step so that calls are synchronised with
minute boundaries.  Offsets (in seconds) are supplied via ``params`` and
default to a small stagger between steps.
"""

import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Optional, Union

import pandas as pd

from mw.live.health import evaluate_freshness
from mw.utils.params import MinuteLoopParams, Params
from mw.utils.persistence import write_parquet
from mw.utils.time import floor_to_minute, now_utc


def run_minute_loop(
    poll_fn: Callable,
    compute_fn: Callable,
    persist_fn: Callable,
    log_fn: Callable,
    plot_fn: Callable,
    health_fn: Callable,
    params: Union["MinuteLoopParams", "Params", None],
    error_fn: Optional[Callable[[str, Exception], None]] = None,
    last_bar_ts_fn: Optional[Callable[[], Optional[datetime]]] = None,
    stale_fn: Optional[Callable[[float], None]] = None,
) -> None:
    """High-level loop; call every minute (t+3s).

    The supplied callables are invoked sequentially with waits applied
    before each invocation to honour per-step offsets from the start of
    the current minute.
    """

    if params is None:
        ml_params = MinuteLoopParams()
    elif isinstance(params, MinuteLoopParams):
        ml_params = params
    elif isinstance(params, Params):
        ml_params = params.minute_loop
    else:
        raise TypeError("params must be Params or MinuteLoopParams")

    offsets = ml_params.offsets

    critical_steps = set(ml_params.critical_steps)

    minute_start = floor_to_minute(now_utc())

    stale = False
    stale_thresh = ml_params.freshness_stale_threshold

    def freshness_check() -> None:
        nonlocal stale
        if last_bar_ts_fn is None:
            return
        last_ts = last_bar_ts_fn()
        freshness, degrade = evaluate_freshness(last_ts, stale_thresh)
        if degrade:
            stale = True
            if stale_fn is not None:
                stale_fn(freshness)

    steps = [
        ("poll", poll_fn),
        ("freshness", freshness_check),
        ("compute", compute_fn),
        ("persist", persist_fn),
        ("log", log_fn),
        ("plot", plot_fn),
        ("health", health_fn),
    ]

    skip_when_stale = {"compute", "persist", "log", "plot"}
    features: Any = None

    for name, fn in steps:
        offset_name = "compute" if name == "freshness" else name
        target = minute_start + timedelta(seconds=offsets.get(offset_name, 0))
        sleep_for = (target - now_utc()).total_seconds()
        time.sleep(max(0.0, sleep_for))
        if stale and name in skip_when_stale:
            continue
        try:
            if name == "compute":
                features = fn()
                _persist_features(features)
            else:
                fn()
        except Exception as exc:
            logging.exception("Exception in %s step", name)
            if error_fn is not None:
                error_fn(name, exc)
            if name in critical_steps:
                break


def _persist_features(features: Any) -> None:
    """Persist feature dataframes to the ``data/`` hierarchy."""

    if features is None:
        return
    base = Path("data")
    if isinstance(features, pd.DataFrame):
        write_parquet(features, str(base / "features.parquet"))
    elif isinstance(features, dict):
        for name, df in features.items():
            if isinstance(df, pd.DataFrame):
                write_parquet(df, str(base / f"{name}.parquet"))
