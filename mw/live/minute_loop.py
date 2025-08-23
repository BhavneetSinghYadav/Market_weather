"""Live minute loop orchestration.

This module coordinates the high level steps executed every minute:

``poll -> compute -> persist -> plot -> health``.

The function waits for configured offsets from the start of the current
minute before invoking each step so that calls are synchronised with
minute boundaries.  Offsets (in seconds) are supplied via ``params`` and
default to a small stagger between steps.
"""

import logging
import time
from datetime import timedelta
from typing import Any, Callable, Dict

from mw.utils.time import floor_to_minute, now_utc


def run_minute_loop(
    poll_fn: Callable,
    compute_fn: Callable,
    persist_fn: Callable,
    plot_fn: Callable,
    health_fn: Callable,
    params: Dict[str, Any],
) -> None:
    """High-level loop; call every minute (t+3s).

    The supplied callables are invoked sequentially with waits applied
    before each invocation to honour per-step offsets from the start of
    the current minute.
    """

    offsets = params.get(
        "minute_loop_offsets",
        {"poll": 3, "compute": 5, "persist": 6, "plot": 7, "health": 8},
    )

    minute_start = floor_to_minute(now_utc())

    steps = [
        ("poll", poll_fn),
        ("compute", compute_fn),
        ("persist", persist_fn),
        ("plot", plot_fn),
        ("health", health_fn),
    ]

    for name, fn in steps:
        target = minute_start + timedelta(seconds=offsets.get(name, 0))
        sleep_for = (target - now_utc()).total_seconds()
        time.sleep(max(0.0, sleep_for))
        try:
            fn()
        except Exception:
            logging.exception("Exception in %s step", name)
