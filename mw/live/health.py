"""Health & freshness utilities.

Exports:
- compute_freshness(last_bar_ts) -> float seconds
- should_degrade(freshness_s: float, thresh: float = 180) -> bool
"""

from datetime import datetime, timezone
from typing import Optional


def compute_freshness(last_bar_ts: Optional[datetime]) -> float:
    """Return seconds since ``last_bar_ts``.

    ``last_bar_ts`` is expected to be an aware ``datetime`` in UTC. If the
    timestamp is missing ``float('inf')`` is returned. The calculation uses the
    current UTC time.
    """

    if last_bar_ts is None:
        return float("inf")

    now = datetime.now(timezone.utc)
    return (now - last_bar_ts).total_seconds()


def should_degrade(freshness_s: float, thresh: float = 180.0) -> bool:
    """Return ``True`` when ``freshness_s`` exceeds ``thresh``.

    Parameters
    ----------
    freshness_s: float
        Age of the last data point in seconds.
    thresh: float
        Threshold in seconds before degradation should occur. The default of
        ``180`` seconds (three minutes) represents the service's acceptable age
        for data before degradation.
    """

    return freshness_s > thresh
