"""
Health & freshness (stub).

Exports:
- compute_freshness(last_bar_ts) -> float seconds
- should_degrade(freshness_s: float, thresh: float=180) -> bool
"""
from datetime import datetime, timezone

def compute_freshness(last_bar_ts: datetime) -> float:
    # TODO: implement
    raise NotImplementedError

def should_degrade(freshness_s: float, thresh: float = 180.0) -> bool:
    # TODO: implement
    raise NotImplementedError
