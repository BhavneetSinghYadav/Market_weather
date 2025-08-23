"""
Time helpers (stub).
"""
from datetime import datetime, timezone, timedelta


def now_utc() -> datetime:
    """Return the current time as an aware ``datetime`` in UTC."""
    return datetime.now(timezone.utc)

def floor_to_minute(dt: datetime) -> datetime:
    # TODO: implement
    raise NotImplementedError
