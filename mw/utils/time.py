"""Time helpers."""

from datetime import datetime, timezone


def now_utc() -> datetime:
    """Return the current time as an aware ``datetime`` in UTC."""
    return datetime.now(timezone.utc)


def floor_to_minute(dt: datetime) -> datetime:
    """Return ``dt`` floored to the start of the minute.

    The original timezone information is preserved.
    """

    return dt.replace(second=0, microsecond=0)
