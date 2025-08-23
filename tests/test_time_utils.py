import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from mw.utils.time import floor_to_minute, now_utc  # noqa: E402


def test_now_utc_timezone_and_proximity():
    current = now_utc()
    assert current.tzinfo == timezone.utc
    system_now = datetime.now(timezone.utc)
    assert abs((system_now - current).total_seconds()) < 1


@pytest.mark.parametrize(
    "dt_raw, expected_raw",
    [
        (
            datetime(2024, 1, 1, 12, 34, 56, 789),
            datetime(2024, 1, 1, 12, 34),
        ),
        (
            datetime(2024, 1, 1, 23, 59, 59, 999_999),
            datetime(2024, 1, 1, 23, 59),
        ),
        (
            datetime(2024, 1, 1, 0, 0, 0, 0),
            datetime(2024, 1, 1, 0, 0),
        ),
    ],
)
@pytest.mark.parametrize(
    "tz",
    [
        None,
        timezone.utc,
        timezone(timedelta(hours=-5)),
        timezone(timedelta(hours=5, minutes=30)),
    ],
)
def test_floor_to_minute_preserves_timezone_and_floors(
    dt_raw,
    expected_raw,
    tz,
):
    dt = dt_raw.replace(tzinfo=tz)
    expected = expected_raw.replace(tzinfo=tz)
    assert floor_to_minute(dt) == expected
