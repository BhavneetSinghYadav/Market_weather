import sys
from pathlib import Path
from datetime import datetime, timezone

sys.path.append(str(Path(__file__).resolve().parents[1]))

from mw.utils.time import now_utc  # noqa: E402


def test_now_utc_timezone_and_proximity():
    current = now_utc()
    assert current.tzinfo == timezone.utc
    system_now = datetime.now(timezone.utc)
    assert abs((system_now - current).total_seconds()) < 1
