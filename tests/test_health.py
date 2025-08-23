from datetime import datetime, timezone
from unittest.mock import patch

from mw.live.health import compute_freshness


def test_compute_freshness_elapsed_seconds():
    now = datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    last_bar = datetime(2023, 1, 1, 11, 59, 0, tzinfo=timezone.utc)
    with patch("mw.live.health.datetime") as mock_datetime:
        mock_datetime.now.return_value = now
        result = compute_freshness(last_bar)
    assert result == 60.0


def test_compute_freshness_missing_timestamp():
    assert compute_freshness(None) == float("inf")
