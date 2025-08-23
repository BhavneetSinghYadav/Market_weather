import sys
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from mw.live.logger import SessionLogger  # noqa: E402


def test_session_logger_writes_csv_and_summary(tmp_path):
    csv_path = tmp_path / "log.csv"
    summary_path = tmp_path / "summary.json"
    logger = SessionLogger(csv_path, summary_path)

    ts1 = datetime(2024, 1, 1, tzinfo=timezone.utc)
    ts2 = datetime(2024, 1, 1, 0, 1, tzinfo=timezone.utc)

    logger.log_minute(ts1, 0.1, "GREEN", {"a": 1})
    logger.log_minute(ts2, -0.2, "RED", {"b": 2})
    logger.close(session="abc")

    with csv_path.open() as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    assert reader.fieldnames == ["timestamp", "score", "state", "diagnostics"]
    assert rows[0]["timestamp"] == ts1.isoformat()
    assert float(rows[0]["score"]) == 0.1
    assert json.loads(rows[0]["diagnostics"]) == {"a": 1}
    assert rows[1]["timestamp"] == ts2.isoformat()
    assert rows[1]["state"] == "RED"

    summary = json.loads(summary_path.read_text())
    assert summary["rows"] == 2
    assert "start" in summary and "end" in summary
    assert summary["session"] == "abc"
