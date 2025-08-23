"""Session logging utilities.

Provides :class:`SessionLogger` to append per-minute observations to a CSV
and write a JSON session summary on shutdown.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from mw.utils.time import now_utc


@dataclass
class SessionLogger:
    """Log per-minute results and summarise the session."""

    csv_path: Path
    summary_path: Path
    start: datetime = field(default_factory=now_utc)
    rows: int = 0

    def log_minute(
        self,
        timestamp: datetime,
        score: float,
        state: str,
        diagnostics: Dict[str, Any],
    ) -> None:
        """Append a record for ``timestamp`` to the CSV file."""

        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        exists = self.csv_path.exists()
        with self.csv_path.open("a", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=["timestamp", "score", "state", "diagnostics"]
            )
            if not exists:
                writer.writeheader()
            writer.writerow(
                {
                    "timestamp": timestamp.isoformat(),
                    "score": score,
                    "state": state,
                    "diagnostics": json.dumps(diagnostics),
                }
            )
        self.rows += 1

    def close(self, **extra: Any) -> None:
        """Write a JSON summary of the session."""

        summary = {
            "start": self.start.isoformat(),
            "end": now_utc().isoformat(),
            "rows": self.rows,
        }
        summary.update(extra)
        self.summary_path.parent.mkdir(parents=True, exist_ok=True)
        with self.summary_path.open("w") as f:
            json.dump(summary, f)
