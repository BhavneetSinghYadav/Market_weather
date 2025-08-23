"""Session logging utilities.

Provides :class:`SessionLogger` to append per-minute observations to a CSV
and write a JSON session summary on shutdown.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from mw.utils.persistence import append_csv, write_json
from mw.utils.time import now_utc


@dataclass
class GapEvent:
    """Represents a missing minute bar in the data feed."""

    timestamp: datetime
    symbol: str
    reason: str


@dataclass
class SessionLogger:
    """Log per-minute results, gaps and summarise the session."""

    csv_path: Path
    summary_path: Path
    start: datetime = field(default_factory=now_utc)
    rows: int = 0
    gap_events: List[GapEvent] = field(default_factory=list)
    gap_count: int = 0

    def log_minute(
        self,
        timestamp: datetime,
        score: float,
        state: str,
        diagnostics: Dict[str, Any],
    ) -> None:
        """Append a record for ``timestamp`` to the CSV file."""

        append_csv(
            {
                "timestamp": timestamp.isoformat(),
                "score": score,
                "state": state,
                "diagnostics": json.dumps(diagnostics),
            },
            self.csv_path,
            ["timestamp", "score", "state", "diagnostics"],
        )
        self.rows += 1

    def close(self, **extra: Any) -> None:
        """Write a JSON summary of the session."""

        summary = {
            "start": self.start.isoformat(),
            "end": now_utc().isoformat(),
            "rows": self.rows,
            "gap_count": self.gap_count,
        }
        summary.update(extra)
        write_json(summary, self.summary_path.as_posix())

    def log_gap(self, event: GapEvent) -> None:
        """Record a missing minute bar event."""

        self.gap_events.append(event)
        self.gap_count += 1
