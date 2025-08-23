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
class DuplicateDrop:
    """Represents a dropped duplicate bar."""

    timestamp: datetime
    symbol: str


@dataclass
class LateBar:
    """Represents a bar that arrived too late to be processed."""

    timestamp: datetime
    symbol: str


@dataclass
class SessionLogger:
    """Log per-minute results, gaps and summarise the session."""

    csv_path: Path
    summary_path: Path
    start: datetime = field(default_factory=now_utc)
    rows: int = 0
    gap_events: List[GapEvent] = field(default_factory=list)
    gap_count: int = 0
    duplicate_events: List[DuplicateDrop] = field(default_factory=list)
    duplicate_count: int = 0
    late_bar_events: List[LateBar] = field(default_factory=list)
    late_bar_count: int = 0
    seen_bars: int = 0
    max_api_latency: float = 0.0

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
            "duplicate_count": self.duplicate_count,
            "late_bar_count": self.late_bar_count,
            "health": self.health_summary(),
        }
        summary.update(extra)
        write_json(summary, self.summary_path.as_posix())

    def log_gap(self, event: GapEvent) -> None:
        """Record a missing minute bar event."""

        self.gap_events.append(event)
        self.gap_count += 1

    def log_duplicate(self, event: DuplicateDrop) -> None:
        """Record a dropped duplicate bar."""

        self.duplicate_events.append(event)
        self.duplicate_count += 1

    def log_late_bar(self, event: LateBar) -> None:
        """Record a bar that arrived too late to be processed."""

        self.late_bar_events.append(event)
        self.late_bar_count += 1

    def log_seen_bars(self, count: int) -> None:
        """Increment the count of bars seen this session."""

        self.seen_bars += count

    def record_api_latency(self, latency_s: float) -> None:
        """Update the maximum observed API latency."""

        if latency_s > self.max_api_latency:
            self.max_api_latency = latency_s

    def health_summary(self) -> Dict[str, Any]:
        """Return a summary of health metrics collected during the session."""

        return {
            "seen_bars": self.seen_bars,
            "gap_count": self.gap_count,
            "duplicate_count": self.duplicate_count,
            "late_bar_count": self.late_bar_count,
            "max_api_latency": self.max_api_latency,
        }
