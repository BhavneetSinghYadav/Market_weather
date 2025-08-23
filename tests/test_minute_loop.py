import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from mw.live.minute_loop import run_minute_loop  # noqa: E402


def test_run_minute_loop_calls_functions_in_order(monkeypatch):
    call_order = []

    def mk(name):
        return lambda: call_order.append(name)

    poll = mk("poll")
    compute = mk("compute")
    persist = mk("persist")
    plot = mk("plot")
    health = mk("health")

    sleeps = []
    monkeypatch.setattr("time.sleep", lambda x: sleeps.append(x))

    start = datetime(2024, 1, 1, 0, 0, 2, tzinfo=timezone.utc)
    times_iter = iter(
        [
            start,
            start,
            start + timedelta(seconds=2),
            start + timedelta(seconds=5),
            start + timedelta(seconds=6),
            start + timedelta(seconds=7),
        ]
    )
    monkeypatch.setattr(
        "mw.live.minute_loop.now_utc",
        lambda: next(times_iter),
    )

    params = {
        "minute_loop_offsets": {
            "poll": 3,
            "compute": 6,
            "persist": 7,
            "plot": 8,
            "health": 9,
        }
    }

    run_minute_loop(poll, compute, persist, plot, health, params)

    assert call_order == ["poll", "compute", "persist", "plot", "health"]
    assert sleeps == [1.0, 2.0, 0.0, 0.0, 0.0]


def test_run_minute_loop_continues_after_failure(monkeypatch):
    call_order = []

    def poll():
        raise ValueError("boom")

    def compute():
        call_order.append("compute")

    def persist():
        call_order.append("persist")

    def plot():
        call_order.append("plot")

    def health():
        call_order.append("health")

    monkeypatch.setattr("time.sleep", lambda x: None)

    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    monkeypatch.setattr("mw.live.minute_loop.now_utc", lambda: start)

    params = {"minute_loop_offsets": {}}

    run_minute_loop(poll, compute, persist, plot, health, params)

    assert call_order == ["compute", "persist", "plot", "health"]
