import pandas as pd
import pytest
import requests

from mw.adapters import polygon_rest


class DummyResponse:
    def __init__(self, payload):
        self.payload = payload
        self.status_code = 200

    def raise_for_status(self):
        pass

    def json(self):
        return self.payload


class FailingResponse:
    def __init__(self, message: str = "boom"):
        self.message = message

    def raise_for_status(self):
        raise requests.HTTPError(self.message)

    def json(self):
        return {}


def test_fetch_fx_agg_minute(monkeypatch):
    payload = {
        "results": [
            {"t": 0, "o": 1.0, "h": 1.5, "l": 0.5, "c": 1.2, "v": 10},
            {"t": 60_000, "o": 1.2, "h": 1.6, "l": 1.1, "c": 1.3, "v": 12},
        ]
    }
    called = {}

    def fake_get(url, params, timeout):
        called["url"] = url
        called["params"] = params
        return DummyResponse(payload)

    monkeypatch.setenv("POLYGON_API_KEY", "KEY")
    monkeypatch.setattr(requests, "get", fake_get)

    df = polygon_rest.fetch_fx_agg_minute("EURUSD", "2020-01-01", "2020-01-01")

    expected_url = (
        "https://api.polygon.io/v2/aggs/ticker/C:EURUSD/range/1/minute/"
        "2020-01-01/2020-01-01"
    )
    assert called["url"] == expected_url
    assert called["params"]["apiKey"] == "KEY"
    expected = pd.DataFrame(
        {
            "timestamp": pd.to_datetime([0, 60_000], unit="ms", utc=True),
            "open": [1.0, 1.2],
            "high": [1.5, 1.6],
            "low": [0.5, 1.1],
            "close": [1.2, 1.3],
            "volume": [10, 12],
        }
    )
    pd.testing.assert_frame_equal(df, expected)


def test_backfill_fx_agg_minute(monkeypatch):
    calls = []

    def fake_fetch(symbol, start, end, limit=50_000):
        calls.append((start, end))
        return pd.DataFrame(
            {
                "timestamp": pd.to_datetime([0], unit="ms", utc=True),
                "open": [1.0],
                "high": [1.0],
                "low": [1.0],
                "close": [1.0],
                "volume": [1],
            }
        )

    monkeypatch.setattr(polygon_rest, "fetch_fx_agg_minute", fake_fetch)

    frames = list(
        polygon_rest.backfill_fx_agg_minute(
            "EURUSD", "2020-01-01", "2020-01-04", chunk_days=2
        )
    )

    assert calls == [
        ("2020-01-01", "2020-01-02"),
        ("2020-01-03", "2020-01-04"),
    ]
    assert len(frames) == 2
    for frame in frames:
        assert list(frame.columns) == [
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "volume",
        ]


def test_fetch_fx_agg_minute_retries_then_succeeds(monkeypatch):
    payload = {
        "results": [
            {"t": 0, "o": 1.0, "h": 1.0, "l": 1.0, "c": 1.0, "v": 1},
        ]
    }
    responses = [FailingResponse(), DummyResponse(payload)]
    calls = {"n": 0}

    def fake_get(url, params, timeout):
        resp = responses[calls["n"]]
        calls["n"] += 1
        return resp

    monkeypatch.setenv("POLYGON_API_KEY", "KEY")
    monkeypatch.setattr(requests, "get", fake_get)
    monkeypatch.setattr(polygon_rest.time, "sleep", lambda s: None)

    df = polygon_rest.fetch_fx_agg_minute("EURUSD", "2020-01-01", "2020-01-01")

    assert calls["n"] == 2
    expected = pd.DataFrame(
        {
            "timestamp": pd.to_datetime([0], unit="ms", utc=True),
            "open": [1.0],
            "high": [1.0],
            "low": [1.0],
            "close": [1.0],
            "volume": [1],
        }
    )
    pd.testing.assert_frame_equal(df, expected)


def test_fetch_fx_agg_minute_retries_and_fails(monkeypatch):
    calls = {"n": 0}

    def fake_get(url, params, timeout):
        calls["n"] += 1
        return FailingResponse("fail")

    monkeypatch.setenv("POLYGON_API_KEY", "KEY")
    monkeypatch.setattr(requests, "get", fake_get)
    monkeypatch.setattr(polygon_rest.time, "sleep", lambda s: None)

    with pytest.raises(RuntimeError, match="fail"):
        polygon_rest.fetch_fx_agg_minute("EURUSD", "2020-01-01", "2020-01-01")

    assert calls["n"] == 3
