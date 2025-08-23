import pandas as pd
import requests
import pytest

from mw.adapters import polygon_quotes


class DummyResponse:
    def __init__(self, payload, status_code=200):
        self.payload = payload
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code != 200:
            raise requests.HTTPError(str(self.status_code))

    def json(self):
        return self.payload


class FailingResponse:
    def __init__(self, status_code):
        self.status_code = status_code

    def raise_for_status(self):
        raise requests.HTTPError(str(self.status_code))

    def json(self):
        return {}


def test_fetch_quotes_paginates(monkeypatch):
    payload1 = {
        "results": [
            {"sip_timestamp": 0, "bid_price": 1.0, "ask_price": 2.0, "participant_exchange": "V"},
            {"sip_timestamp": 1, "bid_price": 1.1, "ask_price": 2.1, "participant_exchange": "V"},
        ],
        "next_url": "https://api.polygon.io/v3/quotes/XYZ?cursor=abc",
    }
    payload2 = {
        "results": [
            {"sip_timestamp": 2, "bid_price": 1.2, "ask_price": 2.2},
            {"sip_timestamp": 3, "bid_price": 1.3, "ask_price": 2.3},
        ]
    }
    responses = [DummyResponse(payload1), DummyResponse(payload2)]
    calls = []

    def fake_get(url, params=None, timeout=10):
        calls.append((url, params))
        return responses.pop(0)

    monkeypatch.setenv("POLYGON_API_KEY", "KEY")
    monkeypatch.setattr(requests, "get", fake_get)

    df = polygon_quotes.fetch_quotes("XYZ", 0, 3, limit=2)

    assert calls[0][0] == "https://api.polygon.io/v3/quotes/XYZ"
    assert calls[0][1]["apiKey"] == "KEY"
    assert calls[1][0] == "https://api.polygon.io/v3/quotes/XYZ?cursor=abc&apiKey=KEY"
    assert calls[1][1] is None

    expected_ts = pd.to_datetime([0, 1, 2, 3], unit="ns", utc=True)
    assert df["ts_utc"].tolist() == list(expected_ts)
    assert df["bid"].tolist() == [1.0, 1.1, 1.2, 1.3]
    assert df["ask"].tolist() == [2.0, 2.1, 2.2, 2.3]
    assert df["mid"].tolist() == pytest.approx([1.5, 1.6, 1.7, 1.8])
    assert list(df.columns) == ["ts_utc", "bid", "ask", "mid", "venue"]
    assert df["ts_utc"].is_monotonic_increasing


def test_fetch_quotes_retries_on_429(monkeypatch):
    payload = {"results": [{"sip_timestamp": 0, "bid_price": 1.0, "ask_price": 2.0}]}
    responses = [FailingResponse(429), DummyResponse(payload)]
    calls = {"n": 0}

    def fake_get(url, params=None, timeout=10):
        resp = responses[calls["n"]]
        calls["n"] += 1
        return resp

    monkeypatch.setenv("POLYGON_API_KEY", "KEY")
    monkeypatch.setattr(requests, "get", fake_get)
    monkeypatch.setattr(polygon_quotes.time, "sleep", lambda s: None)

    df = polygon_quotes.fetch_quotes("XYZ", 0, 0)

    assert calls["n"] == 2
    expected_ts = pd.to_datetime([0], unit="ns", utc=True)
    assert df["ts_utc"].tolist() == list(expected_ts)
    assert df["bid"].tolist() == [1.0]
    assert df["ask"].tolist() == [2.0]
    assert df["mid"].tolist() == [1.5]
