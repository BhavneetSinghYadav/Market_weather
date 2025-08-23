import pandas as pd
import pytest
import requests

from mw.adapters import polygon_quotes, polygon_rest


class DummyResponse:
    def __init__(self, payload, status_code=200, headers=None):
        self.payload = payload
        self.status_code = status_code
        self.headers = headers or {}

    def raise_for_status(self):
        if self.status_code != 200:
            raise requests.HTTPError(str(self.status_code))

    def json(self):
        return self.payload


class FailingResponse:
    def __init__(self, status_code, headers=None):
        self.status_code = status_code
        self.headers = headers or {}

    def raise_for_status(self):
        raise requests.HTTPError(str(self.status_code))

    def json(self):
        return {}


def test_fetch_quotes_paginates(monkeypatch):
    payload1 = {
        "results": [
            {
                "sip_timestamp": 0,
                "bid_price": 1.0,
                "ask_price": 2.0,
                "participant_exchange": "V",
            },
            {
                "sip_timestamp": 1,
                "bid_price": 1.1,
                "ask_price": 2.1,
                "participant_exchange": "V",
            },
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
    expected_url = "https://api.polygon.io/v3/quotes/XYZ?cursor=abc&apiKey=KEY"
    assert calls[1][0] == expected_url
    assert calls[1][1] is None

    expected_ts = pd.to_datetime([0, 1, 2, 3], unit="ns", utc=True)
    assert df["ts_utc"].tolist() == list(expected_ts)
    assert df["bid"].tolist() == [1.0, 1.1, 1.2, 1.3]
    assert df["ask"].tolist() == [2.0, 2.1, 2.2, 2.3]
    assert df["mid"].tolist() == pytest.approx([1.5, 1.6, 1.7, 1.8])
    assert list(df.columns) == ["ts_utc", "bid", "ask", "mid", "venue"]
    assert df["ts_utc"].is_monotonic_increasing


def test_fetch_quotes_three_pages(monkeypatch):
    payload1 = {
        "results": [
            {"sip_timestamp": 0, "bid_price": 1.0, "ask_price": 2.0},
            {"sip_timestamp": 1, "bid_price": 1.1, "ask_price": 2.1},
        ],
        "next_url": "https://api.polygon.io/v3/quotes/XYZ?cursor=abc",
    }
    payload2 = {
        "results": [
            {"sip_timestamp": 2, "bid_price": 1.2, "ask_price": 2.2},
            {"sip_timestamp": 3, "bid_price": 1.3, "ask_price": 2.3},
        ],
        "next_url": "https://api.polygon.io/v3/quotes/XYZ?cursor=def",
    }
    payload3 = {
        "results": [
            {"sip_timestamp": 4, "bid_price": 1.4, "ask_price": 2.4},
            {"sip_timestamp": 5, "bid_price": 1.5, "ask_price": 2.5},
        ]
    }
    responses = [
        DummyResponse(payload1),
        DummyResponse(payload2),
        DummyResponse(payload3),
    ]
    calls = []

    def fake_get(url, params=None, timeout=10):
        calls.append((url, params))
        return responses.pop(0)

    monkeypatch.setenv("POLYGON_API_KEY", "KEY")
    monkeypatch.setattr(requests, "get", fake_get)

    df = polygon_quotes.fetch_quotes("XYZ", 0, 5, limit=2)

    assert calls[0][0] == "https://api.polygon.io/v3/quotes/XYZ"
    assert calls[0][1]["apiKey"] == "KEY"
    expected_url2 = "https://api.polygon.io/v3/quotes/XYZ?cursor=abc&apiKey=KEY"
    expected_url3 = "https://api.polygon.io/v3/quotes/XYZ?cursor=def&apiKey=KEY"
    assert calls[1][0] == expected_url2
    assert calls[1][1] is None
    assert calls[2][0] == expected_url3
    assert calls[2][1] is None

    expected_ts = pd.to_datetime([0, 1, 2, 3, 4, 5], unit="ns", utc=True)
    assert df["ts_utc"].tolist() == list(expected_ts)
    assert df["bid"].tolist() == [1.0, 1.1, 1.2, 1.3, 1.4, 1.5]
    assert df["ask"].tolist() == [2.0, 2.1, 2.2, 2.3, 2.4, 2.5]
    assert df["mid"].tolist() == pytest.approx([1.5, 1.6, 1.7, 1.8, 1.9, 2.0])
    assert df["ts_utc"].is_monotonic_increasing


def test_fetch_quotes_retries_on_429(monkeypatch):
    payload = {
        "results": [
            {
                "sip_timestamp": 0,
                "bid_price": 1.0,
                "ask_price": 2.0,
            },
        ]
    }
    responses = [
        FailingResponse(429, headers={"Retry-After": "4"}),
        DummyResponse(payload),
    ]
    calls = {"n": 0}

    def fake_get(url, params=None, timeout=10):
        resp = responses[calls["n"]]
        calls["n"] += 1
        return resp

    monkeypatch.setenv("POLYGON_API_KEY", "KEY")
    monkeypatch.setattr(requests, "get", fake_get)
    sleeps: list[float] = []
    monkeypatch.setattr(polygon_rest.time, "sleep", lambda s: sleeps.append(s))

    df = polygon_quotes.fetch_quotes("XYZ", 0, 0)

    assert calls["n"] == 2
    assert sleeps == [4.0]
    expected_ts = pd.to_datetime([0], unit="ns", utc=True)
    assert df["ts_utc"].tolist() == list(expected_ts)
    assert df["bid"].tolist() == [1.0]
    assert df["ask"].tolist() == [2.0]
    assert df["mid"].tolist() == [1.5]
