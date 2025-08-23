"""Polygon REST adapter."""

from __future__ import annotations

import os
import random
import time
from datetime import timedelta
from typing import Iterator

import pandas as pd
import requests

BASE_URL = "https://api.polygon.io"

_SESSION: requests.Session | None = None
_REQUESTS_GET = requests.get


def _get_session() -> requests.Session:
    """Return a global ``requests.Session`` instance."""
    global _SESSION
    if _SESSION is None:
        _SESSION = requests.Session()
    return _SESSION


def _get_api_key() -> str:
    """Return the Polygon API key from the environment."""
    api_key = os.getenv("POLYGON_API_KEY")
    if not api_key:
        raise RuntimeError("POLYGON_API_KEY not set")
    return api_key


def _request_with_retry(
    url: str,
    params: dict | None = None,
    *,
    timeout: int = 10,
    max_attempts: int = 3,
) -> requests.Response:
    """Make a GET request with retry logic.

    Retries on HTTP 429 and 5xx responses using a jittered exponential
    backoff.  If the response includes a ``Retry-After`` header it is used
    verbatim for the sleep interval.
    """

    session = _get_session()
    get_call = session.get if requests.get is _REQUESTS_GET else requests.get

    for attempt in range(max_attempts):
        resp = get_call(url, params=params, timeout=timeout)
        if resp.status_code == 429 or resp.status_code >= 500:
            if attempt == max_attempts - 1:
                resp.raise_for_status()
            retry_after = resp.headers.get("Retry-After")
            if retry_after is not None:
                try:
                    sleep_time = float(retry_after)
                except ValueError:  # pragma: no cover - safety
                    sleep_time = 0
                if sleep_time <= 0:
                    sleep_time = (2**attempt) + random.uniform(0, 1)
            else:
                sleep_time = (2**attempt) + random.uniform(0, 1)
            time.sleep(sleep_time)
            continue
        resp.raise_for_status()
        return resp

    # Should not reach here; raise for completeness.
    resp.raise_for_status()


def fetch_fx_agg_minute(
    symbol: str, start: str | int, end: str | int, limit: int = 50_000
) -> pd.DataFrame:
    """Fetch minute aggregates for ``symbol`` between ``start`` and ``end``.

    Parameters
    ----------
    symbol:
        Currency pair such as ``"EURUSD"``.
    start, end:
        Either ``YYYY-MM-DD`` date strings or millisecond timestamps
        understood by the Polygon aggregates endpoint.
    limit:
        Maximum number of rows to return from Polygon (default 50,000).

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ``timestamp``, ``open``, ``high``, ``low``,
        ``close`` and ``volume``.  ``timestamp`` is timezone aware (UTC).
    """

    api_key = _get_api_key()
    url = f"{BASE_URL}/v2/aggs/ticker/C:{symbol}/range/1/minute/{start}/{end}"
    params = {
        "adjusted": "true",
        "sort": "asc",
        "limit": limit,
        "apiKey": api_key,
    }
    try:
        resp = _request_with_retry(url, params=params)
    except requests.RequestException as exc:  # pragma: no cover - safety
        raise RuntimeError(
            f"Polygon API request failed after 3 attempts: {exc}"
        ) from exc
    try:
        data = resp.json()
    except ValueError as exc:
        raise RuntimeError("Polygon API returned invalid JSON") from exc
    results = data.get("results", [])
    if not results:
        return pd.DataFrame(
            columns=["timestamp", "open", "high", "low", "close", "volume"]
        )
    df = pd.DataFrame(results)
    df = df.rename(
        columns={
            "t": "timestamp",
            "o": "open",
            "h": "high",
            "l": "low",
            "c": "close",
            "v": "volume",
        }
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df.attrs["source_time_basis"] = "UTC"
    return df[["timestamp", "open", "high", "low", "close", "volume"]]


def backfill_fx_agg_minute(
    symbol: str, start: str, end: str, chunk_days: int = 3
) -> Iterator[pd.DataFrame]:
    """Yield minute aggregates for ``symbol`` over a date range.

    The date range ``start`` → ``end`` (inclusive) is split into chunks of
    ``chunk_days`` each.  Each yielded dataframe contains canonical columns as
    described by :func:`fetch_fx_agg_minute`.
    """

    start_dt = pd.to_datetime(start).tz_localize("UTC")
    end_dt = pd.to_datetime(end).tz_localize("UTC")
    current = start_dt
    delta = timedelta(days=chunk_days - 1)

    while current <= end_dt:
        chunk_end = min(current + delta, end_dt)
        df = fetch_fx_agg_minute(
            symbol,
            current.strftime("%Y-%m-%d"),
            chunk_end.strftime("%Y-%m-%d"),
        )
        if not df.empty:
            yield df
        current = chunk_end + timedelta(days=1)
