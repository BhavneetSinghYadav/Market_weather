"""Polygon quotes adapter with pagination."""

from __future__ import annotations

from typing import List

import pandas as pd
import requests

from .polygon_rest import BASE_URL, _get_api_key, _request_with_retry


def fetch_quotes(
    ticker: str, start: str | int, end: str | int, limit: int = 50_000
) -> pd.DataFrame:
    """Fetch NBBO quotes for ``ticker`` between ``start`` and ``end``.

    Parameters
    ----------
    ticker:
        Ticker symbol such as ``"AAPL"``.
    start, end:
        Either ``YYYY-MM-DD`` date strings or nanosecond timestamps understood
        by the Polygon quotes endpoint.
    limit:
        Maximum number of rows to request from Polygon per page
        (default 50,000).

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ``ts_utc``, ``bid``, ``ask``, ``mid`` and
        ``venue`` (if available).
    """

    api_key = _get_api_key()
    url = f"{BASE_URL}/v3/quotes/{ticker}"
    params = {
        "timestamp.gte": start,
        "timestamp.lte": end,
        "limit": limit,
        "sort": "timestamp",
        "order": "asc",
        "apiKey": api_key,
    }
    all_results: List[dict] = []
    while url:
        try:
            resp = _request_with_retry(
                url, params=params if params is not None else None
            )
        except requests.RequestException as exc:  # pragma: no cover - safety
            raise RuntimeError("Polygon API request failed") from exc

        data = resp.json()
        results = data.get("results", [])
        all_results.extend(results)
        next_url = data.get("next_url")
        if next_url:
            if "apiKey" not in next_url:
                next_url = f"{next_url}&apiKey={api_key}"
            url = next_url
            params = None
        else:
            url = None

    if not all_results:
        return pd.DataFrame(columns=["ts_utc", "bid", "ask", "mid", "venue"])

    df = pd.DataFrame(all_results)
    if "sip_timestamp" in df.columns:
        df["ts_utc"] = pd.to_datetime(df["sip_timestamp"], unit="ns", utc=True)
    else:
        df["ts_utc"] = pd.to_datetime(df["timestamp"], unit="ns", utc=True)
    df["bid"] = df["bid_price"]
    df["ask"] = df["ask_price"]
    df["mid"] = (df["bid"] + df["ask"]) / 2

    venue_col = None
    for col in ("participant_exchange", "exchange", "x"):
        if col in df.columns:
            venue_col = col
            break
    if venue_col is not None:
        df["venue"] = df[venue_col]
    else:
        df["venue"] = pd.NA
    df.attrs["source_time_basis"] = "UTC"
    return df[["ts_utc", "bid", "ask", "mid", "venue"]]
