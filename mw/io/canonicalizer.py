"""Canonicalize raw OHLCV data.

This module normalises raw minute level market data so that downstream
components can rely on a stable contract.  The process consists of:

* converting timestamps to UTC based on a declared source timezone;
* enforcing a strict one minute grid, keeping only the last observation
  when duplicates are present and inserting explicit gap rows when data is
  missing;
* validating the ``open``, ``high``, ``low`` and ``close`` relationship for
  every observation; and
* persisting the result as a Parquet file accompanied by a ``.meta.json``
  file containing basic integrity information.

The entry point is :func:`canonicalize` which accepts a dataframe and a
target Parquet path.  The function writes the canonicalised data and returns
the dataframe for convenience.
"""

from __future__ import annotations

import hashlib
from typing import Any, Dict, Optional

import pandas as pd
import pytz

from mw.utils.ohlc_checks import validate_ohlc
from mw.utils.persistence import write_json, write_parquet

ET_TZ = pytz.timezone("America/New_York")


def _hash_df(df: pd.DataFrame) -> str:
    """Return a deterministic SHA256 hash for ``df``.

    The dataframe is hashed row wise using
    :func:`pandas.util.hash_pandas_object` which provides a stable 64bit hash
    for each row. The bytes of these hashes are then digested with SHA256 to
    obtain the final hexadecimal string.
    """

    row_hashes = pd.util.hash_pandas_object(df, index=True).values.tobytes()
    return hashlib.sha256(row_hashes).hexdigest()


def canonicalize(
    df: pd.DataFrame,
    parquet_path: str,
    contract_version: int = 1,
    *,
    source: Optional[str] = None,
    source_time_basis: Optional[str] = None,
) -> pd.DataFrame:
    """Canonicalise ``df`` and persist the result to ``parquet_path``.

    Parameters
    ----------
    df:
        Raw market data.  Must contain ``timestamp`` along with the OHLC
        columns ``open``, ``high``, ``low`` and ``close``.  Timestamps are
        interpreted as Eastern time if naive.
    parquet_path:
        Destination path for the parquet file.  A sibling ``.meta.json`` will
        be written alongside it containing metadata about the canonicalised
        dataset.
    contract_version:
        Integer identifying the schema version of the canonical contract.
    source:
        Identifier describing the origin of ``df``.
    source_time_basis:
        Timezone string describing the timezone of the raw timestamps.

    Returns
    -------
    pd.DataFrame
        The canonicalised dataframe with a UTC ``timestamp`` index and the
        columns ``is_gap``, ``minute_of_day``, ``is_session`` and
        ``quality_score``. If ``df`` is empty an empty canonicalised dataframe
        is returned and metadata files are still written with zero counts.
    """
    if source_time_basis is None:
        source_time_basis = df.attrs.get("source_time_basis", ET_TZ.zone)
    source_tz = pytz.timezone(source_time_basis)

    working = df.copy()
    ohlc_cols = ["open", "high", "low", "close"]

    def _persist_empty_result(clip_count: int, duplicates: int = 0) -> pd.DataFrame:
        empty = pd.DataFrame(
            columns=ohlc_cols
            + ["is_gap", "minute_of_day", "is_session", "quality_score"]
        ).set_index(pd.DatetimeIndex([], name="timestamp", tz="UTC"))
        metadata: Dict[str, Any] = {
            "rows": 0,
            "duplicates": duplicates,
            "gaps": 0,
            "contract_version": contract_version,
            "source": source,
            "source_time_basis": source_time_basis,
            "loaded_at": pd.Timestamp.utcnow().isoformat(),
            "clip_count": clip_count,
        }
        metadata["hash"] = _hash_df(empty)
        out_df = empty.reset_index().rename(columns={"index": "timestamp"})
        write_parquet(out_df, parquet_path)
        write_json(metadata, f"{parquet_path}.meta.json")
        return empty

    if working.empty:
        # Persist empty outputs early and return
        return _persist_empty_result(clip_count=0)

    # ------------------------------------------------------------------
    # Timestamp normalisation
    ts = pd.to_datetime(working["timestamp"])
    if ts.dt.tz is None:
        ts = ts.dt.tz_localize(source_tz)
    else:
        ts = ts.dt.tz_convert(source_tz)
    ts = ts.dt.tz_convert("UTC")
    working["timestamp"] = ts

    # Sort and remove duplicate timestamps, keeping the last observation.
    working = working.sort_values("timestamp")
    duplicate_count = int(working["timestamp"].duplicated(keep="last").sum())
    working = working.drop_duplicates(subset="timestamp", keep="last")

    # ------------------------------------------------------------------
    # OHLC integrity checks
    valid_mask, clipped = validate_ohlc(working[ohlc_cols], return_clipped=True)
    invalid_mask = ~valid_mask
    clip_count = int(invalid_mask.sum())
    if clip_count:
        working.loc[invalid_mask, ohlc_cols] = clipped.loc[invalid_mask]
        final_mask = validate_ohlc(working[ohlc_cols])
        if not bool(final_mask.all()):
            working = working[final_mask]
    else:
        clip_count = 0

    if working.empty:
        return _persist_empty_result(clip_count, duplicate_count)

    # ------------------------------------------------------------------
    # Enforce strict one-minute grid and mark gaps
    working = working.set_index("timestamp")
    full_index = pd.date_range(
        start=working.index.min(),
        end=working.index.max(),
        freq="1min",
        tz="UTC",
    )
    working = working.reindex(full_index)

    gap_mask = working[ohlc_cols].isna().all(axis=1)
    working["is_gap"] = gap_mask
    working["minute_of_day"] = working.index.hour * 60 + working.index.minute
    working["is_session"] = working.index.dayofweek < 5
    working["quality_score"] = 1.0
    gap_count = int(gap_mask.sum())

    # ------------------------------------------------------------------
    # Metadata and persistence
    metadata: Dict[str, Any] = {
        "rows": int(len(working)),
        "duplicates": duplicate_count,
        "gaps": gap_count,
        "contract_version": contract_version,
        "source": source,
        "source_time_basis": source_time_basis,
        "loaded_at": pd.Timestamp.utcnow().isoformat(),
        "clip_count": clip_count,
    }
    metadata["hash"] = _hash_df(working)

    # Write parquet and metadata JSON atomically
    out_df = working.reset_index().rename(columns={"index": "timestamp"})
    write_parquet(out_df, parquet_path)
    write_json(metadata, f"{parquet_path}.meta.json")

    return working


__all__ = ["canonicalize"]
