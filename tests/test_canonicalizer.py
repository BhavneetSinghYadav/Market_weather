import json
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from mw.io.canonicalizer import canonicalize  # noqa: E402


def make_sample_df():
    return pd.DataFrame(
        {
            "timestamp": [
                "2024-01-01 09:30",
                "2024-01-01 09:31",
                "2024-01-01 09:31",  # duplicate
                "2024-01-01 09:33",  # gap at 09:32
            ],
            "open": [10.0, 10.5, 10.6, 11.0],
            "high": [11.0, 11.5, 11.6, 12.0],
            "low": [9.0, 10.0, 10.1, 10.5],
            "close": [10.5, 11.0, 11.1, 11.5],
        }
    )


def test_canonicalize_end_to_end(tmp_path):
    raw = make_sample_df()
    out = tmp_path / "data.parquet"
    raw.attrs["source_time_basis"] = "America/New_York"
    canonicalize(
        raw,
        out.as_posix(),
        source="unit_test",
    )

    result = pd.read_parquet(out)
    result["timestamp"] = pd.to_datetime(result["timestamp"], utc=True)

    expected_index = pd.date_range(
        "2024-01-01 14:30",
        periods=4,
        freq="1min",
        tz="UTC",
    )
    expected = pd.DataFrame(
        {
            "timestamp": expected_index,
            "open": [10.0, 10.6, float("nan"), 11.0],
            "high": [11.0, 11.6, float("nan"), 12.0],
            "low": [9.0, 10.1, float("nan"), 10.5],
            "close": [10.5, 11.1, float("nan"), 11.5],
            "is_gap": [False, False, True, False],
            "minute_of_day": [870, 871, 872, 873],
            "is_session": [True, True, True, True],
            "quality_score": [1.0, 1.0, 1.0, 1.0],
        }
    )
    expected["minute_of_day"] = expected["minute_of_day"].astype(
        result["minute_of_day"].dtype
    )

    pd.testing.assert_frame_equal(result, expected)

    meta_path = Path(str(out) + ".meta.json")
    with meta_path.open() as f:
        meta = json.load(f)

    assert meta["rows"] == 4
    assert meta["duplicates"] == 1
    assert meta["gaps"] == 1
    assert meta["contract_version"] == 1
    assert isinstance(meta["hash"], str) and len(meta["hash"]) == 64
    assert meta["source"] == "unit_test"
    assert meta["source_time_basis"] == "America/New_York"
    assert isinstance(meta["loaded_at"], str)
    assert meta["clip_count"] == 0


def test_respects_source_time_basis_attr(tmp_path):
    df = pd.DataFrame(
        {
            "timestamp": ["2024-01-01 14:30"],
            "open": [1.0],
            "high": [1.0],
            "low": [1.0],
            "close": [1.0],
        }
    )
    df.attrs["source_time_basis"] = "UTC"
    out = tmp_path / "utc.parquet"
    result = canonicalize(df, out.as_posix())

    assert result.index[0] == pd.Timestamp("2024-01-01 14:30", tz="UTC")


def test_canonicalize_invalid_ohlc(tmp_path):
    bad = pd.DataFrame(
        {
            "timestamp": ["2024-01-01 09:30"],
            "open": [1.0],
            "high": [1.0],
            "low": [2.0],  # invalid low > high
            "close": [1.0],
        }
    )
    out = tmp_path / "bad.parquet"
    result = canonicalize(bad, out.as_posix())

    assert not result.empty

    result_df = pd.read_parquet(out)
    result_df["timestamp"] = pd.to_datetime(result_df["timestamp"], utc=True)
    expected = pd.DataFrame(
        {
            "timestamp": [pd.Timestamp("2024-01-01 14:30", tz="UTC")],
            "open": [1.0],
            "high": [2.0],
            "low": [1.0],
            "close": [1.0],
            "is_gap": [False],
            "minute_of_day": [870],
            "is_session": [True],
            "quality_score": [1.0],
        }
    )
    expected["minute_of_day"] = expected["minute_of_day"].astype(
        result_df["minute_of_day"].dtype
    )
    pd.testing.assert_frame_equal(result_df, expected)

    with Path(str(out) + ".meta.json").open() as f:
        meta = json.load(f)

    assert meta["clip_count"] == 1


def test_canonicalize_drop_unfixable_row(tmp_path):
    bad = pd.DataFrame(
        {
            "timestamp": ["2024-01-01 09:30"],
            "open": [1.0],
            "high": [float("nan")],
            "low": [float("nan")],
            "close": [1.0],
        }
    )
    out = tmp_path / "drop.parquet"
    result = canonicalize(bad, out.as_posix())

    assert result.empty

    with Path(str(out) + ".meta.json").open() as f:
        meta = json.load(f)

    assert meta["rows"] == 0
    assert meta["clip_count"] == 1


def test_canonicalize_empty(tmp_path):
    raw = pd.DataFrame(columns=["timestamp", "open", "high", "low", "close"])
    out = tmp_path / "empty.parquet"
    result = canonicalize(raw, out.as_posix())

    assert result.empty
    assert list(result.columns) == [
        "open",
        "high",
        "low",
        "close",
        "is_gap",
        "minute_of_day",
        "is_session",
        "quality_score",
    ]
    assert result.index.name == "timestamp"

    meta_path = Path(str(out) + ".meta.json")
    with meta_path.open() as f:
        meta = json.load(f)

    assert meta["rows"] == 0
    assert meta["duplicates"] == 0
    assert meta["gaps"] == 0
    assert "source" in meta
    assert "source_time_basis" in meta
    assert isinstance(meta["loaded_at"], str)
    assert meta["clip_count"] == 0


def test_is_session_weekend(tmp_path):
    raw = pd.DataFrame(
        {
            "timestamp": ["2024-01-06 09:30"],  # Saturday
            "open": [1.0],
            "high": [1.1],
            "low": [0.9],
            "close": [1.05],
        }
    )
    out = tmp_path / "weekend.parquet"
    canonicalize(raw, out.as_posix())

    result = pd.read_parquet(out)
    result["timestamp"] = pd.to_datetime(result["timestamp"], utc=True)

    expected = pd.DataFrame(
        {
            "timestamp": [pd.Timestamp("2024-01-06 14:30", tz="UTC")],
            "open": [1.0],
            "high": [1.1],
            "low": [0.9],
            "close": [1.05],
            "is_gap": [False],
            "minute_of_day": [870],
            "is_session": [False],
            "quality_score": [1.0],
        }
    )
    expected["minute_of_day"] = expected["minute_of_day"].astype(
        result["minute_of_day"].dtype
    )

    pd.testing.assert_frame_equal(result, expected)
