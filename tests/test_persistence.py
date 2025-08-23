import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from mw.utils.persistence import write_json, write_parquet  # noqa: E402


def test_write_json_round_trip(tmp_path):
    data = {"key": "value", "num": 1, "nested": {"text": "café"}}
    target = tmp_path / "data.json"
    write_json(data, target.as_posix())

    with target.open(encoding="utf-8") as f:
        loaded = json.load(f)

    assert loaded == data
    # Ensure no temporary files remain
    assert {p.name for p in tmp_path.iterdir()} == {target.name}


def test_write_parquet_round_trip(tmp_path):
    df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    target = tmp_path / "nested" / "data.parquet"
    write_parquet(df, target.as_posix())

    loaded = pd.read_parquet(target)
    assert loaded.equals(df)
    # Ensure no temporary files remain
    assert {p.name for p in (tmp_path / "nested").iterdir()} == {target.name}
