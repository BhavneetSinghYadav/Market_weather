from pathlib import Path
import sys
import json

sys.path.append(str(Path(__file__).resolve().parents[1]))

from mw.utils.persistence import write_json


def test_write_json_round_trip(tmp_path):
    data = {"key": "value", "num": 1, "nested": {"text": "café"}}
    target = tmp_path / "data.json"
    write_json(data, target.as_posix())

    with target.open(encoding="utf-8") as f:
        loaded = json.load(f)

    assert loaded == data
    # Ensure no temporary files remain
    assert {p.name for p in tmp_path.iterdir()} == {target.name}

