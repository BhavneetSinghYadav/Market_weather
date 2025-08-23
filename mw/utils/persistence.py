"""Persistence helpers for common file formats.

Currently only JSON writing is implemented. The helper ensures that data is
written atomically so that partially written files are not left behind when a
process is interrupted.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict

import pandas as pd


def write_parquet(df: pd.DataFrame, path: str) -> None:
    # TODO: implement
    raise NotImplementedError


def write_json(obj: Dict[str, Any], path: str) -> None:
    """Serialize ``obj`` as JSON and write it to ``path`` atomically.

    The object is first written to a temporary file using UTF-8 encoding and is
    then moved into place with :func:`os.replace`, providing atomic semantics on
    both POSIX and Windows platforms.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(
        "w", delete=False, dir=str(target.parent), encoding="utf-8"
    ) as tmp:
        json.dump(obj, tmp, ensure_ascii=False)
        tmp.flush()
        os.fsync(tmp.fileno())
        temp_name = tmp.name

    os.replace(temp_name, target)

