"""Persistence helpers for common file formats.

Provides atomic write helpers for Parquet and JSON files to avoid partially
written data when processes are interrupted.

Exports:
- write_parquet(df: pd.DataFrame, path: str) -> None
- write_json(obj: Dict[str, Any], path: str) -> None
"""

from __future__ import annotations

import csv
import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Sequence

import pandas as pd


def write_parquet(df: pd.DataFrame, path: str) -> None:
    """Write ``df`` to ``path`` as a parquet file atomically.

    The dataframe is first written to a temporary file and then moved into
    place with :func:`os.replace` so that partially written files are not left
    behind. Any :class:`OSError` raised during the process is logged and
    re-raised.
    """

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)

    tmp = tempfile.NamedTemporaryFile(
        delete=False, dir=str(target.parent), suffix=".parquet"
    )
    tmp.close()
    temp_name = tmp.name

    try:
        df.to_parquet(temp_name, index=False)
        os.replace(temp_name, target)
    except OSError as err:
        logging.error("Failed to write parquet file %s: %s", path, err)
        try:
            os.remove(temp_name)
        except OSError:
            pass
        raise


def write_json(obj: Dict[str, Any], path: str) -> None:
    """Serialize ``obj`` as JSON and write it to ``path`` atomically.

    The object is first written to a temporary file using UTF-8 encoding and
    is then moved into place with :func:`os.replace`, providing atomic
    semantics on both POSIX and Windows platforms.
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

    try:
        os.replace(temp_name, target)
    except OSError as err:
        logging.error("Failed to write JSON file %s: %s", path, err)
        try:
            os.remove(temp_name)
        except OSError:
            pass
        raise


def append_csv(row: Dict[str, Any], path: str | Path, fieldnames: Sequence[str]) -> None:
    """Append ``row`` to ``path`` ensuring durability.

    The CSV file grows incrementally with each call. Data is flushed and
    fsynced so that interruptions do not leave partially written rows.
    ``fieldnames`` define the column order and header written when the file is
    first created.
    """

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    exists = target.exists()
    with target.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow(row)
        f.flush()
        os.fsync(f.fileno())
