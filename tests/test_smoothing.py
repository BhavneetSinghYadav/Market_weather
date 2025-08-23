import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from mw.features.smoothing import ema  # noqa: E402  # isort: skip


def manual_ema(series: pd.Series, span: int) -> pd.Series:
    alpha = 2 / (span + 1)
    vals = []
    for x in series:
        if not vals:
            vals.append(x)
        else:
            vals.append(alpha * x + (1 - alpha) * vals[-1])
    return pd.Series(vals, index=series.index)


def test_ema_matches_manual_calculation_and_preserves_index():
    idx = pd.to_datetime(
        ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"]
    )
    x = pd.Series([1.0, 2.0, 3.0, 4.0], index=idx)
    span = 3

    result = ema(x, span=span)
    expected = manual_ema(x, span)

    assert result.index.equals(x.index)
    assert result.tolist() == pytest.approx(expected.tolist())

