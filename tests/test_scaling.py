import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from mw.features.scaling import minmax_causal  # noqa: E402


def test_minmax_causal_scales_to_unit_interval():
    x = pd.Series([0, 5, 3, 4])
    result = minmax_causal(x, win=3)
    expected = pd.Series([0, 1, 0.6, 0.5])
    assert result.tolist() == pytest.approx(expected.tolist())
    assert result.between(0, 1).all()


def test_minmax_causal_constant_series():
    x = pd.Series([2, 2, 2])
    result = minmax_causal(x, win=2)
    assert result.eq(0).all()
