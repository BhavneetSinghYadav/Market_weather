import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from mw.features.scaling import (  # noqa: E402  # isort: skip
    minmax_causal,
    tod_percentile_fit,
    tod_percentile_transform,
)


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


def test_tod_percentile_fit_groups_by_minute():
    idx = pd.to_datetime(
        [
            "2024-01-01 00:00",
            "2024-01-01 00:01",
            "2024-01-02 00:00",
            "2024-01-02 00:01",
        ]
    )
    x = pd.Series([1, 2, 3, 4], index=idx)
    model = tod_percentile_fit(x)
    assert sorted(model.keys()) == [0, 1]
    assert model[0].tolist() == [1, 3]
    assert model[1].tolist() == [2, 4]


def test_tod_percentile_transform_computes_percentiles():
    model = {0: np.array([1, 3]), 1: np.array([2, 4])}
    idx = pd.to_datetime(
        [
            "2024-01-03 00:00",
            "2024-01-03 00:01",
            "2024-01-03 00:01",
        ]
    )
    x = pd.Series([2, 3, 5], index=idx)
    result = tod_percentile_transform(x, model)
    expected = pd.Series([0.5, 0.5, 1.0], index=idx)
    assert result.tolist() == pytest.approx(expected.tolist())
    assert result.between(0, 1).all()
