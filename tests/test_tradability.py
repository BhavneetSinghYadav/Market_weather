# isort: skip_file
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))
from mw.scoring.tradability import (  # noqa: E402
    score_tradability,
    state_machine,
)


def test_score_tradability_default_weights_align_and_clip():
    e_hat = pd.Series([0.1, 0.2, 0.3], index=[0, 1, 2])
    l_hat = pd.Series([0.4, 0.5, 0.6], index=[1, 2, 3])
    expected = pd.Series([0.72, 0.62], index=[1, 2])

    result = score_tradability(e_hat, l_hat)

    pd.testing.assert_series_equal(result, expected)


def test_score_tradability_custom_weights_and_clip():
    e_hat = pd.Series([-0.5, 2.0])
    l_hat = pd.Series([-0.5, 2.0])
    weights = {"w1": 0.5, "w2": 0.5}
    expected = pd.Series([1.0, 0.0])

    result = score_tradability(e_hat, l_hat, weights)

    pd.testing.assert_series_equal(result, expected)


def test_state_machine_hysteresis_and_spacing():
    scores = pd.Series([0.5, 0.7, 0.7, 0.3, 0.3, 0.3, 0.7, 0.7, 0.7])
    expected = pd.Series(
        [
            "YELLOW",
            "YELLOW",
            "GREEN",
            "GREEN",
            "GREEN",
            "RED",
            "RED",
            "RED",
            "GREEN",
        ],
        index=scores.index,
    )

    result = state_machine(scores)

    pd.testing.assert_series_equal(result, expected)


def test_state_machine_rejects_timestamps_argument():
    scores = pd.Series([0.5, 0.7])
    ts = pd.Series([1, 2])
    with pytest.raises(TypeError):
        state_machine(scores, timestamps=ts)
