"""Tradability scoring and state classification utilities.

Combines forecast error and latency into a tradability score and maps the
score to ``"RED"``, ``"YELLOW"`` or ``"GREEN"`` states with hysteresis.

Exports:
- score_tradability(e_hat: pd.Series, l_hat: pd.Series, weights: dict = None)
  -> pd.Series
- state_machine(scores: pd.Series, prev_state: str = None,
                thresholds: dict = None, hysteresis: dict = None,
                timestamps: pd.Series | None = None) -> pd.Series
"""

from typing import Optional

import pandas as pd

DEFAULT_WEIGHTS = {"w1": 0.6, "w2": 0.4}
DEFAULT_THRESH = {"tau_y": 0.45, "tau_g": 0.65}
DEFAULT_HYST = {"k_up": 2, "k_down": 1, "min_flip_spacing": 3}


def score_tradability(
    e_hat: pd.Series, l_hat: pd.Series, weights: dict = None
) -> pd.Series:
    """𝒯 = w1*(1 - e_hat) + w2*(1 - l_hat), clipped to [0,1]."""
    if weights is None:
        weights = DEFAULT_WEIGHTS
    else:
        # fall back to defaults if keys missing
        weights = {
            "w1": weights.get("w1", DEFAULT_WEIGHTS["w1"]),
            "w2": weights.get("w2", DEFAULT_WEIGHTS["w2"]),
        }

    e_hat_aligned, l_hat_aligned = e_hat.align(l_hat, join="inner")

    score = weights["w1"] * (1 - e_hat_aligned)
    score += weights["w2"] * (1 - l_hat_aligned)
    return score.clip(0, 1)


def state_machine(
    scores: pd.Series,
    prev_state: Optional[str] = None,
    thresholds: dict = None,
    hysteresis: dict = None,
    timestamps: Optional[pd.Series] = None,
) -> pd.Series:
    """
    Map scores to RED/YELLOW/GREEN with hysteresis:
    - require ``k_up`` consecutive >= ``tau_g`` to turn GREEN
    - require ``k_down`` consecutive <= ``tau_y`` to turn RED
    - otherwise YELLOW
    - throttle flips by ``min_flip_spacing`` observations.

    ``timestamps`` is accepted for future compatibility but spacing is
    currently measured in number of observations.
    """
    thresholds = thresholds or DEFAULT_THRESH
    thresholds = {
        "tau_y": thresholds.get("tau_y", DEFAULT_THRESH["tau_y"]),
        "tau_g": thresholds.get("tau_g", DEFAULT_THRESH["tau_g"]),
    }

    hysteresis = hysteresis or DEFAULT_HYST
    hysteresis = {
        "k_up": hysteresis.get("k_up", DEFAULT_HYST["k_up"]),
        "k_down": hysteresis.get("k_down", DEFAULT_HYST["k_down"]),
        "min_flip_spacing": hysteresis.get(
            "min_flip_spacing", DEFAULT_HYST["min_flip_spacing"]
        ),
    }

    tau_y = thresholds["tau_y"]
    tau_g = thresholds["tau_g"]
    k_up = hysteresis["k_up"]
    k_down = hysteresis["k_down"]
    min_flip_spacing = hysteresis["min_flip_spacing"]

    current_state = prev_state or "YELLOW"
    last_flip_idx = -min_flip_spacing
    up_count = 0
    down_count = 0

    states = []
    for i, score in enumerate(scores):
        if score >= tau_g:
            up_count += 1
            down_count = 0
        elif score <= tau_y:
            down_count += 1
            up_count = 0
        else:
            up_count = 0
            down_count = 0

        desired_state = "YELLOW"
        if up_count >= k_up:
            desired_state = "GREEN"
        elif down_count >= k_down:
            desired_state = "RED"

        spacing_ok = i - last_flip_idx >= min_flip_spacing
        if desired_state != current_state and spacing_ok:
            current_state = desired_state
            last_flip_idx = i
            up_count = 0
            down_count = 0

        states.append(current_state)

    return pd.Series(states, index=scores.index)
