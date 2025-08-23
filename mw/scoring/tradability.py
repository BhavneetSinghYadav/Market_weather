"""Tradability scoring and state classification utilities.

Combines forecast error and latency into a tradability score and maps the
score to ``"RED"``, ``"YELLOW"`` or ``"GREEN"`` states with hysteresis.

Exports:
- score_tradability(
    e_hat: pd.Series, l_hat: pd.Series, params: ScoreParams | None = None
  ) -> pd.Series
- state_machine(scores: pd.Series, prev_state: str = None,
                params: ScoreParams | None = None) -> pd.Series
"""

from typing import Optional

import pandas as pd

from mw.utils.params import ScoreParams


def score_tradability(
    e_hat: pd.Series, l_hat: pd.Series, params: ScoreParams | None = None
) -> pd.Series:
    """𝒯 = w1*(1 - e_hat) + w2*(1 - l_hat), clipped to [0,1]."""
    params = params or ScoreParams()
    e_hat_aligned, l_hat_aligned = e_hat.align(l_hat, join="inner")
    score = params.w1 * (1 - e_hat_aligned)
    score += params.w2 * (1 - l_hat_aligned)
    return score.clip(0, 1)


def state_machine(
    scores: pd.Series,
    prev_state: Optional[str] = None,
    params: ScoreParams | None = None,
) -> pd.Series:
    """
    Map scores to RED/YELLOW/GREEN with hysteresis:
    - require ``k_up`` consecutive >= ``tau_g`` to turn GREEN
    - require ``k_down`` consecutive <= ``tau_y`` to turn RED
    - otherwise YELLOW
    - throttle flips by ``min_flip_spacing`` observations.
    """
    params = params or ScoreParams()

    tau_y = params.tau_y
    tau_g = params.tau_g
    k_up = params.k_up
    k_down = params.k_down
    min_flip_spacing = params.min_flip_spacing

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
