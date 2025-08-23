"""
Tradability score & state machine (stubs).

Exports:
- score_tradability(e_hat: pd.Series, l_hat: pd.Series, weights: dict) -> pd.Series
- state_machine(scores: pd.Series, prev_state: str, thresholds: dict,
                hysteresis: dict, timestamps: pd.Series=None) -> pd.Series
States: "RED", "YELLOW", "GREEN".
"""
from typing import Optional
import pandas as pd

DEFAULT_WEIGHTS = {"w1": 0.6, "w2": 0.4}
DEFAULT_THRESH = {"tau_y": 0.45, "tau_g": 0.65}
DEFAULT_HYST = {"k_up": 2, "k_down": 1, "min_flip_spacing": 3}

def score_tradability(e_hat: pd.Series, l_hat: pd.Series, weights: dict = None) -> pd.Series:
    """𝒯 = w1*(1 - e_hat) + w2*(1 - l_hat), clipped to [0,1]."""
    if weights is None:
        weights = DEFAULT_WEIGHTS
    else:
        # fall back to defaults if keys missing
        weights = {"w1": weights.get("w1", DEFAULT_WEIGHTS["w1"]),
                   "w2": weights.get("w2", DEFAULT_WEIGHTS["w2"])}

    e_hat_aligned, l_hat_aligned = e_hat.align(l_hat, join="inner")

    score = weights["w1"] * (1 - e_hat_aligned) + weights["w2"] * (1 - l_hat_aligned)
    return score.clip(0, 1)

def state_machine(scores: pd.Series, prev_state: Optional[str] = None,
                  thresholds: dict = None, hysteresis: dict = None,
                  timestamps: Optional[pd.Series] = None) -> pd.Series:
    """
    Map scores to RED/YELLOW/GREEN with hysteresis:
    - require k_up consecutive >= tau_g to turn GREEN
    - require k_down consecutive <= tau_y to turn RED
    - else YELLOW; throttle flips by min_flip_spacing.
    """
    # TODO: implement
    raise NotImplementedError
