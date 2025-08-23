"""Configuration loader and structured parameter objects."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class PEParams:
    """Permutation entropy parameters."""

    window: int = 60
    m: int = 3
    tau: int = 1


@dataclass
class FTLEParams:
    """Finite-time Lyapunov exponent parameters."""

    window: int = 200
    m: int = 3
    tau: int = 1
    horizon: int = 10
    theiler: int = 2


@dataclass
class SmoothingParams:
    """Smoothing configuration."""

    ema_span: int = 3


@dataclass
class ScoreParams:
    """Tradability scoring and state machine parameters."""

    w1: float = 0.6
    w2: float = 0.4
    tau_y: float = 0.45
    tau_g: float = 0.65
    k_up: int = 2
    k_down: int = 1
    min_flip_spacing: int = 3


@dataclass
class NormalizationParams:
    """Normalisation parameters."""

    method: str = "minmax"


@dataclass
class MinuteLoopParams:
    """Minute loop orchestration parameters."""

    offsets: Dict[str, int] = field(
        default_factory=lambda: {
            "poll": 3,
            "compute": 5,
            "persist": 6,
            "log": 7,
            "plot": 8,
            "health": 9,
        }
    )
    critical_steps: list[str] = field(default_factory=list)
    freshness_stale_threshold: float = 180.0


@dataclass
class Params:
    """Structured configuration object for the application."""

    granularity: str = "1min"
    symbol: str = "C:XAUUSD"
    pe: PEParams = field(default_factory=PEParams)
    ftle: FTLEParams = field(default_factory=FTLEParams)
    smoothing: SmoothingParams = field(default_factory=SmoothingParams)
    score: ScoreParams = field(default_factory=ScoreParams)
    normalization: NormalizationParams = field(
        default_factory=NormalizationParams,
    )
    minute_loop: MinuteLoopParams = field(default_factory=MinuteLoopParams)


_DEF_PATH = Path(__file__).resolve().parents[2] / "params" / "params_v1.json"


def _merge(defaults: Any, data: Dict[str, Any]) -> Dict[str, Any]:
    base = asdict(defaults)
    for k, v in data.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            base[k].update(v)
        else:
            base[k] = v
    return base


def load_params(path: Optional[Path] = None) -> Params:
    """Load configuration from ``path`` into a :class:`Params` instance."""
    path = path or _DEF_PATH
    with Path(path).open() as f:
        raw = json.load(f)
    defaults = Params()
    return Params(
        granularity=raw.get("granularity", defaults.granularity),
        symbol=raw.get("symbol", defaults.symbol),
        pe=PEParams(**_merge(defaults.pe, raw.get("pe", {}))),
        ftle=FTLEParams(**_merge(defaults.ftle, raw.get("ftle", {}))),
        smoothing=SmoothingParams(
            **_merge(defaults.smoothing, raw.get("smoothing", {}))
        ),
        score=ScoreParams(**_merge(defaults.score, raw.get("score", {}))),
        normalization=NormalizationParams(
            **_merge(defaults.normalization, raw.get("normalization", {}))
        ),
        minute_loop=MinuteLoopParams(
            **_merge(defaults.minute_loop, raw.get("minute_loop", {}))
        ),
    )
