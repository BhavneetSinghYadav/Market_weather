"""
Live minute loop (stub).

Orchestrates: poll -> canonical append -> compute -> score -> state -> log -> plot.
This module should NOT depend on Colab specifics; pure Python orchestrator.
"""
from typing import Callable, Dict, Any

def run_minute_loop(poll_fn: Callable, compute_fn: Callable,
                    persist_fn: Callable, plot_fn: Callable,
                    health_fn: Callable, params: Dict[str, Any]) -> None:
    """High-level loop; call every minute (t+3s)."""
    # TODO: implement skeleton orchestration
    raise NotImplementedError
