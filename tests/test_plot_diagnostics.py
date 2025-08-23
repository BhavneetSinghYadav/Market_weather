import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))
from mw.viz.plots import plot_diagnostics


def test_plot_diagnostics_subplots_and_labels():
    e_hat = pd.Series([0.1, 0.2, 0.3])
    l_hat = pd.Series([0.2, 0.3, 0.4])
    score = pd.Series([0.3, 0.4, 0.5])

    fig = plot_diagnostics(e_hat, l_hat, score)

    try:
        assert len(fig.axes) == 3
        assert fig.axes[0].get_ylabel() == "e_hat"
        assert fig.axes[1].get_ylabel() == "l_hat"
        assert fig.axes[2].get_ylabel() == "score"
        assert fig.axes[2].get_xlabel() == "index"
        assert fig.axes[0].get_shared_x_axes().joined(fig.axes[0], fig.axes[1])
        assert fig.axes[0].get_shared_x_axes().joined(fig.axes[0], fig.axes[2])
    finally:
        fig.clf()
