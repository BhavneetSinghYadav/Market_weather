import sys
from pathlib import Path

import matplotlib
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))
matplotlib.use("Agg")

from mw.viz.plots import plot_price_with_state  # noqa: E402


def test_plot_price_with_state_line_and_patches():
    df = pd.DataFrame({"price": [1, 2, 3, 4]}, index=[0, 1, 2, 3])
    states = pd.Series(["A", "A", "B", "B"], index=df.index)

    fig = plot_price_with_state(df, states)
    try:
        ax = fig.axes[0]
        assert ax.lines[0].get_ydata().tolist() == [1, 2, 3, 4]
        assert len(ax.patches) == 2
        # ensure different states produce different colors
        colors = {tuple(p.get_facecolor()) for p in ax.patches}
        assert len(colors) == 2
    finally:
        fig.clf()
