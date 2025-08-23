import sys
from pathlib import Path

import matplotlib
import pandas as pd
from matplotlib.colors import to_hex

sys.path.append(str(Path(__file__).resolve().parents[1]))
matplotlib.use("Agg")

from mw.viz.plots import plot_price_with_state  # noqa: E402


def test_plot_price_with_state_color_mapping():
    df = pd.DataFrame({"price": [1, 2, 3, 4, 5, 6]}, index=range(6))
    states = pd.Series(
        ["GREEN", "GREEN", "YELLOW", "YELLOW", "RED", "RED"], index=df.index
    )

    fig = plot_price_with_state(df, states)
    try:
        ax = fig.axes[0]
        assert ax.lines[0].get_ydata().tolist() == [1, 2, 3, 4, 5, 6]
        assert len(ax.patches) == 3
        assert [to_hex(p.get_facecolor()) for p in ax.patches] == [
            "#00ff00",
            "#ffff00",
            "#ff0000",
        ]
    finally:
        fig.clf()
