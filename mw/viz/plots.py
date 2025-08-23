"""Visualization utilities.

Exports:
- plot_price_with_state(df, state_series) -> figure
- plot_diagnostics(e_hat, l_hat, score) -> figure
"""

import pandas as pd
from matplotlib import pyplot as plt


def plot_price_with_state(df: pd.DataFrame, state_series: pd.Series):
    """Plot price with background colored by state.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing a ``price`` column or single column of prices.
    state_series : pd.Series
        Series of state labels aligned to ``df``'s index. Consecutive regions
        with the same label are shaded with the same color.

    Returns
    -------
    matplotlib.figure.Figure
        Figure containing a single axes with the price line and state shading.
    """

    # Ensure alignment and extract the price series
    states = state_series.reindex(df.index)
    price = df["price"] if "price" in df.columns else df.iloc[:, 0]

    fig, ax = plt.subplots()
    ax.plot(df.index, price)
    ax.set_ylabel("price")

    # Map each distinct state to a color from the matplotlib color cycle
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    state_colors = {
        state: color_cycle[i % len(color_cycle)]
        for i, state in enumerate(pd.unique(states.dropna()))
    }

    # Group consecutive identical states and shade the corresponding regions
    groups = (states != states.shift()).cumsum()
    for _, g in states.groupby(groups):
        state = g.iloc[0]
        if pd.isna(state):
            continue
        start = g.index[0]
        end = g.index[-1]
        # Extend the shading to the next index to avoid gaps
        next_pos = df.index.get_indexer([end])[0] + 1
        if next_pos < len(df.index):
            end = df.index[next_pos]
        ax.axvspan(start, end, color=state_colors[state], alpha=0.3)

    return fig


def plot_diagnostics(e_hat: pd.Series, l_hat: pd.Series, score: pd.Series):
    """Plot diagnostic series on three vertically aligned subplots.

    Parameters
    ----------
    e_hat : pd.Series
        Error series to plot on the first subplot.
    l_hat : pd.Series
        Leakage series to plot on the second subplot.
    score : pd.Series
        Score series to plot on the third subplot.

    Returns
    -------
    matplotlib.figure.Figure
        Figure containing three aligned subplots with a shared x-axis.
    """

    fig, axes = plt.subplots(3, 1, sharex=True)

    axes[0].plot(e_hat)
    axes[1].plot(l_hat)
    axes[2].plot(score)

    axes[0].set_ylabel("e_hat")
    axes[1].set_ylabel("l_hat")
    axes[2].set_ylabel("score")
    axes[2].set_xlabel("index")

    return fig
