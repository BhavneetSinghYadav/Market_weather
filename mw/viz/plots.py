"""Visualization utilities.

Exports:
- plot_price_with_state(df, state_series) -> figure
- plot_diagnostics(e_hat, l_hat, score) -> figure
"""

from matplotlib import pyplot as plt
import pandas as pd


def plot_price_with_state(df, state_series):
    # TODO: implement (matplotlib; background shading by state)
    raise NotImplementedError


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

