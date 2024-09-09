"""Script with code to generate visualisalisations of spreading potentials."""

import matplotlib
import pandas as pd

from nsl_data_utils.loaders.constants import *
from matplotlib import pyplot as plt
from matplotlib.axes import Axes


matplotlib.use("inline")


def plot_ax(
    partial_result: pd.DataFrame, metric: str, y_lim: tuple[int, int], ax: Axes, top_k: int = 5
) -> None:
    top_k_pr = partial_result.nlargest(n=top_k, columns=f"{metric}_avg")
    ax.errorbar(
        x=partial_result[ACTOR],
        y=partial_result[f"{metric}_avg"],
        yerr=partial_result[f"{metric}_std"],
        fmt="o",
        color = "royalblue",
        alpha = 0.4,
    )
    ax.scatter(
        x=partial_result[ACTOR],
        y=partial_result[f"{metric}_avg"],
        color = "royalblue",
    )
    ax.errorbar(
        x=top_k_pr[ACTOR],
        y=top_k_pr[f"{metric}_avg"],
        yerr=top_k_pr[f"{metric}_std"],
        fmt="o",
        color = "orange",
    )
    ax.set_title(f"{metric}, top {top_k}: \n {list(top_k_pr[ACTOR].sort_values())}", fontsize=10)
    ax.set_ylim(y_lim)


def plot_partial_result(
    partial_result: pd.DataFrame,
    suptitle: str,
    sl_range: tuple[int, int] = None,
    ex_range: tuple[int, int] = None,
    pi_range: tuple[int, int] = None,
    pt_range: tuple[int, int] = None,
):
    fig, ax = plt.subplots(nrows=1, ncols=4)
    fig.set_size_inches(w=16, h=3)
    for idx, (metric, lim) in enumerate(
        zip(
            [EXPOSED, SIMULATION_LENGTH, PEAK_INFECTED, PEAK_ITERATION],
            [ex_range, sl_range, pi_range, pt_range]
        )
    ):
        plot_ax(partial_result, metric, lim, ax[idx])
    fig.subplots_adjust(top=0.9)
    fig.suptitle(suptitle, fontsize=10)
    fig.tight_layout()
    return fig


def plot_title_page(title: str):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(16, 2))
    ax.set_visible(False)
    fig.suptitle(title, x=0.5, y=.5, fontsize = 15)
    return fig
