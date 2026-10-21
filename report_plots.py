"""Extra plots for the Exercise 2 report (bucket_model_plotter.py is kept unchanged)."""

from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd

MONTH_LABELS = ["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"]


def plot_monthly_variable_boxplots(
    results: pd.DataFrame,
    S_max: float,
    observed: Optional[pd.DataFrame] = None,
    title: str = "",
    figsize: tuple[int, int] = (12, 8),
) -> pd.DataFrame:
    """Monthly boxplots of runoff, ET, relative soil moisture and snow storage.

    One box per calendar month, showing the spread over the simulated years.
    Fluxes (Q_s + Q_gw, ET) are monthly totals [mm/month]; states (S/S_max,
    Snow_accum) are monthly means, because summing a storage is meaningless.

    Args:
        results (pd.DataFrame): Model output with columns 'Q_s', 'Q_gw', 'ET', 'S', 'Snow_accum'.
        S_max (float): Soil storage capacity [mm] used for this run.
        observed (pd.DataFrame): Optional, data with an observed 'Q' column [mm/d],
            plotted next to the simulated runoff.
        title (str): Figure title, if empty, no title will be shown.
        figsize (tuple): The size of the figure.

    Returns:
        pd.DataFrame: The monthly values (one row per month) used for the plot.
    """
    df = pd.DataFrame(index=results.index)
    df["Total_Runoff"] = results["Q_s"] + results["Q_gw"]
    df["ET"] = results["ET"]
    df["S_rel"] = results["S"] / S_max
    df["Snow_accum"] = results["Snow_accum"]
    flux_columns = ["Total_Runoff", "ET"]
    if observed is not None:
        df["Q_obs"] = observed["Q"]
        flux_columns.append("Q_obs")

    monthly_sum = df[flux_columns].resample("MS").sum(min_count=1)
    monthly_mean = df[["S_rel", "Snow_accum"]].resample("MS").mean()
    monthly = pd.concat([monthly_sum, monthly_mean], axis=1)
    monthly["month"] = monthly.index.month
    months = range(1, 13)

    def boxes(ax, column, positions, color, width):
        data = [monthly.loc[monthly["month"] == m, column].dropna() for m in months]
        ax.boxplot(data, positions=positions, widths=width, patch_artist=True,
                   boxprops=dict(facecolor=color, alpha=0.6), medianprops=dict(color="k"),
                   flierprops=dict(markersize=2))

    fig, axes = plt.subplots(2, 2, figsize=figsize, sharex=True)

    ax = axes[0, 0]
    if observed is not None:
        boxes(ax, "Total_Runoff", [m - 0.2 for m in months], "tab:red", 0.35)
        boxes(ax, "Q_obs", [m + 0.2 for m in months], "tab:blue", 0.35)
        ax.legend([plt.Rectangle((0, 0), 1, 1, fc="tab:red", alpha=0.6),
                   plt.Rectangle((0, 0), 1, 1, fc="tab:blue", alpha=0.6)],
                  ["Simulated (Q_s + Q_gw)", "Observed"], loc="upper left")
    else:
        boxes(ax, "Total_Runoff", list(months), "tab:red", 0.6)
    ax.set_ylabel("Runoff [mm/month]")

    panels = [(axes[0, 1], "ET", "tab:green", "ET [mm/month]"),
              (axes[1, 0], "S_rel", "tab:brown", "S / S_max [-] (monthly mean)"),
              (axes[1, 1], "Snow_accum", "tab:cyan", "Snow accumulation [mm SWE] (monthly mean)")]
    for ax, column, color, label in panels:
        boxes(ax, column, list(months), color, 0.6)
        ax.set_ylabel(label)
    axes[1, 0].set_ylim(0, 1.05)

    for ax in axes.flat:
        ax.set_xticks(list(months))
        ax.set_xticklabels(MONTH_LABELS)
        ax.grid(axis="y", alpha=0.3)
    for ax in axes[1]:
        ax.set_xlabel("Month")
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    plt.show()
    return monthly
