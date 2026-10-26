"""Extra plots for the Exercise 2 report (bucket_model_plotter.py is kept unchanged)."""

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
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


def budyko_curve(aridity_index):
    """Budyko (1974) curve: AET/P as a function of the aridity index PET/P."""
    phi = np.asarray(aridity_index, dtype=float)
    return np.sqrt(phi * np.tanh(1 / phi) * (1 - np.exp(-phi)))


def compare_aet_pet(
    results: pd.DataFrame,
    observed: Optional[pd.DataFrame] = None,
    snow_threshold: float = 1.0,
    title: str = "",
    figsize: tuple[int, int] = (15, 4.5),
) -> dict:
    """Compare actual ET (AET) with potential ET (PET, Hamon) of a model run.

    Panels: (a) mean monthly PET and AET, with AET split into days with and
    without snow cover, and AET/PET on a second axis; (b) annual PET, AET and,
    if observed runoff is given, the water-balance ET (P - Q_obs);
    (c) Budyko plot with the annual and mean model points (and P - Q_obs).

    Args:
        results (pd.DataFrame): Model output with columns 'PET', 'ET', 'Precip', 'Q_s', 'Q_gw', 'Snow_accum'.
        observed (pd.DataFrame): Optional, data with an observed 'Q' column [mm/d].
        snow_threshold (float): SWE [mm] above which a day counts as snow-covered.
        title (str): Figure title, if empty, no title will be shown.
        figsize (tuple): The size of the figure.

    Returns:
        dict: 'annual' (yearly totals and ratios, with a 'mean' row) and
        'monthly' (mean monthly totals and AET/PET) DataFrames.
    """
    df = pd.DataFrame(index=results.index)
    df["P"] = results["Precip"]
    df["PET"] = results["PET"]
    df["AET"] = results["ET"]
    df["AET_snow"] = results["ET"].where(results["Snow_accum"] > snow_threshold, 0.0)
    df["Q_sim"] = results["Q_s"] + results["Q_gw"]
    if observed is not None:
        df["Q_obs"] = observed["Q"]

    # Annual totals and ratios
    annual = df.resample("YS").sum(min_count=1)
    annual.index = annual.index.year
    annual.index.name = "Year"
    annual.loc["mean"] = annual.mean()
    annual["AET/PET"] = annual["AET"] / annual["PET"]
    annual["PET/P"] = annual["PET"] / annual["P"]
    annual["AET/P"] = annual["AET"] / annual["P"]
    if observed is not None:
        annual["ET_wb"] = annual["P"] - annual["Q_obs"]  # water-balance ET (storage change neglected)
        annual["ET_wb/P"] = annual["ET_wb"] / annual["P"]

    # Mean monthly totals
    n_years = df.index.year.nunique()
    monthly = df.groupby(df.index.month).sum() / n_years
    monthly.index.name = "Month"
    monthly["AET/PET"] = monthly["AET"] / monthly["PET"]

    fig, (ax_m, ax_y, ax_b) = plt.subplots(1, 3, figsize=figsize)

    # (a) monthly PET vs AET
    months = np.arange(1, 13)
    ax_m.bar(months, monthly["AET"] - monthly["AET_snow"], color="tab:green", alpha=0.7, label="AET, snow-free")
    ax_m.bar(months, monthly["AET_snow"], bottom=monthly["AET"] - monthly["AET_snow"],
             color="tab:cyan", alpha=0.7, label=f"AET, SWE > {snow_threshold:g} mm")
    ax_m.plot(months, monthly["PET"], "k-o", markersize=4, label="PET (Hamon)")
    ax_m.set_xticks(months, MONTH_LABELS)
    ax_m.set_ylabel("Mean monthly total [mm/month]")
    ax_m.set_title("(a) Mean monthly PET and AET")
    ax_r = ax_m.twinx()
    ax_r.plot(months, monthly["AET/PET"], "r--", label="AET/PET")
    ax_r.set_ylim(0, 1.05)
    ax_r.set_ylabel("AET/PET [-]", color="r")
    handles = ax_m.get_legend_handles_labels()[0] + ax_r.get_legend_handles_labels()[0]
    ax_m.legend(handles, [h.get_label() for h in handles], loc="upper left", fontsize=8)

    # (b) annual PET, AET and water-balance ET
    years = annual.index[:-1]
    ax_y.plot(years, annual.loc[years, "PET"], "k-o", markersize=4, label="PET (Hamon)")
    ax_y.plot(years, annual.loc[years, "AET"], "g-o", markersize=4, label="AET (model)")
    if observed is not None:
        ax_y.plot(years, annual.loc[years, "ET_wb"], "b-s", markersize=4, label="P - Q$_{obs}$ (water balance)")
    ax_y.set_ylabel("Annual total [mm/yr]")
    ax_y.set_title("(b) Annual ET")
    ax_y.legend(fontsize=8)

    # (c) Budyko plot
    phi = np.linspace(0.05, 2, 200)
    ax_b.plot(phi, budyko_curve(phi), "k-", label="Budyko curve")
    ax_b.plot([0, 1], [0, 1], "k:", lw=1, label="Energy limit")
    ax_b.plot([1, 2], [1, 1], "k:", lw=1)
    ax_b.scatter(annual.loc[years, "PET/P"], annual.loc[years, "AET/P"], s=15, color="tab:green", alpha=0.5)
    ax_b.scatter(annual.loc["mean", "PET/P"], annual.loc["mean", "AET/P"], s=60, color="tab:green",
                 edgecolor="k", label="Model AET (annual / mean)")
    if observed is not None:
        ax_b.scatter(annual.loc[years, "PET/P"], annual.loc[years, "ET_wb/P"], s=15, color="tab:blue", alpha=0.5)
        ax_b.scatter(annual.loc["mean", "PET/P"], annual.loc["mean", "ET_wb/P"], s=60, color="tab:blue",
                     edgecolor="k", label="P - Q$_{obs}$ (annual / mean)")
    ax_b.set_xlim(0, max(0.5, annual["PET/P"].max() * 1.5))
    ax_b.set_ylim(0, max(0.5, annual[["AET/P"] + (["ET_wb/P"] if observed is not None else [])].max().max() * 1.5))
    ax_b.set_xlabel("Aridity index PET/P [-]")
    ax_b.set_ylabel("Evaporative index ET/P [-]")
    ax_b.set_title("(c) Budyko space")
    ax_b.legend(fontsize=8, loc="lower right")

    for ax in (ax_m, ax_y, ax_b):
        ax.grid(alpha=0.3)
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    plt.show()

    return {"annual": annual, "monthly": monthly}


def plot_hru_snow_accumulation(
    hru_results: dict,
    hru_table: Optional[pd.DataFrame] = None,
    title: str = "",
    figsize: tuple[int, int] = (14, 4.5),
) -> pd.DataFrame:
    """Snow accumulation of each HRU, to show where snow piles up over the years.

    Panels: (a) daily snow storage (SWE) of each HRU; (b) snow left at the end
    of each melt season (30 September). Snow that survives every summer and keeps
    growing is a "snow tower": the HRU never melts out.

    Args:
        hru_results (dict): HRU id -> model output with a 'Snow_accum' column
            (the second output of run_hru_model).
        hru_table (pd.DataFrame): Optional, table with columns 'HRU', 'name' and
            'mean_elevation', used for the legend.
        title (str): Figure title, if empty, no title will be shown.
        figsize (tuple): The size of the figure.

    Returns:
        pd.DataFrame: Snow left on 30 September [mm] (one row per year, one column per HRU).
    """
    labels = {}
    for hru_id in hru_results:
        if hru_table is not None:
            row = hru_table.loc[hru_table["HRU"] == hru_id].iloc[0]
            labels[hru_id] = f"HRU {hru_id} {row['name']} ({row['mean_elevation']:.0f} m)"
        else:
            labels[hru_id] = f"HRU {hru_id}"

    # Snow left at the end of the melt season
    end_of_season = pd.DataFrame(
        {
            labels[hru_id]: results["Snow_accum"][
                (results.index.month == 9) & (results.index.day == 30)
            ].set_axis(results.index[(results.index.month == 9) & (results.index.day == 30)].year)
            for hru_id, results in hru_results.items()
        }
    )
    end_of_season.index.name = "Year"

    fig, (ax_ts, ax_end) = plt.subplots(1, 2, figsize=figsize, gridspec_kw={"width_ratios": [2, 1]})

    # (a) daily snow storage
    for hru_id, results in hru_results.items():
        ax_ts.plot(results.index, results["Snow_accum"], lw=0.8, label=labels[hru_id])
    ax_ts.set_ylabel("Snow accumulation (SWE) [mm]")
    ax_ts.set_title("(a) Daily snow storage per HRU")
    ax_ts.legend(fontsize=8, loc="upper left")

    # (b) snow surviving the summer
    for column in end_of_season.columns:
        ax_end.plot(end_of_season.index, end_of_season[column], "o-", markersize=3, label=column)
    ax_end.set_ylabel("Snow left on 30 September [mm]")
    ax_end.set_title("(b) Snow surviving the melt season")

    for ax in (ax_ts, ax_end):
        ax.grid(alpha=0.3)
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    plt.show()

    return end_of_season
