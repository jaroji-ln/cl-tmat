"""Figure 1: TMAT at the 8 stations across regimes, rainfall, and static-model degradation.

The degradation panel reads the panel Ridge, suspect_excluded rows of
results/drift_regression.csv.

Output: figures/fig1_tmat_regimes_drift.{pdf,png}
"""
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D

from common import CHIRPS_CSV, DRIFT_CSV, GWT_CLEAN_CSV, PEATLANDS, REGIMES, ROOT

OUTPUT_STEM = ROOT / "figures" / "fig1_tmat_regimes_drift"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
DROUGHT_BAND, OTHER_BAND = "#e3e1da", "#f5f4f1"
COLORS = {
    "BRG3": "#2a78d6", "BRG4": "#eb6834", "BRG5": "#1baf7a", "BRG6": "#eda100",
    "BRG17": "#e87ba4", "BRG18": "#008300", "BRG19": "#6250d6", "BRG20": "#e34948",
}
PANELS = [
    ("(a) Groundwater table: Batanghari, Jambi", ["BRG3", "BRG4", "BRG5", "BRG6"]),
    ("Groundwater table: Kubu Raya, West Kalimantan", ["BRG17", "BRG18", "BRG19", "BRG20"]),
]
STATION_REGION = {"BRG3": "Batanghari", "BRG4": "Batanghari", "BRG5": "Batanghari", "BRG6": "Batanghari"}
RAIN_COLORS = dict(zip(PEATLANDS, [INK, "#a3a29c"]))


def shade_regimes(ax) -> None:
    for index, (label, start, end, _) in enumerate(REGIMES):
        start, end = pd.Timestamp(start), pd.Timestamp(end) + pd.Timedelta(days=1)
        ax.axvspan(start, end, color=DROUGHT_BAND if "kemarau" in label else OTHER_BAND, lw=0, zorder=0)
        ax.axvline(start, color="white", lw=1, zorder=1)
        ax.text(start + (end - start) / 2, 1.0, f"T{index + 1}", transform=ax.get_xaxis_transform(),
                ha="center", va="bottom", color=MUTED, fontsize=7.5)


def plot_series(ax, gwt: pd.DataFrame, title: str, stations: list[str]) -> None:
    shade_regimes(ax)
    ax.axhline(0, color=MUTED, lw=0.6, zorder=1)
    for station in stations:
        rows = gwt[gwt["station"] == station]
        ax.plot(rows["date"], rows["tmat_m"].mask(rows["is_suspect"]), color=COLORS[station], lw=1.1, label=station, zorder=2)
        suspect = rows[rows["is_suspect"]]
        if len(suspect):
            ax.plot(suspect["date"], suspect["tmat_m"], color=MUTED, lw=1.1, ls=(0, (1, 1.5)), zorder=2)
            ax.annotate(
                f"{station} suspect segment,\nexcluded", (suspect["date"].iloc[-1], suspect["tmat_m"].iloc[-1]),
                xytext=(5, -2), textcoords="offset points", va="top", color=MUTED, fontsize=7,
            )
    ax.set_ylabel("Groundwater table (m)")
    ax.set_title(title, loc="left", color=INK, fontsize=9, pad=13)
    ax.legend(loc="lower left", ncol=4, frameon=False, handlelength=1.4, columnspacing=1.0)
    ax.grid(axis="y", color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    ax.set_ylim(-1.95, 0.3)  # shared scale, with room for the legend below the data


def plot_rain(ax, chirps: pd.DataFrame) -> None:
    shade_regimes(ax)
    region = chirps["station"].map(STATION_REGION).fillna("Kubu Raya")
    monthly = chirps.assign(region=region, month=chirps["date"].dt.to_period("M")).groupby(["month", "station", "region"])["precip_mm"].sum()
    monthly = monthly.groupby(["month", "region"]).mean().unstack()
    monthly = monthly["2018-04":"2019-12"]
    for offset, peatland in zip((-5.5, 5.5), PEATLANDS):
        centers = monthly.index.to_timestamp() + pd.Timedelta(days=14)
        ax.bar(centers + pd.Timedelta(days=offset), monthly[peatland], width=9.5, color=RAIN_COLORS[peatland], label=peatland, zorder=2)
    ax.set_ylabel("Monthly rainfall (mm)")
    ax.set_title("(b) CHIRPS rainfall, station mean per peatland", loc="left", color=INK, fontsize=9, pad=13)
    ax.legend(loc="upper left", ncol=2, frameon=False, handlelength=1.0)
    ax.set_ylim(0, 620)
    ax.grid(axis="y", color=GRID, lw=0.5)
    ax.set_axisbelow(True)


def plot_drift(ax, drift: pd.DataFrame) -> None:
    rows = drift[(drift["data"] == "suspect_excluded") & (drift["mode"] == "panel")].sort_values("ratio")
    for y, row in enumerate(rows.itertuples()):
        ax.plot([row.mae_pre_m, row.mae_drought_m], [y, y], color=COLORS[row.station], lw=1.5, zorder=2)
        ax.scatter(row.mae_pre_m, y, s=34, facecolor="white", edgecolor=COLORS[row.station], linewidth=1.5, zorder=3)
        ax.scatter(row.mae_drought_m, y, s=34, color=COLORS[row.station], zorder=3)
        ax.text(max(row.mae_pre_m, row.mae_drought_m) + 0.004, y, f"{row.ratio:.2f}×  (n = {row.n_drought})", va="center", color=INK, fontsize=8)
    ax.set_yticks(range(len(rows)), rows["station"])
    ax.set_xlim(0, 0.24)
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.set_xlabel("Static panel Ridge MAE, 7-day horizon (m)")
    ax.set_title("(c) Static-model degradation, trained on 2018, suspect segment excluded", loc="left", color=INK, fontsize=9)
    ax.grid(axis="x", color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    ax.legend(
        handles=[
            Line2D([], [], ls="", marker="o", mfc="white", mec=MUTED, mew=1.5, ms=6, label="Pre-drought 2019"),
            Line2D([], [], ls="", marker="o", color=MUTED, ms=6, label="2019 El Niño drought (T4)"),
        ],
        loc="lower right", frameon=False,
    )


def main() -> int:
    missing = [p for p in (GWT_CLEAN_CSV, DRIFT_CSV, CHIRPS_CSV) if not p.exists()]
    if missing:
        print(f"GAGAL: belum ada: {', '.join(map(str, missing))}", file=sys.stderr)
        return 1
    gwt = pd.read_csv(GWT_CLEAN_CSV, parse_dates=["date"])
    chirps = pd.read_csv(CHIRPS_CSV, parse_dates=["date"])
    drift = pd.read_csv(DRIFT_CSV)
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": MUTED, "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
        "legend.fontsize": 7.5, "pdf.fonttype": 42,
    })
    fig, axes = plt.subplots(4, 1, figsize=(7.09, 9.2), gridspec_kw={"height_ratios": [1.15, 1.15, 0.7, 0.95]}, constrained_layout=True)
    fig.get_layout_engine().set(h_pad=0.08)
    for ax, (title, stations) in zip(axes[:2], PANELS):
        plot_series(ax, gwt, title, stations)
    plot_rain(axes[2], chirps)
    for ax in axes[:3]:
        ax.set_xlim(pd.Timestamp("2018-04-01"), pd.Timestamp("2020-01-01"))
        ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    plot_drift(axes[3], drift)
    fig.savefig(OUTPUT_STEM.with_suffix(".pdf"), metadata={"CreationDate": None, "ModDate": None})
    fig.savefig(OUTPUT_STEM.with_suffix(".png"), dpi=300, metadata={"Software": None})
    print(f"Ditulis: {OUTPUT_STEM}.pdf dan .png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
