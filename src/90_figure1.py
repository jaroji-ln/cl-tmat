"""Figure 1: TMAT at the 8 stations across regimes, and static-model degradation.

The degradation panel plots DRIFT_TABLE from the research brief verbatim, without BRG3,
whose drought errors are measured against a suspect segment. See
results/drift_regression.csv for whether the current pipeline reproduces it.

Output: figures/fig1_tmat_regimes_drift.{pdf,png}
"""
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D

from common import DRIFT_TABLE, GWT_CLEAN_CSV, REGIMES, ROOT, SUSPECT_SEGMENTS

OUTPUT_STEM = ROOT / "figures" / "fig1_tmat_regimes_drift"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
DROUGHT_BAND, OTHER_BAND = "#e3e1da", "#f5f4f1"
COLORS = {
    "BRG3": "#2a78d6", "BRG4": "#eb6834", "BRG5": "#1baf7a", "BRG6": "#eda100",
    "BRG17": "#e87ba4", "BRG18": "#008300", "BRG19": "#6250d6", "BRG20": "#e34948",
}
PANELS = [
    ("(a) Batanghari, Jambi", ["BRG3", "BRG4", "BRG5", "BRG6"]),
    ("(b) Kubu Raya, West Kalimantan", ["BRG17", "BRG18", "BRG19", "BRG20"]),
]


def plot_series(ax, gwt: pd.DataFrame, title: str, stations: list[str]) -> None:
    for index, (label, start, end, _) in enumerate(REGIMES):
        start, end = pd.Timestamp(start), pd.Timestamp(end) + pd.Timedelta(days=1)
        ax.axvspan(start, end, color=DROUGHT_BAND if "kemarau" in label else OTHER_BAND, lw=0, zorder=0)
        ax.axvline(start, color="white", lw=1, zorder=1)
        ax.text(start + (end - start) / 2, 1.0, f"T{index + 1}", transform=ax.get_xaxis_transform(),
                ha="center", va="bottom", color=MUTED, fontsize=7.5)
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


def plot_drift(ax) -> None:
    excluded = {station for station, *_ in SUSPECT_SEGMENTS}
    table = sorted(((s, v) for s, v in DRIFT_TABLE.items() if s not in excluded), key=lambda item: item[1][2])
    for y, (station, (pre, drought, ratio)) in enumerate(table):
        ax.plot([pre, drought], [y, y], color=COLORS[station], lw=1.5, zorder=2)
        ax.scatter(pre, y, s=34, facecolor="white", edgecolor=COLORS[station], linewidth=1.5, zorder=3)
        ax.scatter(drought, y, s=34, color=COLORS[station], zorder=3)
        ax.text(drought + 0.006, y, f"{ratio:.2f}×", va="center", color=INK, fontsize=8)
    ax.set_yticks(range(len(table)), [station for station, _ in table])
    ax.set_xlim(0, 0.18)
    ax.set_ylim(-0.6, len(table) - 0.4)
    ax.set_xlabel("Static Ridge MAE, 7-day horizon (m)")
    ax.set_title("(c) Static-model degradation, trained on 2018", loc="left", color=INK, fontsize=9)
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
    if not GWT_CLEAN_CSV.exists():
        print(f"GAGAL: {GWT_CLEAN_CSV} belum ada. Jalankan src/01_prepare_gwt.py dulu.", file=sys.stderr)
        return 1
    gwt = pd.read_csv(GWT_CLEAN_CSV, parse_dates=["date"])
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": MUTED, "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
        "legend.fontsize": 7.5, "pdf.fonttype": 42,
    })
    fig, axes = plt.subplots(3, 1, figsize=(7.09, 7.6), gridspec_kw={"height_ratios": [1.15, 1.15, 0.95]}, constrained_layout=True)
    fig.get_layout_engine().set(h_pad=0.08)
    for ax, (title, stations) in zip(axes[:2], PANELS):
        plot_series(ax, gwt, title, stations)
        ax.set_xlim(pd.Timestamp("2018-04-01"), pd.Timestamp("2020-01-01"))
        ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    plot_drift(axes[2])
    fig.savefig(OUTPUT_STEM.with_suffix(".pdf"), metadata={"CreationDate": None, "ModDate": None})
    fig.savefig(OUTPUT_STEM.with_suffix(".png"), dpi=300, metadata={"Software": None})
    print(f"Ditulis: {OUTPUT_STEM}.pdf dan .png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
