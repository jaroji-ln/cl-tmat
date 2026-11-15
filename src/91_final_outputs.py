"""Final figures and tables (protocol v6, 2.6 item 3), read only from results/. Nothing is recomputed.

Figures (figures/*.pdf, *.png):
  fig2_skill_horizon      online skill against persistence vs horizon: pooled T3-T5, T3, T4, T5 (locked batch
                          + damped trend, exploratory)
  fig3_cl_t4              CL methods vs B5 LSTM: online skill T4 and pooled T3-T5
  fig4_diagnostics_t4     online skill T4 of the best candidate per pipeline (locked, D1-D4; exploratory)
  fig5_crossing_f1        crossing F1 (D5 definition, T3-T5) vs horizon, with the damped trend
Tables (results/tables/*.csv and *.md): table2_criteria_locked, table3_diagnostics, table4_cl_vs_b5,
table5_damped_trend
"""
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from common import ROOT  # noqa: E402

RESULTS = ROOT / "results"
TABLES = RESULTS / "tables"
FIGURES = ROOT / "figures"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
HORIZONS = [3, 7, 14, 21, 30]
# Color follows the entity in every figure (validated: scripts/validate_palette.js, light surface).
CANDIDATES = {
    "B5_retrain30d_lstm": ("B5 LSTM", "#2a78d6", "o"),
    "B5_retrain30d_ridge": ("B5 Ridge", "#eb6834", "s"),
    "B5_retrain30d_lightgbm": ("B5 LightGBM", "#1baf7a", "^"),
    "B2_static_lightgbm": ("B2 LightGBM", "#eda100", "D"),
    "B2_static_ridge": ("B2 Ridge", "#e87ba4", "v"),
}
CL = {
    "B5_retrain30d_lstm": ("B5 LSTM (retrain)", "#2a78d6", "o"),
    "CL_finetune30d_lstm": ("Fine-tune", "#008300", "s"),
    "CL_replay30d_lstm": ("Replay", "#6250d6", "^"),
    "CL_ewc30d_lstm": ("EWC", "#e34948", "D"),
}
DAMPED = ("Damped trend φ=0.9 (exploratory)", INK, "x")
FLOOR = -1.0  # skill axis floor; values below are drawn at the floor with an arrow and their value


def style(ax, title: str, ylabel: str | None = None) -> None:
    ax.set_title(title, loc="left", color=INK, fontsize=8.5, pad=6)
    ax.set_xticks(HORIZONS)
    ax.set_xlabel("Horizon (days)", color=MUTED, fontsize=7.5)
    if ylabel:
        ax.set_ylabel(ylabel, color=MUTED, fontsize=7.5)
    ax.grid(axis="y", color=GRID, lw=0.5)
    ax.tick_params(colors=MUTED, labelsize=7, length=2)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)


def line(ax, x, y, label, color, marker, floor=None, ls="-"):
    y = pd.Series(list(y), dtype=float)
    shown = y if floor is None else y.clip(lower=floor)
    ax.plot(x, shown, color=color, lw=2 if ls == "-" else 1.4, ls=ls, marker=marker, ms=4.5, label=label, zorder=3,
            markeredgecolor="white" if marker != "x" else color, markeredgewidth=0.8)
    if floor is not None:  # off-scale points: a down-pointing mark at the floor; exact values are in the tables
        below = y < floor
        if below.any():
            ax.scatter(pd.Series(x)[below.to_numpy()], [floor - 0.06] * int(below.sum()), marker="v", s=14, color=color, zorder=4, clip_on=False)


def zero(ax):
    ax.axhline(0, color=MUTED, lw=0.9, zorder=2, label="Persistence (skill = 0)")


def percent(ax):
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v * 100:.0f}%"))


def save(fig, stem: str) -> None:
    FIGURES.mkdir(exist_ok=True)
    fig.savefig(FIGURES / f"{stem}.pdf", metadata={"CreationDate": None, "ModDate": None})
    fig.savefig(FIGURES / f"{stem}.png", dpi=300, metadata={"Software": None})
    plt.close(fig)


def pivot(frame, value, methods):
    return frame.pivot(index="method", columns="horizon_days", values=value).reindex(index=list(methods), columns=HORIZONS)


def fig2(summary, criteria, damped):
    pooled = criteria.pivot(index="candidate", columns="horizon_days", values="pooled_online_skill")
    primary = damped[damped["role"] == "primary"].set_index("horizon_days")
    panels = [("(a) Pooled online T3–T5", pooled, "pooled_online_skill")]
    for task in ("T3", "T4", "T5"):
        panels.append((f"({'bcd'[len(panels) - 1]}) Online {task}", pivot(summary, f"skill_online_{task}_mean", CANDIDATES), f"skill_{task}"))
    fig, axes = plt.subplots(1, 4, figsize=(10, 3.1), sharey=True, constrained_layout=True)
    for ax, (title, table, damped_column) in zip(axes, panels):
        zero(ax)
        for method, (label, color, marker) in CANDIDATES.items():
            line(ax, HORIZONS, table.loc[method, HORIZONS], label, color, marker, floor=FLOOR)
        line(ax, HORIZONS, primary.loc[HORIZONS, damped_column], DAMPED[0], DAMPED[1], DAMPED[2], floor=FLOOR, ls="--")
        style(ax, title, "Skill vs persistence" if ax is axes[0] else None)
        percent(ax)
    axes[0].set_ylim(FLOOR - 0.1, 0.55)
    handles, labels = axes[0].get_legend_handles_labels()
    handles.append(matplotlib.lines.Line2D([], [], color=MUTED, marker="v", ls="none", ms=4))
    labels.append("Below −100% (values in results/horizon_summary.csv)")
    fig.legend(handles, labels, loc="outside lower center", ncol=4, frameon=False, fontsize=7)
    save(fig, "fig2_skill_horizon")


def fig3(cl):
    fig, axes = plt.subplots(1, 2, figsize=(7, 3.1), sharey=True, constrained_layout=True)
    for ax, (title, column) in zip(axes, [("(a) Online T4 (El Niño drought)", "skill_online_T4_mean"), ("(b) Pooled online T3–T5", "pooled")]):
        zero(ax)
        table = pivot(cl, column, CL) if column != "pooled" else cl.attrs["pooled"].reindex(index=list(CL), columns=HORIZONS)
        for method, (label, color, marker) in CL.items():
            line(ax, HORIZONS, table.loc[method, HORIZONS], label, color, marker, floor=FLOOR)
        style(ax, title, "Skill vs persistence" if ax is axes[0] else None)
        percent(ax)
    axes[0].set_ylim(-0.7, 0.2)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=5, frameon=False, fontsize=7)
    save(fig, "fig3_cl_t4")


def fig4(locked_summary, diagnostics):
    pipelines = {"locked": ("Locked", INK, "o", "-"), "D1": ("D1 change target", "#2a78d6", "s", "--"),
                 "D2": ("D2 expanding norm.", "#eb6834", "^", "--"), "D3": ("D3 = D1 + D2 (= D2 where B5 LSTM is best)", "#1baf7a", "D", ":"),
                 "D4": ("D4 B5 + holdout rows", "#6250d6", "v", "--")}
    frames = [locked_summary.assign(diagnostic="locked"), diagnostics]
    allrows = pd.concat(frames, ignore_index=True)
    allrows = allrows[allrows["method"].isin(CANDIDATES)]
    best = allrows.groupby(["diagnostic", "horizon_days"])["skill_online_T4_mean"].max().unstack().reindex(columns=HORIZONS)
    fig, ax = plt.subplots(figsize=(4.6, 3.8), constrained_layout=True)
    zero(ax)
    for key, (label, color, marker, ls) in pipelines.items():
        ax.plot(HORIZONS, best.loc[key], color=color, lw=2 if ls == "-" else 1.4, ls=ls, marker=marker, ms=4.5, label=label,
                markeredgecolor="white", markeredgewidth=0.8, zorder=3)
    style(ax, "Online T4: best of the five candidates per pipeline\n(D1–D4 exploratory post hoc)", "Skill vs persistence")
    percent(ax)
    fig.legend(*ax.get_legend_handles_labels(), loc="outside lower center", ncol=2, frameon=False, fontsize=6.5)
    save(fig, "fig4_diagnostics_t4")
    return best


def fig5(crossing, damped):
    locked = crossing[(crossing["diagnostic"] == "locked") & (crossing["scope"] == "T3_T5")]
    table = pivot(locked, "crossing_f1_mean", CANDIDATES)
    primary = damped[damped["role"] == "primary"].set_index("horizon_days")
    events = locked.groupby("horizon_days")["true_crossings"].first().reindex(HORIZONS)
    fig, ax = plt.subplots(figsize=(4.6, 3.2), constrained_layout=True)
    ax.axhline(0, color=MUTED, lw=0.8, zorder=2)
    ax.annotate("persistence (F1 = 0 by construction)", (HORIZONS[0], 0), xytext=(0, 3), textcoords="offset points", fontsize=6, color=MUTED)
    for method, (label, color, marker) in CANDIDATES.items():
        line(ax, HORIZONS, table.loc[method, HORIZONS], label, color, marker)
    line(ax, HORIZONS, primary.loc[HORIZONS, "crossing_f1_T3_T5"], DAMPED[0], DAMPED[1], DAMPED[2], ls="--")
    style(ax, "Crossing −0.4 m, online T3–T5 (definition post hoc)", "F1")
    ax.set_xticks(HORIZONS, [f"{h}\nn={int(events[h])}" for h in HORIZONS])
    ax.set_ylim(-0.03, 0.75)
    ax.legend(frameon=False, fontsize=6.5, ncol=2, loc="upper left")
    save(fig, "fig5_crossing_f1")


def write_table(frame: pd.DataFrame, stem: str, note: str) -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    frame.to_csv(TABLES / f"{stem}.csv", index=False, float_format="%.4f")
    (TABLES / f"{stem}.md").write_text(f"<!-- Written by src/91_final_outputs.py. {note} -->\n" + frame.to_markdown(index=False, floatfmt=".3f") + "\n")


def main() -> int:
    criteria = pd.read_csv(RESULTS / "horizon_criteria.csv")
    summary = pd.read_csv(RESULTS / "horizon_summary.csv")
    diagnostics = pd.read_csv(RESULTS / "diagnostics_summary.csv")
    diag_criteria = pd.read_csv(RESULTS / "diagnostics_criteria.csv")
    crossing = pd.read_csv(RESULTS / "diagnostics_crossing_f1.csv")
    damped = pd.read_csv(RESULTS / "damped_trend.csv")
    cl = pd.read_csv(RESULTS / "cl_summary.csv")
    cl_criteria = pd.read_csv(RESULTS / "cl_criteria.csv")
    cl_t4 = pd.read_csv(RESULTS / "cl_t4_vs_b5.csv")

    fig2(summary, criteria, damped)
    cl.attrs["pooled"] = cl_criteria.pivot(index="candidate", columns="horizon_days", values="pooled_online_skill")
    fig3(cl)
    best_t4 = fig4(summary, diagnostics)
    fig5(crossing, damped)

    columns = ["horizon_days", "candidate", "n_cells", "median_cell_skill", "wilcoxon_one_sided_p", "p_holm", "pooled_online_skill",
               "oracle", "oracle_holdout_skill", "pass_a", "pass_b", "pass_c", "passes"]
    write_table(criteria[columns], "table2_criteria_locked", "Rule 2.4 as locked (tag horizon-rule-locked).")
    t4 = diagnostics[diagnostics["method"].isin(CANDIDATES)].groupby(["diagnostic", "horizon_days"])["skill_online_T4_mean"].max().rename("best_skill_online_T4")
    diag = diag_criteria.groupby(["diagnostic", "horizon_days"]).agg(best_pooled_online_skill=("pooled_online_skill", "max"),
                                                                     min_p_holm=("p_holm", "min"), any_candidate_passes=("passes", "any"))
    write_table(diag.join(t4).reset_index().assign(label="exploratory_post_hoc"), "table3_diagnostics", "Exploratory post hoc; cannot overturn 2.4.")
    cl_table = cl[cl["method"].isin(CL)][["horizon_days", "method", "n_adaptations_mean", "skill_online_T3_mean", "skill_online_T4_mean",
                                          "skill_online_T5_mean", "final_skill_mean", "forgetting_mean"]]
    cl_table = cl_table.merge(cl_criteria[["horizon_days", "candidate", "pooled_online_skill"]].rename(columns={"candidate": "method"}), on=["horizon_days", "method"], how="left")
    cl_table = cl_table.merge(cl_t4.drop(columns=["label"]), on=["horizon_days", "method"], how="left")
    write_table(cl_table.assign(label="secondary_planned_v5"), "table4_cl_vs_b5", "Replay and EWC vs B5 (config/closing.yaml).")
    damped_columns = ["horizon_days", "phi", "role", "pooled_online_skill", "median_cell_skill", "wilcoxon_one_sided_p", "skill_T3", "skill_T4",
                      "skill_T5", "crossing_f1_T3_T5", "true_crossings_T3_T5", "crossing_f1_T4", "true_crossings_T4"]
    write_table(damped[damped_columns].assign(label="exploratory_post_hoc"), "table5_damped_trend", "Exploratory post hoc; conservative direction.")
    print(best_t4.round(3).to_string())
    print(f"Ditulis: {FIGURES}/fig2..fig5, {TABLES}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
