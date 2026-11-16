"""Manuscript tables: formats Tables 2-5 and Appendix Table A1 for docs/manuscript/manuscript.md.

Formatting only. Reads files already in results/ (written at tag analysis-frozen) and computes nothing
new except percentages and mean +/- sd strings. Analysis code and results are untouched.

Output: docs/manuscript/tables.md (pasted into the manuscript between TABLE markers by --insert).
"""
import re
import sys

import pandas as pd

from common import ROOT

TABLES = ROOT / "results" / "tables"
OUT = ROOT / "docs" / "manuscript" / "tables.md"
MANUSCRIPT = ROOT / "docs" / "manuscript" / "manuscript.md"

NAMES = {
    "B2_static_ridge": "B2 Ridge", "B2_static_lightgbm": "B2 LightGBM",
    "B5_retrain30d_ridge": "B5 Ridge", "B5_retrain30d_lightgbm": "B5 LightGBM", "B5_retrain30d_lstm": "B5 LSTM",
    "CL_finetune30d_lstm": "Fine-tune (control)", "CL_replay30d_lstm": "Replay", "CL_ewc30d_lstm": "EWC",
}


def pct(x: float, signed: bool = True, digits: int = 0) -> str:
    if pd.isna(x):
        return "–"
    return f"{100 * x:+.{digits}f}".replace("-", "−") if signed else f"{100 * x:.{digits}f}"


def num(x: float, digits: int = 2) -> str:
    return "–" if pd.isna(x) else f"{x:.{digits}f}".replace("-", "−")


def yes(flag) -> str:
    return "✓" if str(flag) == "True" else "✗"


def md(rows: list[list[str]], header: list[str], align: str) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(":---" if a == "l" else "---:" for a in align) + "|"]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines)


def table2() -> str:
    t = pd.read_csv(TABLES / "table2_criteria_locked.csv")
    rows = [[str(r.horizon_days), NAMES[r.candidate], pct(r.median_cell_skill), num(r.wilcoxon_one_sided_p), num(r.p_holm),
             pct(r.pooled_online_skill), pct(r.oracle_holdout_skill, digits=1), yes(r.pass_a), yes(r.pass_b), yes(r.pass_c)] for r in t.itertuples()]
    header = ["*h* (d)", "Candidate", "Median cell skill (%)", "*p*", "*p*-Holm", "Pooled skill (%)", "Oracle skill (%)", "(a)", "(b)", "(c)"]
    return md(rows, header, "llrrrrrlll")


def table3() -> str:
    t = pd.read_csv(TABLES / "table3_diagnostics.csv")
    rows = [[r.diagnostic, str(r.horizon_days), pct(r.best_pooled_online_skill), pct(r.best_skill_online_T4), num(r.min_p_holm),
             "yes" if str(r.any_candidate_passes) == "True" else "no"] for r in t.itertuples()]
    return md(rows, ["Diagnostic", "*h* (d)", "Best pooled skill (%)", "Best T4 skill (%)", "Min. *p*-Holm", "Any candidate passes"], "llrrrl")


def table4() -> str:
    t = pd.read_csv(TABLES / "table4b_cl_vs_b5_seeds.csv")
    word = {"reference": "reference", "indistinguishable": "indist.", "worse": "worse", "better": "better"}
    rows = [[str(r.horizon_days), NAMES[r.method], f"{pct(r.skill_T4_mean)} ± {pct(r.skill_T4_sd, False)}", word[r.skill_T4_vs_b5],
             f"{pct(r.pooled_T3_T5_mean)} ± {pct(r.pooled_T3_T5_sd, False)}", word[r.pooled_T3_T5_vs_b5]] for r in t.itertuples()]
    return md(rows, ["*h* (d)", "Method", "T4 skill (%)", "vs B5", "Pooled T3–T5 skill (%)", "vs B5"], "llrlrl")


def table5() -> str:
    t = pd.read_csv(TABLES / "table5_damped_trend.csv")
    rows = [[str(r.horizon_days), num(r.phi), pct(r.pooled_online_skill), pct(r.skill_T3), pct(r.skill_T4), pct(r.skill_T5),
             num(r.crossing_f1_T3_T5), str(int(r.true_crossings_T3_T5)), str(int(r.true_crossings_T4))] for r in t.itertuples()]
    header = ["*h* (d)", "φ", "Pooled skill (%)", "T3 (%)", "T4 (%)", "T5 (%)", "Crossing F1, T3–T5", "Crossings, T3–T5", "Crossings, T4"]
    return md(rows, header, "llrrrrrrr")


def table_a1() -> str:
    d = pd.read_csv(ROOT / "results" / "drift_regression.csv")
    wide = d.pivot_table(index="station", columns=["data", "mode"], values="ratio")
    counts = d[(d["data"] == "suspect_excluded") & (d["mode"] == "panel")].set_index("station")
    order = sorted(wide.index, key=lambda s: int(re.sub(r"\D", "", s)))
    rows = [[s, str(int(counts.loc[s, "n_pre"])), str(int(counts.loc[s, "n_drought"])),
             num(wide.loc[s, ("suspect_excluded", "panel")]), num(wide.loc[s, ("suspect_excluded", "per_station")]),
             num(wide.loc[s, ("with_suspect", "panel")]), num(wide.loc[s, ("with_suspect", "per_station")])] for s in order]
    header = ["Station", "Pre-drought rows", "Drought rows", "Panel", "Per station", "Panel, suspect kept", "Per station, suspect kept"]
    return md(rows, header, "lrrrrrr")


BLOCKS = {"TABLE2": table2, "TABLE3": table3, "TABLE4": table4, "TABLE5": table5, "TABLEA1": table_a1}


def main() -> int:
    blocks = {name: fn() for name, fn in BLOCKS.items()}
    OUT.write_text("\n\n".join(f"<!-- {k} -->\n{v}" for k, v in blocks.items()) + "\n", encoding="utf-8")
    if "--insert" in sys.argv:
        text = MANUSCRIPT.read_text(encoding="utf-8")
        for name, body in blocks.items():
            pattern = re.compile(rf"<!-- BEGIN {name} -->.*?<!-- END {name} -->", re.S)
            if not pattern.search(text):
                raise SystemExit(f"marker {name} missing in manuscript")
            text = pattern.sub(lambda _: f"<!-- BEGIN {name} (src/92_manuscript_tables.py) -->\n{body}\n<!-- END {name} -->", text)
        MANUSCRIPT.write_text(text, encoding="utf-8")
    print(OUT.read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
