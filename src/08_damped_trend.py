"""Closing item 2 (protocol v6, 2.6): damped-trend naive baseline. EXPLORATORY POST-HOC.

Forecast = TMAT(t) + slope * sum_{k=1..h} phi^k, slope = tmat_diff_7 / 7 (config/closing.yaml).
phi = 0.9 primary (fixed in advance), 0.8 and 0.95 as sensitivity; all reported. Nothing is trained.
Added post hoc, but its direction is conservative: a stronger reference makes claims harder.
MAE skill against persistence on the rows of the locked batch, and crossing F1 with the D5 definition,
compared with the locked candidates (results/diagnostics_crossing_f1.csv, diagnostic = locked).

Output: results/damped_trend.csv, results/damped_trend_vs_models_crossing.csv
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from common import ROOT


def _load(filename: str, name: str):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(filename))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


bl = _load("03_baselines.py", "baselines")
hp = _load("04_hp_selection.py", "hp_selection")
hb = _load("05_horizon_batch.py", "horizon_batch")
dg = _load("06_diagnostics.py", "diagnostics")

CLOSING_YAML = ROOT / "config" / "closing.yaml"
OUTPUT_CSV = ROOT / "results" / "damped_trend.csv"
COMPARISON_CSV = ROOT / "results" / "damped_trend_vs_models_crossing.csv"
LOCKED_CROSSING_CSV = ROOT / "results" / "diagnostics_crossing_f1.csv"
T4 = "T4_kemarau_2019"


def damped_trend(rows: pd.DataFrame, horizon: int, phi: float) -> np.ndarray:
    multiplier = phi * (1 - phi**horizon) / (1 - phi)
    return rows["tmat_lag_0"].to_numpy() + rows["tmat_diff_7"].to_numpy() / 7 * multiplier


def main() -> int:
    plan = hp.read_locked(CLOSING_YAML)["damped_trend"]
    rule = hp.read_locked(hp.RULE_YAML)
    phis = [("primary", plan["phi_primary"])] + [("sensitivity", p) for p in plan["phi_sensitivity"]]
    online_tasks = rule["online_tasks"]
    records = []
    for horizon in rule["horizons_days"]:
        data = bl.load_data(horizon)
        rows, y = data.rows, data.y
        persistence = rows["tmat_lag_0"].to_numpy()
        in_scope = rows["task"].isin(online_tasks).to_numpy()
        for role, phi in phis:
            forecast = damped_trend(rows, horizon, phi)
            run = [{"online": forecast}]
            tests = hb.candidate_tests(data, run, online_tasks)
            record = {"label": plan["label"], "note": "post hoc; conservative direction", "horizon_days": horizon, "phi": phi, "role": role, **tests}
            for task in bl.TASK_ORDER:
                mask = (rows["task"] == task).to_numpy()
                record[f"skill_{task.split('_')[0]}"] = 1 - np.abs(forecast[mask] - y[mask]).mean() / np.abs(persistence[mask] - y[mask]).mean()
            for scope, tasks in (("T3_T5", online_tasks), ("T4", [T4])):
                mask = rows["task"].isin(tasks).to_numpy()
                f1, events, flagged = dg.crossing_f1(persistence[mask], y[mask], forecast[mask])
                record[f"crossing_f1_{scope}"], record[f"true_crossings_{scope}"], record[f"predicted_crossings_{scope}"] = f1, events, flagged
            records.append(record)
    table = pd.DataFrame(records)
    table.to_csv(OUTPUT_CSV, index=False, float_format="%.5f")

    locked = pd.read_csv(LOCKED_CROSSING_CSV)
    locked = locked[(locked["diagnostic"] == "locked") & locked["method"].isin(rule["candidates"])]
    primary = table[table["role"] == "primary"].set_index("horizon_days")
    comparison = []
    for _, row in locked.iterrows():
        reference = primary.loc[row["horizon_days"], f"crossing_f1_{row['scope']}"]
        comparison.append({"label": plan["label"], "horizon_days": row["horizon_days"], "scope": row["scope"], "method": row["method"],
                           "true_crossings": row["true_crossings"], "model_crossing_f1": row["crossing_f1_mean"],
                           "damped_trend_crossing_f1": reference, "model_minus_damped": row["crossing_f1_mean"] - reference})
    pd.DataFrame(comparison).to_csv(COMPARISON_CSV, index=False, float_format="%.5f")
    view = ["horizon_days", "phi", "pooled_online_skill", "median_cell_skill", "wilcoxon_one_sided_p", "skill_T3", "skill_T4", "skill_T5",
            "crossing_f1_T3_T5", "true_crossings_T3_T5", "crossing_f1_T4", "true_crossings_T4"]
    print(table[view].round(3).to_string(index=False))
    print(pd.DataFrame(comparison).pivot_table(index=["scope", "method"], columns="horizon_days", values="model_minus_damped").round(2).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
