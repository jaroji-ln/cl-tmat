"""Horizon batch H = {3, 7, 14, 21, 30} and the decision of protocol v4, rule 2.4, applied as locked.

Same pipeline as src/03_baselines.py (B1-B5, same splits, T1-frozen normalization), with the
hyperparameters chosen per horizon by src/04_hp_selection.py (config/models.yaml).
The decision rule is read from config/horizon_rule.yaml (tag horizon-rule-locked):
  a  online T3-T5, cells station x task, median cell skill > 0 and one-sided paired Wilcoxon,
     Holm over the 5 realistic candidates, p-Holm < 0.05
  b  pooled online skill on T3-T5 >= 0.05
  c  B3 of the same family: final holdout skill > 0
A horizon passes if at least one candidate passes a, b and c; the primary horizon is the shortest passing one.

Output: results/horizon_criteria.csv, results/horizon_summary.csv, results/horizon_R.csv,
results/horizon_decision.yaml, data/processed/horizon_predictions.csv, rows in results/experiments.csv
"""
import importlib.util
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import wilcoxon

from common import ROOT


def _load(filename: str, name: str):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(filename))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


bl = _load("03_baselines.py", "baselines")
hp = _load("04_hp_selection.py", "hp_selection")

CRITERIA_CSV = ROOT / "results" / "horizon_criteria.csv"
SUMMARY_CSV = ROOT / "results" / "horizon_summary.csv"
R_CSV = ROOT / "results" / "horizon_R.csv"
DECISION_YAML = ROOT / "results" / "horizon_decision.yaml"
PREDICTIONS_CSV = ROOT / "data" / "processed" / "horizon_predictions.csv"
FAMILIES = ("ridge", "lightgbm", "lstm")


def config_for(models: dict, horizon: int) -> dict:
    config = {k: (dict(v) if isinstance(v, dict) else v) for k, v in bl.CONFIG.items()}
    for family in FAMILIES:
        chosen = {k: v for k, v in models["horizons"][horizon][family].items() if k != "validation_mae"}
        config[family] = {**config[family], **chosen}
    config["horizon_days"] = horizon
    config["hyperparameters"] = "config/models.yaml"
    return config


def _seed_worker(job: tuple[int, dict, int]) -> tuple[int, list[dict]]:
    horizon, config, seed = job
    return horizon, bl.run_lstm_methods(bl.load_data(horizon, config), seed)


def family_of(method: str) -> str:
    return next(f for f in FAMILIES if method.endswith(f))


def holm(p_values: np.ndarray) -> np.ndarray:
    order = np.argsort(p_values)
    m = len(p_values)
    adjusted = np.empty(m)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p_values[index]))
        adjusted[index] = running
    return adjusted


def one_sided_p(model: pd.Series, reference: pd.Series) -> float:
    if (model == reference).all():
        return 1.0
    return float(wilcoxon(model, reference, alternative="less").pvalue)


def candidate_tests(data, runs: list[dict], online_tasks: list[str]) -> dict:
    """Criteria a and b for one candidate: online rows of T3-T5, errors averaged over seeds."""
    rows = data.rows
    persistence = rows["tmat_lag_0"].to_numpy()
    in_scope = rows["task"].isin(online_tasks).to_numpy()
    mask = in_scope & ~np.isnan(runs[0]["online"])
    if any((in_scope & ~np.isnan(r["online"]) != mask).any() for r in runs) or (in_scope & ~mask).any():
        raise AssertionError("Prakiraan online tidak mencakup seluruh baris T3-T5")
    model_cells = pd.concat([bl.cell_mae(data, r["online"], mask) for r in runs], axis=1).mean(axis=1)
    reference_cells = bl.cell_mae(data, persistence, mask).reindex(model_cells.index)
    skill = 1 - model_cells / reference_cells
    model_mae = float(np.mean([np.abs(r["online"][mask] - data.y[mask]).mean() for r in runs]))
    reference_mae = float(np.abs(persistence[mask] - data.y[mask]).mean())
    return {
        "n_rows": int(mask.sum()), "n_cells": len(skill), "median_cell_skill": float(skill.median()),
        "cells_better_than_persistence": int((skill > 0).sum()), "wilcoxon_one_sided_p": one_sided_p(model_cells, reference_cells),
        "online_mae": model_mae, "persistence_online_mae": reference_mae, "pooled_online_skill": 1 - model_mae / reference_mae,
    }


def online_f1(data, online: np.ndarray | None, tasks: list[str]) -> float:
    if online is None:
        return np.nan
    mask = data.rows["task"].isin(tasks).to_numpy() & ~np.isnan(online)
    return bl.f1_score(data.y[mask], online[mask])


def evaluate_horizon(horizon: int, data, results: list[dict], rule: dict, config: dict) -> tuple[list, list, list, list]:
    """R records, per-run summaries (logged), criteria rows, and online predictions for one horizon."""
    online_tasks = rule["online_tasks"]
    records, summaries, predictions = [], [], []
    persistence = pd.DataFrame(bl.r_records(data, "B1_persistence", bl.DETERMINISTIC_SEED, {bl.TASK_ORDER[-1]: data.rows["tmat_lag_0"].to_numpy()}))
    for result in results:
        method_records = pd.DataFrame(bl.r_records(data, result["method"], result["seed"], result["by_task"]))
        records.append(method_records.assign(horizon_days=horizon))
        metrics = {
            "horizon_days": horizon, "method": result["method"], "seed": result["seed"], "n_adaptations": result["adaptations"],
            **bl.summarize_run(method_records, persistence), **bl.online_metrics(data, result["online"]),
            "f1_online_T3_T5": online_f1(data, result["online"], online_tasks), "f1_online_T4": online_f1(data, result["online"], ["T4_kemarau_2019"]),
            "runtime_seconds": result["seconds"],
        }
        summaries.append(metrics)
        bl.log_experiment(f"horizon_h{horizon}_{result['method']}_seed{result['seed']}", config, result["seed"],
                          {k: v for k, v in metrics.items() if k not in ("method", "seed")}, result["seconds"])
        if result["online"] is not None:
            predictions.append(pd.DataFrame({"horizon_days": horizon, "method": result["method"], "seed": result["seed"], "station": data.rows["station"],
                                             "date": data.rows["date"], "target_date": data.rows["target_date"], "prediction_m": result["online"]}).dropna())
    per_run = pd.DataFrame(summaries)
    final_skill = per_run.groupby("method")["final_skill"].mean()
    criteria = []
    for method in rule["candidates"]:
        runs = [r for r in results if r["method"] == method]
        oracle = f"B3_joint_{family_of(method)}"
        criteria.append({"horizon_days": horizon, "candidate": method, "n_seeds": len(runs), **candidate_tests(data, runs, online_tasks),
                         "oracle": oracle, "oracle_holdout_skill": float(final_skill[oracle])})
    p_holm = holm(np.array([c["wilcoxon_one_sided_p"] for c in criteria]))
    a_rule, b_rule, c_rule = rule["criteria"]["a"], rule["criteria"]["b"], rule["criteria"]["c"]
    for row, p in zip(criteria, p_holm):
        row["p_holm"] = float(p)
        row["pass_a"] = bool(row["median_cell_skill"] > a_rule["median_cell_skill_gt"] and p < a_rule["p_holm_lt"])
        row["pass_b"] = bool(row["pooled_online_skill"] >= b_rule["pooled_online_skill_ge"])
        row["pass_c"] = bool(row["oracle_holdout_skill"] > c_rule["holdout_skill_gt"])
        row["passes"] = row["pass_a"] and row["pass_b"] and row["pass_c"]
    return records, summaries, criteria, predictions


def main() -> int:
    rule = hp.read_locked(hp.RULE_YAML)
    models = hp.read_locked(hp.MODELS_YAML)
    horizons = rule["horizons_days"]
    if sorted(models["horizons"]) != sorted(horizons):
        raise ValueError("config/models.yaml tidak mencakup semua horizon")
    configs = {h: config_for(models, h) for h in horizons}
    started = time.perf_counter()
    datasets = {h: bl.load_data(h, configs[h]) for h in horizons}
    results = {h: bl.run_deterministic_methods(datasets[h]) for h in horizons}
    jobs = [(h, configs[h], seed) for h in horizons for seed in bl.SEEDS]
    with ProcessPoolExecutor(max_workers=7) as pool:
        for horizon, batch in pool.map(_seed_worker, jobs):
            results[horizon].extend(batch)

    records, summaries, criteria, predictions = [], [], [], []
    for horizon in horizons:
        parts = evaluate_horizon(horizon, datasets[horizon], results[horizon], rule, configs[horizon])
        for target, part in zip((records, summaries, criteria, predictions), parts):
            target.extend(part)

    pd.concat(records, ignore_index=True).to_csv(R_CSV, index=False, float_format="%.5f")
    per_run = pd.DataFrame(summaries)
    columns = [c for c in per_run.columns if c not in ("horizon_days", "method", "seed", "runtime_seconds")]
    summary = per_run.groupby(["horizon_days", "method"], sort=False)[columns].agg(["mean", "std"])
    summary.columns = [f"{name}_{stat}" for name, stat in summary.columns]
    summary.insert(0, "n_seeds", per_run.groupby(["horizon_days", "method"], sort=False)["seed"].nunique())
    summary.reset_index().to_csv(SUMMARY_CSV, index=False, float_format="%.5f")
    table = pd.DataFrame(criteria)
    table.to_csv(CRITERIA_CSV, index=False, float_format="%.5f")
    pd.concat(predictions, ignore_index=True).to_csv(PREDICTIONS_CSV, index=False, float_format="%.5f", date_format="%Y-%m-%d")

    passing = sorted(table.loc[table["passes"], "horizon_days"].unique().tolist())
    decision = {
        "rule": "protocol v4 2.4, config/horizon_rule.yaml at tag horizon-rule-locked",
        "horizons_passing": [int(h) for h in passing],
        "passing_candidates": {int(h): table.loc[table["passes"] & (table["horizon_days"] == h), "candidate"].tolist() for h in passing},
        "primary_horizon_days": int(passing[0]) if passing else None,
        "outcome": "primary horizon chosen (2.4.4)" if passing else "no horizon passes: benchmark framing (2.4.5)",
    }
    DECISION_YAML.write_text("# Written by src/05_horizon_batch.py\n" + yaml.safe_dump(decision, sort_keys=False))
    view = ["horizon_days", "candidate", "median_cell_skill", "wilcoxon_one_sided_p", "p_holm", "pooled_online_skill",
            "oracle", "oracle_holdout_skill", "pass_a", "pass_b", "pass_c", "passes"]
    print(table[view].round(4).to_string(index=False))
    print(yaml.safe_dump(decision, sort_keys=False))
    print(f"Selesai dalam {time.perf_counter() - started:.0f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
