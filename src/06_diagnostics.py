"""Diagnostics D1-D5 of protocol v5, section 2.5. EXPLORATORY POST-HOC: cannot overturn the 2.4 decision.

Definitions: config/diagnostics.yaml (tag diagnostics-plan-locked). Each diagnostic changes one thing
against the locked pipeline; candidates, H, grid, selection procedure, and criteria a-c are as locked.
  D1  change target for Ridge/LightGBM (the LSTM is already residual on TMAT t)
  D2  expanding-window normalization, only data <= t
  D3  D1 + D2
  D4  B5 retrains on every row whose target is observed by the checkpoint (holdouts included)
  D5  F1 of crossing the -0.4 m threshold, on the locked batch and on D1-D4

Output (all with label = exploratory_post_hoc): results/diagnostics_criteria.csv,
results/diagnostics_summary.csv, results/diagnostics_hp_selection.csv, results/diagnostics_models.yaml,
results/diagnostics_crossing_f1.csv, rows diagnostics_D*_... in results/experiments.csv
"""
import importlib.util
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from common import ROOT


def _load(filename: str, name: str):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(filename))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


bl = _load("03_baselines.py", "baselines")
hp = _load("04_hp_selection.py", "hp_selection")
hb = _load("05_horizon_batch.py", "horizon_batch")

PLAN_YAML = ROOT / "config" / "diagnostics.yaml"
CRITERIA_CSV = ROOT / "results" / "diagnostics_criteria.csv"
SUMMARY_CSV = ROOT / "results" / "diagnostics_summary.csv"
SELECTION_CSV = ROOT / "results" / "diagnostics_hp_selection.csv"
MODELS_YAML = ROOT / "results" / "diagnostics_models.yaml"
CROSSING_CSV = ROOT / "results" / "diagnostics_crossing_f1.csv"
LOCKED_PREDICTIONS = ROOT / "data" / "processed" / "horizon_predictions.csv"
LABEL = "exploratory_post_hoc"
FLAGS = {
    "D1": {"target": "change"},
    "D2": {"normalization_mode": "expanding"},
    "D3": {"target": "change", "normalization_mode": "expanding"},
    "D4": {"b5_rows": "all_observed"},
}
RESELECT = ("D1", "D2", "D3")  # D4 changes nothing the selection sees
TMAT_ALERT_M = bl.TMAT_ALERT_M


def base_config(diagnostic: str) -> dict:
    config = {k: (dict(v) if isinstance(v, dict) else v) for k, v in bl.CONFIG.items()}
    config.update(FLAGS[diagnostic])
    config["diagnostic"] = diagnostic
    config["label"] = LABEL
    return config


def with_family(base: dict, family: str, settings: dict) -> dict:
    config = {k: (dict(v) if isinstance(v, dict) else v) for k, v in base.items()}
    config[family] = {**config[family], **settings}
    return config


_DATA = {}


def cached_data(diagnostic: str, horizon: int):
    key = (diagnostic, horizon)
    if key not in _DATA:
        _DATA[key] = bl.load_data(horizon, base_config(diagnostic))
    return _DATA[key]


def selection_score(diagnostic: str, horizon: int, family: str, settings: dict) -> float:
    """The locked selection procedure (src/04_hp_selection.py), under one diagnostic's pipeline."""
    data = cached_data(diagnostic, horizon)
    data.config = with_family(base_config(diagnostic), family, settings)
    fit, validate = hp.selection_split(data)
    if family == "ridge":
        prediction = bl.fit_ridge(data, fit)
    elif family == "lightgbm":
        prediction = bl.fit_lightgbm(data, fit)
    else:
        groups = data.rows["task"].to_numpy()[fit]
        model = bl.fit_lstm(bl.new_lstm(data, hp.SELECTION_SEED), data, fit, groups, hp.SELECTION_SEED)
        prediction = bl.lstm_predictor(model, data)
    return float(np.abs(prediction[validate] - data.y[validate]).mean())


def _selection_worker(job):
    diagnostic, horizon, family, number, settings = job
    return diagnostic, horizon, family, number, selection_score(diagnostic, horizon, family, settings)


def _seed_worker(job):
    diagnostic, horizon, config, seed = job
    return diagnostic, horizon, bl.run_lstm_methods(bl.load_data(horizon, config), seed)


def crossing_f1(actual_now: np.ndarray, actual: np.ndarray, predicted: np.ndarray) -> tuple[float, int, int]:
    """D5: event = TMAT(t) >= -0.4 m and TMAT(t+h) < -0.4 m; predicted the same way from the forecast."""
    above = actual_now >= TMAT_ALERT_M
    true_event, predicted_event = above & (actual < TMAT_ALERT_M), above & (predicted < TMAT_ALERT_M)
    hits = int((true_event & predicted_event).sum())
    denominator = int(true_event.sum() + predicted_event.sum())
    return (2 * hits / denominator if denominator else float("nan")), int(true_event.sum()), int(predicted_event.sum())


def crossing_rows(diagnostic: str, horizon: int, data, results: list[dict]) -> list[dict]:
    rows = data.rows
    now = rows["tmat_lag_0"].to_numpy()
    records = []
    for result in results:
        if result["online"] is None:
            continue
        for scope, tasks in (("T3_T5", bl.TASK_ORDER[2:]), ("T4", ["T4_kemarau_2019"])):
            mask = rows["task"].isin(tasks).to_numpy() & ~np.isnan(result["online"])
            f1, events, predicted = crossing_f1(now[mask], data.y[mask], result["online"][mask])
            records.append({"label": LABEL, "diagnostic": diagnostic, "horizon_days": horizon, "method": result["method"], "seed": result["seed"],
                            "scope": scope, "n_rows": int(mask.sum()), "true_crossings": events, "predicted_crossings": predicted, "crossing_f1": f1})
    return records


def locked_results(horizon: int, data, predictions: pd.DataFrame) -> list[dict]:
    """Online predictions of the locked batch, aligned with this horizon's rows, for D5."""
    part = predictions[predictions["horizon_days"] == horizon]
    key = data.rows[["station", "date"]].reset_index()
    results = []
    for (method, seed), group in part.groupby(["method", "seed"]):
        merged = key.merge(group, on=["station", "date"], how="left").sort_values("index")
        results.append({"method": method, "seed": seed, "online": merged["prediction_m"].to_numpy()})
    persistence = data.rows["tmat_lag_0"].to_numpy()
    results.append({"method": "B1_persistence", "seed": 0, "online": persistence})
    return results


def main() -> int:
    plan = hp.read_locked(PLAN_YAML)
    rule = hp.read_locked(hp.RULE_YAML)
    locked_models = hp.read_locked(hp.MODELS_YAML)
    configs_grid = hp.grid_configs(hp.read_locked(hp.GRID_YAML))
    horizons = rule["horizons_days"]
    if plan["label"] != LABEL or sorted(plan["diagnostics"]) != ["D1", "D2", "D3", "D4", "D5"]:
        raise ValueError("config/diagnostics.yaml tidak sesuai skrip")
    started = time.perf_counter()

    # Selection under D1-D3, same grid and split as locked.
    jobs = [(d, h, f, n, s) for d in RESELECT for h in horizons for f in ("ridge", "lightgbm", "lstm") for n, s in enumerate(configs_grid[f])]
    selection = []
    with ProcessPoolExecutor(max_workers=7) as pool:
        for diagnostic, horizon, family, number, mae in pool.map(_selection_worker, jobs, chunksize=4):
            selection.append({"label": LABEL, "diagnostic": diagnostic, "horizon_days": horizon, "family": family,
                              "config_number": number, "config": repr(configs_grid[family][number]), "validation_mae": mae})
    selection = pd.DataFrame(selection).sort_values(["diagnostic", "horizon_days", "family", "config_number"]).reset_index(drop=True)
    selection.to_csv(SELECTION_CSV, index=False, float_format="%.6f")
    chosen = {"D4": {h: locked_models["horizons"][h] for h in horizons}}
    for diagnostic in RESELECT:
        chosen[diagnostic] = {}
        for horizon in horizons:
            chosen[diagnostic][horizon] = {}
            for family in ("ridge", "lightgbm", "lstm"):
                part = selection[(selection["diagnostic"] == diagnostic) & (selection["horizon_days"] == horizon) & (selection["family"] == family)]
                best = part.loc[part["validation_mae"].idxmin()]
                chosen[diagnostic][horizon][family] = {**configs_grid[family][int(best["config_number"])], "validation_mae": round(float(best["validation_mae"]), 6)}
    MODELS_YAML.write_text(f"# Written by src/06_diagnostics.py. label: {LABEL}. D4 = config/models.yaml unchanged.\n"
                           + yaml.safe_dump({"label": LABEL, "diagnostics": chosen}, sort_keys=False))
    print(f"Seleksi selesai ({time.perf_counter() - started:.0f} s)", flush=True)

    # Batch per diagnostic and horizon.
    run_configs, datasets, results = {}, {}, {}
    for diagnostic in FLAGS:
        for horizon in horizons:
            config = base_config(diagnostic)
            for family in hb.FAMILIES:
                config[family] = {**config[family], **{k: v for k, v in chosen[diagnostic][horizon][family].items() if k != "validation_mae"}}
            config["horizon_days"] = horizon
            run_configs[diagnostic, horizon] = config
            datasets[diagnostic, horizon] = bl.load_data(horizon, config)
            results[diagnostic, horizon] = bl.run_deterministic_methods(datasets[diagnostic, horizon])
    jobs = [(d, h, run_configs[d, h], seed) for d in FLAGS for h in horizons for seed in bl.SEEDS]
    with ProcessPoolExecutor(max_workers=7) as pool:
        for diagnostic, horizon, batch in pool.map(_seed_worker, jobs):
            results[diagnostic, horizon].extend(batch)

    criteria, summaries, crossings = [], [], []
    for (diagnostic, horizon), run in results.items():
        data = datasets[diagnostic, horizon]
        _, summary, crit, _ = hb.evaluate_horizon(horizon, data, run, rule, run_configs[diagnostic, horizon], prefix=f"diagnostics_{diagnostic}")
        for row in crit:
            criteria.append({"label": LABEL, "diagnostic": diagnostic, **row})
        for row in summary:
            holdout_valid = not (diagnostic == "D4" and row["method"].startswith("B5_"))
            summaries.append({"label": LABEL, "diagnostic": diagnostic, "holdout_out_of_sample": holdout_valid, **row})
        crossings.extend(crossing_rows(diagnostic, horizon, data, run))
    predictions = pd.read_csv(LOCKED_PREDICTIONS, parse_dates=["date"])
    for horizon in horizons:
        data = bl.load_data(horizon)
        crossings.extend(crossing_rows("locked", horizon, data, locked_results(horizon, data, predictions)))

    pd.DataFrame(criteria).to_csv(CRITERIA_CSV, index=False, float_format="%.5f")
    per_run = pd.DataFrame(summaries)
    columns = [c for c in per_run.columns if c not in ("label", "diagnostic", "holdout_out_of_sample", "horizon_days", "method", "seed", "runtime_seconds")]
    keys = ["label", "diagnostic", "holdout_out_of_sample", "horizon_days", "method"]
    summary = per_run.groupby(keys, sort=False)[columns].agg(["mean", "std"])
    summary.columns = [f"{name}_{stat}" for name, stat in summary.columns]
    summary.insert(0, "n_seeds", per_run.groupby(keys, sort=False)["seed"].nunique())
    summary.reset_index().to_csv(SUMMARY_CSV, index=False, float_format="%.5f")
    crossing = pd.DataFrame(crossings)
    crossing.groupby(["label", "diagnostic", "horizon_days", "method", "scope"], sort=False).agg(
        n_seeds=("seed", "nunique"), n_rows=("n_rows", "first"), true_crossings=("true_crossings", "first"),
        predicted_crossings_mean=("predicted_crossings", "mean"), crossing_f1_mean=("crossing_f1", "mean"), crossing_f1_std=("crossing_f1", "std"),
    ).reset_index().to_csv(CROSSING_CSV, index=False, float_format="%.5f")

    table = pd.DataFrame(criteria)
    view = ["diagnostic", "horizon_days", "candidate", "median_cell_skill", "p_holm", "pooled_online_skill", "oracle_holdout_skill", "pass_a", "pass_b", "pass_c", "passes"]
    print(table[view].round(4).to_string(index=False))
    print(f"Selesai dalam {time.perf_counter() - started:.0f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
