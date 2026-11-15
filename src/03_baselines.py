"""Phase 2 baselines B1-B5 (protocol 2.1). Hyperparameters are fixed, never tuned.

Each task is split once by target date. R[i, j] is the holdout error of task j for the model
available after training on task i. mae_online is the error on every row of a task using only
the model that existed when that forecast was issued, so it covers drought onset too.

Skill is 1 - MAE_model / MAE_persistence on the same rows (protocol 3.4), and
results/baseline_skill.csv pairs it by station x task cell (protocol 3.5).

Output: results/baseline_R.csv, results/baseline_summary.csv, results/baseline_skill.csv, results/normalization_t1.json,
data/processed/baseline_predictions.csv, and one row per method and seed in results/experiments.csv
"""
import argparse
import importlib.util
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import wilcoxon
from sklearn.linear_model import Ridge

from common import CHIRPS_CSV, FEATURES_CSV, GWT_CLEAN_CSV, ROOT
from normalization import FrozenScaler
from splits import TASK_ORDER, add_folds, available_for_training

torch.set_num_threads(1)  # deterministic, and avoids an OpenMP clash with LightGBM


def _load(filename: str, name: str):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(filename))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


bf = _load("02_build_features.py", "build_features")
log_experiment = _load("80_drift_regression.py", "drift_regression").log_experiment

R_CSV = ROOT / "results" / "baseline_R.csv"
SKILL_CSV = ROOT / "results" / "baseline_skill.csv"
SUMMARY_CSV = ROOT / "results" / "baseline_summary.csv"
NORMALIZATION_JSON = ROOT / "results" / "normalization_t1.json"
PREDICTIONS_CSV = ROOT / "data" / "processed" / "baseline_predictions.csv"

SEEDS = [0, 1, 2, 3, 4]
DETERMINISTIC_SEED = 0
RETRAIN_EVERY_DAYS = 30
SEQUENCE_DAYS = 30
TMAT_ALERT_M = -0.4  # PP 57/2016
STATIC_TASKS = TASK_ORDER[:2]
TARGET_SCALE = "tmat_lag_0"
RAW_COLUMNS = bf.SEASON_FEATURES + bf.STATION_FEATURES + bf.PEATLAND_FEATURES  # bounded, no statistics needed
SCALED_COLUMNS = bf.TMAT_FEATURES + bf.RAIN_FEATURES
CONTEXT_COLUMNS = bf.RAIN_FEATURES + RAW_COLUMNS
# Fixed before the cutoff-t run: material skill needs both, per station x task cell (protocol 3.5).
MATERIAL_SKILL = {"median_cell_skill_min": 0.05, "wilcoxon_p_max": 0.05}
CONFIG = {
    "information_cutoff": "t",
    "ridge": {"alpha": 1.0},
    "lightgbm": {"n_estimators": 300, "learning_rate": 0.05, "num_leaves": 15, "min_child_samples": 10},
    "lstm": {"layers": 2, "units": 64, "sequence_days": SEQUENCE_DAYS, "residual_on": "tmat_lag_0", "lr": 1e-3,
             "batch": 64, "max_epochs": 60, "patience": 8, "validation": "last 15% by target date", "grad_clip": 1.0},
    "holdout": "last 30% of each task by target date",
    "normalization": "T1 training fold, frozen; season and one-hot columns stay raw",
}


class Data:
    """Scaled arrays aligned with rows."""

    def __init__(self, rows: pd.DataFrame, sequences: np.ndarray, scaler: FrozenScaler, config: dict | None = None):
        self.config = config or CONFIG  # hyperparameters; the horizon batch passes the selected ones
        self.rows = rows.reset_index(drop=True)
        self.y = rows["target_tmat_m"].to_numpy()
        scaled = {c: scaler.transform(rows[c], c) for c in SCALED_COLUMNS}
        self.tabular = np.column_stack([scaled[c] for c in SCALED_COLUMNS] + [rows[c].to_numpy(float) for c in RAW_COLUMNS])
        self.context = np.column_stack([scaled.get(c, rows[c].to_numpy(float)) for c in CONTEXT_COLUMNS])
        self.sequence = np.stack(
            [scaler.transform(sequences[:, :, 0], TARGET_SCALE), scaler.transform(sequences[:, :, 1], "rain_1d")], axis=2
        )
        self.y_scaled = scaler.transform(self.y, TARGET_SCALE)
        self.scaler = scaler


def build_sequences(rows: pd.DataFrame, gwt: pd.DataFrame, chirps: pd.DataFrame) -> np.ndarray:
    """TMAT and rain for t-29..t per row. Suspect values are masked and must not be reached."""
    tmat = {s: g.set_index("date")["tmat_m"].mask(g.set_index("date")["is_suspect"]) for s, g in gwt.groupby("station")}
    rain = {s: g.set_index("date")["precip_mm"] for s, g in chirps.groupby("station")}
    output = np.empty((len(rows), SEQUENCE_DAYS, 2))
    for i, (station, date) in enumerate(zip(rows["station"], rows["date"])):
        window = pd.date_range(date - pd.Timedelta(days=SEQUENCE_DAYS - 1), periods=SEQUENCE_DAYS)
        output[i, :, 0] = tmat[station].reindex(window).to_numpy()
        output[i, :, 1] = rain[station].reindex(window).to_numpy()
    if np.isnan(output).any():
        raise ValueError("Jendela input memuat NaN")
    return output


class LSTMForecaster(torch.nn.Module):
    def __init__(self, context_size: int, settings: dict):
        super().__init__()
        self.sequence_days = settings["sequence_days"]
        self.lstm = torch.nn.LSTM(2, settings["units"], settings["layers"], batch_first=True)
        # Dropout on the LSTM output before the head; skipped at 0 so the fixed t+7 run is unchanged.
        self.dropout = torch.nn.Dropout(settings["dropout"]) if settings.get("dropout", 0.0) > 0 else torch.nn.Identity()
        self.head = torch.nn.Linear(settings["units"] + context_size, 1)

    def forward(self, sequence, context):
        sequence = sequence[:, -self.sequence_days :]  # inputs hold t-29..t; shorter windows keep the last days
        encoded, _ = self.lstm(sequence)
        change = self.head(torch.cat([self.dropout(encoded[:, -1]), context], dim=1)).squeeze(1)
        return sequence[:, -1, 0] + change  # the last observed level, plus a learned change


def validation_mask(rows: pd.DataFrame) -> np.ndarray:
    dates = rows["target_date"].sort_values()
    cut = dates.iloc[int(len(dates) * 0.85)]
    return (rows["target_date"] >= cut).to_numpy()


def fit_lstm(model, data: Data, index: np.ndarray, groups: np.ndarray, seed: int, max_epochs: int | None = None):
    """Train on rows in index. Validation is the last 15% of each group, so nothing shuffles across time."""
    settings = data.config["lstm"]
    rows = data.rows.iloc[index]
    validation = np.zeros(len(index), dtype=bool)
    for group in np.unique(groups):
        member = groups == group
        validation[member] = validation_mask(rows[member])
    tensors = [torch.tensor(a, dtype=torch.float32) for a in (data.sequence[index], data.context[index], data.y_scaled[index])]
    fit_tensors = [t[~validation] for t in tensors]
    val_tensors = [t[validation] for t in tensors]
    generator = torch.Generator().manual_seed(seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=settings["lr"])
    best, best_loss, waited = None, float("inf"), 0
    for _ in range(max_epochs or settings["max_epochs"]):
        model.train()
        order = torch.randperm(len(fit_tensors[2]), generator=generator)
        for start in range(0, len(order), settings["batch"]):
            batch = order[start : start + settings["batch"]]
            optimizer.zero_grad()
            loss = torch.nn.functional.mse_loss(model(fit_tensors[0][batch], fit_tensors[1][batch]), fit_tensors[2][batch])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), settings["grad_clip"])
            optimizer.step()
        model.eval()
        with torch.no_grad():
            val_loss = torch.nn.functional.mse_loss(model(val_tensors[0], val_tensors[1]), val_tensors[2]).item()
        if val_loss < best_loss:
            best, best_loss, waited = {k: v.clone() for k, v in model.state_dict().items()}, val_loss, 0
        else:
            waited += 1
            if waited >= settings["patience"]:
                break
    model.load_state_dict(best)
    return model


def lstm_predictor(model, data: Data):
    model.eval()
    sequence = torch.tensor(data.sequence, dtype=torch.float32)
    context = torch.tensor(data.context, dtype=torch.float32)
    with torch.no_grad():
        scaled = model(sequence, context).numpy()
    return data.scaler.inverse(scaled, TARGET_SCALE)


def new_lstm(data: Data, seed: int) -> LSTMForecaster:
    torch.manual_seed(seed)
    return LSTMForecaster(data.context.shape[1], data.config["lstm"])


def train_index(data: Data, tasks: list[str]) -> np.ndarray:
    rows = data.rows
    return np.flatnonzero((rows["fold"] == "train").to_numpy() & rows["task"].isin(tasks).to_numpy())


def fit_ridge(data: Data, index: np.ndarray) -> np.ndarray:
    return Ridge(alpha=data.config["ridge"]["alpha"]).fit(data.tabular[index], data.y[index]).predict(data.tabular)


def fit_ridge_lags(data: Data, index: np.ndarray) -> np.ndarray:
    """Control for B2: only TMAT lags and station, the inputs of Figure 1c."""
    lags = [SCALED_COLUMNS.index(f"tmat_lag_{lag}") for lag in bf.TMAT_LAGS]
    stations = [len(SCALED_COLUMNS) + RAW_COLUMNS.index(c) for c in bf.STATION_FEATURES]
    inputs = data.tabular[:, lags + stations]
    return Ridge(alpha=data.config["ridge"]["alpha"]).fit(inputs[index], data.y[index]).predict(inputs)


def fit_lightgbm(data: Data, index: np.ndarray) -> np.ndarray:
    import lightgbm as lgb  # imported here so LSTM workers never load a second OpenMP runtime

    model = lgb.LGBMRegressor(**data.config["lightgbm"], random_state=DETERMINISTIC_SEED, deterministic=True, force_row_wise=True, n_jobs=1, verbose=-1)
    return model.fit(data.tabular[index], data.y[index]).predict(data.tabular)


def checkpoints(data: Data) -> list[pd.Timestamp]:
    """First fit at the end of the T1 training fold, then every RETRAIN_EVERY_DAYS."""
    rows = data.rows
    start = rows.loc[(rows["task"] == TASK_ORDER[0]) & (rows["fold"] == "train"), "target_date"].max()
    last = rows["target_date"].max()
    return list(pd.date_range(start, last, freq=f"{RETRAIN_EVERY_DAYS}D"))


def task_train_ends(data: Data) -> dict[str, pd.Timestamp]:
    train = data.rows[data.rows["fold"] == "train"]
    return train.groupby("task")["target_date"].max().to_dict()


def scheduled_baseline(data: Data, fit_predict) -> tuple[dict, np.ndarray, int]:
    """Retrain from scratch on each checkpoint. Returns snapshots per task, online predictions, adaptations."""
    dates = checkpoints(data)
    snapshots = []
    for as_of in dates:
        available = available_for_training(data.rows, as_of)
        snapshots.append(fit_predict(available.index.to_numpy()))
    online = np.full(len(data.rows), np.nan)
    issued = data.rows["date"]
    for as_of, prediction in zip(dates, snapshots):
        usable = (issued >= as_of).to_numpy()
        online[usable] = prediction[usable]
    ends = task_train_ends(data)
    by_task = {}
    for task in TASK_ORDER:
        position = max(i for i, as_of in enumerate(dates) if as_of <= ends[task])
        by_task[task] = snapshots[position]
    return by_task, online, len(dates) - 1


def f1_score(actual: np.ndarray, predicted: np.ndarray) -> float:
    true_alert, predicted_alert = actual < TMAT_ALERT_M, predicted < TMAT_ALERT_M
    hits = (true_alert & predicted_alert).sum()
    denominator = true_alert.sum() + predicted_alert.sum()
    return float(2 * hits / denominator) if denominator else float("nan")


def r_records(data: Data, method: str, seed: int, by_task: dict[str, np.ndarray]) -> list[dict]:
    records = []
    rows = data.rows
    for after, prediction in by_task.items():
        for task in TASK_ORDER:
            mask = ((rows["task"] == task) & (rows["fold"] == "holdout")).to_numpy() & ~np.isnan(prediction)
            error = prediction[mask] - data.y[mask]
            records.append(
                {
                    "method": method, "seed": seed, "after_task": after, "eval_task": task, "n": int(mask.sum()),
                    "mae": float(np.abs(error).mean()), "rmse": float(np.sqrt((error**2).mean())),
                    "f1_alert": f1_score(data.y[mask], prediction[mask]), "alert_prevalence": float((data.y[mask] < TMAT_ALERT_M).mean()),
                }
            )
    return records


def run_lstm_methods(data: Data, seed: int, max_epochs: int | None = None) -> list[dict]:
    """B3, B4, and B5 for one seed. Each result: method, seed, by_task, online, adaptations, seconds."""
    results = []
    rows = data.rows

    started = time.perf_counter()
    index = train_index(data, TASK_ORDER)
    model = fit_lstm(new_lstm(data, seed), data, index, rows["task"].to_numpy()[index], seed, max_epochs)
    results.append({"method": "B3_joint_lstm", "seed": seed, "by_task": {TASK_ORDER[-1]: lstm_predictor(model, data)},
                    "online": None, "adaptations": 0, "seconds": time.perf_counter() - started})

    started = time.perf_counter()
    model, by_task = new_lstm(data, seed), {}
    online = np.full(len(rows), np.nan)
    previous = None
    for task in TASK_ORDER:
        if previous is not None:
            mask = (rows["task"] == task).to_numpy()
            online[mask] = previous[mask]
        index = train_index(data, [task])
        fit_lstm(model, data, index, np.zeros(len(index)), seed, max_epochs)
        previous = by_task[task] = lstm_predictor(model, data)
    results.append({"method": "B4_finetune_lstm", "seed": seed, "by_task": by_task, "online": online,
                    "adaptations": len(TASK_ORDER) - 1, "seconds": time.perf_counter() - started})

    started = time.perf_counter()
    def fit_predict(index):
        fresh = fit_lstm(new_lstm(data, seed), data, index, np.zeros(len(index)), seed, max_epochs)
        return lstm_predictor(fresh, data)
    by_task, online, adaptations = scheduled_baseline(data, fit_predict)
    results.append({"method": "B5_retrain30d_lstm", "seed": seed, "by_task": by_task, "online": online,
                    "adaptations": adaptations, "seconds": time.perf_counter() - started})
    return results


def run_deterministic_methods(data: Data) -> list[dict]:
    results = []
    rows = data.rows
    seed = DETERMINISTIC_SEED
    static_index = train_index(data, STATIC_TASKS)
    static_online_mask = rows["task"].isin(TASK_ORDER[2:]).to_numpy()

    persistence = rows["tmat_lag_0"].to_numpy()
    results.append({"method": "B1_persistence", "seed": seed, "by_task": dict.fromkeys(TASK_ORDER, persistence),
                    "online": persistence, "adaptations": 0, "seconds": 0.0})
    started = time.perf_counter()
    prediction = fit_ridge_lags(data, static_index)
    results.append({"method": "B2_static_ridge_lags_control", "seed": seed, "by_task": dict.fromkeys(TASK_ORDER[1:], prediction),
                    "online": np.where(static_online_mask, prediction, np.nan), "adaptations": 0, "seconds": time.perf_counter() - started})
    for name, fit in (("ridge", fit_ridge), ("lightgbm", fit_lightgbm)):
        started = time.perf_counter()
        prediction = fit(data, static_index)
        results.append({"method": f"B2_static_{name}", "seed": seed, "by_task": dict.fromkeys(TASK_ORDER[1:], prediction),
                        "online": np.where(static_online_mask, prediction, np.nan), "adaptations": 0, "seconds": time.perf_counter() - started})
        started = time.perf_counter()
        results.append({"method": f"B3_joint_{name}", "seed": seed, "by_task": {TASK_ORDER[-1]: fit(data, train_index(data, TASK_ORDER))},
                        "online": None, "adaptations": 0, "seconds": time.perf_counter() - started})
        started = time.perf_counter()
        by_task, online, adaptations = scheduled_baseline(data, lambda index, fit=fit: fit(data, index))
        results.append({"method": f"B5_retrain30d_{name}", "seed": seed, "by_task": by_task, "online": online,
                        "adaptations": adaptations, "seconds": time.perf_counter() - started})
    return results


def cell_mae(data: Data, prediction: np.ndarray, mask: np.ndarray) -> pd.Series:
    """Mean absolute error per station x task cell over the rows in mask."""
    rows = data.rows[mask]
    return pd.Series(np.abs(prediction[mask] - data.y[mask]), index=rows.index).groupby([rows["station"], rows["task"]]).mean()


def skill_tests(data: Data, results: list[dict]) -> pd.DataFrame:
    """Skill against persistence per method, paired by station x task cell, errors averaged over seeds."""
    persistence = data.rows["tmat_lag_0"].to_numpy()
    holdout = (data.rows["fold"] == "holdout").to_numpy()
    scopes = {"final_holdout": lambda r: (r["by_task"][TASK_ORDER[-1]], holdout), "online": lambda r: (r["online"], None)}
    records = []
    for method in dict.fromkeys(r["method"] for r in results):
        if method == "B1_persistence":
            continue
        runs = [r for r in results if r["method"] == method]
        for scope, pick in scopes.items():
            if scope == "final_holdout" and TASK_ORDER[-1] not in runs[0]["by_task"]:
                continue
            if scope == "online" and runs[0]["online"] is None:
                continue
            cells = []
            for run in runs:
                prediction, base_mask = pick(run)
                mask = ~np.isnan(prediction) if base_mask is None else base_mask & ~np.isnan(prediction)
                cells.append(cell_mae(data, prediction, mask))
            model = pd.concat(cells, axis=1).mean(axis=1)
            reference = cell_mae(data, persistence, mask).reindex(model.index)
            skill = 1 - model / reference
            if (model == reference).all():
                p_value = 1.0
            else:
                p_value = wilcoxon(model, reference).pvalue if len(model) >= 6 else np.nan
            material = bool(skill.median() >= MATERIAL_SKILL["median_cell_skill_min"] and p_value < MATERIAL_SKILL["wilcoxon_p_max"])
            records.append({
                "method": method, "scope": scope, "n_cells": len(model), "median_cell_skill": skill.median(),
                "mean_cell_skill": skill.mean(), "cells_better_than_persistence": int((skill > 0).sum()),
                "wilcoxon_p": p_value, "material_skill": material,
            })
    return pd.DataFrame(records)


def summarize_run(records: pd.DataFrame, persistence: pd.DataFrame) -> dict:
    """Continual-learning metrics on error (protocol 3.4), where lower is better."""
    matrix = records.pivot(index="after_task", columns="eval_task", values="mae").reindex(index=TASK_ORDER, columns=TASK_ORDER)
    last = TASK_ORDER[-1]
    final = records[records["after_task"] == last]
    forgetting, backward, forward = [], [], []
    for position, task in enumerate(TASK_ORDER[:-1]):
        history = matrix[task].iloc[position:-1].dropna()
        if len(history):
            forgetting.append(matrix.loc[last, task] - history.min())  # positive means error grew
            if not np.isnan(matrix.loc[task, task]):
                backward.append(matrix.loc[last, task] - matrix.loc[task, task])
    baseline = persistence.set_index("eval_task")["mae"]
    for position, task in enumerate(TASK_ORDER[1:], start=1):
        ahead = matrix.iloc[position - 1, position]
        if not np.isnan(ahead):
            forward.append(ahead - baseline[task])  # negative means better than persistence
    reference = baseline.reindex(final["eval_task"]).to_numpy()
    return {
        "final_skill": float(np.mean(1 - final["mae"].to_numpy() / reference)),
        "final_mae": final["mae"].mean(), "final_rmse": final["rmse"].mean(), "final_f1_alert": final["f1_alert"].mean(),
        "forgetting": np.mean(forgetting) if forgetting else np.nan, "backward_transfer": np.mean(backward) if backward else np.nan,
        "forward_vs_persistence": np.mean(forward) if forward else np.nan,
        "mae_T4_holdout": final.loc[final["eval_task"] == "T4_kemarau_2019", "mae"].item(),
    }


def online_metrics(data: Data, online: np.ndarray | None) -> dict:
    """Online MAE per task, and skill against persistence on the same rows."""
    metrics = {}
    persistence = data.rows["tmat_lag_0"].to_numpy()
    for number, task in enumerate(TASK_ORDER, start=1):
        mask = (data.rows["task"] == task).to_numpy()
        if online is not None:
            mask &= ~np.isnan(online)
        if online is None or not mask.any():
            metrics[f"mae_online_T{number}"] = metrics[f"skill_online_T{number}"] = np.nan
            continue
        model = float(np.abs(online[mask] - data.y[mask]).mean())
        metrics[f"mae_online_T{number}"] = model
        metrics[f"skill_online_T{number}"] = 1 - model / float(np.abs(persistence[mask] - data.y[mask]).mean())
    return metrics


def load_data(horizon: int | None = None, config: dict | None = None) -> Data:
    """Rows from features.csv (t+7), or rebuilt in memory for another horizon by the same builder."""
    missing = [p for p in (FEATURES_CSV, GWT_CLEAN_CSV, CHIRPS_CSV) if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Belum ada: {', '.join(map(str, missing))}. Jalankan langkah 01 dan 02 dulu.")
    gwt = pd.read_csv(GWT_CLEAN_CSV, parse_dates=["date"])
    chirps = pd.read_csv(CHIRPS_CSV, parse_dates=["date"])
    if horizon is None:
        rows = pd.read_csv(FEATURES_CSV, parse_dates=["date", "target_date"])
    else:
        rain, climatology = bf.load_rain_inputs(gwt)
        rows = bf.build_features(gwt, rain, climatology, horizon)
    rows = add_folds(rows).sort_values(["target_date", "station"]).reset_index(drop=True)
    scaler = FrozenScaler.fit(rows, SCALED_COLUMNS)
    return Data(rows, build_sequences(rows, gwt, chirps), scaler, config)


def _seed_worker(seed: int, max_epochs: int | None) -> list[dict]:
    data = load_data()
    return run_lstm_methods(data, seed, max_epochs)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    parser.add_argument("--max-epochs", type=int, default=None, help="only for smoke runs; results are then not valid")
    args = parser.parse_args()
    try:
        data = load_data()
    except FileNotFoundError as err:
        print(f"GAGAL: {err}", file=sys.stderr)
        return 1

    data.scaler.save(NORMALIZATION_JSON)
    results = run_deterministic_methods(data)
    with ProcessPoolExecutor(max_workers=min(5, len(args.seeds))) as pool:
        for batch in pool.map(_seed_worker, args.seeds, [args.max_epochs] * len(args.seeds)):
            results.extend(batch)

    if args.max_epochs is not None:
        print(f"Run asap selesai untuk {len(results)} hasil; tidak ada yang ditulis.")
        return 0

    records, summaries, predictions = [], [], []
    persistence = pd.DataFrame(r_records(data, "B1_persistence", DETERMINISTIC_SEED, {TASK_ORDER[-1]: data.rows["tmat_lag_0"].to_numpy()}))
    for result in results:
        method_records = pd.DataFrame(r_records(data, result["method"], result["seed"], result["by_task"]))
        records.append(method_records)
        metrics = {
            "method": result["method"], "seed": result["seed"], "n_adaptations": result["adaptations"],
            **summarize_run(method_records, persistence), **online_metrics(data, result["online"]), "runtime_seconds": result["seconds"],
        }
        summaries.append(metrics)
        log_experiment(f"baseline_cut-t_{result['method']}_seed{result['seed']}", CONFIG, result["seed"], {k: v for k, v in metrics.items() if k not in ("method", "seed")}, result["seconds"])
        if result["online"] is not None:
            predictions.append(pd.DataFrame({"method": result["method"], "seed": result["seed"], "station": data.rows["station"], "date": data.rows["date"], "prediction_m": result["online"]}).dropna())

    all_records = pd.concat(records, ignore_index=True)
    all_records.to_csv(R_CSV, index=False, float_format="%.5f")
    per_run = pd.DataFrame(summaries)
    columns = [c for c in per_run.columns if c not in ("method", "seed", "runtime_seconds")]
    summary = per_run.groupby("method", sort=False)[columns].agg(["mean", "std"])
    summary.columns = [f"{name}_{stat}" for name, stat in summary.columns]
    summary.insert(0, "n_seeds", per_run.groupby("method", sort=False)["seed"].nunique())
    summary.reset_index().to_csv(SUMMARY_CSV, index=False, float_format="%.5f")
    skill_tests(data, results).to_csv(SKILL_CSV, index=False, float_format="%.5f")
    pd.concat(predictions, ignore_index=True).to_csv(PREDICTIONS_CSV, index=False, float_format="%.5f", date_format="%Y-%m-%d")
    view = per_run.groupby("method", sort=False)[["final_skill", "final_mae", "skill_online_T4", "mae_online_T4", "n_adaptations"]].mean()
    print(view.round(4).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
