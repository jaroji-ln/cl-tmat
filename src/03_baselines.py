"""Phase 2 baselines B1-B5 (protocol 2.1). Hyperparameters are fixed, never tuned.

Each task is split once by target date. R[i, j] is the holdout error of task j for the model
available after training on task i. mae_online is the error on every row of a task using only
the model that existed when that forecast was issued, so it covers drought onset too.

Output: results/baseline_R.csv, results/baseline_summary.csv, results/normalization_t1.json,
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
SUMMARY_CSV = ROOT / "results" / "baseline_summary.csv"
NORMALIZATION_JSON = ROOT / "results" / "normalization_t1.json"
PREDICTIONS_CSV = ROOT / "data" / "processed" / "baseline_predictions.csv"

SEEDS = [0, 1, 2, 3, 4]
DETERMINISTIC_SEED = 0
RETRAIN_EVERY_DAYS = 30
SEQUENCE_DAYS = 30
TMAT_ALERT_M = -0.4  # PP 57/2016
STATIC_TASKS = TASK_ORDER[:2]
TARGET_SCALE = "tmat_lag_1"
RAW_COLUMNS = bf.SEASON_FEATURES + bf.STATION_FEATURES + bf.PEATLAND_FEATURES  # bounded, no statistics needed
SCALED_COLUMNS = bf.TMAT_FEATURES + bf.RAIN_FEATURES
CONTEXT_COLUMNS = bf.RAIN_FEATURES + RAW_COLUMNS
CONFIG = {
    "ridge": {"alpha": 1.0},
    "lightgbm": {"n_estimators": 300, "learning_rate": 0.05, "num_leaves": 15, "min_child_samples": 10},
    "lstm": {"layers": 2, "units": 64, "sequence_days": SEQUENCE_DAYS, "residual_on": "tmat_lag_1", "lr": 1e-3,
             "batch": 64, "max_epochs": 60, "patience": 8, "validation": "last 15% by target date", "grad_clip": 1.0},
    "holdout": "last 30% of each task by target date",
    "normalization": "T1 training fold, frozen; season and one-hot columns stay raw",
}


class Data:
    """Scaled arrays aligned with rows."""

    def __init__(self, rows: pd.DataFrame, sequences: np.ndarray, scaler: FrozenScaler):
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
    """TMAT and rain for t-30..t-1 per row. Suspect values are masked and must not be reached."""
    tmat = {s: g.set_index("date")["tmat_m"].mask(g.set_index("date")["is_suspect"]) for s, g in gwt.groupby("station")}
    rain = {s: g.set_index("date")["precip_mm"] for s, g in chirps.groupby("station")}
    output = np.empty((len(rows), SEQUENCE_DAYS, 2))
    for i, (station, date) in enumerate(zip(rows["station"], rows["date"])):
        window = pd.date_range(date - pd.Timedelta(days=SEQUENCE_DAYS), periods=SEQUENCE_DAYS)
        output[i, :, 0] = tmat[station].reindex(window).to_numpy()
        output[i, :, 1] = rain[station].reindex(window).to_numpy()
    if np.isnan(output).any():
        raise ValueError("Jendela input memuat NaN")
    return output


class LSTMForecaster(torch.nn.Module):
    def __init__(self, context_size: int):
        super().__init__()
        settings = CONFIG["lstm"]
        self.lstm = torch.nn.LSTM(2, settings["units"], settings["layers"], batch_first=True)
        self.head = torch.nn.Linear(settings["units"] + context_size, 1)

    def forward(self, sequence, context):
        encoded, _ = self.lstm(sequence)
        change = self.head(torch.cat([encoded[:, -1], context], dim=1)).squeeze(1)
        return sequence[:, -1, 0] + change  # the last observed level, plus a learned change


def validation_mask(rows: pd.DataFrame) -> np.ndarray:
    dates = rows["target_date"].sort_values()
    cut = dates.iloc[int(len(dates) * 0.85)]
    return (rows["target_date"] >= cut).to_numpy()


def fit_lstm(model, data: Data, index: np.ndarray, groups: np.ndarray, seed: int, max_epochs: int | None = None):
    """Train on rows in index. Validation is the last 15% of each group, so nothing shuffles across time."""
    settings = CONFIG["lstm"]
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
    return LSTMForecaster(data.context.shape[1])


def train_index(data: Data, tasks: list[str]) -> np.ndarray:
    rows = data.rows
    return np.flatnonzero((rows["fold"] == "train").to_numpy() & rows["task"].isin(tasks).to_numpy())


def fit_ridge(data: Data, index: np.ndarray) -> np.ndarray:
    return Ridge(alpha=CONFIG["ridge"]["alpha"]).fit(data.tabular[index], data.y[index]).predict(data.tabular)


def fit_ridge_lags(data: Data, index: np.ndarray) -> np.ndarray:
    """Control for B2: only TMAT lags and station, the inputs of Figure 1c."""
    lags = [SCALED_COLUMNS.index(f"tmat_lag_{lag}") for lag in bf.TMAT_LAGS]
    stations = [len(SCALED_COLUMNS) + RAW_COLUMNS.index(c) for c in bf.STATION_FEATURES]
    inputs = data.tabular[:, lags + stations]
    return Ridge(alpha=CONFIG["ridge"]["alpha"]).fit(inputs[index], data.y[index]).predict(inputs)


def fit_lightgbm(data: Data, index: np.ndarray) -> np.ndarray:
    import lightgbm as lgb  # imported here so LSTM workers never load a second OpenMP runtime

    model = lgb.LGBMRegressor(**CONFIG["lightgbm"], random_state=DETERMINISTIC_SEED, deterministic=True, force_row_wise=True, n_jobs=1, verbose=-1)
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


def run_deterministic_methods(data: Data, tmat_at_t: np.ndarray) -> list[dict]:
    results = []
    rows = data.rows
    seed = DETERMINISTIC_SEED
    static_index = train_index(data, STATIC_TASKS)
    static_online_mask = rows["task"].isin(TASK_ORDER[2:]).to_numpy()

    persistence = rows["tmat_lag_1"].to_numpy()
    results.append({"method": "B1_persistence", "seed": seed, "by_task": dict.fromkeys(TASK_ORDER, persistence),
                    "online": persistence, "adaptations": 0, "seconds": 0.0})
    results.append({"method": "B1_persistence_tmat_t", "seed": seed, "by_task": dict.fromkeys(TASK_ORDER, tmat_at_t),
                    "online": tmat_at_t, "adaptations": 0, "seconds": 0.0})
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
    return {
        "final_mae": final["mae"].mean(), "final_rmse": final["rmse"].mean(), "final_f1_alert": final["f1_alert"].mean(),
        "forgetting": np.mean(forgetting) if forgetting else np.nan, "backward_transfer": np.mean(backward) if backward else np.nan,
        "forward_vs_persistence": np.mean(forward) if forward else np.nan,
        "mae_T4_holdout": final.loc[final["eval_task"] == "T4_kemarau_2019", "mae"].item(),
    }


def online_metrics(data: Data, online: np.ndarray | None) -> dict:
    metrics = {}
    for number, task in enumerate(TASK_ORDER, start=1):
        mask = (data.rows["task"] == task).to_numpy()
        values = None if online is None else online[mask] - data.y[mask]
        usable = None if values is None else values[~np.isnan(values)]
        metrics[f"mae_online_T{number}"] = float(np.abs(usable).mean()) if usable is not None and len(usable) else np.nan
    return metrics


def load_data() -> tuple[Data, np.ndarray]:
    missing = [p for p in (FEATURES_CSV, GWT_CLEAN_CSV, CHIRPS_CSV) if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Belum ada: {', '.join(map(str, missing))}. Jalankan langkah 01 dan 02 dulu.")
    rows = add_folds(pd.read_csv(FEATURES_CSV, parse_dates=["date", "target_date"]))
    rows = rows.sort_values(["target_date", "station"]).reset_index(drop=True)
    gwt = pd.read_csv(GWT_CLEAN_CSV, parse_dates=["date"])
    chirps = pd.read_csv(CHIRPS_CSV, parse_dates=["date"])
    scaler = FrozenScaler.fit(rows, SCALED_COLUMNS)
    observed = gwt[~gwt["is_imputed"] & ~gwt["is_suspect"]].set_index(["station", "date"])["tmat_m"]
    tmat_at_t = observed.reindex(pd.MultiIndex.from_frame(rows[["station", "date"]])).to_numpy()
    return Data(rows, build_sequences(rows, gwt, chirps), scaler), tmat_at_t


def _seed_worker(seed: int, max_epochs: int | None) -> list[dict]:
    data, _ = load_data()
    return run_lstm_methods(data, seed, max_epochs)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    parser.add_argument("--max-epochs", type=int, default=None, help="only for smoke runs; results are then not valid")
    args = parser.parse_args()
    try:
        data, tmat_at_t = load_data()
    except FileNotFoundError as err:
        print(f"GAGAL: {err}", file=sys.stderr)
        return 1

    data.scaler.save(NORMALIZATION_JSON)
    results = run_deterministic_methods(data, tmat_at_t)
    with ProcessPoolExecutor(max_workers=min(5, len(args.seeds))) as pool:
        for batch in pool.map(_seed_worker, args.seeds, [args.max_epochs] * len(args.seeds)):
            results.extend(batch)

    if args.max_epochs is not None:
        print(f"Run asap selesai untuk {len(results)} hasil; tidak ada yang ditulis.")
        return 0

    records, summaries, predictions = [], [], []
    persistence = pd.DataFrame(r_records(data, "B1_persistence", DETERMINISTIC_SEED, {TASK_ORDER[-1]: data.rows["tmat_lag_1"].to_numpy()}))
    for result in results:
        method_records = pd.DataFrame(r_records(data, result["method"], result["seed"], result["by_task"]))
        records.append(method_records)
        metrics = {
            "method": result["method"], "seed": result["seed"], "n_adaptations": result["adaptations"],
            **summarize_run(method_records, persistence), **online_metrics(data, result["online"]), "runtime_seconds": result["seconds"],
        }
        summaries.append(metrics)
        log_experiment(f"baseline_{result['method']}_seed{result['seed']}", CONFIG, result["seed"], {k: v for k, v in metrics.items() if k not in ("method", "seed")}, result["seconds"])
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
    pd.concat(predictions, ignore_index=True).to_csv(PREDICTIONS_CSV, index=False, float_format="%.5f", date_format="%Y-%m-%d")
    view = per_run.groupby("method", sort=False)[["final_mae", "final_rmse", "forgetting", "mae_T4_holdout", "mae_online_T4", "n_adaptations"]].mean()
    print(view.round(4).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
