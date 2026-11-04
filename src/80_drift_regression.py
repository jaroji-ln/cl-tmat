"""Static-model degradation evidence for Figure 1c.

Ridge on TMAT lags only, 7-day horizon, targets in 2018 for training and 2019 for testing.
panel with suspect_excluded is the result to cite. per_station is for the appendix, and
with_suspect only measures how much the BRG3 segment moves the numbers.

Output: results/drift_regression.csv
"""
import importlib.util
import json
import sys
import time
from pathlib import Path

import pandas as pd
from sklearn.compose import make_column_transformer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from common import DRIFT_CSV, EXPERIMENTS_CSV, GWT_CLEAN_CSV, SEED, STATIONS, SUSPECT_SEGMENTS
from splits import split_by_target_date

ALPHA = 1.0
DROUGHT_STATIONS = ["BRG3", "BRG5", "BRG6", "BRG19", "BRG20"]  # the others have no T4 rows
CONFIG = {
    "model": "Ridge", "alpha": ALPHA, "scaler": "StandardScaler fit on train", "horizon_days": 7,
    "features": "tmat lags 1,2,3,5,7,14,21,30", "train": "target_date < 2019-01-01",
    "pre_drought": "target_date 2019-01-01..2019-06-30", "drought": "target_date 2019-07-01..2019-10-31",
    "suspect_segments": SUSPECT_SEGMENTS,
}

_spec = importlib.util.spec_from_file_location("build_features", Path(__file__).with_name("02_build_features.py"))
bf = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bf)
LAGS = [f"tmat_lag_{lag}" for lag in bf.TMAT_LAGS]


def lag_rows(gwt: pd.DataFrame) -> pd.DataFrame:
    return bf.filter_rows(bf.add_tmat_columns(gwt), LAGS)


def fit_ridge(train: pd.DataFrame, with_station: bool):
    columns = LAGS + (["station"] if with_station else [])
    steps = [(StandardScaler(), LAGS)]
    if with_station:
        steps.append((OneHotEncoder(categories=[STATIONS]), ["station"]))
    model = make_pipeline(make_column_transformer(*steps), Ridge(alpha=ALPHA, random_state=SEED))
    model.fit(train[columns], train["target_tmat_m"])
    return lambda rows: model.predict(rows[columns])


def mae(predict, rows: pd.DataFrame) -> float:
    return float((predict(rows) - rows["target_tmat_m"]).abs().mean())


def evaluate_rows(gwt: pd.DataFrame, data: str) -> pd.DataFrame:
    train, test = split_by_target_date(lag_rows(gwt), "2019-01-01")
    panel = fit_ridge(train, with_station=True)
    records = []
    for station in DROUGHT_STATIONS:
        rows = test[test["station"] == station]
        pre = rows[rows["target_date"] <= "2019-06-30"]
        drought = rows[(rows["target_date"] >= "2019-07-01") & (rows["target_date"] <= "2019-10-31")]
        per_station = fit_ridge(train[train["station"] == station], with_station=False)
        for mode, predict in (("per_station", per_station), ("panel", panel)):
            mae_pre, mae_drought = mae(predict, pre), mae(predict, drought)
            records.append(
                {
                    "data": data, "mode": mode, "station": station, "n_pre": len(pre), "n_drought": len(drought),
                    "mae_pre_m": mae_pre, "mae_drought_m": mae_drought, "ratio": mae_drought / mae_pre,
                }
            )
    return pd.DataFrame(records)


def evaluate(gwt: pd.DataFrame) -> pd.DataFrame:
    return pd.concat(
        [evaluate_rows(gwt.assign(is_suspect=False), "with_suspect"), evaluate_rows(gwt, "suspect_excluded")],
        ignore_index=True,
    )


def log_experiment(experiment_id: str, config: dict, seed: int, metrics, seconds: float, path=EXPERIMENTS_CSV) -> None:
    """Upsert one row per experiment id so reruns do not grow the log."""
    row = {
        "experiment_id": experiment_id, "config": json.dumps(config, sort_keys=True), "seed": seed,
        "metrics": json.dumps(metrics, sort_keys=True), "runtime_seconds": round(seconds, 2),
    }
    log = pd.DataFrame([row])
    if path.exists():
        previous = pd.read_csv(path)
        log = pd.concat([previous[previous["experiment_id"] != experiment_id], log], ignore_index=True)
    log.to_csv(path, index=False)


def main() -> int:
    if not GWT_CLEAN_CSV.exists():
        print(f"GAGAL: {GWT_CLEAN_CSV} belum ada. Jalankan src/01_prepare_gwt.py dulu.", file=sys.stderr)
        return 1
    started = time.perf_counter()
    result = evaluate(pd.read_csv(GWT_CLEAN_CSV, parse_dates=["date"]))
    result.to_csv(DRIFT_CSV, index=False, float_format="%.4f")
    log_experiment("drift_regression_ridge_lags", CONFIG, SEED, result.round(4).to_dict("records"), time.perf_counter() - started)
    print(result.round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
