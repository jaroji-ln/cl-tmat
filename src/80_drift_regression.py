"""Regression check of the static-model drift table in the research brief.

Ridge on TMAT lags only, 7-day horizon, targets in 2018 for training and 2019 for testing.
The configuration is fixed here and is not tuned towards the table. Both per-station and
panel fits run because the brief does not say which produced it; per-station exists only
for this comparison (research models stay panel, rule 4). The table predates is_suspect, so
it is compared on data with_suspect; suspect_excluded is the result to cite.
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

from common import DRIFT_CSV, DRIFT_TABLE, EXPERIMENTS_CSV, GWT_CLEAN_CSV, SEED, STATIONS
from splits import split_by_target_date

ALPHA = 1.0
MAE_TOLERANCE_M = 0.02
CONFIG = {
    "model": "Ridge", "alpha": ALPHA, "scaler": "StandardScaler fit on train", "horizon_days": 7,
    "features": "tmat lags 1,2,3,5,7,14,21,30", "train": "target_date < 2019-01-01",
    "pre_drought": "target_date 2019-01-01..2019-06-30", "drought": "target_date 2019-07-01..2019-10-31",
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
    for station, (ref_pre, ref_drought, ref_ratio) in DRIFT_TABLE.items():
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
                    "brief_mae_pre_m": ref_pre, "brief_mae_drought_m": ref_drought, "brief_ratio": ref_ratio,
                    "max_abs_diff_m": max(abs(mae_pre - ref_pre), abs(mae_drought - ref_drought)),
                }
            )
    return pd.DataFrame(records)


def evaluate(gwt: pd.DataFrame) -> pd.DataFrame:
    return pd.concat(
        [evaluate_rows(gwt.assign(is_suspect=False), "with_suspect"), evaluate_rows(gwt, "suspect_excluded")],
        ignore_index=True,
    )


def log_experiment(result: pd.DataFrame, seconds: float) -> None:
    """Upsert one row per experiment id so reruns do not grow the log."""
    row = {
        "experiment_id": "drift_regression_ridge_lags", "config": json.dumps(CONFIG, sort_keys=True), "seed": SEED,
        "metrics": json.dumps(result.round(4).to_dict("records"), sort_keys=True), "runtime_seconds": round(seconds, 2),
    }
    log = pd.DataFrame([row])
    if EXPERIMENTS_CSV.exists():
        previous = pd.read_csv(EXPERIMENTS_CSV)
        log = pd.concat([previous[previous["experiment_id"] != row["experiment_id"]], log], ignore_index=True)
    log.to_csv(EXPERIMENTS_CSV, index=False)


def main() -> int:
    if not GWT_CLEAN_CSV.exists():
        print(f"GAGAL: {GWT_CLEAN_CSV} belum ada. Jalankan src/01_prepare_gwt.py dulu.", file=sys.stderr)
        return 1
    started = time.perf_counter()
    result = evaluate(pd.read_csv(GWT_CLEAN_CSV, parse_dates=["date"]))
    result.to_csv(DRIFT_CSV, index=False, float_format="%.4f")
    log_experiment(result, time.perf_counter() - started)
    print(result.drop(columns=["brief_ratio"]).round(3).to_string(index=False))
    worst = result[result["data"] == "with_suspect"].groupby("mode")["max_abs_diff_m"].max()
    print(f"\nSelisih terbesar terhadap tabel (toleransi {MAE_TOLERANCE_M} m):\n{worst.round(3).to_string()}")
    if (worst > MAE_TOLERANCE_M).all():
        print("TIDAK TEREPRODUKSI: laporkan apa adanya, jangan menyetel konfigurasi agar cocok.", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
