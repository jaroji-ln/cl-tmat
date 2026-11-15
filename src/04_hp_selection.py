"""Hyperparameter selection per horizon (protocol v4, rules 2.2 and 2.4.2), one round, then locked.

Grid: config/hp_grid.yaml (32 configurations per family, locked by tag horizon-rule-locked).
Only training folds of T1 and T2 are used:
  fit      = T1 training fold + first 70% of the T2 training fold (by target date)
  validate = last 30% of the T2 training fold
Criterion: validation MAE; ties go to the first configuration in grid order. LSTM uses seed 0.
Nothing from T3-T5 or any holdout is read by the models here.

Output: config/models.yaml (chosen configuration per horizon and family),
results/hp_selection.csv (validation MAE of every configuration), one row per horizon in results/experiments.csv
"""
import importlib.util
import itertools
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from common import ROOT

GRID_YAML = ROOT / "config" / "hp_grid.yaml"
RULE_YAML = ROOT / "config" / "horizon_rule.yaml"
MODELS_YAML = ROOT / "config" / "models.yaml"
SELECTION_CSV = ROOT / "results" / "hp_selection.csv"
SELECTION_SEED = 0
FIT_FRACTION_OF_T2 = 0.7


def _load(filename: str, name: str):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(filename))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


bl = _load("03_baselines.py", "baselines")


def read_locked(path: Path) -> dict:
    config = yaml.safe_load(path.read_text())
    if config.get("status") != "locked":
        raise ValueError(f"{path} belum berstatus locked")
    return config


def grid_configs(grid: dict) -> dict[str, list[dict]]:
    ridge_text = grid["ridge"]["alpha"]
    if ridge_text != "logspace(-3, 3, 32)":
        raise ValueError(f"Grid Ridge tak dikenal: {ridge_text}")
    ridge = [{"alpha": float(a)} for a in np.logspace(-3, 3, 32)]
    def product(section: dict) -> list[dict]:
        keys = list(section)
        return [dict(zip(keys, values)) for values in itertools.product(*(section[k] for k in keys))]
    configs = {"ridge": ridge, "lightgbm": product(grid["lightgbm"]), "lstm": product(grid["lstm"])}
    for family, items in configs.items():
        if len(items) != 32:
            raise ValueError(f"{family}: {len(items)} konfigurasi, bukan 32")
    return configs


def selection_split(data) -> tuple[np.ndarray, np.ndarray]:
    """Indices of fit and validation rows, both inside the training folds of T1 and T2."""
    rows = data.rows
    train = (rows["fold"] == "train").to_numpy()
    t1 = train & (rows["task"] == bl.TASK_ORDER[0]).to_numpy()
    t2 = train & (rows["task"] == bl.TASK_ORDER[1]).to_numpy()
    dates = rows.loc[t2, "target_date"].sort_values()
    cut = dates.iloc[int(len(dates) * FIT_FRACTION_OF_T2)]
    late = t2 & (rows["target_date"] >= cut).to_numpy()
    fit = np.flatnonzero(t1 | (t2 & ~late))
    validate = np.flatnonzero(late)
    if rows["target_date"].iloc[fit].max() >= rows["target_date"].iloc[validate].min():
        raise AssertionError("Target fit tumpang tindih dengan validasi")
    return fit, validate


def full_config(family: str, settings: dict) -> dict:
    config = {k: (dict(v) if isinstance(v, dict) else v) for k, v in bl.CONFIG.items()}
    config[family] = {**config[family], **settings}
    return config


_DATA = {}


def horizon_data(horizon: int):
    """Rows and inputs per horizon, built once per process; only the configuration changes."""
    if horizon not in _DATA:
        _DATA[horizon] = bl.load_data(horizon)
    return _DATA[horizon]


def score(horizon: int, family: str, settings: dict) -> float:
    data = horizon_data(horizon)
    data.config = full_config(family, settings)
    fit, validate = selection_split(data)
    if family == "ridge":
        prediction = bl.fit_ridge(data, fit)
    elif family == "lightgbm":
        prediction = bl.fit_lightgbm(data, fit)
    else:
        groups = data.rows["task"].to_numpy()[fit]
        model = bl.fit_lstm(bl.new_lstm(data, SELECTION_SEED), data, fit, groups, SELECTION_SEED)
        prediction = bl.lstm_predictor(model, data)
    return float(np.abs(prediction[validate] - data.y[validate]).mean())


def _lstm_worker(job: tuple[int, int, dict]) -> tuple[int, int, float, float]:
    horizon, number, settings = job
    started = time.perf_counter()
    return horizon, number, score(horizon, "lstm", settings), time.perf_counter() - started


def main() -> int:
    rule, grid = read_locked(RULE_YAML), read_locked(GRID_YAML)
    horizons = rule["horizons_days"]
    configs = grid_configs(grid)
    records = []
    started = time.perf_counter()
    for horizon in horizons:
        data = horizon_data(horizon)
        fit, validate = selection_split(data)
        persistence = float(np.abs(data.rows["tmat_lag_0"].to_numpy()[validate] - data.y[validate]).mean())
        records.append({"horizon_days": horizon, "family": "persistence", "config_number": -1, "config": "{}",
                        "n_fit": len(fit), "n_validate": len(validate), "validation_mae": persistence})
        for family in ("ridge", "lightgbm"):
            for number, settings in enumerate(configs[family]):
                records.append({"horizon_days": horizon, "family": family, "config_number": number, "config": repr(settings),
                                "n_fit": len(fit), "n_validate": len(validate), "validation_mae": score(horizon, family, settings)})
        print(f"h={horizon}: Ridge dan LightGBM selesai", flush=True)
    sizes = {r["horizon_days"]: (r["n_fit"], r["n_validate"]) for r in records}
    jobs = [(h, n, s) for h in horizons for n, s in enumerate(configs["lstm"])]
    with ProcessPoolExecutor(max_workers=7) as pool:
        for horizon, number, mae, _ in pool.map(_lstm_worker, jobs):
            records.append({"horizon_days": horizon, "family": "lstm", "config_number": number, "config": repr(configs["lstm"][number]),
                            "n_fit": sizes[horizon][0], "n_validate": sizes[horizon][1], "validation_mae": mae})
    table = pd.DataFrame(records).sort_values(["horizon_days", "family", "config_number"]).reset_index(drop=True)
    SELECTION_CSV.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(SELECTION_CSV, index=False, float_format="%.6f")

    chosen = {}
    for horizon in horizons:
        chosen[horizon] = {}
        for family in ("ridge", "lightgbm", "lstm"):
            part = table[(table["horizon_days"] == horizon) & (table["family"] == family)]
            best = part.loc[part["validation_mae"].idxmin()]  # idxmin returns the first minimum
            chosen[horizon][family] = {**configs[family][int(best["config_number"])], "validation_mae": round(float(best["validation_mae"]), 6)}
    document = {
        "status": "locked",
        "selected_by": "src/04_hp_selection.py",
        "grid": "config/hp_grid.yaml",
        "selection_data": "fit T1 train fold + first 70% of T2 train fold; validate last 30% of T2 train fold",
        "lstm_selection_seed": SELECTION_SEED,
        "fixed_lstm_settings": {k: v for k, v in bl.CONFIG["lstm"].items() if k not in grid["lstm"]},
        "horizons": chosen,
    }
    MODELS_YAML.write_text("# Written by src/04_hp_selection.py. Do not edit by hand.\n" + yaml.safe_dump(document, sort_keys=False))
    elapsed = time.perf_counter() - started
    for horizon in horizons:
        bl.log_experiment(f"hp_selection_h{horizon}", {"grid": grid, "selection": document["selection_data"]}, SELECTION_SEED,
                          {f"{family}_{key}": value for family, settings in chosen[horizon].items() for key, value in settings.items()}, elapsed)
    print(table.groupby(["horizon_days", "family"])["validation_mae"].min().unstack().round(4).to_string())
    print(f"Ditulis: {MODELS_YAML}, {SELECTION_CSV} ({elapsed:.0f} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
