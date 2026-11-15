"""Closing item 1 (protocol v6, 2.6): Experience Replay and EWC against B5, focus online skill on T4, all H.

Definitions in config/closing.yaml (committed before this ran). LSTM backbone with the locked
per-horizon configuration; same checkpoints and row rule as locked B5; warm-start updates.
  CL_finetune30d_lstm  control: update on new rows only
  CL_replay30d_lstm    new rows + an equal number from a reservoir buffer of past rows
  CL_ewc30d_lstm       new rows + online EWC penalty
Buffer size and lambda are selected inside T1-T2 with the locked selection split (seed 0).
B3, B4 and B5 LSTM are rerun in the same worker as the reference (they must equal the locked batch).

Output: results/cl_selection.csv, results/cl_summary.csv, results/cl_criteria.csv (2.4 table, reference
only: Holm over the 4 methods tested here), results/cl_t4_vs_b5.csv, rows closing_cl_* in results/experiments.csv
"""
import importlib.util
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch
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
hb = _load("05_horizon_batch.py", "horizon_batch")

CLOSING_YAML = ROOT / "config" / "closing.yaml"
SELECTION_CSV = ROOT / "results" / "cl_selection.csv"
SUMMARY_CSV = ROOT / "results" / "cl_summary.csv"
CRITERIA_CSV = ROOT / "results" / "cl_criteria.csv"
T4_CSV = ROOT / "results" / "cl_t4_vs_b5.csv"
MIN_NEW_ROWS = 16
FISHER_ROWS = 512
METHODS = ("CL_finetune30d_lstm", "CL_replay30d_lstm", "CL_ewc30d_lstm")
T4 = "T4_kemarau_2019"


class Learner:
    """One continual learner; update() adds rows that became available."""

    def __init__(self, data, method: str, seed: int, buffer_rows: int = 0, ewc_lambda: float = 0.0):
        self.data, self.method, self.seed = data, method, seed
        self.buffer_rows, self.ewc_lambda = buffer_rows, ewc_lambda
        self.rng = np.random.default_rng(seed)
        self.model, self.used, self.pending = None, set(), []
        self.buffer, self.seen = [], 0
        self.fisher, self.anchor = None, None
        self.updates = 0

    def _reservoir(self, rows: np.ndarray) -> None:
        for row in rows:
            self.seen += 1
            if len(self.buffer) < self.buffer_rows:
                self.buffer.append(int(row))
            else:
                slot = int(self.rng.integers(self.seen))
                if slot < self.buffer_rows:
                    self.buffer[slot] = int(row)

    def _penalty(self):
        if self.fisher is None or self.ewc_lambda == 0:
            return None
        fisher, anchor, scale = self.fisher, self.anchor, self.ewc_lambda / 2

        def penalty(model):
            return scale * sum((fisher[n] * (p - anchor[n]) ** 2).sum() for n, p in model.named_parameters())
        return penalty

    def _accumulate_fisher(self, rows: np.ndarray) -> None:
        data, model = self.data, self.model
        sample = rows if len(rows) <= FISHER_ROWS else self.rng.choice(rows, FISHER_ROWS, replace=False)
        sequence = torch.tensor(data.sequence[sample], dtype=torch.float32)
        context = torch.tensor(data.context[sample], dtype=torch.float32)
        target = torch.tensor(data.y_scaled[sample], dtype=torch.float32)
        totals = {n: torch.zeros_like(p) for n, p in model.named_parameters()}
        model.eval()
        for i in range(len(sample)):
            model.zero_grad()
            loss = (model(sequence[i : i + 1], context[i : i + 1]) - target[i : i + 1]).pow(2).sum()
            loss.backward()
            for n, p in model.named_parameters():
                totals[n] += p.grad.detach() ** 2
        new = {n: t / len(sample) for n, t in totals.items()}
        self.fisher = new if self.fisher is None else {n: self.fisher[n] + new[n] for n in new}
        self.anchor = {n: p.detach().clone() for n, p in model.named_parameters()}

    def update(self, available: np.ndarray) -> bool:
        new = np.array([i for i in available if i not in self.used], dtype=int)
        self.pending = np.unique(np.concatenate([np.asarray(self.pending, dtype=int), new]))
        if self.model is None:
            if len(self.pending) == 0:
                return False
            rows = self.pending
            self.model = bl.fit_lstm(bl.new_lstm(self.data, self.seed), self.data, rows, np.zeros(len(rows)), self.seed)
        elif len(self.pending) >= MIN_NEW_ROWS:
            rows = self.pending
            if self.method == "CL_replay30d_lstm" and self.buffer:
                take = min(len(self.buffer), len(rows))
                rows = np.concatenate([rows, self.rng.choice(np.array(self.buffer), take, replace=False)])
            bl.fit_lstm(self.model, self.data, rows, np.zeros(len(rows)), self.seed, penalty=self._penalty())
            self.updates += 1
        else:
            return False
        fresh = self.pending
        self.used.update(int(i) for i in fresh)
        if self.method == "CL_replay30d_lstm":
            self._reservoir(fresh)
        if self.method == "CL_ewc30d_lstm":
            self._accumulate_fisher(fresh)
        self.pending = []
        return True


def run_cl(data, method: str, seed: int, settings: dict) -> dict:
    """Online predictions, per-task snapshots, and adaptations on the locked B5 schedule."""
    started = time.perf_counter()
    learner = Learner(data, method, seed, **settings)
    dates = bl.checkpoints(data)
    snapshots = []
    for as_of in dates:
        learner.update(bl.available_for_training(data.rows, as_of).index.to_numpy())
        snapshots.append(bl.lstm_predictor(learner.model, data))
    online = np.full(len(data.rows), np.nan)
    for as_of, prediction in zip(dates, snapshots):
        usable = (data.rows["date"] >= as_of).to_numpy()
        online[usable] = prediction[usable]
    ends = bl.task_train_ends(data)
    by_task = {task: snapshots[max(i for i, d in enumerate(dates) if d <= ends[task])] for task in bl.TASK_ORDER}
    return {"method": method, "seed": seed, "by_task": by_task, "online": online, "adaptations": learner.updates,
            "seconds": time.perf_counter() - started}


def selection_score(horizon: int, config: dict, method: str, settings: dict) -> float:
    """Initial fit on the T1 training fold, 30-day updates through the selection fit rows, validate on the rest."""
    data = bl.load_data(horizon, config)
    fit, validate = hp.selection_split(data)
    fit_rows = data.rows.iloc[fit]
    start = data.rows.loc[(data.rows["task"] == bl.TASK_ORDER[0]) & (data.rows["fold"] == "train"), "target_date"].max()
    end = fit_rows["target_date"].max()
    dates = list(pd.date_range(start, end, freq=f"{bl.RETRAIN_EVERY_DAYS}D")) + [end]
    learner = Learner(data, method, hp.SELECTION_SEED, **settings)
    for as_of in dates:
        learner.update(fit_rows.index[fit_rows["target_date"] <= as_of].to_numpy())
    prediction = bl.lstm_predictor(learner.model, data)
    return float(np.abs(prediction[validate] - data.y[validate]).mean())


def _selection_worker(job):
    horizon, config, method, settings = job
    return horizon, method, settings, selection_score(horizon, config, method, settings)


def _run_worker(job):
    horizon, config, seed, chosen = job
    data = bl.load_data(horizon, config)
    results = bl.run_lstm_methods(data, seed)
    for method in METHODS:
        results.append(run_cl(data, method, seed, chosen[method]))
    return horizon, results


def main() -> int:
    closing = hp.read_locked(CLOSING_YAML)["cl_methods"]
    rule = hp.read_locked(hp.RULE_YAML)
    models = hp.read_locked(hp.MODELS_YAML)
    horizons = closing["horizons_days"]
    configs = {h: hb.config_for(models, h) for h in horizons}
    started = time.perf_counter()

    grid = [("CL_replay30d_lstm", {"buffer_rows": int(b)}) for b in closing["selection"]["replay_buffer_rows"]]
    grid += [("CL_ewc30d_lstm", {"ewc_lambda": float(l)}) for l in closing["selection"]["ewc_lambda"]]
    jobs = [(h, configs[h], m, s) for h in horizons for m, s in grid]
    selection = []
    with ProcessPoolExecutor(max_workers=7) as pool:
        for horizon, method, settings, mae in pool.map(_selection_worker, jobs):
            selection.append({"horizon_days": horizon, "method": method, "settings": repr(settings), "validation_mae": mae, **settings})
    selection = pd.DataFrame(selection)
    selection.to_csv(SELECTION_CSV, index=False, float_format="%.6f")
    chosen = {}
    for horizon in horizons:
        chosen[horizon] = {"CL_finetune30d_lstm": {}}
        for method, key in (("CL_replay30d_lstm", "buffer_rows"), ("CL_ewc30d_lstm", "ewc_lambda")):
            part = selection[(selection["horizon_days"] == horizon) & (selection["method"] == method)]
            value = part.loc[part["validation_mae"].idxmin(), key]  # first minimum in grid order
            chosen[horizon][method] = {key: int(value) if key == "buffer_rows" else float(value)}
    print(f"Seleksi CL selesai ({time.perf_counter() - started:.0f} s): {chosen}", flush=True)

    results = {h: [] for h in horizons}
    jobs = [(h, configs[h], seed, chosen[h]) for h in horizons for seed in bl.SEEDS]
    with ProcessPoolExecutor(max_workers=7) as pool:
        for horizon, batch in pool.map(_run_worker, jobs):
            results[horizon].extend(batch)

    reference_rule = {**rule, "candidates": ["B5_retrain30d_lstm", *METHODS]}
    summaries, criteria, t4_rows = [], [], []
    for horizon in horizons:
        data = bl.load_data(horizon, configs[horizon])
        persistence = data.rows["tmat_lag_0"].to_numpy()
        run = results[horizon] + [{"method": "B1_persistence", "seed": 0, "by_task": dict.fromkeys(bl.TASK_ORDER, persistence),
                                   "online": persistence, "adaptations": 0, "seconds": 0.0}]
        config = {**configs[horizon], "cl_settings": chosen[horizon], "closing": "config/closing.yaml"}
        _, summary, crit, _ = hb.evaluate_horizon(horizon, data, run, reference_rule, config, prefix="closing_cl")
        summaries.extend(summary)
        criteria.extend({"label": closing["label"], "holm_over": "B5 LSTM + 3 CL methods (reference only)", **row} for row in crit)
        t4 = (data.rows["task"] == T4).to_numpy()
        def cells(method):
            runs = [r for r in results[horizon] if r["method"] == method]
            return pd.concat([bl.cell_mae(data, r["online"], t4) for r in runs], axis=1).mean(axis=1)
        b5 = cells("B5_retrain30d_lstm")
        for method in METHODS:
            other = cells(method).reindex(b5.index)
            difference = other - b5
            p = 1.0 if (difference == 0).all() else float(wilcoxon(other, b5).pvalue)
            t4_rows.append({"label": closing["label"], "horizon_days": horizon, "method": method, "n_cells": len(b5),
                            "median_mae_difference_vs_b5_m": float(difference.median()), "cells_better_than_b5": int((difference < 0).sum()),
                            "wilcoxon_two_sided_p": p, **{f"settings_{k}": v for k, v in chosen[horizon][method].items()}})
    per_run = pd.DataFrame(summaries)
    columns = [c for c in per_run.columns if c not in ("horizon_days", "method", "seed", "runtime_seconds")]
    summary = per_run.groupby(["horizon_days", "method"], sort=False)[columns].agg(["mean", "std"])
    summary.columns = [f"{name}_{stat}" for name, stat in summary.columns]
    summary.insert(0, "n_seeds", per_run.groupby(["horizon_days", "method"], sort=False)["seed"].nunique())
    summary.insert(0, "label", closing["label"])
    summary.reset_index().to_csv(SUMMARY_CSV, index=False, float_format="%.5f")
    pd.DataFrame(criteria).to_csv(CRITERIA_CSV, index=False, float_format="%.5f")
    pd.DataFrame(t4_rows).to_csv(T4_CSV, index=False, float_format="%.5f")
    view = summary.reset_index().pivot(index="method", columns="horizon_days", values="skill_online_T4_mean")
    print("skill online T4:\n" + view.round(3).to_string())
    print(pd.DataFrame(t4_rows).round(4).to_string(index=False))
    print(f"Selesai dalam {time.perf_counter() - started:.0f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
