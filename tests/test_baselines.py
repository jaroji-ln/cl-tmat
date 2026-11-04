import numpy as np
import pandas as pd
import pytest

from common import CHIRPS_CSV, FEATURES_CSV, ROOT
from conftest import load_script
from normalization import FrozenScaler
from splits import TASK_ORDER, available_for_training

SUMMARY_CSV = ROOT / "results" / "baseline_summary.csv"


@pytest.fixture(scope="module")
def module():
    return load_script("03_baselines.py")


@pytest.fixture(scope="module")
def data(module):
    if not (FEATURES_CSV.exists() and CHIRPS_CSV.exists()):
        pytest.skip("features.csv atau CHIRPS belum ada")
    return module.load_data()[0]


def test_holdout_is_later_than_train_within_every_task(data):
    for task in TASK_ORDER:
        rows = data.rows[data.rows["task"] == task]
        train, holdout = rows[rows["fold"] == "train"], rows[rows["fold"] == "holdout"]
        assert len(train) and len(holdout)
        assert train["target_date"].max() < holdout["target_date"].min()


def test_scaler_uses_only_the_t1_training_fold(module, data):
    changed = data.rows.copy()
    mask = ~((changed["task"] == TASK_ORDER[0]) & (changed["fold"] == "train"))
    changed.loc[mask, "rain_deficit_90d"] += 1000.0
    other = FrozenScaler.fit(changed, module.SCALED_COLUMNS)
    assert other.means == data.scaler.means and other.stds == data.scaler.stds


def test_scheduled_training_sees_only_observed_targets_and_no_holdout(module, data):
    for as_of in module.checkpoints(data):
        available = available_for_training(data.rows, as_of)
        assert (available["target_date"] <= as_of).all()
        assert (available["fold"] == "train").all()


def test_online_prediction_comes_from_a_model_older_than_the_forecast(module, data):
    seen = []
    def fit_predict(index):
        seen.append(data.rows.loc[index, "target_date"].max())
        return np.full(len(data.rows), float(len(seen)))
    _, online, adaptations = module.scheduled_baseline(data, fit_predict)
    dates = module.checkpoints(data)
    assert adaptations == len(dates) - 1
    used = ~np.isnan(online)
    positions = online[used].astype(int) - 1
    assert (pd.DatetimeIndex(seen)[positions] < pd.DatetimeIndex(data.rows.loc[used, "target_date"])).all()
    assert (pd.DatetimeIndex(dates)[positions] <= pd.DatetimeIndex(data.rows.loc[used, "date"])).all()


def test_sequences_end_at_t_minus_1_and_match_lags(data):
    lag1 = data.scaler.transform(data.rows["tmat_lag_1"], "tmat_lag_1")
    lag30 = data.scaler.transform(data.rows["tmat_lag_30"], "tmat_lag_1")
    assert np.allclose(data.sequence[:, -1, 0], lag1, atol=1e-4)
    assert np.allclose(data.sequence[:, 0, 0], lag30, atol=1e-4)


def test_lstm_is_deterministic_for_a_seed(module, data):
    index = module.train_index(data, TASK_ORDER[:1])
    runs = []
    for _ in range(2):
        model = module.fit_lstm(module.new_lstm(data, 3), data, index, np.zeros(len(index)), 3, max_epochs=2)
        runs.append(module.lstm_predictor(model, data))
    assert np.array_equal(runs[0], runs[1])


def test_persistence_scores_by_hand(module):
    assert module.f1_score(np.array([-0.5, -0.5, -0.1]), np.array([-0.5, -0.1, -0.1])) == pytest.approx(2 / 3)


def test_forgetting_is_positive_when_error_grows(module):
    records = []
    for after in TASK_ORDER:
        for task in TASK_ORDER:
            value = 0.2
            if task == TASK_ORDER[0]:
                value = 0.1 if after == TASK_ORDER[0] else 0.3
            records.append({"after_task": after, "eval_task": task, "mae": value, "rmse": value, "f1_alert": 1.0})
    frame = pd.DataFrame(records)
    persistence = pd.DataFrame({"eval_task": TASK_ORDER, "mae": 0.25})
    summary = module.summarize_run(frame, persistence)
    assert summary["forgetting"] == pytest.approx(0.2 / 4)
    assert summary["backward_transfer"] == pytest.approx(0.2 / 4)
    assert summary["forward_vs_persistence"] == pytest.approx(-0.05)


def test_summary_covers_every_baseline_with_five_lstm_seeds():
    summary = pd.read_csv(SUMMARY_CSV).set_index("method")
    for method in ["B1_persistence", "B2_static_ridge", "B2_static_lightgbm", "B3_joint_lstm", "B4_finetune_lstm", "B5_retrain30d_lstm"]:
        assert method in summary.index
    assert (summary.loc[summary.index.str.contains("lstm"), "n_seeds"] == 5).all()


def test_persistence_is_rebuilt_from_data(module, data):
    frame = pd.DataFrame(module.r_records(data, "B1", 0, {TASK_ORDER[-1]: data.rows["tmat_lag_1"].to_numpy()}))
    summary = pd.read_csv(SUMMARY_CSV).set_index("method")
    assert frame["mae"].mean() == pytest.approx(summary.loc["B1_persistence", "final_mae_mean"], abs=1e-4)
    assert frame["n"].tolist() == [112, 296, 396, 66, 42]
