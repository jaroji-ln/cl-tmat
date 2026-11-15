import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from common import CHIRPS_CSV, FEATURES_CSV, ROOT
from conftest import load_script
from splits import TASK_ORDER


@pytest.fixture(scope="module")
def hp():
    return load_script("04_hp_selection.py")


@pytest.fixture(scope="module")
def batch():
    return load_script("05_horizon_batch.py")


def test_rule_and_grid_are_locked_and_match_protocol(hp):
    rule = hp.read_locked(hp.RULE_YAML)
    assert rule["horizons_days"] == [3, 7, 14, 21, 30]
    assert "B4_finetune_lstm" not in rule["candidates"] and len(rule["candidates"]) == 5
    assert rule["online_tasks"] == TASK_ORDER[2:]
    configs = hp.grid_configs(hp.read_locked(hp.GRID_YAML))
    assert {k: len(v) for k, v in configs.items()} == {"ridge": 32, "lightgbm": 32, "lstm": 32}


def test_target_follows_horizon(features_module, gwt, rain_stub):
    chirps, climatology = rain_stub
    for horizon in (3, 30):
        rows = features_module.build_features(gwt, chirps, climatology, horizon)
        assert ((rows["target_date"] - rows["date"]).dt.days == horizon).all()
        sample = rows.iloc[len(rows) // 2]
        observed = gwt[(gwt["station"] == sample["station"]) & (gwt["date"] == sample["target_date"])]["tmat_m"].item()
        assert sample["target_tmat_m"] == pytest.approx(observed, abs=1e-9)


def test_default_horizon_rebuild_matches_features_csv(features_module, gwt):
    if not (FEATURES_CSV.exists() and CHIRPS_CSV.exists()):
        pytest.skip("features.csv atau CHIRPS belum ada")
    chirps, climatology = features_module.load_rain_inputs(gwt)
    rebuilt = features_module.build_features(gwt, chirps, climatology, 7)
    stored = pd.read_csv(FEATURES_CSV, parse_dates=["date", "target_date"])
    pd.testing.assert_frame_equal(rebuilt.reset_index(drop=True), stored, check_dtype=False, atol=1e-5)


def test_selection_uses_only_t1_t2_training_folds(hp):
    if not (FEATURES_CSV.exists() and CHIRPS_CSV.exists()):
        pytest.skip("features.csv atau CHIRPS belum ada")
    data = hp.horizon_data(14)
    fit, validate = hp.selection_split(data)
    used = data.rows.iloc[np.concatenate([fit, validate])]
    assert (used["fold"] == "train").all() and set(used["task"]) == set(TASK_ORDER[:2])
    assert data.rows["target_date"].iloc[fit].max() < data.rows["target_date"].iloc[validate].min()
    t2_train = ((data.rows["task"] == TASK_ORDER[1]) & (data.rows["fold"] == "train")).sum()
    assert abs(len(validate) - 0.3 * t2_train) <= 8  # same-date rows stay together


def test_short_window_ignores_older_days(hp):
    bl = hp.bl
    settings = {**bl.CONFIG["lstm"], "units": 8, "layers": 1, "dropout": 0.2, "sequence_days": 14}
    torch.manual_seed(0)
    model = bl.LSTMForecaster(3, settings).eval()
    sequence, context = torch.randn(4, 30, 2), torch.randn(4, 3)
    changed = sequence.clone()
    changed[:, :16] += 10.0
    with torch.no_grad():
        assert torch.allclose(model(sequence, context), model(changed, context))


def test_holm_matches_definition(batch):
    p = np.array([0.01, 0.04, 0.03, 0.2, 0.005])
    expected = np.array([0.04, 0.09, 0.09, 0.2, 0.025])
    np.testing.assert_allclose(batch.holm(p), expected)


def test_one_sided_wilcoxon_direction(batch):
    reference = pd.Series(np.linspace(1.0, 2.0, 10))
    assert batch.one_sided_p(reference * 0.8, reference) < 0.01
    assert batch.one_sided_p(reference * 1.2, reference) > 0.99
    assert batch.one_sided_p(reference, reference) == 1.0


def test_models_yaml_covers_every_horizon_when_present(hp):
    if not hp.MODELS_YAML.exists():
        pytest.skip("config/models.yaml belum dibuat")
    models = yaml.safe_load(hp.MODELS_YAML.read_text())
    assert models["status"] == "locked" and sorted(models["horizons"]) == [3, 7, 14, 21, 30]
    grid = hp.grid_configs(hp.read_locked(hp.GRID_YAML))
    for chosen in models["horizons"].values():
        for family, settings in chosen.items():
            assert {k: v for k, v in settings.items() if k != "validation_mae"} in grid[family]


@pytest.fixture(scope="module")
def diag():
    return load_script("06_diagnostics.py")


def test_expanding_scaler_only_sees_the_past(hp):
    if not (FEATURES_CSV.exists() and CHIRPS_CSV.exists()):
        pytest.skip("features.csv atau CHIRPS belum ada")
    from normalization import ExpandingScaler, FrozenScaler

    rows = hp.horizon_data(7).rows
    columns = hp.bl.SCALED_COLUMNS
    scaler = ExpandingScaler.fit(rows, columns)
    frozen = FrozenScaler.fit(rows, columns)
    t1_end = rows.loc[(rows["task"] == TASK_ORDER[0]) & (rows["fold"] == "train"), "date"].max()
    early = (rows["date"] <= t1_end).to_numpy()
    assert np.allclose(scaler.means["rain_deficit_90d"][early], frozen.means["rain_deficit_90d"])
    cut = pd.Timestamp("2019-07-01")
    changed = rows.copy()
    changed.loc[changed["date"] > cut, columns] += 1000.0
    other = ExpandingScaler.fit(changed, columns)
    past = (rows["date"] <= cut).to_numpy()
    for column in ("tmat_lag_0", "rain_deficit_90d", "dry_spell_days"):
        assert np.allclose(other.means[column][past], scaler.means[column][past])
        assert np.allclose(other.stds[column][past], scaler.stds[column][past])


def test_change_target_adds_back_tmat_t(hp):
    if not (FEATURES_CSV.exists() and CHIRPS_CSV.exists()):
        pytest.skip("features.csv atau CHIRPS belum ada")
    from sklearn.linear_model import Ridge

    data = hp.horizon_data(7)
    index = hp.bl.train_index(data, TASK_ORDER[:2])
    data.config = {**hp.bl.CONFIG, "target": "change"}
    prediction = hp.bl.fit_ridge(data, index)
    base = data.rows["tmat_lag_0"].to_numpy()
    expected = Ridge(alpha=1.0).fit(data.tabular[index], data.y[index] - base[index]).predict(data.tabular) + base
    data.config = hp.bl.CONFIG
    np.testing.assert_allclose(prediction, expected)


def test_crossing_f1(diag):
    now = np.array([-0.3, -0.3, -0.5, -0.35])
    actual = np.array([-0.45, -0.3, -0.6, -0.41])
    predicted = np.array([-0.42, -0.45, -0.6, -0.3])
    f1, events, flagged = diag.crossing_f1(now, actual, predicted)
    assert (events, flagged) == (2, 2) and f1 == pytest.approx(0.5)
    assert diag.crossing_f1(now, actual, now)[0] == 0.0  # persistence never predicts a crossing


@pytest.fixture(scope="module")
def cl():
    return load_script("07_cl_methods.py")


def test_reservoir_and_ewc_penalty(cl, hp):
    if not (FEATURES_CSV.exists() and CHIRPS_CSV.exists()):
        pytest.skip("features.csv atau CHIRPS belum ada")
    data = hp.horizon_data(7)
    learner = cl.Learner(data, "CL_replay30d_lstm", 0, buffer_rows=10)
    learner._reservoir(np.arange(100))
    assert len(learner.buffer) == 10 and learner.seen == 100 and set(learner.buffer) <= set(range(100))
    ewc = cl.Learner(data, "CL_ewc30d_lstm", 0, ewc_lambda=100.0)
    ewc.model = hp.bl.new_lstm(data, 0)
    ewc._accumulate_fisher(np.arange(20))
    assert float(ewc._penalty()(ewc.model)) == 0.0  # zero at the anchor
    with torch.no_grad():
        next(ewc.model.parameters()).add_(0.1)
    assert float(ewc._penalty()(ewc.model)) > 0.0


def test_damped_trend_formula():
    module = load_script("08_damped_trend.py")
    rows = pd.DataFrame({"tmat_lag_0": [-0.5], "tmat_diff_7": [-0.07]})
    expected = -0.5 + (-0.01) * sum(0.9**k for k in range(1, 15))
    assert module.damped_trend(rows, 14, 0.9)[0] == pytest.approx(expected)
