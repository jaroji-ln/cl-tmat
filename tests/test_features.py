import numpy as np
import pandas as pd
import pytest

from common import FEATURES_CSV, HORIZON_DAYS, regime_of

LEAK_CUTOFFS = ["2018-09-15", "2019-03-21", "2019-04-02", "2019-04-05", "2019-06-08", "2019-07-10"]


@pytest.fixture(scope="module")
def table(features_module, gwt, rain_stub):
    return features_module.build_features(gwt, *rain_stub)


def test_no_imputed_target_survives(table, gwt):
    flags = gwt.set_index(["station", "date"])["is_imputed"]
    targets = pd.MultiIndex.from_frame(table[["station", "target_date"]].rename(columns={"target_date": "date"}))
    assert not flags.loc[targets].any()


def test_imputed_days_exist_so_the_filter_is_exercised(gwt):
    assert gwt["is_imputed"].sum() > 0


def test_suspect_values_reach_neither_targets_nor_inputs(table, gwt):
    flags = gwt.set_index(["station", "date"])["is_suspect"]
    targets = pd.MultiIndex.from_frame(table[["station", "target_date"]].rename(columns={"target_date": "date"}))
    assert not flags.loc[targets].any()
    first_suspect = gwt.loc[gwt["is_suspect"], "date"].min()
    assert (table.loc[table["station"] == "BRG3", "target_date"] < first_suspect).all()


def test_target_is_tmat_seven_days_ahead(table, gwt):
    values = gwt.set_index(["station", "date"])["tmat_m"]
    keys = pd.MultiIndex.from_arrays([table["station"], table["date"] + pd.Timedelta(days=HORIZON_DAYS)])
    assert np.allclose(values.loc[keys].to_numpy(), table["target_tmat_m"].to_numpy())


def test_lag_is_calendar_based(table, gwt):
    values = gwt.set_index(["station", "date"])["tmat_m"]
    keys = pd.MultiIndex.from_arrays([table["station"], table["date"] - pd.Timedelta(days=30)])
    assert np.allclose(values.loc[keys].to_numpy(), table["tmat_lag_30"].to_numpy())


def test_task_follows_target_date(table):
    assert (table["task"] == regime_of(table["target_date"])).all()
    assert (table["target_date"] - table["date"]).dt.days.eq(HORIZON_DAYS).all()


def test_no_missing_feature_and_full_one_hot(features_module, table):
    assert table[features_module.FEATURE_COLUMNS].notna().all().all()
    assert table[features_module.STATION_FEATURES].sum(axis=1).eq(1).all()
    assert table[features_module.PEATLAND_FEATURES].sum(axis=1).eq(1).all()


def perturb_after(raw: pd.DataFrame, cutoff: str) -> pd.DataFrame:
    changed = raw.copy()
    changed.loc[changed["date"] > cutoff, "tmat_m"] += 1.0
    return changed


@pytest.mark.parametrize("cutoff", LEAK_CUTOFFS)
def test_future_tmat_cannot_change_past_features(prepare, features_module, raw, cutoff):
    columns = features_module.TMAT_FEATURES
    def rows(frame):
        built = features_module.filter_rows(features_module.add_tmat_columns(prepare.clean(frame)), columns)
        return built[built["date"] <= pd.Timestamp(cutoff) + pd.Timedelta(days=1)].set_index(["station", "date"])[columns]
    base, changed = rows(raw), rows(perturb_after(raw, cutoff))
    pd.testing.assert_frame_equal(base, changed)


def test_leak_check_detects_interpolation_anchor_without_the_filter(prepare, features_module, raw):
    # Proves the test above has teeth: without dropping imputed t-1, the future does leak in.
    cutoff = "2019-04-05"
    def rows(frame):
        built = features_module.add_tmat_columns(prepare.clean(frame))
        selected = built[(built["station"] == "BRG5") & (built["date"] == pd.Timestamp(cutoff))]
        return selected["tmat_lag_1"].item()
    assert rows(raw) != pytest.approx(rows(perturb_after(raw, cutoff)))


def test_future_rain_cannot_change_past_features(features_module, rain_stub):
    chirps, climatology = rain_stub
    cutoff = pd.Timestamp("2019-02-01")
    changed = chirps.copy()
    changed.loc[changed["date"] > cutoff, "precip_mm"] = 99.0
    base = features_module.rain_features(chirps, climatology)
    other = features_module.rain_features(changed, climatology)
    keep = base["date"] <= cutoff + pd.Timedelta(days=1)
    pd.testing.assert_frame_equal(base[keep], other[keep])


def test_rain_windows_and_deficit_by_hand(features_module):
    dates = pd.date_range("2018-01-01", periods=120)
    rain = np.zeros(120)
    rain[[100, 101]] = [4.0, 6.0]
    chirps = pd.DataFrame({"station": "BRG3", "date": dates, "precip_mm": rain})
    climatology = pd.DataFrame({"station": "BRG3", "doy": range(1, 367), "precip_clim_mm": 1.5})
    out = features_module.rain_features(chirps, climatology).set_index("date")
    t = dates[102]  # inputs stop at dates[101]
    assert out.loc[t, "rain_1d"] == 6.0
    assert out.loc[t, "rain_sum_3d"] == 10.0
    assert out.loc[t, "rain_normal_90d"] == pytest.approx(135.0)
    assert out.loc[t, "rain_deficit_90d"] == pytest.approx(125.0)  # positive means drier than normal
    assert out.loc[t, "dry_spell_days"] == 0
    assert out.loc[dates[110], "dry_spell_days"] == 8
    assert np.isnan(out.loc[dates[88], "rain_deficit_90d"])


def test_missing_chirps_fails_clearly(features_module, gwt, monkeypatch, tmp_path):
    monkeypatch.setattr(features_module, "CHIRPS_CSV", tmp_path / "chirps_stations.csv")
    with pytest.raises(FileNotFoundError, match="Hujan sintetis tidak dipakai"):
        features_module.load_rain_inputs(gwt)


def test_real_features_file_if_present(features_module):
    if not FEATURES_CSV.exists():
        pytest.skip("features.csv belum ada: menunggu ekspor CHIRPS")
    real = pd.read_csv(FEATURES_CSV, parse_dates=["date", "target_date"])
    assert real[features_module.FEATURE_COLUMNS + ["target_tmat_m"]].notna().all().all()
    assert not real.duplicated(["station", "date"]).any()
