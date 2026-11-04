import pandas as pd
import pytest

from common import MAX_INTERP_DAYS, REGIMES


def test_missing_raw_file_fails_clearly(prepare, tmp_path):
    with pytest.raises(FileNotFoundError, match="tidak membuat data sintetis"):
        prepare.load_raw(tmp_path / "mmc1.xlsx")


def test_no_duplicate_dates_after_cleaning(gwt):
    assert not gwt.duplicated(["station", "date"]).any()


def test_brg5_duplicate_is_merged_by_mean(raw, gwt):
    assert raw[(raw.station == "BRG5") & (raw.date == "2019-03-21")].shape[0] == 2
    value = gwt[(gwt.station == "BRG5") & (gwt.date == "2019-03-21")]["tmat_m"].item()
    assert value == pytest.approx((-0.362 - 0.316) / 2)


@pytest.mark.parametrize("station", ["BRG3", "BRG5", "BRG20"])
def test_daily_calendar_is_complete(gwt, station):
    assert (gwt[gwt.station == station]["date"].diff().dropna().dt.days == 1).all()


def test_units_stay_in_meters(gwt):
    assert gwt["tmat_m"].dropna().between(-2.0, 0.5).all()


def test_imputation_is_limited_and_never_overwrites_observations(raw, gwt):
    merged = raw.groupby(["station", "date"])["tmat_m"].mean()
    observed = gwt.set_index(["station", "date"])
    common = observed.index.intersection(merged.dropna().index)
    assert not observed.loc[common, "is_imputed"].any()
    assert (observed.loc[common, "tmat_m"] - merged.loc[common]).abs().max() < 1e-12
    for _, group in gwt.groupby("station"):
        runs = group["is_imputed"].ne(group["is_imputed"].shift()).cumsum()
        longest = group["is_imputed"].groupby(runs).sum().max()
        assert longest <= MAX_INTERP_DAYS


def test_suspect_flag_covers_only_the_brg3_segment(gwt):
    suspect = gwt[gwt["is_suspect"]]
    assert set(suspect["station"]) == {"BRG3"}
    assert suspect["date"].min() == pd.Timestamp("2019-07-10")
    assert suspect["date"].max() == pd.Timestamp("2019-08-27")
    assert len(suspect) == 49
    assert suspect["tmat_m"].notna().all()  # kept for transparency, excluded downstream


def test_objective_rule_finds_only_the_flagged_straight_run(prepare, gwt):
    runs = prepare.straight_runs(gwt)
    assert runs[["station", "days", "ends_station_record"]].to_dict("records") == [
        {"station": "BRG3", "days": 45, "ends_station_record": True}
    ]
    assert str(runs["start"].item()) == "2019-07-14" and str(runs["end"].item()) == "2019-08-27"
    assert runs["residual_sd_mm"].item() < 0.5


@pytest.mark.parametrize("label,start,end,expected", REGIMES)
def test_regime_day_counts_match_brief(gwt, label, start, end, expected):
    usable = gwt[gwt["tmat_m"].notna() & (gwt["regime"] == label)]
    assert len(usable) == expected, f"{label}: pipeline {len(usable)} hari, tabel brief {expected}"
    assert usable["date"].between(pd.Timestamp(start), pd.Timestamp(end)).all()
