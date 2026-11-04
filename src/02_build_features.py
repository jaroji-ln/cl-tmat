"""Build the 7-day-ahead panel feature table.

A row dated t is a forecast whose inputs stop at t-1 and whose target is TMAT at t+7.
`task` follows the target date, so a training target never falls in a test regime.
Rows are dropped when the target is missing or imputed, when TMAT at t-1 is imputed
(its interpolation anchor lies in the future), or when any feature is missing.

Output: data/processed/features.csv
"""
import sys

import numpy as np
import pandas as pd

from common import (
    CHIRPS_CSV,
    CLIMATOLOGY_CSV,
    FEATURES_CSV,
    GWT_CLEAN_CSV,
    HORIZON_DAYS,
    PEATLANDS,
    STATIONS,
    regime_of,
)

TMAT_LAGS = [1, 2, 3, 5, 7, 14, 21, 30]
RAIN_WINDOWS = [3, 7, 14, 30, 60]
DEFICIT_WINDOW = 90
DRY_DAY_MM = 1.0

TMAT_FEATURES = [f"tmat_lag_{lag}" for lag in TMAT_LAGS] + ["tmat_diff_1", "tmat_diff_7", "tmat_ma_7", "tmat_ma_30"]
RAIN_FEATURES = ["rain_1d"] + [f"rain_sum_{w}d" for w in RAIN_WINDOWS] + ["dry_spell_days", "rain_deficit_90d"]
SEASON_FEATURES = ["doy_sin", "doy_cos"]
STATION_FEATURES = [f"station_{s}" for s in STATIONS]
PEATLAND_FEATURES = ["peatland_" + p.replace(" ", "_") for p in PEATLANDS]
FEATURE_COLUMNS = TMAT_FEATURES + RAIN_FEATURES + SEASON_FEATURES + STATION_FEATURES + PEATLAND_FEATURES


def tmat_features(group: pd.DataFrame) -> pd.DataFrame:
    """TMAT inputs and target for one station on a gap-free daily calendar."""
    past = group["tmat_m"].shift(1)
    out = pd.DataFrame(index=group.index)
    for lag in TMAT_LAGS:
        out[f"tmat_lag_{lag}"] = group["tmat_m"].shift(lag)
    out["tmat_diff_1"] = past - group["tmat_m"].shift(2)
    out["tmat_diff_7"] = past - group["tmat_m"].shift(8)
    out["tmat_ma_7"] = past.rolling(7, min_periods=7).mean()
    out["tmat_ma_30"] = past.rolling(30, min_periods=30).mean()
    out["last_input_imputed"] = group["is_imputed"].shift(1, fill_value=False).astype(bool)
    out["n_imputed_inputs"] = group["is_imputed"].astype(int).shift(1).rolling(30, min_periods=1).sum().fillna(0).astype(int)
    out["target_date"] = group["date"] + pd.Timedelta(days=HORIZON_DAYS)
    out["target_tmat_m"] = group["tmat_m"].shift(-HORIZON_DAYS)
    out["target_is_imputed"] = group["is_imputed"].shift(-HORIZON_DAYS, fill_value=False).astype(bool)
    return out


def add_tmat_columns(gwt: pd.DataFrame) -> pd.DataFrame:
    parts = [pd.concat([group, tmat_features(group)], axis=1) for _, group in gwt.groupby("station", sort=False)]
    return pd.concat(parts, ignore_index=True)


def filter_rows(df: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    keep = df["target_tmat_m"].notna() & ~df["target_is_imputed"] & ~df["last_input_imputed"]
    return df[keep & df[feature_columns].notna().all(axis=1)].reset_index(drop=True)


def rain_features(chirps: pd.DataFrame, climatology: pd.DataFrame) -> pd.DataFrame:
    """Rain inputs per station-date, all ending at t-1, from complete daily series."""
    normals = climatology.set_index(["station", "doy"])["precip_clim_mm"]
    parts = []
    for station, group in chirps.groupby("station", sort=False):
        group = group.sort_values("date")
        rain = group["precip_mm"].reset_index(drop=True)
        past = rain.shift(1)
        out = pd.DataFrame({"station": station, "date": group["date"].to_numpy(), "rain_1d": past})
        for window in RAIN_WINDOWS:
            out[f"rain_sum_{window}d"] = past.rolling(window, min_periods=window).sum()
        dry = (rain < DRY_DAY_MM).astype(int)
        spell = dry.groupby((dry == 0).cumsum()).cumsum()
        # A spell already running when the series starts has unknown length.
        spell[(dry == 0).cumsum() == 0] = np.nan
        out["dry_spell_days"] = spell.shift(1)
        normal_daily = normals.reindex(pd.MultiIndex.from_arrays([[station] * len(group), group["date"].dt.dayofyear])).to_numpy()
        out["rain_sum_90d"] = past.rolling(DEFICIT_WINDOW, min_periods=DEFICIT_WINDOW).sum()
        out["rain_normal_90d"] = pd.Series(normal_daily).shift(1).rolling(DEFICIT_WINDOW, min_periods=DEFICIT_WINDOW).sum()
        # Positive means drier than the 1988-2017 normal.
        out["rain_deficit_90d"] = out["rain_normal_90d"] - out["rain_sum_90d"]
        parts.append(out)
    return pd.concat(parts, ignore_index=True)


def static_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    angle = 2 * np.pi * out["date"].dt.dayofyear / 365.25
    out["doy_sin"], out["doy_cos"] = np.sin(angle), np.cos(angle)
    for station, column in zip(STATIONS, STATION_FEATURES):
        out[column] = (out["station"] == station).astype(int)
    for peatland, column in zip(PEATLANDS, PEATLAND_FEATURES):
        out[column] = (out["peatland"] == peatland).astype(int)
    return out


def build_features(gwt: pd.DataFrame, chirps: pd.DataFrame, climatology: pd.DataFrame) -> pd.DataFrame:
    df = add_tmat_columns(gwt)
    df = df.merge(rain_features(chirps, climatology), on=["station", "date"], how="left", validate="one_to_one")
    df = filter_rows(static_features(df), FEATURE_COLUMNS)
    df["task"] = regime_of(df["target_date"])
    id_columns = ["date", "target_date", "station", "peatland", "task", "n_imputed_inputs"]
    return df[id_columns + FEATURE_COLUMNS + ["rain_sum_90d", "rain_normal_90d", "target_tmat_m"]]


def load_rain_inputs(gwt: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load CHIRPS exports, refusing anything that would silently create gaps."""
    missing = [p for p in (CHIRPS_CSV, CLIMATOLOGY_CSV) if not p.exists()]
    if missing:
        names = ", ".join(str(p) for p in missing)
        raise FileNotFoundError(
            f"{names} belum ada. Jalankan src/00_fetch_chirps.js di Earth Engine (lihat README) "
            "dan letakkan hasilnya di data/external/. Hujan sintetis tidak dipakai."
        )
    chirps = pd.read_csv(CHIRPS_CSV, parse_dates=["date"])
    climatology = pd.read_csv(CLIMATOLOGY_CSV).astype({"doy": int})
    needed_start = gwt["date"].min() - pd.Timedelta(days=DEFICIT_WINDOW)
    for station in STATIONS:
        rain = chirps[chirps["station"] == station].sort_values("date")
        if rain.empty or rain["precip_mm"].isna().any() or (rain["precip_mm"] < 0).any():
            raise ValueError(f"CHIRPS {station}: kosong, NaN, atau negatif")
        if rain["date"].duplicated().any() or (rain["date"].diff().dt.days.dropna() != 1).any():
            raise ValueError(f"CHIRPS {station}: kalender harian tidak lengkap atau ganda")
        if rain["date"].min() > needed_start or rain["date"].max() < gwt["date"].max():
            raise ValueError(f"CHIRPS {station}: harus mencakup {needed_start.date()} s.d. {gwt['date'].max().date()}")
        clim = climatology[climatology["station"] == station]
        if sorted(clim["doy"]) != list(range(1, 367)) or clim["precip_clim_mm"].isna().any():
            raise ValueError(f"Klimatologi {station}: harus lengkap untuk hari-dalam-tahun 1-366")
    return chirps, climatology


def main() -> int:
    if not GWT_CLEAN_CSV.exists():
        print(f"GAGAL: {GWT_CLEAN_CSV} belum ada. Jalankan src/01_prepare_gwt.py dulu.", file=sys.stderr)
        return 1
    gwt = pd.read_csv(GWT_CLEAN_CSV, parse_dates=["date"])
    try:
        chirps, climatology = load_rain_inputs(gwt)
    except (FileNotFoundError, ValueError) as err:
        print(f"GAGAL: {err}", file=sys.stderr)
        return 1
    features = build_features(gwt, chirps, climatology)
    features.to_csv(FEATURES_CSV, index=False, float_format="%.6f", date_format="%Y-%m-%d")
    print(f"{len(features)} baris, {len(FEATURE_COLUMNS)} fitur")
    print(features.groupby("task").size().to_string())
    print(f"Ditulis: {FEATURES_CSV}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
