"""Clean the Taufik et al. (2022) groundwater table series.

Merges duplicate dates by mean, reindexes each station to a full daily calendar,
interpolates at most MAX_INTERP_DAYS consecutive days inside observed spans,
flags those days in is_imputed, flags SUSPECT_SEGMENTS in is_suspect, and labels the
hydrological regime.

Output: data/processed/gwt_clean.csv, results/straight_runs.csv
"""
import sys

import numpy as np
import pandas as pd

from common import (
    GWT_CLEAN_CSV,
    MAX_INTERP_DAYS,
    RAW_XLSX,
    STATIONS,
    STRAIGHT_RUNS_CSV,
    SUSPECT_SEGMENTS,
    regime_of,
)

TMAT_RAW = "groundwater table (m)"
STRAIGHT_RUN_MIN_DAYS = 14
STRAIGHT_RUN_SPREAD_M = 0.002


def load_raw(path=RAW_XLSX) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} tidak ditemukan. Letakkan mmc1.xlsx (Taufik dkk. 2022, Data in Brief "
            "41:107903) di data/raw/. Pipeline ini tidak membuat data sintetis."
        )
    raw = pd.read_excel(path, sheet_name="gwt")
    raw.columns = raw.columns.str.strip()  # the sheet has a 'peatland ' column
    return raw.rename(columns={TMAT_RAW: "tmat_m"})[["date", "tmat_m", "station", "peatland"]]


def clean(raw: pd.DataFrame) -> pd.DataFrame:
    merged = raw.groupby(["station", "peatland", "date"], as_index=False)["tmat_m"].mean()
    frames = []
    for (station, peatland), group in merged.groupby(["station", "peatland"]):
        series = group.set_index("date")["tmat_m"]
        series = series.reindex(pd.date_range(series.index.min(), series.index.max(), freq="D"))
        filled = series.interpolate(method="linear", limit=MAX_INTERP_DAYS, limit_area="inside")
        frames.append(
            pd.DataFrame(
                {
                    "date": series.index,
                    "station": station,
                    "peatland": peatland,
                    "tmat_m": filled.to_numpy(),
                    "is_imputed": (series.isna() & filled.notna()).to_numpy(),
                }
            )
        )
    out = pd.concat(frames, ignore_index=True)
    out["is_suspect"] = False
    for station, start, end in SUSPECT_SEGMENTS:
        out.loc[(out["station"] == station) & out["date"].between(start, end), "is_suspect"] = True
    out["regime"] = regime_of(out["date"])
    order = {station: i for i, station in enumerate(STATIONS)}
    return out.sort_values(["station", "date"], key=lambda c: c.map(order) if c.name == "station" else c).reset_index(drop=True)


def straight_runs(cleaned: pd.DataFrame) -> pd.DataFrame:
    """Observed runs of at least STRAIGHT_RUN_MIN_DAYS whose daily changes stay within 2 mm."""
    records = []
    for station, group in cleaned.groupby("station", sort=False):
        series = group.set_index("date")["tmat_m"].where(~group.set_index("date")["is_imputed"])
        changes = series.diff().round(3).to_numpy()
        first = 1
        while first < len(changes):
            last = first
            while (
                not np.isnan(changes[first])
                and last + 1 < len(changes)
                and not np.isnan(changes[last + 1])
                and np.ptp(changes[first : last + 2]) <= STRAIGHT_RUN_SPREAD_M
            ):
                last += 1
            run = series.iloc[first - 1 : last + 1]
            if len(run) >= STRAIGHT_RUN_MIN_DAYS:
                days = np.arange(len(run))
                slope, intercept = np.polyfit(days, run.to_numpy(), 1)
                residual = run.to_numpy() - (slope * days + intercept)
                records.append(
                    {
                        "station": station, "start": run.index[0].date(), "end": run.index[-1].date(), "days": len(run),
                        "slope_mm_per_day": round(slope * 1000, 2), "residual_sd_mm": round(residual.std() * 1000, 2),
                        "change_sd_m": round(float(np.std(changes[first : last + 1], ddof=1)), 5),
                        "ends_station_record": run.index[-1] == series.index[-1],
                    }
                )
            first = last + 1
    return pd.DataFrame(records)


def main() -> int:
    try:
        raw = load_raw()
    except FileNotFoundError as err:
        print(f"GAGAL: {err}", file=sys.stderr)
        return 1
    out = clean(raw)
    GWT_CLEAN_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(GWT_CLEAN_CSV, index=False, float_format="%.6f", date_format="%Y-%m-%d")

    runs = straight_runs(out)
    runs.to_csv(STRAIGHT_RUNS_CSV, index=False)

    used = out[out["tmat_m"].notna()]
    print(f"Baris mentah {len(raw)} -> kalender harian {len(out)}, terpakai {len(used)}, imputasi {int(out['is_imputed'].sum())}")
    print(used.groupby("regime", sort=True).agg(hari=("tmat_m", "size"), imputasi=("is_imputed", "sum")).to_string())
    print(out.groupby("station", sort=False).agg(hari=("date", "size"), nan=("tmat_m", lambda s: int(s.isna().sum())), imputasi=("is_imputed", "sum")).to_string())
    print(f"Hari suspect: {int(out['is_suspect'].sum())}; segmen lurus >= {STRAIGHT_RUN_MIN_DAYS} hari:")
    print(runs.to_string(index=False))
    print(f"Ditulis: {GWT_CLEAN_CSV}, {STRAIGHT_RUNS_CSV}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
