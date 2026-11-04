"""Rebuild every data-quality number quoted in the protocol and the paper.

Output: results/data_audit.csv (metric, value)
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from common import AUDIT_CSV, CHIRPS_CSV, FEATURES_CSV, GWT_CLEAN_CSV, REGIMES

_spec = importlib.util.spec_from_file_location("build_features", Path(__file__).with_name("02_build_features.py"))
bf = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bf)

JUMP_DATE = "2019-07-10"
LINE_START, LINE_END = "2019-07-14", "2019-08-27"
BRG6_START, BRG6_END = "2019-07-23", "2019-08-26"


def line_stats(series: pd.Series) -> tuple[float, float]:
    """Residual sd from a straight line in mm, and the quadratic coefficient."""
    days = np.arange(len(series))
    residual = series.to_numpy() - np.polyval(np.polyfit(days, series.to_numpy(), 1), days)
    return float(residual.std() * 1000), float(np.polyfit(days, series.to_numpy(), 2)[0])


def segment_metrics(gwt: pd.DataFrame, chirps: pd.DataFrame) -> dict:
    observed = {s: g.set_index("date")["tmat_m"].where(~g.set_index("date")["is_imputed"]) for s, g in gwt.groupby("station")}
    brg3 = observed["BRG3"]
    line = brg3[LINE_START:LINE_END]
    line_sd = line.diff().std()
    outside_sd = brg3[:JUMP_DATE].diff().std()
    residual_mm, curvature = line_stats(line)
    metrics = {
        "brg3_jump_m": brg3[JUMP_DATE] - brg3[:JUMP_DATE].iloc[-2],
        "brg3_line_days": len(line),
        "brg3_line_slope_mm_per_day": np.polyfit(np.arange(len(line)), line.to_numpy(), 1)[0] * 1000,
        "brg3_line_residual_mm": residual_mm,
        "brg3_line_curvature": curvature,
        "brg3_line_change_sd_m": line_sd,
        "brg3_outside_change_sd_m": outside_sd,
        "brg3_outside_over_line_sd": outside_sd / line_sd,
    }
    rain = chirps[chirps["station"] == "BRG3"].set_index("date")["precip_mm"]
    metrics["brg3_rain_during_line_mm"] = rain[LINE_START:LINE_END].sum()
    metrics["brg3_rain_jump_day_mm"] = rain[JUMP_DATE]

    window = len(line) - 1
    ratios = []
    for station, series in observed.items():
        changes = series[:"2019-07-09"].diff() if station == "BRG3" else series.diff()
        ratios.append(changes.rolling(window, min_periods=window).std().min() / line_sd)
    metrics["same_window_smoothest_ratio_min"] = np.nanmin(ratios)
    metrics["same_window_smoothest_ratio_max"] = np.nanmax(ratios)

    brg6 = observed["BRG6"][BRG6_START:BRG6_END]
    metrics["brg6_recession_change_sd_m"] = brg6.diff().std()
    metrics["brg6_recession_residual_mm"], metrics["brg6_recession_curvature"] = line_stats(brg6)
    return metrics


def deficit_metrics(features: pd.DataFrame, chirps: pd.DataFrame) -> dict:
    metrics = {}
    for label, *_ in REGIMES:
        rows = features[features["task"] == label]
        metrics[f"{label}_rows"] = len(rows)
        metrics[f"{label}_deficit_mean_mm"] = rows["rain_deficit_90d"].mean()
        metrics[f"{label}_deficit_positive_share"] = (rows["rain_deficit_90d"] > 0).mean()
    t2 = features[features["task"] == "T2_kemarau_2018"].groupby("station")["rain_deficit_90d"].mean()
    for station in ["BRG17", "BRG18", "BRG19", "BRG20"]:
        metrics[f"T2_deficit_mean_{station}_mm"] = t2[station]

    numeric = bf.TMAT_FEATURES + bf.RAIN_FEATURES
    t1 = features[features["task"] == REGIMES[0][0]][numeric]
    t4 = features[features["task"] == "T4_kemarau_2019"][numeric]
    scaled = (t4 - t1.mean()) / t1.std(ddof=0)
    metrics["T4_deficit_z_mean"] = scaled["rain_deficit_90d"].mean()
    metrics["T4_deficit_z_max"] = scaled["rain_deficit_90d"].max()
    metrics["T4_share_rows_any_abs_z_gt_5"] = (scaled.abs() > 5).any(axis=1).mean()
    metrics["T4_max_abs_z"] = scaled.abs().max().max()

    monthly = chirps.assign(month=chirps["date"].dt.to_period("M")).groupby(["month", "station"])["precip_mm"].sum()
    for month in ["2018-10", "2019-10"]:
        values = monthly[month][["BRG17", "BRG18", "BRG19", "BRG20"]]
        metrics[f"kubu_raya_rain_{month}_min_mm"] = values.min()
        metrics[f"kubu_raya_rain_{month}_max_mm"] = values.max()
    return metrics


def main() -> int:
    missing = [p for p in (GWT_CLEAN_CSV, FEATURES_CSV, CHIRPS_CSV) if not p.exists()]
    if missing:
        print(f"GAGAL: belum ada: {', '.join(map(str, missing))}", file=sys.stderr)
        return 1
    gwt = pd.read_csv(GWT_CLEAN_CSV, parse_dates=["date"])
    chirps = pd.read_csv(CHIRPS_CSV, parse_dates=["date"])
    features = pd.read_csv(FEATURES_CSV, parse_dates=["date", "target_date"])
    metrics = {**segment_metrics(gwt, chirps), **deficit_metrics(features, chirps)}
    audit = pd.DataFrame({"metric": list(metrics), "value": [float(v) for v in metrics.values()]})
    audit.to_csv(AUDIT_CSV, index=False, float_format="%.6g")
    print(audit.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
