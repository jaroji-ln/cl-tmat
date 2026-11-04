"""Shared constants and paths for the cl-tmat pipeline."""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_XLSX = ROOT / "data" / "raw" / "mmc1.xlsx"
COORDS_CSV = ROOT / "data" / "external" / "brg_station_coordinates.csv"
CHIRPS_CSV = ROOT / "data" / "external" / "chirps_stations.csv"
CLIMATOLOGY_CSV = ROOT / "data" / "external" / "chirps_climatology.csv"
GWT_CLEAN_CSV = ROOT / "data" / "processed" / "gwt_clean.csv"
FEATURES_CSV = ROOT / "data" / "processed" / "features.csv"
EXPERIMENTS_CSV = ROOT / "results" / "experiments.csv"
DRIFT_CSV = ROOT / "results" / "drift_regression.csv"

SEED = 42
STATIONS = ["BRG3", "BRG4", "BRG5", "BRG6", "BRG17", "BRG18", "BRG19", "BRG20"]
PEATLANDS = ["Batanghari", "Kubu Raya"]
HORIZON_DAYS = 7
MAX_INTERP_DAYS = 5

# (label, first day, last day, usable station-days stated in the research brief)
REGIMES = [
    ("T1_transisi_2018", "2018-04-08", "2018-06-30", 649),
    ("T2_kemarau_2018", "2018-07-01", "2018-10-31", 984),
    ("T3_basah_201819", "2018-11-01", "2019-06-30", 1215),
    ("T4_kemarau_2019", "2019-07-01", "2019-10-31", 381),
    ("T5_pemulihan_2019", "2019-11-01", "2019-12-31", 136),
]

# Station: (MAE pre-drought m, MAE 2019 drought m, ratio) as stated in the brief.
DRIFT_TABLE = {
    "BRG3": (0.051, 0.298, 5.84),
    "BRG6": (0.057, 0.144, 2.53),
    "BRG5": (0.059, 0.117, 1.98),
    "BRG20": (0.053, 0.088, 1.66),
    "BRG19": (0.043, 0.059, 1.37),
}


def regime_of(dates: pd.Series) -> pd.Series:
    """Label each date with its regime; dates outside every regime become NaN."""
    labels = pd.Series(pd.NA, index=dates.index, dtype="object")
    for label, start, end, _ in REGIMES:
        labels[(dates >= start) & (dates <= end)] = label
    return labels
