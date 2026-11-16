#!/usr/bin/env bash
# Rebuild every result, table, and figure of the article from the raw data.
#
# Prerequisites (see README.md):
#   data/raw/mmc1.xlsx                      Taufik et al. (2022), Data in Brief 41:107903 (not redistributed here)
#   data/external/chirps_stations.csv       CHIRPS export from src/00_fetch_chirps.js (committed)
#   data/external/chirps_climatology.csv    CHIRPS 1988-2017 climatology export (committed)
#   data/external/taufik2022_fig1.jpg       high-resolution Fig. 1 of Taufik et al. (2022), CC BY 4.0 (committed)
#
# Not rerun here: src/00_fetch_chirps.js (Google Earth Engine, manual) and src/82_verify_bilinear.py
# (downloads one CHIRPS raster); their outputs are committed and checked by tests/.
#
# Run time: several hours on a laptop CPU; scripts 04-07 use 7 worker processes.
# Every script appends a row to results/experiments.csv; all other outputs are deterministic.
set -euo pipefail
cd "$(dirname "$0")/src"

run() { echo "== $(date '+%F %T') $*"; python "$@"; }

run extract_station_coords.py   # data/external/brg_station_coordinates.csv, results/coord_validation.csv
run 01_prepare_gwt.py           # data/processed/gwt_clean.csv, results/straight_runs.csv
run 02_build_features.py        # data/processed/features.csv
run 03_baselines.py             # results/baseline_*.csv, results/normalization_t1.json
run 80_drift_regression.py      # results/drift_regression.csv
run 81_data_audit.py            # results/data_audit.csv
run 90_figure1.py               # figures/fig1_*, results/degradation_ratios.csv
run 04_hp_selection.py          # config/models.yaml, results/hp_selection.csv
run 05_horizon_batch.py         # results/horizon_*.csv, results/horizon_decision.yaml
run 06_diagnostics.py           # results/diagnostics_*
run 07_cl_methods.py            # results/cl_*
run 08_damped_trend.py          # results/damped_trend*.csv
run 91_final_outputs.py         # figures/fig2-5, results/tables/
run 92_manuscript_tables.py     # docs/manuscript/tables.md

cd .. && python -m pytest -q tests
echo "== done $(date '+%F %T')"
