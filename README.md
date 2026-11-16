# cl-tmat

Code, configuration, logs, and results for the article

> *Persistence is hard to beat: a locked-protocol benchmark of adaptive groundwater-table forecasting in tropical peatlands under El Niño drought* (manuscript in preparation for *Environmental Modelling & Software*).

The study asks whether static, periodically retrained, and continual-learning models keep groundwater-table (GWT) forecasts accurate through a hydrological regime shift, using eight peatland monitoring stations in Jambi and West Kalimantan, Indonesia (April 2018 to December 2019), which include the 2019 El Niño drought. Under a decision rule locked before the horizons were run, no realistic model beat persistence at any horizon from 3 to 30 days.

Working notes for the research team are in Indonesian ([CLAUDE.md](CLAUDE.md), [docs/protokol-riset-cl-tmat.md](docs/protokol-riset-cl-tmat.md)). The manuscript draft is [docs/manuscript/manuscript.md](docs/manuscript/manuscript.md).

## Staged locking

Each stage of the analysis was fixed in an annotated git tag before it was run.

| Tag | Commit | What was fixed |
|---|---|---|
| `horizon-rule-locked` | 440d3df | Decision rule for the main horizon (`config/horizon_rule.yaml`) and the hyperparameter grid (`config/hp_grid.yaml`), before any horizon other than 7 days was run |
| `diagnostics-plan-locked` | 2f2273b | Diagnostics D1–D5 (`config/diagnostics.yaml`), before any diagnostic code was written |
| `analysis-frozen` | 4def636 | End of analysis on this dataset. Afterwards, code changes only for openly reported bug fixes |

The continual-learning and damped-trend analyses were defined in `config/closing.yaml` and committed (2ab77d8) before they were run. The tags show when each decision was fixed; they do not show that the data were unseen (see Section 3.10 of the manuscript).

Commits after `analysis-frozen` change only the manuscript, documentation, `reproduce.sh`, and `src/92_manuscript_tables.py`, which formats existing result files into manuscript tables and computes nothing new.

## Data

| File | Source | In this repository |
|---|---|---|
| `data/raw/mmc1.xlsx` | Taufik et al. (2022), *Data in Brief* 41:107903, https://doi.org/10.1016/j.dib.2022.107903 (supplementary file) | No. Download it from the article and place it in `data/raw/` |
| `data/external/chirps_stations.csv`, `chirps_climatology.csv` | CHIRPS v2.0 daily rainfall, extracted with `src/00_fetch_chirps.js` in Google Earth Engine (bilinear) | Yes |
| `data/external/taufik2022_fig1.jpg` | Fig. 1 of Taufik et al. (2022), CC BY 4.0, used to estimate six station coordinates | Yes |
| `data/external/brg_station_coordinates.csv` | Built by `src/extract_station_coords.py` | Yes |

To re-extract CHIRPS: open https://code.earthengine.google.com, paste `src/00_fetch_chirps.js`, run it, check that the console reports 761 daily images, start the two export tasks, and move the CSV files to `data/external/`. The daily range starts on 2018-01-01 so that the 90-day window of the first GWT day is complete; the climatology covers 1988–2017, before the study period.

## Reproducing the results

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-lock.txt     # exact versions; requirements.txt lists the packages without pins
# place data/raw/mmc1.xlsx, then:
./reproduce.sh
```

`reproduce.sh` runs every script in order and then the test suite. It takes several hours on a laptop CPU (scripts 04–07 use seven worker processes). All outputs except the run log `results/experiments.csv` are deterministic for a given environment; LSTM runs use fixed seeds and a single PyTorch thread. LightGBM and multithreaded PyTorch can crash when combined in one process, so LightGBM is imported lazily.

## Where each element of the article comes from

| Article element | Script | Output |
|---|---|---|
| Table 1, data-quality numbers (Section 2) | `src/81_data_audit.py` | `results/data_audit.csv` |
| BRG3 suspect segment rule | `src/01_prepare_gwt.py` | `results/straight_runs.csv` |
| Coordinate validation | `src/extract_station_coords.py` | `results/coord_validation.csv` |
| Fig. 1, Table A1 | `src/80_drift_regression.py`, `src/90_figure1.py` | `results/drift_regression.csv`, `results/degradation_ratios.csv`, `figures/fig1_*` |
| Hyperparameters | `src/04_hp_selection.py` | `config/models.yaml`, `results/hp_selection.csv` |
| Table 2, Fig. 2, Sections 4.2–4.3 | `src/05_horizon_batch.py` | `results/horizon_criteria.csv`, `horizon_summary.csv`, `horizon_decision.yaml` |
| Table 3, Fig. 4, Section 4.4 | `src/06_diagnostics.py` | `results/diagnostics_*.csv` |
| Table 4, Fig. 3, Section 4.5 | `src/07_cl_methods.py` | `results/cl_*.csv`, per-seed rows in `results/experiments.csv` |
| Table 5, Fig. 5, Section 4.6 | `src/08_damped_trend.py`, `src/06_diagnostics.py` | `results/damped_trend*.csv`, `results/diagnostics_crossing_f1.csv` |
| Figures 2–5, tables as CSV/Markdown | `src/91_final_outputs.py` | `figures/`, `results/tables/` |
| Tables as printed in the manuscript | `src/92_manuscript_tables.py` | `docs/manuscript/tables.md` |

## Layout

```
config/          locked rules, grids, selected hyperparameters, diagnostics and closing definitions
data/raw/        raw data, never committed
data/external/   CHIRPS exports, station coordinates, source figure
data/processed/  pipeline outputs, rebuilt by the scripts (not committed)
src/             pipeline scripts; common.py (constants), splits.py (time-based splits), normalization.py
results/         all numeric results and the run log experiments.csv
figures/         article figures (PDF and PNG)
tests/           pytest suite
docs/            protocol (Indonesian) and manuscript draft
```

## Rules the code enforces

- All splits are by time, never random; each row's task is set by its target date.
- Normalisation statistics come from the T1 training fold only and are frozen.
- Rows whose target or most recent input is interpolated are dropped; gaps longer than five days stay missing.
- Models are panel models over all stations, with station identity as an input.
- Every run is logged with its configuration, seed, metrics, and run time in `results/experiments.csv`.

## Citation and licence

Citation metadata: [CITATION.cff](CITATION.cff). Licence: to be decided by the authors before the archive is published. The raw groundwater data remain under the terms of Taufik et al. (2022).
