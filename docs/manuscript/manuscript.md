<!--
Target journal: Environmental Modelling & Software (research article). Language: English.
Every number in this file must come from a file in results/ produced by a script at tag analysis-frozen (4def636).
The source of each number is given in an HTML comment next to it. [TODO] marks text not yet written.
Status: Sections 2 and 3 drafted; others outlined.
-->

# Persistence is hard to beat: a locked-protocol benchmark of adaptive groundwater-table forecasting in tropical peatlands under El Niño drought

<!-- Working title. Alternative: "When does adaptive learning pay off? Forecasting tropical peat groundwater tables through a regime shift" -->

**Authors:** [TODO]

## Highlights

[TODO: 3–5 bullets, ≤ 85 characters each, written after Results are final.]

## Abstract

[TODO: write last, ≤ 250 words.]

**Keywords:** tropical peatland; groundwater table; forecasting; persistence baseline; continual learning; El Niño; pre-specified analysis

---

## 1. Introduction

[TODO — outline]

1. Why the groundwater table (GWT) in drained tropical peat matters: fire risk, subsidence, carbon loss; the −0.4 m threshold in Indonesian regulation (PP 57/2016).
2. Data-driven GWT forecasting is growing, but reported skill is usually measured as error, not against a naive persistence forecast, and evaluation often uses random or in-regime splits.
3. Hydrological regimes shift (dry season, El Niño drought, re-wetting), so a model trained in one regime may fail in the next. Continual learning (CL) and periodic retraining are the usual answers.
4. Gap: no study tests, under a time-ordered and pre-specified protocol, whether adaptive models beat persistence for tropical peat GWT, and when.
5. This study: a panel benchmark on eight stations in Jambi and West Kalimantan (2018–2019), with decision rules locked in version control before each stage of analysis. Contributions: (i) skill-versus-horizon curves against persistence; (ii) the behaviour of online retraining and CL through the 2019 El Niño drought; (iii) diagnostics that rule out three pipeline artefacts; (iv) the gap between error-based and event-based skill; (v) a data-quality finding at one station; (vi) the staged locking protocol itself.

Literature to cover [TODO: search]: GWT forecasting in peatlands (ML and process models); persistence and naive baselines in hydrological forecasting; CL for time series and environmental streams; pre-registration and analytic flexibility in environmental modelling.

---

## 2. Data

### 2.1 Groundwater table records

We used the daily groundwater table records published by Taufik et al. (2022), collected under the Indonesian Peatland Restoration Agency (BRG) monitoring network. The data cover eight stations in two peatland landscapes: BRG3–BRG6 in Batanghari, Jambi (Sumatra), and BRG17–BRG20 in Kubu Raya, West Kalimantan (Borneo), from 8 April 2018 to 31 December 2019. Values are in metres and negative below the ground surface. The text of the source article uses metres; its Figure 2 mislabels the unit, and we kept metres throughout.

Cleaning followed fixed rules (`src/01_prepare_gwt.py`). Each series was placed on a gap-free daily calendar. Gaps of up to five consecutive days lying between two observations were filled by linear interpolation and flagged; longer gaps stayed missing. One station-date carried two readings, which we averaged. Rows whose forecast target was interpolated were discarded, as were rows in which the most recent input (the level on the forecast issue date) was itself interpolated, because its interpolation anchor lies in the future.

### 2.2 A suspect segment at BRG3

During the 2019 drought, the BRG3 record shows a 45-day run from 14 July to 27 August in which the level falls along an almost perfect straight line <!-- data_audit.csv: brg3_line_days = 45 -->: the slope is −7.17 mm d⁻¹ and the residual about the line is 0.28 mm <!-- brg3_line_slope_mm_per_day, brg3_line_residual_mm -->. The standard deviation of daily changes in this run is 115 times smaller than in the rest of the BRG3 record <!-- brg3_outside_over_line_sd = 115.3 -->, and 11 to 56 times smaller than the smoothest window of the same length at any other station <!-- same_window_smoothest_ratio_min/max -->. Real peat recession decelerates as the water table drops; this run has no curvature, and 58 mm of rain during the segment produced no response <!-- brg3_rain_during_line_mm = 57.5 -->. A comparable recession at BRG6 over the same weeks is visibly concave, with a residual about a straight line of 5.4 mm <!-- brg6_recession_residual_mm = 5.41 -->. The segment starts four days after an unexplained drop of 0.254 m on a day without rain <!-- brg3_jump_m, brg3_rain_jump_day_mm -->, and it ends where the record ends.

An objective rule, runs of at least 14 days whose daily changes vary by no more than 2 mm, finds this segment and nothing else in the dataset (`results/straight_runs.csv`). We therefore flagged BRG3 from 10 July to 27 August 2019 (49 days, including the four post-jump days) as suspect. The values stay in the cleaned file but are treated as missing in every feature, target, and metric. We judge the segment very unlikely to be a measurement and report it here so that other users of the dataset can decide for themselves.

### 2.3 Rainfall

The groundwater dataset contains no rainfall. We extracted daily rainfall from CHIRPS v2.0 (Funk et al., 2015) at each station with Google Earth Engine (`src/00_fetch_chirps.js`), using bilinear interpolation between the four surrounding 0.05° cells. We verified the extraction against the raw CHIRPS raster for one day: the exported values match a manual bilinear calculation at the centre of Earth Engine's 100 m sampling pixel, and differ from the nearest-cell values (`src/82_verify_bilinear.py`, `tests/test_bilinear.py`). The rainfall normal used for the 90-day deficit is the 1988–2017 CHIRPS climatology, which ends before the study period.

### 2.4 Station coordinates

Only two station coordinates are published (BRG6 and BRG18). We estimated the other six from the high-resolution map in Taufik et al. (2022, Fig. 1; CC BY 4.0) with a scripted photogrammetric procedure (`src/extract_station_coords.py`). Against the two published coordinates, the estimates were off by 214.3 m and 208.1 m <!-- coord_validation.csv -->, measured before a longitude bias correction derived from those same two stations. The remaining six are therefore not independently validated. The two validated errors are small relative to a CHIRPS cell (about 5.5 km), and bilinear sampling avoids discrete jumps when a station lies near a cell boundary.

### 2.5 Hydrological regimes

We split the record into five consecutive regimes, used as the tasks of the continual-learning setting (Table 1). The boundaries follow the seasonal pattern of the data: a transition period (T1), the 2018 dry season (T2), the 2018–2019 wet season (T3), the 2019 El Niño drought (T4), and the late-2019 recovery (T5). The same calendar is applied to both provinces. This fits Kubu Raya less well: October in both years is labelled dry, yet the Kubu Raya stations received 453–488 mm of rain in October 2018 and 341–386 mm in October 2019 <!-- data_audit.csv: kubu_raya_rain_2018-10_min/max, kubu_raya_rain_2019-10_min/max -->. We kept the single calendar because it was fixed before any model was trained, and we return to it in Section 5.

**Table 1.** Hydrological regimes and usable forecast rows (seven-day horizon, after all filters). <!-- data_audit.csv: *_rows, *_deficit_mean_mm, *_deficit_positive_share -->

| Task | Period | Usable rows (t+7) | Mean 90-day rainfall deficit (mm) | Rows with positive deficit |
|---|---|---:|---:|---:|
| T1 transition | 8 Apr – 30 Jun 2018 | 353 | −147.8 | 0% |
| T2 dry season | 1 Jul – 31 Oct 2018 | 984 | −16.8 | 58% |
| T3 wet season | 1 Nov 2018 – 30 Jun 2019 | 1,315 | −46.9 | 34% |
| T4 El Niño drought | 1 Jul – 31 Oct 2019 | 216 | +169.6 | 82% |
| T5 recovery | 1 Nov – 31 Dec 2019 | 136 | +192.9 | 93% |

*A row is a station-day on which a forecast can be issued and scored. Positive deficit means drier than the 1988–2017 normal. T4 excludes the BRG3 suspect segment.*

---

## 3. Methods

### 3.1 Forecasting task and information cut-off

A forecast is issued at the end of day *t* for the groundwater level at *t* + *h*. Every input uses data up to and including day *t*. The main horizon set was H = {3, 7, 14, 21, 30} days. Each row is assigned to a task by its target date, not its issue date, so a training target can never fall inside a later test regime.

All models are panel (global) models trained on the eight stations together, with station identity as one-hot inputs. T4 and T5 are too short for station-specific models.

### 3.2 Features

Each row carries 33 inputs (`src/02_build_features.py`): the level at lags 0, 1, 2, 3, 5, 7, 14, 21, and 30 days; the 1-day and 7-day changes; 7-day and 30-day moving means; rainfall on day *t* and summed over 3, 7, 14, 30, and 60 days; the length of the current dry spell (days with less than 1 mm); the 90-day rainfall deficit (normal minus observed, so positive means drier than usual); the sine and cosine of the day of year; and one-hot station and peatland indicators.

### 3.3 Splits and normalisation

Every split is by time. Each task was divided once: the last 30% of its rows by target date form a holdout, and the first 70% form the training fold. Inputs were standardised with means and standard deviations from the T1 training fold only, then frozen. Seasonal and one-hot inputs were left unscaled. Frozen T1 statistics are leakage-free, but they put T4 far outside the training range: the mean standardised deficit in T4 was +4.4, and 63% of T4 rows had at least one input beyond five standard deviations <!-- data_audit.csv: T4_deficit_z_mean = 4.43, T4_share_rows_any_abs_z_gt_5 = 0.634 -->. Diagnostic D2 (Section 3.8) tests whether this matters.

### 3.4 Baselines and models

Five baselines were defined before any model was run.

- **B1 Persistence.** The forecast for *t* + *h* is the level at *t*.
- **B2 Static.** Ridge regression and LightGBM (Ke et al., 2017), trained once on the T1 and T2 training folds. This mirrors common practice: fit once, then deploy.
- **B3 Joint (oracle).** The same model families trained on the training folds of all five tasks. B3 sees future regimes and serves only as an upper reference.
- **B4 Naive fine-tuning.** An LSTM updated task by task, to show forgetting.
- **B5 Periodic retraining.** Ridge, LightGBM, and LSTM retrained from scratch every 30 days on all training-fold rows whose targets had been observed by the retraining date. The first fit is at the end of the T1 training fold, which gives 18 retrainings.

The LSTM (Hochreiter and Schmidhuber, 1997) reads the last 14 or 30 days of level and rainfall, concatenates its final hidden state with the rainfall, seasonal, and station inputs, and predicts the change from the level at *t*; the forecast is that level plus the predicted change. It was trained with Adam on mean squared error (batch 64, up to 60 epochs, early stopping with patience 8 on the last 15% of the training rows by target date, gradient clipping at 1.0). PyTorch was restricted to one thread so that runs are deterministic for a given seed. LSTM results are averaged over five seeds; Ridge, LightGBM, and persistence are deterministic.

### 3.5 Hyperparameter selection

Hyperparameters were selected once per horizon and model family, using only the T1 and T2 training folds (`src/04_hp_selection.py`). Models were fitted on the T1 training fold plus the first 70% of the T2 training fold, and validated on the last 30% of the T2 training fold by mean absolute error (MAE). Each family had the same budget of 32 configurations (`config/hp_grid.yaml`): Ridge α on 32 log-spaced values from 10⁻³ to 10³; LightGBM over number of leaves {7, 15, 31, 63}, learning rate {0.03, 0.1}, number of trees {200, 500}, and minimum child samples {10, 30}; the LSTM over hidden units {32, 64}, layers {1, 2}, dropout before the output layer {0, 0.2}, learning rate {10⁻³, 3 × 10⁻³}, and window length {14, 30} days. LSTM selection used a single seed. The selected values are in `config/models.yaml`.

### 3.6 Evaluation

**Online (prequential) forecasts.** Each forecast is scored with the model that existed on its issue date (Dawid, 1984). For B2 this is the static model; for B5 it is the most recent retraining. T3–T5 are out-of-sample for every realistic candidate, so they form a common evaluation period.

**Skill against persistence** is the primary metric: skill = 1 − MAE_model / MAE_persistence, computed on identical rows. Positive skill means the model beats persistence. We report it per task and pooled over all online rows of T3–T5.

**Holdout evaluation.** For comparison with the continual-learning literature, we also computed the matrix R, whose entry R(*i*, *j*) is the holdout MAE on task *j* of the model available after task *i*, and forgetting as the final error minus the smallest earlier error on each task (positive means forgetting).

**Event metric.** The −0.4 m threshold is the regulatory limit for peat GWT in Indonesia. Because almost every T4 row lies below it, a simple below-threshold F1 score carries no information during the drought. We therefore scored *crossings*: an event at row *t* occurs when the level at *t* is at or above −0.4 m and the level at *t* + *h* is below it, and a forecast predicts a crossing in the same way. Persistence can never predict a crossing, so its F1 is zero by construction. The name of this metric was fixed before the horizon results were seen; its operational definition was fixed afterwards, during the diagnostics stage (Section 3.8).

### 3.7 Decision rule for the main horizon

Before any horizon other than seven days was run, we wrote a decision rule into the protocol and the configuration file `config/horizon_rule.yaml`, and locked both with the annotated git tag `horizon-rule-locked` (commit 440d3df). A horizon *h* passes if at least one of the five realistic candidates (B2 Ridge, B2 LightGBM, B5 Ridge, B5 LightGBM, B5 LSTM) meets all three criteria:

- **(a)** its median skill over station × task cells (online, T3–T5) is above zero, and a one-sided paired Wilcoxon signed-rank test of cell MAE against persistence gives *p* < 0.05 after Holm correction over the five candidates (Wilcoxon, 1945; Holm, 1979);
- **(b)** its pooled online skill over T3–T5 is at least +5%;
- **(c)** the B3 oracle of the same family has positive holdout skill.

There is no "pick the best model" step: every candidate is tested. The main horizon was to be the shortest passing horizon, not the one with the largest skill. If no horizon passed, the study was to be framed as a benchmark, with no further extension of the horizon set. Cells were station × task combinations; errors were averaged over seeds before testing. We applied no correction across horizons.

### 3.8 Diagnostics

If persistence wins, a reviewer can reasonably ask whether the models failed because of how the pipeline was built. We tested three such explanations, each by changing exactly one element of the locked pipeline (`config/diagnostics.yaml`, tag `diagnostics-plan-locked`, commit 2f2273b). All diagnostics are exploratory and could not overturn the horizon decision.

- **D1 Change target.** Ridge and LightGBM predict the change from the level at *t* instead of the level itself. The LSTM already predicts a change and is unaffected.
- **D2 Expanding normalisation.** Each row is standardised with statistics from all rows issued up to its own date (never earlier than the end of the T1 training fold), instead of frozen T1 statistics.
- **D3** combines D1 and D2.
- **D4 Holdout rows in retraining.** B5 also uses holdout rows whose targets had been observed by the retraining date. B5 holdout scores are then no longer out-of-sample; criterion (c) is unaffected because it uses B3.
- **D5** is the crossing metric described in Section 3.6.

D1–D3 repeated the locked hyperparameter selection under the modified pipeline. D4 changes nothing the selection sees, so it reused the locked hyperparameters.

### 3.9 Continual learning and a damped-trend reference

Two further analyses were specified in `config/closing.yaml` and committed before they were run.

**Continual learning.** We compared experience replay and elastic weight consolidation (EWC; Kirkpatrick et al., 2017) with B5 LSTM, asking one question: do they soften the loss of skill during T4? Both use the locked LSTM configuration and the B5 schedule, but update the existing network instead of retraining it, waiting until at least 16 new rows are available. Replay mixes the new rows with an equal number drawn from a reservoir buffer of past rows (Chaudhry et al., 2019). EWC adds the quadratic penalty λ/2 Σ F_i(θ_i − θ*_i)², with the diagonal Fisher information F accumulated after each update. Fine-tuning without either mechanism served as a control. Buffer size {128, 256, 512, 1024} and λ {1, 10, 100, 1000} were selected inside T1–T2 with the same split as in Section 3.5. With five seeds, we call a difference between a CL method and B5 *indistinguishable* when the difference in means is no larger than the larger of the two standard deviations across seeds.

**Damped trend.** Because persistence cannot predict a crossing, it is too weak a reference for the event metric. We added a damped-trend forecast (Gardner and McKenzie, 1985): the level at *t* plus the mean daily change over the last seven days, multiplied by Σ_{k=1}^{h} φ^k, with φ = 0.9 fixed in advance (0.8 and 0.95 as sensitivity checks). It needs no training. It was added after the horizon results were known, but it can only make our claims harder to support, not easier.

### 3.10 Reproducibility

The analysis was locked in three stages, each marked by an annotated git tag: the horizon rule (`horizon-rule-locked`, 440d3df), the diagnostics plan (`diagnostics-plan-locked`, 2f2273b), and the end of analysis on this dataset (`analysis-frozen`, 4def636). After the last tag, code may change only to fix bugs, and every such fix is reported. Every run is logged with its configuration, seed, metrics, and run time in `results/experiments.csv`, and every number in this paper is produced by a script in the repository. We were not blind to the data: descriptive statistics of the drought period and the seven-day baseline results were seen before the horizon rule was locked, and the regime boundaries were drawn from the seasonal pattern of the data. The tags show when each decision was fixed; they do not show that the data were unseen.

---

## 4. Results

[TODO — outline and sources]

- 4.1 Degradation during the drought (Fig. 1c; `results/degradation_ratios.csv`).
- 4.2 Horizon decision: no horizon passes (Table 2, Fig. 2; `results/horizon_criteria.csv`, `horizon_decision.yaml`).
- 4.3 Skill by regime: negative in T3 and T4, positive in T5 (Fig. 2b–d; `horizon_summary.csv`).
- 4.4 Diagnostics D1–D4 (Table 3, Fig. 4). D1 is inconclusive for shrinkage towards persistence because the selected α was 0.001 at h = 14 and 21.
- 4.5 Continual learning vs B5 (Table 4b, Fig. 3): indistinguishable in 13 of 15 T4 cells.
- 4.6 Event metric and damped trend (Table 5, Fig. 5): damped trend is the only forecaster with positive T4 skill; models beat it on crossings in most cells.

## 5. Discussion

[TODO — outline]

- Why persistence is strong for slow peat dynamics; what that implies for operational early warning.
- Online retraining fails at drought onset: learning from the past cannot anticipate a regime the model has not seen.
- MAE and event skill tell different stories; operational value depends on the metric.
- The best forecaster changes with the regime (H5; to be tested on SiTMAT data, not claimed here).
- Limitations (from CLAUDE.md list 1–8): one severe drought; 4–5 station cells in T4; selected α at grid edges; B5 without holdout rows in the locked run; single-seed LSTM selection; D1 inconclusive; D5 definition post hoc; one regime calendar for two provinces; six unvalidated coordinates.
- Confirmatory test on SiTMAT (H1–H5, pre-registered).

## 6. Conclusions

[TODO]

## Data and code availability

[TODO: Zenodo DOI for the repository at tag analysis-frozen. Raw data: Taufik et al. (2022), Data in Brief 41:107903.]

## References

[TODO: verify every entry before submission.]

- Chaudhry, A., et al. (2019). On tiny episodic memories in continual learning. arXiv:1902.10486.
- Dawid, A.P. (1984). Statistical theory: the prequential approach. Journal of the Royal Statistical Society A, 147, 278–292.
- Funk, C., et al. (2015). The climate hazards infrared precipitation with stations — a new environmental record for monitoring extremes. Scientific Data, 2, 150066.
- Gardner, E.S., McKenzie, E. (1985). Forecasting trends in time series. Management Science, 31, 1237–1246.
- Hochreiter, S., Schmidhuber, J. (1997). Long short-term memory. Neural Computation, 9, 1735–1780.
- Holm, S. (1979). A simple sequentially rejective multiple test procedure. Scandinavian Journal of Statistics, 6, 65–70.
- Ke, G., et al. (2017). LightGBM: a highly efficient gradient boosting decision tree. Advances in Neural Information Processing Systems 30.
- Kirkpatrick, J., et al. (2017). Overcoming catastrophic forgetting in neural networks. PNAS, 114, 3521–3526.
- Taufik, M., et al. (2022). [TODO: exact title]. Data in Brief, 41, 107903.
- Wilcoxon, F. (1945). Individual comparisons by ranking methods. Biometrics Bulletin, 1, 80–83.
