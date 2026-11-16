<!--
Target journal: Environmental Modelling & Software (research article). Language: English.
Every number in this file must come from a file in results/ produced by a script at tag analysis-frozen (4def636).
The source of each number is given in an HTML comment next to it. [TODO] marks text not yet written.
Status: full draft; see [TODO] and VERIFY comments before submission.
-->

# Persistence is hard to beat: a locked-protocol benchmark of adaptive groundwater-table forecasting in tropical peatlands under El Niño drought

<!-- Working title. Alternative: "When does adaptive learning pay off? Forecasting tropical peat groundwater tables through a regime shift" -->

**Authors:** [TODO]

## Highlights

- Under a locked rule, no realistic model beat persistence at 3 to 30 days ahead
- All five candidate models lost to persistence in the 2019 El Niño drought
- Replay and EWC were indistinguishable from simple 30-day retraining
- Models partly anticipated −0.4 m crossings, which persistence cannot predict
- Decision rules were locked in git tags before each stage of the analysis

<!-- horizon_criteria.csv; horizon_summary.csv; tables/table4b_cl_vs_b5_seeds.csv; damped_trend_vs_models_crossing.csv -->

## Abstract

Groundwater-table (GWT) forecasts could support fire prevention in drained tropical peatlands, but they are useful only if they beat a forecast that repeats today's level and hold up when the hydrological regime changes. We asked whether static, periodically retrained, and continual-learning models keep GWT forecasts accurate through a regime shift. Using daily records from eight monitoring stations in Jambi and West Kalimantan, Indonesia (April 2018 to December 2019), we split the data into five consecutive regimes, including the 2019 El Niño drought. Panel Ridge, LightGBM, and LSTM models were evaluated online against persistence at horizons of 3 to 30 days, under a decision rule fixed in a git tag before the horizons were run. No candidate met the rule at any horizon: pooled skill over the evaluation period was negative throughout, and the best value per horizon ranged from −16% to −2%. During the drought every candidate lost to persistence, by 17% to 42% for the best one. Four diagnostics of the pipeline did not change the outcome, and experience replay and elastic weight consolidation were indistinguishable from periodic retraining in 13 of 15 drought comparisons. On crossings of the −0.4 m regulatory threshold, which persistence cannot predict, the trained models scored F1 up to 0.47 and beat a damped-trend reference in 19 of 25 cases. Persistence is therefore a demanding benchmark for slow peat dynamics, and adaptation that learns only from past regimes cannot anticipate a drought. We state the regime dependence of the best forecaster as a hypothesis for confirmatory testing on independent data.

<!-- pooled best per horizon −8/−16/−14/−9/−2% (horizon_criteria.csv); T4 best −32/−39/−42/−32/−17% (horizon_summary.csv); 13/15 (table4b); F1 0.14–0.47, 19/25 (diagnostics_crossing_f1.csv, damped_trend_vs_models_crossing.csv) -->

**Keywords:** tropical peatland; groundwater table; forecasting; persistence baseline; continual learning; El Niño; pre-specified analysis

---

## 1. Introduction

Drained tropical peatlands in Indonesia burn when their water table falls. The 1997 fires released carbon equivalent to 13–40% of the mean annual global emissions from fossil fuels (Page et al., 2002), the 2015 fires were the largest since then (Huijnen et al., 2016), and drainage that does not lead to fire still causes continuous subsidence and carbon loss through oxidation (Hooijer et al., 2012). The problem is not historical. In 2026, an El Niño year, the Ministry of Forestry reported more than 202,000 ha burned by 18 August (Reuters, 2026a), and independent satellite-based estimates were several times larger: 667,133 ha by 20 August, including 212,739 ha of peat (Madani Berkelanjutan, 2026), and 1.92 million ha by the end of September (Nusantara Atlas, reported by Reuters, 2026b). Fire carbon emissions reached about 76 Mt C by early September, on course to match the highest annual total of this century (Carbon Brief, 2026). The Ministry of Health recorded 283,270 cases of acute respiratory infection in seven fire-affected provinces between 18 August and 13 October, although not all of them can be attributed to smoke (CNN Indonesia, 2026), and one analysis estimated about 1,700 deaths from fire smoke in August alone (CREA, 2026). Haze closed schools in Indonesia and in Sarawak, Malaysia (Save the Children, 2026). <!-- add primary source --> These figures differ by source and method, and the official and satellite-based burned areas in particular are not directly comparable; we cite them only to show the scale of a recurring problem. <!-- VERIFY each 2026 figure against the primary source before submission; retrieved 2026-11-16 --> The groundwater table (GWT) is the variable that links these processes. Fire danger rises steeply as the water table drops, and field studies place the critical depths for tropical peat between about −0.3 and −0.85 m (Taufik et al., 2015; Hayasaka et al., 2020; Irfan et al., 2024). Indonesian regulation sets the limit at 0.4 m below the peat surface (Government Regulation No. 57/2016), and both restoration and fire prevention are evaluated against it (Putra et al., 2021; Ismail et al., 2021). A forecast that warns, a week or more ahead, that the water table is about to cross that limit would therefore be useful to the agencies that manage rewetting infrastructure and fire response.

Data-driven GWT forecasting has grown quickly. A recent systematic review counted 223 machine-learning studies of groundwater level prediction, most using neural networks, support vector machines, LSTM networks, or tree ensembles (Gilbert et al., 2025), and comparisons of network architectures are now common (Wunsch et al., 2021). In tropical peatlands the literature is younger but follows the same pattern: neural networks and gradient-boosted trees have been trained on monitoring-station data from Indonesia and Malaysia (Li et al., 2024; Yonekura et al., 2025; Utomo et al., 2026). These studies report accuracy as RMSE, MAE, or R². <!-- VERIFY: persistence baseline in Li 2024, Yonekura 2025, Utomo 2026 --> Such scores say little about a slow-moving series, because a forecast that simply repeats today's level can already achieve a low error. In hydrological forecasting more broadly, the case for judging a forecast against simple benchmarks such as persistence has long been made (Pappenberger et al., 2015), and where such benchmarks are reported, naive forecasts can be hard to beat for groundwater (Jibhakate and Katpatal, 2024).

Evaluation design matters as much as the benchmark. Random splits of a time series let information from the future leak into training, a problem that has inflated reported performance in many fields that use machine learning (Kapoor and Narayanan, 2023). Even with time-ordered splits, a test period drawn from the same hydrological regime as the training period does not show how a model behaves when conditions change. Klemeš (1986) proposed the differential split-sample test for this reason: a model intended for changed conditions should be tested on a period whose climate differs from that of its calibration. For tropical peat the relevant change is not a slow trend but a sequence of regimes, including wet seasons, ordinary dry seasons, El Niño droughts, and re-wetting, each with its own relation between rainfall and water-table response.

A model trained in one regime may therefore fail in the next, and the machine-learning answer to this problem is adaptation. Periodic retraining is the simplest option. Continual learning (CL) offers methods designed to learn from a stream of changing tasks without forgetting earlier ones, among them experience replay and elastic weight consolidation (Kirkpatrick et al., 2017; Chaudhry et al., 2019; De Lange et al., 2022), and the broader literature on concept drift offers many ways to detect and respond to changes in the data-generating process (Gama et al., 2014). Whether any of these methods helps a GWT forecast through a drought has, to our knowledge, not been tested. The question has a practical edge: the forecast matters most at drought onset, when a model can only have learned from regimes that came before.

A further difficulty is analytic flexibility. A study that compares several models, horizons, and adaptation schemes on a short record has many ways to find an apparent winner, and the choices made after seeing results are rarely visible to readers. Pre-registration addresses this by fixing the analysis before the data are examined (Nosek et al., 2018). Full pre-registration is seldom possible with an existing public dataset, but the decisions that matter most can still be fixed before their results are known, and the record of when each was fixed can be made public.

This study asks whether adaptive models keep tropical peat GWT forecasts accurate through a hydrological regime shift. We use the public record of eight BRG monitoring stations in Jambi and West Kalimantan from April 2018 to December 2019, which includes the 2019 El Niño drought (Taufik et al., 2022). We frame the question as a benchmark against persistence, evaluate every forecast online and in time order across five hydrological regimes, and fix each stage of the analysis with an annotated git tag before running it. The study makes six contributions:

1. skill against persistence, as a function of forecast horizon from 3 to 30 days, for static, periodically retrained, and continual-learning panel models;
2. an account of how online retraining and CL behave through the onset and course of the 2019 El Niño drought;
3. four diagnostics that test whether the result is an artefact of the pipeline (target definition, normalisation, and training data);
4. a comparison of error-based skill with skill at forecasting crossings of the −0.4 m threshold, against both persistence and a damped-trend reference;
5. a data-quality finding at one station that would otherwise distort any evaluation of the drought period; and
6. the staged locking protocol itself, together with the code and logs that reproduce every number in the paper.

The result is negative: no model beat persistence under the locked decision rule. We report it as such, and we state the patterns it suggests as hypotheses for an independent confirmatory study rather than as findings.

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

### 4.1 A static model degrades unevenly during the drought

A static panel Ridge model on level lags, trained on targets in 2018, made larger errors in the 2019 drought than in the first half of 2019 at four of the five stations with enough data in both periods (Fig. 1c). The ratio of drought to pre-drought MAE ranged from 0.99 at BRG20 to 2.68 at BRG6 <!-- degradation_ratios.csv: ratio -->. Degradation was therefore real but heterogeneous: at BRG19 and BRG20 the panel model barely deteriorated. Station-specific models degraded more evenly (ratios 1.14–2.20; Appendix A, Table A1), which suggests that pooling across stations already absorbs part of the shift at some sites.

### 4.2 No horizon meets the decision rule

No candidate passed criterion (a) or (b) at any horizon (Table 2, Fig. 2a), so no main horizon could be chosen and the benchmark framing applied (`results/horizon_decision.yaml`). The smallest one-sided Wilcoxon *p*-value before correction was 0.40 (B5 LightGBM, h = 30), and every Holm-corrected value was 1.0 <!-- horizon_criteria.csv: wilcoxon_one_sided_p, p_holm -->. Pooled online skill over T3–T5 was negative for all five candidates at all five horizons. The least negative value per horizon was −8%, −16%, −14%, −9%, and −2% for h = 3, 7, 14, 21, and 30 days <!-- horizon_criteria.csv: max pooled_online_skill per horizon -->. At most 10 of the 16 station × task cells favoured any model over persistence <!-- cells_better_than_persistence, n_cells -->.

Static Ridge was the most fragile candidate. With the selected α = 0.001 at h = 14 and 21 <!-- config/models.yaml -->, its pooled skill fell to −388% and −525% <!-- horizon_criteria.csv: B2_static_ridge pooled_online_skill -->, as near-unregularised coefficients met inputs far outside the T1 range (Section 3.3). Criterion (c) passed in several cells: the B3 LSTM oracle, which is trained on all regimes, reached holdout skill of +3%, +5%, and +8% at h = 7, 14, and 21 <!-- horizon_summary.csv: B3_joint_lstm final_skill_mean -->. Even with access to every regime, the margin over persistence was small.

**Table 2.** Decision criteria of the locked horizon rule for every candidate and horizon (online evaluation, T3–T5, 16 station × task cells). Skill = 1 − MAE/MAE_persistence. *p*: one-sided Wilcoxon signed-rank test of cell MAE against persistence; *p*-Holm: corrected over the five candidates. Oracle: holdout skill of the B3 model of the same family. A horizon passes if any candidate meets (a), (b), and (c); none does.

<!-- BEGIN TABLE2 (src/92_manuscript_tables.py) -->
| *h* (d) | Candidate | Median cell skill (%) | *p* | *p*-Holm | Pooled skill (%) | Oracle skill (%) | (a) | (b) | (c) |
|:---|:---|---:|---:|---:|---:|---:|:---|:---|:---|
| 3 | B2 Ridge | −42 | 1.00 | 1.00 | −46 | +3.0 | ✗ | ✗ | ✓ |
| 3 | B2 LightGBM | −21 | 0.97 | 1.00 | −24 | −2.2 | ✗ | ✗ | ✗ |
| 3 | B5 Ridge | −1 | 0.90 | 1.00 | −8 | +3.0 | ✗ | ✗ | ✓ |
| 3 | B5 LightGBM | −14 | 0.99 | 1.00 | −14 | −2.2 | ✗ | ✗ | ✗ |
| 3 | B5 LSTM | −6 | 0.99 | 1.00 | −10 | +0.1 | ✗ | ✗ | ✓ |
| 7 | B2 Ridge | −119 | 1.00 | 1.00 | −98 | +0.3 | ✗ | ✗ | ✓ |
| 7 | B2 LightGBM | −26 | 0.97 | 1.00 | −33 | −7.4 | ✗ | ✗ | ✗ |
| 7 | B5 Ridge | +3 | 0.84 | 1.00 | −21 | +0.3 | ✗ | ✗ | ✓ |
| 7 | B5 LightGBM | −18 | 0.99 | 1.00 | −18 | −7.4 | ✗ | ✗ | ✗ |
| 7 | B5 LSTM | −16 | 0.98 | 1.00 | −16 | +3.2 | ✗ | ✗ | ✓ |
| 14 | B2 Ridge | −448 | 1.00 | 1.00 | −388 | −12.1 | ✗ | ✗ | ✗ |
| 14 | B2 LightGBM | +4 | 0.67 | 1.00 | −14 | −6.2 | ✗ | ✗ | ✗ |
| 14 | B5 Ridge | −31 | 0.98 | 1.00 | −88 | −12.1 | ✗ | ✗ | ✗ |
| 14 | B5 LightGBM | −9 | 0.95 | 1.00 | −16 | −6.2 | ✗ | ✗ | ✗ |
| 14 | B5 LSTM | −27 | 1.00 | 1.00 | −22 | +5.4 | ✗ | ✗ | ✓ |
| 21 | B2 Ridge | −589 | 1.00 | 1.00 | −525 | −10.1 | ✗ | ✗ | ✗ |
| 21 | B2 LightGBM | +8 | 0.55 | 1.00 | −10 | +2.0 | ✗ | ✗ | ✓ |
| 21 | B5 Ridge | −115 | 0.99 | 1.00 | −148 | −10.1 | ✗ | ✗ | ✗ |
| 21 | B5 LightGBM | +7 | 0.47 | 1.00 | −9 | +2.0 | ✗ | ✗ | ✓ |
| 21 | B5 LSTM | −20 | 1.00 | 1.00 | −18 | +7.6 | ✗ | ✗ | ✓ |
| 30 | B2 Ridge | −22 | 0.92 | 1.00 | −22 | +2.5 | ✗ | ✗ | ✓ |
| 30 | B2 LightGBM | −24 | 0.76 | 1.00 | −24 | +15.0 | ✗ | ✗ | ✓ |
| 30 | B5 Ridge | −10 | 0.74 | 1.00 | −11 | +2.5 | ✗ | ✗ | ✓ |
| 30 | B5 LightGBM | −1 | 0.40 | 1.00 | −2 | +15.0 | ✗ | ✗ | ✓ |
| 30 | B5 LSTM | −18 | 0.96 | 1.00 | −14 | −5.8 | ✗ | ✗ | ✗ |
<!-- END TABLE2 -->

### 4.3 Skill depends on the regime

Skill differed sharply between regimes (Fig. 2b–d). In the wet season (T3), most candidates were slightly to moderately worse than persistence; in the El Niño drought (T4), every candidate was worse at every horizon. The best T4 skill per horizon belonged to B5 LSTM: −32%, −39%, −42%, −32%, and −17% <!-- horizon_summary.csv: skill_online_T4_mean -->. Periodic retraining therefore did not protect against the drought; each retraining could only learn from regimes that had already passed.

The recovery period (T5) was the exception. B5 Ridge beat persistence at every horizon, from +10% at h = 3 to +45% at h = 14 <!-- skill_online_T5_mean, B5_retrain30d_ridge -->, and B5 LSTM was positive from h = 7 onwards. T5 is short (136 rows at h = 7; Table 1), and it is pooled with T3 and T4 in the decision rule, so this pattern did not change the outcome. We treat it as a hypothesis for independent data (Section 5) rather than a finding of this study.

### 4.4 Diagnostics do not rescue the models

None of the four pipeline changes produced a candidate that passed the decision rule (Table 3). The best pooled online skill across all diagnostics, candidates, and horizons was −1.5% (D3, B5 Ridge, h = 3), and the smallest uncorrected *p*-value was 0.36 <!-- diagnostics_criteria.csv -->.

The diagnostics differed in what they changed. D1 (change target) left the results nearly unchanged. At h = 14 and 21, selection again chose α = 0.001 <!-- results/diagnostics_models.yaml -->, and an almost unregularised Ridge on the change is equivalent to one on the level when the current level is an input. D1 therefore did not test shrinkage towards persistence, which was its purpose, and we regard it as inconclusive on that point. D2 (expanding normalisation) reduced the T4 loss of the best candidate at every horizon, by between a quarter and two thirds: from −32/−39/−42/−32/−17% to −24/−19/−19/−19/−6% across the five horizons (Fig. 4) <!-- diagnostics_summary.csv vs horizon_summary.csv: max skill_online_T4_mean over candidates -->. The gain came with losses elsewhere: for the tree and linear models, T3 skill fell under D2, for example from −14% to −80% for B2 LightGBM at h = 3 <!-- skill_online_T3_mean -->. D4 (holdout rows in retraining) moved the best T4 skill by up to nine percentage points, in either direction, without changing the overall picture.

**Table 3.** Diagnostics D1–D4 (exploratory, post hoc; they cannot overturn the horizon decision). Best pooled and best T4 online skill over the five candidates, and the smallest Holm-corrected *p*-value. D1: change target; D2: expanding normalisation; D3: D1 + D2; D4: holdout rows in B5 retraining.

<!-- BEGIN TABLE3 (src/92_manuscript_tables.py) -->
| Diagnostic | *h* (d) | Best pooled skill (%) | Best T4 skill (%) | Min. *p*-Holm | Any candidate passes |
|:---|:---|---:|---:|---:|:---|
| D1 | 3 | −5 | −32 | 1.00 | no |
| D1 | 7 | −11 | −39 | 1.00 | no |
| D1 | 14 | −14 | −42 | 1.00 | no |
| D1 | 21 | −8 | −32 | 1.00 | no |
| D1 | 30 | −4 | −17 | 1.00 | no |
| D2 | 3 | −10 | −24 | 1.00 | no |
| D2 | 7 | −15 | −19 | 1.00 | no |
| D2 | 14 | −9 | −19 | 1.00 | no |
| D2 | 21 | −17 | −19 | 1.00 | no |
| D2 | 30 | −5 | −6 | 1.00 | no |
| D3 | 3 | −2 | −21 | 1.00 | no |
| D3 | 7 | −4 | −19 | 1.00 | no |
| D3 | 14 | −12 | −19 | 1.00 | no |
| D3 | 21 | −17 | −19 | 1.00 | no |
| D3 | 30 | −12 | −6 | 1.00 | no |
| D4 | 3 | −6 | −31 | 1.00 | no |
| D4 | 7 | −16 | −30 | 1.00 | no |
| D4 | 14 | −14 | −39 | 1.00 | no |
| D4 | 21 | −10 | −30 | 1.00 | no |
| D4 | 30 | −2 | −21 | 1.00 | no |
<!-- END TABLE3 -->

### 4.5 Continual learning does not soften the drought

Across five seeds, replay and EWC were indistinguishable from B5 LSTM in 13 of the 15 T4 comparisons (Table 4, Fig. 3a) <!-- tables/table4b_cl_vs_b5_seeds.csv: skill_T4_vs_b5 -->. Seed-to-seed variation was large: the standard deviation of B5 LSTM's T4 skill reached 39 percentage points at h = 7 <!-- skill_T4_sd -->. Of the two distinguishable cells, replay was worse than B5 at h = 3 (−53 ± 16% vs −32 ± 20%), and the fine-tuning control was better at h = 21 (+11 ± 10% vs −32 ± 36%). The same fine-tuning control lost heavily over T3–T5 at that horizon (−53 ± 15% vs −18 ± 8%; Fig. 3b) <!-- pooled_T3_T5_mean, pooled_T3_T5_sd -->, so its T4 gain reflects a trade between regimes rather than a better model. On pooled skill, no CL method was better than B5 at any horizon; fine-tuning and EWC were each worse at three of the five horizons <!-- pooled_T3_T5_vs_b5 -->.

**Table 4.** Continual learning against B5 LSTM: online skill, mean ± standard deviation over five seeds. A difference is called indistinguishable (indist.) when the absolute difference in means is no larger than the larger of the two standard deviations.

<!-- BEGIN TABLE4 (src/92_manuscript_tables.py) -->
| *h* (d) | Method | T4 skill (%) | vs B5 | Pooled T3–T5 skill (%) | vs B5 |
|:---|:---|---:|:---|---:|:---|
| 3 | B5 LSTM | −32 ± 20 | reference | −10 ± 4 | reference |
| 3 | Fine-tune (control) | −34 ± 17 | indist. | −16 ± 4 | worse |
| 3 | Replay | −53 ± 16 | worse | −15 ± 6 | indist. |
| 3 | EWC | −58 ± 28 | indist. | −18 ± 9 | indist. |
| 7 | B5 LSTM | −39 ± 39 | reference | −16 ± 6 | reference |
| 7 | Fine-tune (control) | −4 ± 10 | indist. | −16 ± 3 | indist. |
| 7 | Replay | −52 ± 21 | indist. | −16 ± 5 | indist. |
| 7 | EWC | −25 ± 22 | indist. | −24 ± 7 | worse |
| 14 | B5 LSTM | −42 ± 36 | reference | −22 ± 7 | reference |
| 14 | Fine-tune (control) | −18 ± 7 | indist. | −23 ± 8 | indist. |
| 14 | Replay | −35 ± 21 | indist. | −20 ± 7 | indist. |
| 14 | EWC | −20 ± 7 | indist. | −27 ± 10 | indist. |
| 21 | B5 LSTM | −32 ± 36 | reference | −18 ± 7 | reference |
| 21 | Fine-tune (control) | +11 ± 10 | better | −53 ± 15 | worse |
| 21 | Replay | −44 ± 31 | indist. | −26 ± 10 | indist. |
| 21 | EWC | −10 ± 22 | indist. | −57 ± 23 | worse |
| 30 | B5 LSTM | −17 ± 25 | reference | −14 ± 8 | reference |
| 30 | Fine-tune (control) | −19 ± 8 | indist. | −30 ± 11 | worse |
| 30 | Replay | −23 ± 19 | indist. | −18 ± 9 | indist. |
| 30 | EWC | −2 ± 11 | indist. | −32 ± 18 | worse |
<!-- END TABLE4 -->

### 4.6 Event skill and the damped trend

On crossings of the −0.4 m threshold during T3–T5, the trained models scored F1 values between 0.14 and 0.47 against 47 to 109 observed crossings per horizon (Fig. 5) <!-- diagnostics_crossing_f1.csv, diagnostic = locked, scope = T3_T5 -->, whereas persistence scores zero by construction. The damped trend (φ = 0.9) scored 0.17–0.25 <!-- damped_trend.csv: crossing_f1_T3_T5 -->. Models beat it in 19 of the 25 candidate × horizon cells; B5 Ridge and B5 LSTM did so at every horizon, while static Ridge did so only at h = 21 and 30 <!-- damped_trend_vs_models_crossing.csv: model_minus_damped -->. The event results carry no uncertainty estimate, and T4 contained only zero to five crossings per horizon, too few to interpret <!-- true_crossings_T4 -->.

On MAE, the damped trend lost to persistence overall (pooled skill −16% to −36% for φ = 0.9; Table 5), because it extrapolated recessions into re-wetting events in T3 and T5. In T4, however, it was the only forecaster with positive skill: +5% to +9% for φ = 0.9 across the five horizons, with similar values for φ = 0.8 <!-- damped_trend.csv: skill_T4 -->. This result is descriptive: T4 contains four or five station cells, the baseline was added after the horizon results were known, and only φ was fixed in advance. Read together with Section 4.3, it suggests that the best forecaster depends on the regime: trend continuation during recession, trained models during re-wetting, and persistence otherwise. We do not claim this here; it is hypothesis H5 for the confirmatory study (Section 5).

**Table 5.** Damped-trend reference (exploratory, post hoc). φ = 0.9 was fixed in advance as the primary value; 0.8 and 0.95 are sensitivity checks. Skill is online MAE skill against persistence; crossing F1 uses the −0.4 m definition of Section 3.6. With zero to five crossings, F1 in T4 is not reported.

<!-- BEGIN TABLE5 (src/92_manuscript_tables.py) -->
| *h* (d) | φ | Pooled skill (%) | T3 (%) | T4 (%) | T5 (%) | Crossing F1, T3–T5 | Crossings, T3–T5 | Crossings, T4 |
|:---|:---|---:|---:|---:|---:|---:|---:|---:|
| 3 | 0.90 | −16 | −19 | +8 | −20 | 0.24 | 47 | 0 |
| 3 | 0.80 | −11 | −14 | +10 | −15 | 0.22 | 47 | 0 |
| 3 | 0.95 | −20 | −23 | +5 | −23 | 0.28 | 47 | 0 |
| 7 | 0.90 | −29 | −35 | +5 | −22 | 0.22 | 74 | 0 |
| 7 | 0.80 | −16 | −21 | +9 | −11 | 0.21 | 74 | 0 |
| 7 | 0.95 | −39 | −46 | +1 | −30 | 0.25 | 74 | 0 |
| 14 | 0.90 | −34 | −46 | +5 | −11 | 0.17 | 90 | 4 |
| 14 | 0.80 | −13 | −20 | +9 | +2 | 0.14 | 90 | 4 |
| 14 | 0.95 | −58 | −73 | −5 | −28 | 0.17 | 90 | 4 |
| 21 | 0.90 | −36 | −51 | +8 | −12 | 0.25 | 84 | 5 |
| 21 | 0.80 | −12 | −19 | +9 | −1 | 0.16 | 84 | 5 |
| 21 | 0.95 | −70 | −94 | +2 | −31 | 0.25 | 84 | 5 |
| 30 | 0.90 | −36 | −53 | +9 | −11 | 0.22 | 109 | 5 |
| 30 | 0.80 | −12 | −18 | +5 | −1 | 0.13 | 109 | 5 |
| 30 | 0.95 | −77 | −109 | +11 | −32 | 0.25 | 109 | 5 |
<!-- END TABLE5 -->

## 5. Discussion

### 5.1 Why persistence is hard to beat

The central result is negative. Under a rule fixed before the horizons were run, no adaptive or static model beat persistence over the evaluation period. This is less surprising than it may look. The water table in tropical peat moves slowly: dry-season drawdown proceeds at millimetres to a few centimetres per day (Wösten et al., 2008; Putra et al., 2021; Ismail et al., 2021), so day-to-day changes are small compared with the seasonal range. Under those conditions the level at *t* already explains most of the level at *t* + *h*, and a model must predict the remaining change better than "no change" to earn positive skill. That change is driven largely by rainfall in the days after the forecast is issued, which no input available at *t* contains. Our models did not use rainfall forecasts, and in the absence of such inputs, the information ceiling may sit close to persistence.

The margins of the oracle make the same point. B3, trained on every regime including those it was later scored on, reached only +3% to +8% holdout skill at h = 7–21 (Section 4.2). If a model that has seen the drought gains so little, the gap between persistence and a realistic model is narrow from the outset, and the regime shift only has to cost a few percentage points to erase it.

A practical consequence follows. Studies of GWT forecasting in peatlands report MAE, RMSE, or R² as their headline scores (e.g. Li et al., 2024; Yonekura et al., 2025; see also Gilbert et al., 2025). <!-- VERIFY before submission: whether these papers include a persistence baseline --> On slow-moving series these scores can look good for a model that adds nothing beyond the current level. We suggest that skill against persistence, on identical rows and with online evaluation across regimes, should be the minimum standard for such studies.

### 5.2 Retraining cannot anticipate a new regime

Periodic retraining (B5) was the strongest candidate in the drought, yet it still lost to persistence by 17% to 42% depending on the horizon (Section 4.3). Continual learning did not change this: replay and EWC were indistinguishable from B5 in 13 of 15 T4 comparisons (Section 4.5). The explanation is structural rather than methodological. Every update, whether a full retraining, a replay step, or an EWC-regularised fine-tune, learns from rows whose targets have already been observed. At the onset of the 2019 drought those rows came from the wet season and the earlier, milder dry season. No update schedule can supply a regime that has not yet occurred, and the methods we tested differ mainly in how they weigh the past, not in whether they can see the future.

The scale problem adds to this. Inputs in T4 lay far outside the range of the T1 statistics used for normalisation (Section 3.3), and D2 showed that letting the statistics expand with time reduced the drought loss by between a quarter and two thirds. It did so at a cost in the wet season for the tree and linear models (Section 4.4). We read this as evidence that out-of-range inputs explain part of the drought failure, but not as a remedy: an expanding scaler also shifts the meaning of every input during the period when the model most needs stable inputs.

The one exception, fine-tuning at h = 21, illustrates the trade-off rather than escaping it. Its gain in T4 was paid for in T3 and T5, and its pooled skill was far worse than that of B5. A method that adapts quickly to the latest regime will tend to win during a persistent regime and lose at each transition. For a forecasting service the transitions, not the persistent stretches, are where forecasts matter most.

### 5.3 Error skill and event skill disagree

MAE skill and crossing F1 gave different rankings. On MAE, persistence beat every trained model; on crossings of −0.4 m, persistence scores zero by construction, and the trained models outperformed the damped trend in 19 of 25 cells (Section 4.6). Neither metric is wrong. MAE rewards small errors on ordinary days, which persistence delivers; the crossing metric rewards anticipating the rare day when the level falls below the regulatory limit, which persistence cannot do at all.

The choice of metric therefore depends on the use. For peat fire early warning, the operational question is whether the water table will fall below −0.4 m, not how close the forecast is on a day when it stays well above or well below that level (Taufik et al., 2015, 2019; Irfan et al., 2024). We stress two caveats, however. The event results carry no uncertainty estimate, and the operational definition of a crossing was fixed after the horizon results were known (Section 3.6). They indicate where an operational evaluation should look, not a finding we can defend on its own.

### 5.4 The best forecaster may depend on the regime

Read together, the results suggest a pattern. During the drought, when levels fell steadily, the damped trend was the only forecaster with positive MAE skill. During the recovery, trained models beat persistence, sometimes by a wide margin. In the wet season, neither the candidates nor the damped trend beat persistence at any horizon <!-- horizon_summary.csv: skill_online_T3_mean; damped_trend.csv: skill_T3 -->. This is consistent with the physics: a recession is close to a smooth decay, which a damped trend continues, whereas re-wetting depends on rainfall events that a model with rainfall inputs can partly capture once they have begun.

We do not claim this pattern. It emerged from analyses added after the main results were known, T4 contains only four or five station cells, and T5 is short. Testing it on this dataset again would mean selecting the regime-switching rule on the data used to discover it. We have instead stated it as hypothesis H5 for a confirmatory study on independent data: that a regime switch with rules fixed in advance (trend during recession, model during re-wetting, persistence otherwise) beats persistence on pooled skill. The same study will test whether adaptive models fail to beat persistence on pooled skill (H1), whether model skill is positive during re-wetting and negative during recession (H2), whether models beat the damped trend on crossings (H3), and whether a change model shrunk towards persistence with a fixed penalty is no worse than persistence (H4). The data will come from the SiTMAT monitoring network in Riau, which was not seen during this study, and the operational definitions will be pre-registered before the series is analysed.

The regime-aware adaptation trigger we originally planned (adaptation driven by the 90-day rainfall deficit and the 14-day rate of water-table decline instead of a fixed schedule) was suspended for the same reason. It presupposed a model that beats persistence and would only decide when to retrain it. Without such a model, calibrating the trigger on this dataset would add another set of choices made with the results in view. If it is revisited, it will be labelled exploratory.

### 5.5 Limitations

Several limitations bound these conclusions. We list them in full because some of them would otherwise be easy to miss.

*Data.* The record covers one severe drought, at eight stations in two provinces, over less than two years. T4 has only 216 usable rows at h = 7 and four or five station cells per test. With five cells, the smallest two-sided Wilcoxon *p*-value is 0.0625, so no per-cell comparison in T4 can be significant at 0.05 by construction; we therefore report the T4 comparisons between CL and B5 descriptively. One regime calendar was applied to both provinces, although October was wet at the Kubu Raya stations in both years (Section 2.5). Only two of the eight station coordinates are published; the other six were estimated from a map and could not be validated independently (Section 2.4). The validated errors (about 210 m) are small relative to a CHIRPS cell (0.05°, about 5.5 km), and bilinear interpolation softens the effect of cell boundaries, but we cannot show that the same holds for the six estimated stations. The suspect segment at BRG3 was excluded from every feature, target, and metric (Section 2.2); the degradation analysis with the segment included is retained in `results/drift_regression.csv`.

*Model selection.* The selected Ridge penalty lay at the edge of the grid at three horizons (α = 0.001 at h = 14 and 21, α = 1000 at h = 30). The grid was locked before selection and was not widened afterwards, so better values may exist outside it. LSTM hyperparameters were selected with a single seed. In the locked runs, B5 was retrained without the holdout rows even when their targets had already been observed; D4 showed that adding them moved the best T4 skill by up to nine percentage points without changing the picture.

*Diagnostics.* D1 was meant to test a change model shrunk towards persistence, but selection chose a near-zero penalty at h = 14 and 21, so the shrinkage never occurred. D1 is therefore inconclusive on that mechanism and should not be read as evidence that a change target does not help. H4 tests the mechanism directly with a penalty fixed in advance. The name of the event metric was specified before the horizon results, but its definition was not.

*Researcher degrees of freedom.* We were not blind to the data (Section 3.10). Descriptive statistics of the drought and the seven-day baseline results were seen before the horizon rule was locked, and the regime boundaries were drawn from the seasonal pattern of the data. The staged tags limit, but do not remove, the scope for choices informed by the results. We consider the negative result robust to this, since we found no model that beats persistence despite every opportunity to make one; the positive patterns (T5 skill, the damped trend in T4) are the results most exposed to it, which is why we have moved them to the confirmatory study.

## 6. Conclusions

We asked whether adaptive models keep tropical peat GWT forecasts accurate through a hydrological regime shift. On eight stations in Sumatra and Borneo through the 2019 El Niño drought, and under decision rules locked before each stage of the analysis, the answer was no: no static, periodically retrained, or continual-learning model beat persistence over the evaluation period at any horizon from 3 to 30 days.

Four points follow from this study.

1. Persistence is a demanding reference for slow peat dynamics. GWT forecasting studies should report skill against it, on identical rows, with time-ordered and online evaluation across regimes.
2. Periodic retraining and continual learning cannot anticipate a regime they have not seen. In the drought, every candidate lost to persistence, and replay and EWC were indistinguishable from simple retraining.
3. Error-based and event-based skill can disagree. A model that loses on MAE may still anticipate threshold crossings that persistence cannot, so the evaluation should match the operational use.
4. The best forecaster may depend on the regime. This is a hypothesis, not a finding. We have stated it in advance and will test it on independent data.

The staged locking protocol made the negative result possible to report. Each stage fixed the next decision before its results were known, so the absence of a winning model is an outcome of the design rather than the end of a search.

## Appendix A. Degradation ratios per station

**Table A1.** Ratio of drought (T4) to pre-drought (first half of 2019) MAE for a static panel Ridge model on level lags (seven-day horizon) trained on 2018 targets, at the five stations with T4 rows <!-- results/drift_regression.csv -->. A ratio above 1 means the model degraded in the drought. The main text uses the panel model with the BRG3 suspect segment excluded. The last two columns keep the segment, to show its effect: at BRG3 it would raise the panel ratio from 1.95 to 3.40, which is why the segment is excluded from every feature, target, and metric. With the segment excluded, only nine drought rows remain at BRG3.

<!-- BEGIN TABLEA1 (src/92_manuscript_tables.py) -->
| Station | Pre-drought rows | Drought rows | Panel | Per station | Panel, suspect kept | Per station, suspect kept |
|:---|---:|---:|---:|---:|---:|---:|
| BRG3 | 181 | 9 | 1.95 | 1.89 | 3.40 | 4.74 |
| BRG5 | 119 | 35 | 2.43 | 2.05 | 2.43 | 2.05 |
| BRG6 | 181 | 99 | 2.68 | 1.74 | 2.68 | 1.74 |
| BRG19 | 138 | 53 | 1.17 | 1.14 | 1.17 | 1.14 |
| BRG20 | 59 | 20 | 0.99 | 2.20 | 0.99 | 2.20 |
<!-- END TABLEA1 -->

## Data and code availability

[TODO: Zenodo DOI for the repository at tag analysis-frozen. Raw data: Taufik et al. (2022), Data in Brief 41:107903.]

## References

<!-- DOIs checked against Crossref on 2026-11-16, except entries marked [no DOI]. Verify page ranges and formatting to EMS style before submission. -->

- Carbon Brief (2026). El Niño: Indonesia fire emissions in 2026 'on track' to match record for this century. 11 September 2026. <!-- add URL; verify 76 MtC -->
- CNN Indonesia (2026). Kemenkes catat 283.270 orang idap ISPA akibat karhutla. 14 October 2026. <!-- add URL; prefer Ministry of Health primary release -->
- Chaudhry, A., Rohrbach, M., Elhoseiny, M., Ajanthan, T., Dokania, P.K., Torr, P.H.S., Ranzato, M. (2019). On tiny episodic memories in continual learning. arXiv:1902.10486. [no DOI]
- CREA (Centre for Research on Energy and Clean Air) (2026). Choking on inaction: Indonesia's fires led to 1,700 deaths in August. 14 September 2026. https://energyandcleanair.org/?p=16305 <!-- verify method -->
- Dawid, A.P. (1984). Statistical theory: the prequential approach. Journal of the Royal Statistical Society A, 147, 278–292. <!-- DOI to check -->
- De Lange, M., Aljundi, R., Masana, M., Parisot, S., Jia, X., Leonardis, A., Slabaugh, G., Tuytelaars, T. (2022). A continual learning survey: defying forgetting in classification tasks. IEEE Transactions on Pattern Analysis and Machine Intelligence, 44, 3366–3385. https://doi.org/10.1109/TPAMI.2021.3057446
- Funk, C., et al. (2015). The climate hazards infrared precipitation with stations — a new environmental record for monitoring extremes. Scientific Data, 2, 150066. <!-- DOI to check -->
- Gama, J., Žliobaitė, I., Bifet, A., Pechenizkiy, M., Bouchachia, A. (2014). A survey on concept drift adaptation. ACM Computing Surveys, 46(4), 1–37. https://doi.org/10.1145/2523813
- Gardner, E.S., McKenzie, E. (1985). Forecasting trends in time series. Management Science, 31, 1237–1246. <!-- DOI to check -->
- Gilbert, J., Boateng, C.D., Aryee, J.N.A., et al. (2025). A systematic review of machine learning models for groundwater level prediction. Applied Computing and Geosciences, 100303. https://doi.org/10.1016/j.acags.2025.100303
- Government of Indonesia (2016). Government Regulation No. 57/2016 amending Government Regulation No. 71/2014 on the protection and management of peat ecosystems. <!-- verify article number of the 0.4 m criterion -->
- Hayasaka, H., Usup, A., Naito, D. (2020). New approach evaluating peatland fires in Indonesian factors. Remote Sensing, 12, 2055. https://doi.org/10.3390/rs12122055
- Hochreiter, S., Schmidhuber, J. (1997). Long short-term memory. Neural Computation, 9, 1735–1780.
- Holm, S. (1979). A simple sequentially rejective multiple test procedure. Scandinavian Journal of Statistics, 6, 65–70.
- Hooijer, A., Page, S., Jauhiainen, J., Lee, W.A., Lu, X.X., Idris, A., Anshari, G. (2012). Subsidence and carbon loss in drained tropical peatlands. Biogeosciences, 9, 1053–1071. https://doi.org/10.5194/bg-9-1053-2012
- Huijnen, V., Wooster, M.J., Kaiser, J.W., et al. (2016). Fire carbon emissions over maritime southeast Asia in 2015 largest since 1997. Scientific Reports, 6, 26886. https://doi.org/10.1038/srep26886
- Irfan, M., Koriyanti, E., Saleh, K., et al. (2024). Dynamics of peatland fires in South Sumatra in 2019: role of groundwater levels. Land, 13, 373. https://doi.org/10.3390/land13030373
- Ismail, I., Haghighi, A.T., Marttila, H., et al. (2021). Water table variations on different land use units in a drained tropical peatland island of Indonesia. Hydrology Research, 52, 1372–1388. https://doi.org/10.2166/nh.2021.062 <!-- check end page -->
- Jibhakate, T.M., Katpatal, Y.B. (2024). Prediction of rainfall and groundwater using machine learning algorithms for Nagpur Division. MAUSAM, 75(3). https://doi.org/10.54302/mausam.v75i3.6272
- Kapoor, S., Narayanan, A. (2023). Leakage and the reproducibility crisis in machine-learning-based science. Patterns, 4, 100804. https://doi.org/10.1016/j.patter.2023.100804
- Ke, G., et al. (2017). LightGBM: a highly efficient gradient boosting decision tree. Advances in Neural Information Processing Systems 30. [no DOI]
- Kirkpatrick, J., et al. (2017). Overcoming catastrophic forgetting in neural networks. PNAS, 114, 3521–3526. <!-- DOI to check -->
- Klemeš, V. (1986). Operational testing of hydrological simulation models. Hydrological Sciences Journal, 31, 13–24. https://doi.org/10.1080/02626668609491024
- Li, L., Sali, A.B., Liew, J.T., et al. (2024). Machine learning for peatland ground water level (GWL) prediction via IoT system. IEEE Access, 12. https://doi.org/10.1109/ACCESS.2024.3419237
- Madani Berkelanjutan (2026). August 2026: the peak of the forest and land fire crisis and a real threat to Indonesia's climate commitments. 28 August 2026. https://madaniberkelanjutan.id/news-event/august-2026-the-peak-of-the-forest-and-land-fire-crisis-and-a-real-threat-to-indonesia-s-climate-commitments
- Nosek, B.A., Ebersole, C.R., DeHaven, A.C., Mellor, D.T. (2018). The preregistration revolution. PNAS, 115, 2600–2606. https://doi.org/10.1073/pnas.1708274114
- Page, S.E., Siegert, F., Rieley, J.O., Boehm, H.-D.V., Jaya, A., Limin, S. (2002). The amount of carbon released from peat and forest fires in Indonesia during 1997. Nature, 420, 61–65. https://doi.org/10.1038/nature01131
- Pappenberger, F., Ramos, M.H., Cloke, H.L., Wetterhall, F., Alfieri, L., Bogner, K., Mueller, A., Salamon, P. (2015). How do I know if my forecasts are better? Using benchmarks in hydrological ensemble prediction. Journal of Hydrology, 522, 697–713. https://doi.org/10.1016/j.jhydrol.2015.01.024
- Putra, S.S., Holden, J., Baird, A.J. (2021). The effects of ditch dams on water-level dynamics in tropical peatlands. Hydrological Processes, 35, e14174. https://doi.org/10.1002/hyp.14174
- Reuters (2026a). [Ministry of Forestry burned-area figure, 202,000 ha as of 18 August 2026.] <!-- find exact article; Wikipedia cites Reuters 11 Aug and Kumparan 18 Aug 2026; prefer SiPongi -->
- Reuters (2026b). [Nusantara Atlas: 1.92 million ha burned by end of September 2026.] 20 October 2026. <!-- add title and URL -->
- Save the Children (2026). Indonesia wildfire haze threatens children across Southeast Asia and closes schools. <!-- add date and URL -->
- Taufik, M., Setiawan, B.I., Van Lanen, H.A.J. (2015). Modification of a fire drought index for tropical wetland ecosystems by including water table depth. Agricultural and Forest Meteorology, 203, 1–10. https://doi.org/10.1016/j.agrformet.2014.12.006 <!-- check volume/pages -->
- Taufik, M., Veldhuizen, A.A., Wösten, J.H.M., Van Lanen, H.A.J. (2019). Exploration of the importance of physical properties of Indonesian peatlands to assess critical groundwater table depths, associated drought and fire hazard. Geoderma, 347, 160–169. https://doi.org/10.1016/j.geoderma.2019.04.001 <!-- check authors/volume/pages -->
- Taufik, M., TW, M., Awaluddin, Mukharomah, A.K., Minasny, B. (2022). Groundwater table and soil-hydrological properties datasets of Indonesian peatlands. Data in Brief, 41, 107903. https://doi.org/10.1016/j.dib.2022.107903
- Utomo, W.Y., Aini, F.K., Askary, M., et al. (2026). Machine learning-aided groundwater level and CO2 emission estimations in Indonesian tropical peatland. Environmental Advances, 100762. https://doi.org/10.1016/j.envadv.2026.100762
- Wilcoxon, F. (1945). Individual comparisons by ranking methods. Biometrics Bulletin, 1, 80–83.
- Wösten, J.H.M., Clymans, E., Page, S.E., Rieley, J.O., Limin, S.H. (2008). Peat–water interrelationships in a tropical peatland ecosystem in Southeast Asia. Catena, 73, 212–224. https://doi.org/10.1016/j.catena.2007.07.010
- Wunsch, A., Liesch, T., Broda, S. (2021). Groundwater level forecasting with artificial neural networks: a comparison of long short-term memory (LSTM), convolutional neural networks (CNNs), and non-linear autoregressive networks with exogenous input (NARX). Hydrology and Earth System Sciences, 25, 1671–1687. https://doi.org/10.5194/hess-25-1671-2021
- Yonekura, K., Miyazaki, S., Aichi, M., et al. (2025). Prediction of groundwater level in Indonesian tropical peatland forest plantations using machine learning. Artificial Intelligence in Geosciences, 100148. https://doi.org/10.1016/j.aiig.2025.100148
