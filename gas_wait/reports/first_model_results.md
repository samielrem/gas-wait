# First model results — exp01 weekly retail Δ

Research backtest only. Ridge alpha fixed at 1.0. Thresholds 3/4/5¢ were **not** tuned on test.

## Dataset

- Kept rows: **1747**
- Dropped rows: **132**
- Prediction dates: 1993-01-19 to 2026-08-11
- Chronological cutoff: **2016-07-19** (first 70% train / last 30% test)
- Train weeks: 1222; test weeks: 525

### Dropped-row reasons

| Reason | Count |
| --- | ---: |
| missing_required_features | 124 |
| missing_retail_at_t | 6 |
| missing_or_nonweekly_retail_at_tplus7 | 2 |

Warm-up drops are expected: 20-session volatility, 5-year inventory seasonal z, and the 1990–91 retail gap.

## Leakage checks

These assertions ran on the kept dataset before training and passed:

- `spot_feature_timestamp <= retail_monday` and `< prediction_date` (Monday close, not Tuesday close)
- `inventory_release_ts_utc <= prediction_ts_utc`
- inventory Friday is at least 10 calendar days before the Tuesday prediction
- `target_timestamp` is Monday T+7 and after `prediction_date`
- retail features use only Monday T and earlier prints (`retail_d7`, `retail_d14`)
- rolling 1/3/5-session changes and 20-session vol are computed on the daily series in time order, then as-of joined to Monday T
- no interpolation; 7-calendar-day WTI change is last price on or before T-7, not a filled path
- LA RBOB is not in the feature set

Documented limitation: EIA API values are latest vintage, not original prints.

## Holdout regression (final 30%)

| Model | N | MAE | RMSE | R² | Corr | Directional acc. |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| MODEL 1: Momentum baseline (retail_d7) | 525 | 0.0395 | 0.0621 | 0.0891 | 0.5444 | 0.681 |
| MODEL 2: Ridge, retail momentum only | 525 | 0.0350 | 0.0549 | 0.2890 | 0.5385 | 0.695 |
| MODEL 3: Ridge, retail + daily markets | 525 | 0.0293 | 0.0464 | 0.4914 | 0.7164 | 0.775 |
| MODEL 4: Ridge, retail + markets + inventory + seasonality | 525 | 0.0286 | 0.0455 | 0.5112 | 0.7251 | 0.773 |

Directional accuracy ignores actual zeros. Majority-class baseline on this problem is ~52%.

## Walk-forward (expanding, predict each year from all prior years)

| Model | N | MAE | RMSE | R² | Corr | Directional acc. |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| MODEL 1: Momentum baseline (retail_d7) | 1021 | 0.0400 | 0.0581 | 0.1772 | 0.5884 | 0.700 |
| MODEL 2: Ridge, retail momentum only | 1021 | 0.0362 | 0.0524 | 0.3304 | 0.5758 | 0.704 |
| MODEL 3: Ridge, retail + daily markets | 1021 | 0.0294 | 0.0445 | 0.5161 | 0.7270 | 0.792 |
| MODEL 4: Ridge, retail + markets + inventory + seasonality | 1021 | 0.0291 | 0.0438 | 0.5309 | 0.7349 | 0.791 |

## Decision metrics on holdout

WAIT if predicted Δ ≤ −δ; FILL UP if predicted Δ ≥ +δ; else NO CLEAR SIGNAL.
A WAIT/FILL call is counted correct if the actual next-week change has the intended sign.
Theoretical savings vs always-fill-now: WAIT weeks contribute `15 × (p_t − p_{t+7})`; FILL UP and NO SIGNAL contribute 0.

### δ = $0.03

| Model | WAIT | FILL | No signal | % no signal | % correct among signals | Mean actual after WAIT | Mean actual after FILL | Mean $ / 15 gal vs always-fill |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| MODEL 1: Momentum baseline (retail_d7) | 129 | 138 | 258 | 0.491 | 0.787 | -0.0437 | 0.0469 | 0.161 |
| MODEL 2: Ridge, retail momentum only | 64 | 72 | 389 | 0.741 | 0.824 | -0.0608 | 0.0602 | 0.111 |
| MODEL 3: Ridge, retail + daily markets | 102 | 119 | 304 | 0.579 | 0.896 | -0.0562 | 0.0655 | 0.164 |
| MODEL 4: Ridge, retail + markets + inventory + seasonality | 104 | 123 | 298 | 0.568 | 0.899 | -0.0550 | 0.0637 | 0.164 |

### δ = $0.04

| Model | WAIT | FILL | No signal | % no signal | % correct among signals | Mean actual after WAIT | Mean actual after FILL | Mean $ / 15 gal vs always-fill |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| MODEL 1: Momentum baseline (retail_d7) | 95 | 109 | 321 | 0.611 | 0.809 | -0.0520 | 0.0507 | 0.141 |
| MODEL 2: Ridge, retail momentum only | 35 | 47 | 443 | 0.844 | 0.817 | -0.0864 | 0.0637 | 0.086 |
| MODEL 3: Ridge, retail + daily markets | 74 | 93 | 358 | 0.682 | 0.928 | -0.0712 | 0.0740 | 0.150 |
| MODEL 4: Ridge, retail + markets + inventory + seasonality | 74 | 93 | 358 | 0.682 | 0.922 | -0.0670 | 0.0774 | 0.142 |

### δ = $0.05

| Model | WAIT | FILL | No signal | % no signal | % correct among signals | Mean actual after WAIT | Mean actual after FILL | Mean $ / 15 gal vs always-fill |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| MODEL 1: Momentum baseline (retail_d7) | 71 | 81 | 373 | 0.710 | 0.816 | -0.0570 | 0.0555 | 0.116 |
| MODEL 2: Ridge, retail momentum only | 26 | 26 | 473 | 0.901 | 0.865 | -0.0968 | 0.0582 | 0.072 |
| MODEL 3: Ridge, retail + daily markets | 52 | 71 | 402 | 0.766 | 0.935 | -0.0806 | 0.0854 | 0.120 |
| MODEL 4: Ridge, retail + markets + inventory + seasonality | 55 | 68 | 402 | 0.766 | 0.935 | -0.0777 | 0.0878 | 0.122 |

## Answers

**A. Does Ridge beat the momentum baseline?** Yes on holdout MAE (0.0286 vs 0.0395 for MODEL 4 vs MODEL 1). Directional accuracy: 0.773 vs 0.681.

**B. Do daily wholesale/crude features add value?** Yes vs retail-only Ridge (MAE 0.0293 vs 0.0350).

**C. Does inventory add value?** Not in a way we can credit on its own. Inventory features are only in MODEL 4, bundled with seasonality. The joint MAE change vs MODEL 3 is 0.0007 $/gal — about 0.07¢, within noise relative to the market-feature jump of 0.0057 $/gal.

**D. Does seasonality add value?** Same caveat: Fourier day-of-year and `is_summer` only appear in MODEL 4. They do not produce a material holdout MAE gain once daily markets are included. Keep them as cheap controls, not as the story.

**E. Out-of-sample directional accuracy (MODEL 4 holdout):** 0.773 (525 weeks). Walk-forward: 0.791.

**F. Theoretical economic benefit (MODEL 4, δ=$0.03, holdout):** mean 0.164 USD per 15-gallon fill vs always filling now (WAIT-only P&L); total 85.87 USD over 525 test weeks. Momentum is similar on this WAIT-only metric (0.161 USD/fill) because FILL UP does not change P&L vs always-fill. FILL quality is better for MODEL 4: mean actual Δ after FILL UP is 0.0637 vs 0.0469 for momentum.

**G. WAIT/FILL signals (MODEL 4, δ=$0.03):** WAIT 104, FILL UP 123, NO CLEAR SIGNAL 298 (0.568 of weeks silent). Among signaled weeks, 0.899 had the intended sign.

**H. Strong enough to justify pursuing daily retail data?** Weekly national signal is real: daily WTI/NY Harbor/Gulf Coast improve MAE and direction vs momentum. It is **not** strong enough to ship a 3-day local recommendation. Typical weekly move is still a few cents, the label is national Monday-to-Monday, and ~57% of weeks are NO CLEAR SIGNAL at 3¢. Licensed daily/metro retail is still required for the intended product. This result **does** justify paying for that data: wholesale appears to lead the pump on a weekly clock.

**I. What next?** Keep the leakage-safe weekly dataset. Do not jump to gradient boosting. Next scientific step is PADD or city EIA retail against the matching hub, plus as-of vintages. In parallel, request OPIS/PDI quotes for daily retail. Revisit δ on a locked validation slice, not on this test set.

## Charts

- `reports/figures/holdout_actual_vs_predicted.png`
- `reports/figures/holdout_residuals.png`
- `reports/figures/holdout_prediction_distribution.png`
- `reports/figures/holdout_cumulative_savings.png`
- `reports/figures/holdout_mae_by_year.png`

## Features used

- MODEL 1: Momentum baseline (retail_d7): `retail_d7`
- MODEL 2: Ridge, retail momentum only: `retail_d7, retail_d14`
- MODEL 3: Ridge, retail + daily markets: `retail_d7, retail_d14, wti_d1, wti_d3, wti_d5, wti_vol_20, nyh_d1, nyh_d5, nyh_vol_20, gc_d1, gc_d5, crack_nyh_d5`
- MODEL 4: Ridge, retail + markets + inventory + seasonality: `retail_d7, retail_d14, wti_d1, wti_d3, wti_d5, wti_vol_20, nyh_d1, nyh_d5, nyh_vol_20, gc_d1, gc_d5, crack_nyh_d5, inv_wow, inv_seasonal_z, sin_doy, cos_doy, is_summer`

Ridge uses `StandardScaler` + `Ridge(alpha=1.0)`. No test-set tuning.
A 7-calendar-day WTI change (`wti_chg_7cal`) is stored on the dataset using last known price on or before T-7 (no interpolation) but is **not** in Ridge: it is collinear with `wti_d5` on a business-day calendar, as noted in `docs/modeling_design.md`.
