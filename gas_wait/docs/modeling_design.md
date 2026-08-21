# Gas Wait — Feature and Target Design

Status: design only. No model training. No new data collection. No resampling of weekly series to daily.

This document specifies a **point-in-time-correct** first experiment using only verified EIA series already in the repository.

## 1. Recommended target

**Primary target (first experiment): A — one-week-ahead U.S. regular retail gasoline price change.**

```
y_t = retail_monday_{T+7} − retail_monday_T
```

Units: dollars per gallon. Decision mapping later:

- **FILL UP** if predicted change > +δ
- **WAIT** if predicted change < −δ
- **NO CLEAR SIGNAL** otherwise

δ is chosen from historical mean absolute weekly move (~3.3¢), not from in-sample accuracy.

### Why A, not B, as the product target

The product question is whether **pump prices** will be higher or lower. The only consumer-facing series we currently possess is EIA weekly retail (Form EIA-878 / Gasoline and Diesel Fuel Update). Daily WTI, NY Harbor, Gulf Coast, and LA RBOB are **wholesale spots**. Our EDA already showed:

- Retail vs WTI *levels* correlate 0.96 because of shared trend, not because they are the same object.
- Same-week WTI *changes* correlate 0.58 with retail changes; 1-week-ahead WTI still correlates 0.38.
- Mean absolute weekly retail move is 3.3¢; direction is barely unbalanced (~52% down).
- Pass-through is **short** (0–2 weeks), not 6–8 weeks.

A daily wholesale target (option B) is statistically cleaner (≈10k business-day rows, native daily grain) but answers a different question: “Did NY Harbor gasoline go up tomorrow?” Success there would not license a WAIT/FILL recommendation at the pump. Wholesale can lead retail and still fail as a proxy if rack-to-retail margins, taxes, or station stickiness absorb the move.

### Tradeoffs

| Option | What it measures | Strength | Weakness |
| --- | --- | --- | --- |
| **A. Weekly retail Δ, daily features lagged to a public cutoff** | The consumer outcome | Aligns with product; EDA already found signal; ~1,870 labels | Weekly labels; cannot honestly score a 72-hour local move; national average |
| **B. Daily wholesale gasoline Δ** | Hub spot dynamics | Many observations; true daily evaluation; tests 1/3/5-day momentum | Not the pump; regional hubs ≠ U.S. average; would overstate product readiness |
| **C. Daily nowcast of next Monday’s retail, updated each weekday** | Same label as A, scored daily | Closest prototype of a daily recommendation using weekly retail | Same weekly label repeated; mid-week “updates” must not leak the Monday print |

**Recommendation:** train and evaluate on **A**. Use **C only as a scoring protocol** (emit a forecast every weekday for the same upcoming Monday print). Run **B as a diagnostic**, not as the shippable target: it tells us whether daily spots are themselves predictable, which informs feature choice, but it is not the Gas Wait label.

Do not use “cheaper sometime in the next 1–3 weeks” (EDA TARGET D). The min of three future prices is cheaper ~62% of the time even under a near-random walk.

## 2. Point-in-time rules

Every feature is defined by four clocks:

1. **Observation timestamp** — the economic date the number refers to.
2. **Public-available timestamp** — the earliest time a real-time system could have seen it.
3. **Model-allowed timestamp** — `public_available_ts <= prediction_ts`.
4. **Lag** — extra buffer when availability is uncertain (holidays, API lag, revisions).

### Official EIA clocks (do not use observation date as availability)

| Series | Observation | Public availability | Model rule |
| --- | --- | --- | --- |
| U.S. regular retail | Monday pump survey | **Through 2025-04-04:** ~5:00 p.m. ET Monday. **From week of 2025-04-07:** ~10:00 a.m. ET **Tuesday** (Wednesday if Monday is a federal holiday). Always a Monday price. Source: [GDFU publication change](https://www.eia.gov/petroleum/gasdiesel/notice.php), [holiday schedule](https://www.eia.gov/petroleum/gasdiesel/schedule.php). | Conservative first experiment: treat Monday retail as available at **Tuesday 12:00 p.m. ET**. Do not use `retail_T` as a feature before that instant. |
| U.S. gasoline stocks | Week ending Friday 7:00 a.m. | **Wednesday after 10:30 a.m. ET** (CSV/XLS); PDF/HTML after 1:00 p.m. Holiday weeks often slip to Thursday. Source: [WPSR schedule](https://www.eia.gov/petroleum/supply/weekly/schedule.php). | Stocks for Friday *F* are available only at **Wednesday 12:00 p.m. ET** the following week (or Thursday noon on holiday weeks). Never join Friday stocks onto Monday/Tuesday predictions in the same cycle. |
| Daily WTI / NYH / Gulf / LA RBOB | Business-day close | EIA republishes daily spots on the next weekday morning (Today in Energy snapshot typically 7:30–8:30 a.m. ET). The PET_PRI_SPT HTML table is a weekly page; the API series can update intra-week, but we do not have vintage timestamps. | Conservative: close on calendar date *D* is available at **D+1 weekday 12:00 p.m. ET**. Never use same-day close at a morning prediction. Weekends/holidays: last completed session, still lagged one weekday. |

### First-experiment prediction clock

One labeled row per retail week. Prediction time is **Tuesday 12:00 p.m. ET** after Monday *T*’s retail price is public.

- `prediction_date` = Tuesday following Monday *T* (or Wednesday after Monday holidays).
- `target_timestamp` = Monday *T+7*.
- `target` = `retail(T+7) − retail(T)`.
- Features may use retail through *T*, daily spots with observation date ≤ Monday *T* (available Tuesday morning), and inventories from the **previous** WPSR only (week ending Friday *T−10*, released the prior Wednesday).

This is stricter than “as-of Monday.” It matches what a live system can do after the GDFU print.

Later (protocol C): emit the same target from Wednesday–Monday as new daily closes and the Wednesday WPSR arrive. Those extra rows share a label; they are for operational testing, not independent samples.

## 3. Daily market features

Construct from native daily series. **Do not interpolate missing sessions.** Rolling windows count **business days** (rows present), not calendar days. Weekend gaps stay gaps.

### WTI (`RWTC`, $/bbl)

Include (economically justified; EDA found crude changes predictive, levels not):

| Feature | Definition | Why |
| --- | --- | --- |
| `wti_d1` | `p_t − p_{t-1}` | Immediate cost shock |
| `wti_d3`, `wti_d5` | 3- and 5-session changes | Matches 3-day product horizon and one work week |
| `wti_vol_20` | Std of 1-session changes, 20 sessions | Abstain when crude is chaotic |
| `wti_mom_10` | 10-session change / 20-session vol | Scale-free momentum |

Skip 7-calendar-day change as a separate primitive if `wti_d5` is present (they are collinear on a business-day calendar). Skip percent changes of WTI as a default: 2020 near-zero crude explodes percentages (EDA Spearman ≫ Pearson on pct). Skip deviations from long moving averages of **levels**: those recover the 0.96 trend correlation.

Allowed timestamp: last WTI session with `session_date + 1 weekday ≤ prediction_ts`.

### Wholesale gasoline (NY Harbor `EER_EPMRU_PF4_Y35NY_DPG`, Gulf Coast `EER_EPMRU_PF4_RGC_DPG`, LA RBOB `EER_EPMRR_PF4_Y05LA_DPG`, $/gal)

Include:

| Feature | Definition | Why |
| --- | --- | --- |
| `nyh_d1`, `nyh_d5` | NY Harbor 1- and 5-session Δ | East Coast wholesale; largest PADD by population |
| `gc_d1`, `gc_d5` | Gulf Coast 1- and 5-session Δ | Refining-center wholesale |
| `crack_nyh_d5` | `42 × Δ_5 NYH − Δ_5 WTI` | Gasoline vs crude (simple crack, not 3-2-1; we lack heating oil) |
| `spread_nyh_gc` | NYH − Gulf, level **and** 5-session change | Regional dislocation |
| `gas_vol_20` | 20-session std of NYH 1-session Δ | NO CLEAR SIGNAL input |

**LA RBOB:** keep as a **diagnostic / California-specific** feature, not in the national MVP model. CARB/RBOB Los Angeles is a distinct specification and often diverges from the U.S. average. Including it in a national model invites spurious West Coast noise.

Skip: raw wholesale **levels**; same-day NYH and Gulf both as 1/3/5/7-day plus momentum plus MA deviations (collinear soup). First model uses 1-day and 5-day only.

## 4. Inventory features

Series: week-ending Friday, values in **thousand barrels** (EIA series description), despite older metadata saying million barrels.

Public only after Wednesday 10:30 a.m. ET. At Tuesday 12:00 prediction time, **this week’s Friday stocks are not public**. Use the WPSR released the previous Wednesday.

| Feature | Definition | Availability |
| --- | --- | --- |
| `inv_wow` | Latest public week − prior week | Prior Wednesday |
| `inv_yoy` | Latest public week − 52 weeks earlier | Same |
| `inv_seasonal_z` | (latest − mean of that calendar week over prior 5 years) / 5-year std | Expanding, past years only |
| `inv_wow_lag1` | Previous Wow change | Extra lag; EDA peak inventory lead was 1 week |

EDA: inventory 1-week change vs retail 1-week change peaked at lag 1 (r ≈ −0.21), then faded and flipped sign at lags 6–8 (seasonal artifact). Keep Wow and seasonal z; do not treat 6–8 week inventory lags as causal.

Do **not** forward-fill Friday stocks onto Monday/Tuesday as if the number had been released.

## 5. Calendar / seasonality

EDA: weekly retail tends to **rise Jan–May** (March strongest) and **fall Jun–Dec** (Oct–Dec weakest). November: ~70% of weeks down; March: ~33% down.

Include:

- `month` (1–12) or two Fourier terms: `sin(2π doy / 365.25)`, `cos(2π doy / 365.25)`
- `weekofyear` only if month is not also one-hot; prefer Fourier over 52 dummies on ~1,870 rows
- `is_summer` = Memorial Day week through Labor Day week (U.S. driving season)
- `is_holiday_travel_week` = Thanksgiving week, Independence Day week, Labor Day week, Memorial Day week
- `dow_prediction` only if we later emit weekday forecasts (protocol C)

Holiday flags must use **U.S. federal / travel calendar known at prediction time**, not realized demand.

## 6. Regional mismatch

The label is **U.S. national** volume-weighted retail. Inputs are hub spots:

- NY Harbor → PADD 1 (East Coast)
- Gulf Coast → PADD 3
- Los Angeles RBOB → PADD 5 / California

A national target is **appropriate for this experiment** because that is the only retail series we have, and EIA’s national average is dominated by PADDs 1–3 more than California. It is **not** appropriate as the long-run product:

- National average hides $0.20–$0.50 local gaps.
- LA RBOB should pair with California or PADD 5 retail, not U.S. regular.
- A 3-day consumer recommendation needs metro or station retail (licensed OPIS/PDI), not EIA weekly national.

First model: national retail target + NYH + Gulf + WTI. Park LA RBOB for a later regional experiment. Eventually match wholesale hub → PADD/city retail.

## 7. First modeling dataset schema

Grain: one row per retail week. No daily expansion of the weekly label for training.

Example row (illustrative clocks, not a computed feature dump):

| Field | Example | Clock |
| --- | --- | --- |
| `prediction_date` | 2026-08-11 | Tuesday 12:00 p.m. ET |
| `target` | `retail(2026-08-17) − retail(2026-08-10)` | Label uses future Monday; **not a feature** |
| `target_timestamp` | 2026-08-17 | Next GDFU Monday |
| `retail_last` | Monday 2026-08-10 price | Public Tue 2026-08-11 ~10:00 |
| `retail_d7` | `retail(08-10) − retail(08-03)` | Momentum; both prints public |
| `wti_d1`, `wti_d5`, `wti_vol_20`, `wti_mom_10` | From daily WTI through **2026-08-10** | 08-10 close treated public Tue 12:00 |
| `wti_feature_timestamp` | 2026-08-10 | Observation date of last spot used |
| `nyh_d1`, `nyh_d5`, `gc_d1`, `gc_d5` | Same last session 2026-08-10 | |
| `crack_nyh_d5`, `spread_nyh_gc` | From those sessions | |
| `gas_vol_20` | NYH 20-session vol through 2026-08-10 | |
| `inv_wow`, `inv_yoy`, `inv_seasonal_z` | Week ending **2026-08-01** | Released Wed 2026-08-06 10:30; week ending 08-07 **not** allowed |
| `inv_feature_timestamp` | 2026-08-01 | Inventory observation Friday |
| `month`, `sin_doy`, `cos_doy`, `is_summer`, `is_holiday_travel_week` | As of 2026-08-11 | Known calendar |

Required columns on every row: `prediction_date`, `prediction_ts_utc`, `target`, `target_timestamp`, `retail_feature_timestamp`, `spot_feature_timestamp`, `inventory_feature_timestamp`, plus features. Persist timestamps so leakage audits are mechanical.

## 8. Leakage inventory

| Source | How it leaks | Control |
| --- | --- | --- |
| Future daily spots | Using Tuesday–Monday closes to predict a Tuesday-morning forecast | Hard cutoff: last session with availability ≤ `prediction_ts` |
| Same-day close | Morning prediction using that afternoon’s WTI | +1 weekday lag |
| Friday inventory on Monday/Tuesday | WPSR is Wednesday 10:30 | Join inventories only if `wpsr_release_ts ≤ prediction_ts` |
| Monday retail as a Tuesday-morning feature before 10:00 | GDFU is now Tuesday ~10:00 (since 2025-04-07); historically Monday 5:00 p.m. | Conservative Tuesday 12:00; regime-aware later |
| Target in features | `retail(T+7)` anywhere except `target` | Build features, then attach label in a last step |
| Rolling windows | 20-day vol that includes days after cutoff if computed on a full-history series then sliced | Compute rolling stats on the as-of series, or shift the entire rolling column by the availability lag |
| Interpolation | Filling weekend/holiday holes with linear interpolation fabricates intra-week path | No interpolation. Last-observation-carried-forward only as “last known public print,” never as a new observation date |
| Forward-fill of weekly retail onto daily rows | Makes it look like Tuesday–Sunday had a fresh pump survey | Do not upsample retail to daily for training labels |
| Forward-fill of inventories across the unreleased window | Treats Friday stocks as known Saturday–Tuesday | Leave NA until Wednesday noon |
| EIA revisions | API v2 is “updated constantly”; backfilled revisions are not the print a 2019 model would have seen | First experiment: latest vintage, documented limitation. Next: store fetch time / period vintage |
| Holiday WPSR/GDFU slips | Using a Wednesday inventory that actually released Thursday | Holiday calendar table; delay availability one calendar day when EIA schedule says so |
| Random or shuffled train/test | Leaks future weeks through shared seasonality fits | Time-based split only |
| Fitting δ (WAIT/FILL threshold) on the test set | Inflates decision accuracy | Freeze δ from training-period MAE or a preset 3–5¢ |
| Using LA RBOB to “explain” national spikes that were California-only | Geographic leakage of a sort | Exclude from national MVP |
| Protocol C duplicate labels treated as i.i.d. | Five weekday forecasts of the same Monday overstate N | Cluster metrics by `target_timestamp` |

## 9. First experiment

**Name:** `exp01_weekly_retail_d7_tuesday_cutoff`

**Question:** Using only information public at Tuesday 12:00 p.m. ET, can lagged daily WTI and Gulf/NY Harbor gasoline changes improve 1-week-ahead U.S. retail direction and error versus a momentum baseline?

**Why this is enough:** It uses data we already have, respects publication lags, targets the consumer series, and tests whether the new daily spots add anything beyond the weekly EDA (WTI r=0.38, momentum r=0.55).

### Feature groups (keep small)

1. Retail momentum: `retail_d7`
2. Crude: `wti_d1`, `wti_d5`, `wti_vol_20`
3. Wholesale: `nyh_d5`, `gc_d5`, `crack_nyh_d5`
4. Inventories: `inv_wow`, `inv_seasonal_z`
5. Calendar: Fourier doy pair + `is_summer`

No LA RBOB. No level prices. No interpolation.

### Train / test split

Walk-forward, not random:

- **Train:** 1994-01-01 through 2016-12-31 (drop 1990–93 until inventory/retail overlap is dense and the 1990–91 retail gap is behind us)
- **Validation:** 2017-01-01 through 2019-12-31
- **Test:** 2020-01-01 through latest complete week (includes COVID, 2022 spike, 2025 GDFU timing change)

Also report test **excluding 2020-03 through 2020-05** as a sensitivity. Never tune on test.

If a later model uses weekday protocol C, split by `target_timestamp`, not by `prediction_date`.

### Leakage controls in code (when we implement, not now)

- Availability join functions with explicit `prediction_ts`
- Unit tests: a Tuesday prediction must not see Wednesday WPSR or Tuesday afternoon spots
- Assert `spot_feature_timestamp < prediction_date` (or ≤ Monday *T*)
- Assert `inventory_feature_timestamp` Friday is at least 10 calendar days before prediction Tuesday
- No future-derived seasonal statistics (5-year week-of-year mean uses only years `< current year`)

### Baseline model (not trained in this step)

1. **Naive:** predict 0 (always NO CLEAR SIGNAL if \|0\| < δ)
2. **Momentum:** predict `retail_d7`
3. **Crude sign:** predict `sign(wti_d5) × historical mean |retail_d7|`

First statistical model, when authorized: **regularized linear regression** (Ridge) on the feature groups above, and a **logistic / ordinal head** only for direction. Tree models later; they hide leakage.

### Evaluation metrics

| Metric | Role |
| --- | --- |
| MAE, RMSE ($/gal) | Primary regression quality |
| Directional accuracy | vs 51.9% majority and vs momentum (~69% in-sample — expect shrinkage) |
| Brier / log loss if we output probabilities | Calibration for NO CLEAR SIGNAL |
| Utility: mean $ saved vs always-fill, with δ ∈ {0.03, 0.05} frozen from train | Product metric (EDA TARGET E) |
| Coverage: fraction of weeks with \|ŷ\| > δ | If coverage is tiny, the product is silent |
| Stress weeks: Katrina, 2008, 2020, 2022 | Tail failure |

Do not optimize accuracy alone. A model that is 60% directional but wrong by 15¢ in crisis weeks is worse than NO CLEAR SIGNAL.

### Additional data needed next

1. **Licensed daily or metro retail** (OPIS or PDI) — required before any honest 3-day consumer claim.
2. **Release timestamps / vintages** for EIA series (or a self-collected as-of archive).
3. EIA weekly **PADD or selected-city retail** to pair with NYH / Gulf / LA.
4. Heating oil or ULSD daily spot if we want a real 3-2-1 crack.
5. Refinery utilization, gasoline production, product supplied (already on the petroleum roadmap; still weekly).

Until (1) exists, Gas Wait can study **weekly national** WAIT/FILL with a dead zone, and can *update that weekly forecast daily* as spots move. It cannot claim a 72-hour local pump call.

## Decision

| Item | Choice |
| --- | --- |
| Target | Weekly U.S. regular retail Δ, Monday-to-Monday ($/gal) |
| Prediction time | Tuesday 12:00 p.m. ET |
| Daily spots | Features only, lagged +1 weekday; NYH + Gulf + WTI; not LA in MVP |
| Inventories | Prior WPSR only (Wednesday 10:30 clock) |
| First model | Ridge vs momentum/crude baselines |
| Not this round | Training code, daily wholesale-as-label, interpolation, third-party retail APIs |
