# Regional experiment — matched wholesale → retail

Research backtest. Ridge α = 1.0 and δ = $0.03 are **frozen from exp01**. Holdout cutoff is the same Tuesday as the national experiment: **2016-07-19**. The first-model report and `weekly_model_dataset.csv` were not modified.

EIA weekly regular retail (product `EPMR`, process `PTE`) has **29 geographies**. Only six have a defensible match to an EIA daily gasoline spot already in the repo.

## EIA weekly regular retail coverage

Source: EIA API v2 `petroleum/pri/gnd`, frequency=weekly, verified 2026-08-20. Same GDFU publication clock as the national series (treated as available Tuesday 12:00 p.m. ET).

| duoarea | Geography | N | Missing | Start | End |
| --- | --- | ---: | ---: | --- | --- |
| NUS | U.S. | 1885 | 6 | 1990-08-20 | 2026-08-17 |
| R10 | PADD 1 | 1789 | 0 | 1992-05-11 | 2026-08-17 |
| R20 | PADD 2 | 1789 | 0 | 1992-05-11 | 2026-08-17 |
| R30 | PADD 3 | 1785 | 0 | 1992-05-11 | 2026-08-17 |
| R40 | PADD 4 | 1789 | 0 | 1992-05-11 | 2026-08-17 |
| R50 | PADD 5 | 1785 | 0 | 1992-05-11 | 2026-08-17 |
| R1X | PADD 1A | 1743 | 0 | 1993-04-05 | 2026-08-17 |
| R1Y | PADD 1B | 1740 | 0 | 1993-04-05 | 2026-08-17 |
| R1Z | PADD 1C | 1736 | 0 | 1993-04-05 | 2026-08-17 |
| R5XCA | PADD 5 EXCEPT CALIFORNIA | 1469 | 0 | 1998-05-18 | 2026-08-17 |
| SCA | CALIFORNIA | 1372 | 0 | 2000-05-22 | 2026-08-17 |
| SCO | COLORADO | 1364 | 0 | 2000-06-05 | 2026-08-17 |
| SMN | MINNESOTA | 1375 | 0 | 2000-06-05 | 2026-08-17 |
| SNY | NEW YORK | 1372 | 0 | 2000-06-05 | 2026-08-17 |
| STX | TEXAS | 1373 | 0 | 2000-06-05 | 2026-08-17 |
| Y05LA | LOS ANGELES | 1365 | 0 | 2000-06-05 | 2026-08-17 |
| Y05SF | SAN FRANCISCO | 1372 | 0 | 2000-06-05 | 2026-08-17 |
| Y35NY | NEW YORK CITY | 1368 | 0 | 2000-06-05 | 2026-08-17 |
| Y44HO | HOUSTON | 1371 | 1 | 2000-06-05 | 2026-08-17 |
| YDEN | DENVER | 1371 | 0 | 2000-06-05 | 2026-08-17 |
| YORD | CHICAGO | 1367 | 0 | 2000-06-05 | 2026-08-17 |
| SFL | FLORIDA | 1215 | 0 | 2003-05-26 | 2026-08-17 |
| SMA | MASSACHUSETTS | 1217 | 0 | 2003-05-26 | 2026-08-17 |
| SOH | OHIO | 1211 | 0 | 2003-05-26 | 2026-08-17 |
| SWA | WASHINGTON | 1213 | 0 | 2003-05-26 | 2026-08-17 |
| Y48SE | SEATTLE | 1207 | 0 | 2003-05-26 | 2026-08-17 |
| YBOS | BOSTON | 1219 | 0 | 2003-05-26 | 2026-08-17 |
| YCLE | CLEVELAND | 1207 | 0 | 2003-05-26 | 2026-08-17 |
| YMIA | MIAMI | 1210 | 1 | 2003-05-26 | 2026-08-17 |

## Mapping (do not force a bad match)

| Region | Matched wholesale | Why this pair | Mismatch control |
| --- | --- | --- | --- |
| New York City → NY Harbor | New York Harbor | NYC pumps are supplied from the New York Harbor barge/pipeline complex. EIA NY Harbor conventional regular is the local wholesale print. Tightest city-hub pair on the East Coast. | U.S. Gulf Coast |
| Central Atlantic (PADD 1B) → NY Harbor | New York Harbor | PADD 1B is NY, NJ, PA, DE, MD, and DC. Those states price off NY Harbor / northern Colonial, not the Gulf Coast waterborne market. Broader than NYC but still a Harbor market. | U.S. Gulf Coast |
| Houston → Gulf Coast gasoline | U.S. Gulf Coast | Houston sits on the U.S. Gulf refining complex. EIA Gulf Coast conventional regular is the local rack/spot analogue. NY Harbor is a destination market, not the origin. | New York Harbor |
| Gulf Coast (PADD 3) → Gulf Coast gasoline | U.S. Gulf Coast | PADD 3 is the Gulf producing/refining region (TX, LA, AR, AL, MS, NM). The Gulf Coast spot is the regional wholesale. Broader than Houston but the same supply basin. | New York Harbor |
| Los Angeles → LA RBOB | Los Angeles, CA | Los Angeles requires CARB reformulated gasoline. EIA LA RBOB regular is the matching West Coast wholesale spec. NY Harbor conventional is a different molecule and a different logistics system. | New York Harbor |
| California → LA RBOB | Los Angeles, CA | Statewide California retail is still CARB gasoline. LA RBOB is the primary CA wholesale benchmark. Bay Area can diverge from LA, so this is a looser match than the LA city series. | New York Harbor |

### Skipped

- **PADD 1 East Coast** (`R10`): Mixes Harbor-priced 1A/1B with Lower Atlantic barrels that arrive on Colonial from the Gulf. Not a single wholesale market.
- **PADD 1A New England** (`R1X`): NY Harbor is the right wholesale, but NYC/PADD 1B are tighter. Skipped to keep the Northeast set small.
- **PADD 1C Lower Atlantic** (`R1Z`): Supplied northbound on Colonial from the Gulf, not NY Harbor. Do not map to Harbor. Gulf Coast is origin, not the local rack.
- **Miami** (`YMIA`): Lower Atlantic destination with local blend/tax effects. Gulf Coast spot is only a distant origin proxy.
- **PADD 5 West Coast** (`R50`): Blends California CARB with WA/OR/AZ/NV/AK/HI conventional/other specs. LA RBOB is not this average.
- **West Coast less California** (`R5XCA`): Pacific Northwest / Rockies-adjacent. We have no Seattle or PNW gasoline spot.
- **Seattle** (`Y48SE`): Different spec and logistics from LA RBOB. No matching daily wholesale in the repo.
- **San Francisco** (`Y05SF`): CARB like LA, but a distinct local market. LA RBOB is a sibling spec, not a local print. Parked to avoid a forced match.
- **PADD 2 Midwest** (`R20`): Chicago/Group 3 pricing. No Midwest gasoline spot in the repo.
- **PADD 4 Rocky Mountain** (`R40`): Isolated market. No matching daily wholesale.
- **Chicago** (`YORD`): Midwest pipeline market, not NY Harbor, Gulf waterborne, or LA RBOB.

National inventories are used only in MODEL `ridge_full`, as a separate increment. They are U.S. total stocks, not PADD stocks.

## Holdout results (train before 2016-07-19, test after)

Frozen national reference (exp01, same cutoff): momentum MAE 3.95¢ / dir 68.1%; market Ridge MAE 2.93¢ / dir 77.5%; full Ridge MAE 2.86¢ / dir 77.3%.

LA/CA drop more rows because LA RBOB daily only begins 2003-03-11 (no interpolation). PADD 3 drops include early-1990s inventory seasonal-z warmup, same rule as exp01.

| Region | N train | N test | Mom MAE | Matched MAE | Mismatch MAE | Full MAE | Mom dir | Matched dir | Mismatch dir | Matched R² | Signals (δ=$0.03) | % silent |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| New York City → NY Harbor | 836 | 525 | 0.0408 | 0.0315 | 0.0330 | 0.0308 | 0.710 | 0.762 | 0.735 | 0.3980 | 87/118 | 0.610 |
| Central Atlantic (PADD 1B) → NY Harbor | 1209 | 525 | 0.0371 | 0.0273 | 0.0292 | 0.0269 | 0.692 | 0.776 | 0.762 | 0.4382 | 86/113 | 0.621 |
| Houston → Gulf Coast gasoline | 836 | 521 | 0.0440 | 0.0351 | 0.0334 | 0.0343 | 0.673 | 0.747 | 0.774 | 0.3491 | 101/128 | 0.560 |
| Gulf Coast (PADD 3) → Gulf Coast gasoline | 1222 | 525 | 0.0578 | 0.0415 | 0.0402 | 0.0408 | 0.575 | 0.713 | 0.711 | 0.3397 | 103/126 | 0.564 |
| Los Angeles → LA RBOB | 689 | 525 | 0.0505 | 0.0441 | 0.0440 | 0.0440 | 0.742 | 0.757 | 0.753 | 0.4506 | 137/162 | 0.430 |
| California → LA RBOB | 689 | 525 | 0.0451 | 0.0397 | 0.0394 | 0.0395 | 0.770 | 0.793 | 0.789 | 0.5010 | 135/159 | 0.440 |

Signals column is WAIT/FILL counts for the matched-hub Ridge. % silent is NO CLEAR SIGNAL at δ=$0.03.

## Pass-through: is wholesale → retail stronger when geography matches?

Correlations use the modeling rows (not a selected subset of the test set). `matched hub d5` is the 5-session change of the mapped gasoline spot, as-of Monday T.

| Region | corr(target, matched d5) | corr(target, mismatched d5) | corr(target, WTI d5) | corr(target, retail_d7) |
| --- | ---: | ---: | ---: | ---: |
| New York City → NY Harbor | 0.5656 | 0.4649 | 0.3547 | 0.5317 |
| Central Atlantic (PADD 1B) → NY Harbor | 0.5896 | 0.4925 | 0.3787 | 0.5238 |
| Houston → Gulf Coast gasoline | 0.4990 | 0.5494 | 0.3729 | 0.5074 |
| Gulf Coast (PADD 3) → Gulf Coast gasoline | 0.5498 | 0.5932 | 0.4135 | 0.4654 |
| Los Angeles → LA RBOB | 0.5287 | 0.3679 | 0.2480 | 0.5642 |
| California → LA RBOB | 0.5376 | 0.3777 | 0.2623 | 0.6127 |

Matched-hub Ridge beat mismatched-hub Ridge on MAE in **2 of 6** regions. It beat the frozen national market MAE in **1 of 6** regions.

## Incremental groups

| Region | Retail-only MAE | + matched market MAE | + inv/season MAE | Market lift vs retail-only | Inv/season lift |
| --- | ---: | ---: | ---: | ---: | ---: |
| New York City → NY Harbor | 0.0364 | 0.0315 | 0.0308 | 0.0049 | 0.0007 |
| Central Atlantic (PADD 1B) → NY Harbor | 0.0332 | 0.0273 | 0.0269 | 0.0060 | 0.0004 |
| Houston → Gulf Coast gasoline | 0.0382 | 0.0351 | 0.0343 | 0.0031 | 0.0008 |
| Gulf Coast (PADD 3) → Gulf Coast gasoline | 0.0487 | 0.0415 | 0.0408 | 0.0072 | 0.0007 |
| Los Angeles → LA RBOB | 0.0452 | 0.0441 | 0.0440 | 0.0011 | 0.0000 |
| California → LA RBOB | 0.0409 | 0.0397 | 0.0395 | 0.0012 | 0.0002 |

## Charts

- `reports/figures/regional_mae_by_region.png`
- `reports/figures/regional_diracc_by_region.png`
- `reports/figures/regional_passthrough_corr.png`

## Surprises

Matching is **not** uniformly helpful in Ridge MAE. It helps the Northeast (NYC, PADD 1B). On the Gulf, the *mismatched* NY Harbor print beats the local Gulf Coast spot for Houston and PADD 3, both in correlation and in MAE. NY Harbor conventional gasoline is a liquid national benchmark; Gulf Coast spot is a refinery-gate print that does not automatically stick to weekly Houston/PADD 3 retail (taxes, brands, rack-to-retail lag). Do not force Gulf retail onto Gulf spot just because the names match.

California has the highest directional accuracy (**79.3%**), but regional momentum is already **77.0%**. Matched markets add only ~0.12¢ MAE. That is sticky CARB retail, not a large wholesale-lead effect. LA city MAE (**4.41¢**) is worse than the national 2.93¢: the city series is jumpy.

Inventory + seasonality remain a ~0.0–0.08¢ afterthought, same as exp01.

## Answers

**A. Which region has the strongest predictable signal?** It depends on the metric. **California → LA RBOB** has the highest matched-hub directional accuracy (79.3%), mostly from retail momentum (77.0% already). **PADD 1B → NY Harbor** has the best matched MAE (**2.73¢**), beating the frozen national market Ridge (2.93¢), with 77.6% direction. The cleanest *matching* win is the Northeast: Harbor 5-day changes correlate 0.57–0.59 with next-week retail vs 0.46–0.49 for Gulf Coast. PADD 3 is the weakest (MAE 4.15¢, direction 71.3%).

**B. Does matching retail geography to wholesale geography improve performance?** **Sometimes, and mainly in the correlation sense.** Matched-hub Ridge beat mismatched-hub Ridge on MAE in only **2 of 6** regions (NYC and PADD 1B). Pass-through correlations *do* rise when the spec is actually different: Harbor beats Gulf for the Northeast; LA RBOB beats Harbor for LA/CA (0.53 vs 0.37). Gulf retail is the exception: NY Harbor tracks Houston/PADD 3 weekly changes as well or better than Gulf Coast spot. Versus the national weekly model, matched regional Ridge beat national MAE in **1 of 6** regions (PADD 1B only). City labels are noisier than the U.S. average. Matching is a pairing rule, not a free accuracy upgrade.

**C. Is the improvement large enough to justify pursuing licensed daily retail data?** Not as a weekly-EIA accuracy story: regional matching did **not** produce a breakthrough over the national 77.5% / 2.93¢ model, and several city series are *worse*. It **is** enough to justify licensed daily/metro retail as the *next data purchase*, for a different reason: we now know which hub belongs with which city (Harbor with NYC, RBOB with LA), weekly city averages are too noisy for a local product, and a 3-day WAIT/FILL claim still cannot be scored on Monday EIA prints. Do not buy daily retail expecting 90% weekly direction from the same labels.

**D. Which geography should we use for the first real MVP?** **New York City (conventional, NY Harbor)** as the first city, with **PADD 1B** as the broader East Coast check — matching works and MAE is in the same band as the national model. **Los Angeles (CARB, LA RBOB)** as the second geography, because spec matching is real even though weekly MAE is worse. Do not launch on PADD 3, PADD 5, or U.S. average as if they were a local pump. California statewide direction looks strong because prices are sticky, not because LA RBOB suddenly explains the pump.

**E. What is the biggest remaining limitation?** The label is still a **weekly EIA city or PADD average**, published Tuesday, not a station or metro daily pump price. Even a correct hub match cannot see intra-week moves, $0.20 neighborhood gaps, or brand/tax residuals. Gulf results show that a geographically named spot is not automatically the best predictor. Inventories here are national. EIA API values are latest vintage, not original prints.

## Method notes

- Prediction timestamp: Tuesday 12:00 p.m. ET after Monday T.
- Target: regional retail(T+7) − regional retail(T).
- Daily spots: last session with observation date ≤ Monday T (available Tuesday noon).
- No interpolation. No LA RBOB in Northeast/Gulf models except as a diagnostic correlation column.
- Mismatched-hub Ridge is a pre-specified control, not a selected model.
- Thresholds were not re-tuned.
