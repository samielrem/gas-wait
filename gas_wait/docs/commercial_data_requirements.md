# Commercial data requirements — licensed daily retail

Status: research and purchase-readiness only. No new model. No scrape. No purchase. Existing experiments and datasets are unchanged.

Date: 2026-08-20.

This note answers one question: **what is the minimum licensed daily retail dataset that can test whether Gas Wait’s 1–3 day WAIT / FILL UP / NO CLEAR SIGNAL product is real**, given that weekly EIA retail cannot support that claim.

It is not a quote. Public list prices for the relevant retail products were **not found**. Where a number is not published, this document says **contact sales**.

---

## What we already know (do not re-litigate)

| Finding | Implication for buying data |
| --- | --- |
| Daily WTI / NY Harbor / Gulf Coast / LA RBOB improve **weekly national** retail Δ (market Ridge 77.5% dir, 2.93¢ MAE vs 68% / 3.95¢ momentum) | Wholesale features are worth keeping. We do **not** need to buy spots for the first licensed experiment; EIA daily spots already exist. |
| Matching geography helps **pass-through correlation** (NYC/PADD 1B ↔ Harbor; LA/CA ↔ RBOB) | Buy retail that maps to a hub we already have. |
| Matching does **not** automatically beat the national weekly model (only PADD 1B beat national MAE) | Do not buy nationwide station data hoping for a weekly-accuracy miracle. |
| Gulf Coast spot is a weak or inverted match to Houston/PADD 3 weekly retail | Do **not** make Houston the first paid geography. |
| EIA retail is Monday, national or PADD/city, published Tuesday | It cannot score a 72-hour local consumer call. That is the only reason to buy commercial retail. |

Intended MVP (prediction proof, not a full consumer product):

```
User opens app
→ app knows metro / location
→ model predicts retail direction over ~1–3 days
→ WAIT / FILL UP / NO CLEAR SIGNAL
```

Out of scope for the first licensed dataset: station routing, exact station-price prediction, brand recommendations, nationwide station coverage, minute-by-minute updates.

---

## 1. Minimum viable data specification

Request this and nothing more for the first paid contract.

| Field | Spec |
| --- | --- |
| Product | Regular unleaded retail gasoline, all formulations **or** formulation flagged (conventional vs RFG/CARB). Include taxes in the pump price. |
| Grain | **Metro-level daily average** (Census MSA / CBSA). One number per metro per calendar day. Not station-level. |
| Metros (first buy) | **New York–Newark–Jersey City, NY-NJ-PA** and **Los Angeles–Long Beach–Anaheim, CA**. Optional third: a Harbor-adjacent smaller check (e.g. Boston-Cambridge-Newton) only if the two-metro quote is cheap. |
| History | Daily observations, **calendar 2018-01-01 through latest complete day**, preferably back to **2007-01-01** if the vendor already stores it. |
| Cadence | **One snapshot per calendar day** with a documented timestamp (recommend local 08:00 or a fixed vendor “day close”). Intraday ticks are not required. |
| Timestamps | Observation date, snapshot time, timezone, and (if they have it) when the average became available. Point-in-time rules matter as much as they did for EIA. |
| Identifiers | Stable metro ID (CBSA code), metro name, grade, formulation flag, units ($/gal). |
| Delivery | Bulk historical file (CSV/Parquet) **plus** a daily incremental file or API. Snowflake is acceptable. |
| License (non-negotiable) | Written rights to (a) **store** history, (b) **train** statistical / ML models, (c) show **derived** WAIT / FILL UP / NO CLEAR SIGNAL to end users, (d) keep trained weights after the contract if the raw feed is turned off. We will **not** republish raw vendor prices in v1. |
| What we will not buy in v1 | Station-level prices, diesel, midgrade/premium (except as a free add-on), nationwide coverage, rack (we have EIA spots), volume/margin products, connected-car redistribution. |

This is Option A, two metros, one grade, one snapshot/day.

---

## 2. Ideal data specification

Same as the minimum, plus, in priority order:

1. **Third metro** in a different spec/logistics system only after NYC and LA work (do not add PADD 3 / Houston first).
2. **Formulation split** (conventional vs RFG vs CARB) inside the metro if they produce it.
3. **Intraday metro average** at 2–4 fixed clocks (e.g. 08:00 / 12:00 / 17:00 local) so we can test whether a same-day update changes the 72-hour call. Still not station-level.
4. **As-of / vintage** of each average (when it was published, not only the economic date).
5. **Coverage diagnostics**: station count in the average, % of stations reporting that day, missing-day flag. No interpolation by the vendor without a flag.
6. **History to January 2007** (OPIS public retail-history floor) so 2008 / 2020 / 2022 are all in sample.
7. **Optional later**: a **derived-output + display** license (OPISNAVX-class) if we ever show a metro average or nearby stations in the app. That is a different SKU from a research feed.
8. **Optional later**: station-level for **one** metro, regular only, daily close, if and only if metro averages predict 1–3 day moves and we then need to test whether a station residual is large enough to change WAIT/FILL.

Do not put (7) or (8) on the first purchase order.

---

## 3. Vendor comparison

Sources are official public pages, published terms, and one public government OPIS order form. Consumer sites (GasBuddy app, AAA Fuel Prices) are **not** a data license.

### 3.1 OPIS (Oil Price Information Service, a Dow Jones company)

| Topic | Public information | Gap |
| --- | --- | --- |
| Daily retail gasoline | Yes. Monitors ~5 million daily gasoline/diesel prices for nearly **150,000** North American outlets. Custom feeds: national, 8 OPIS regions, **state**, **MSA**, county, city, zip, custom zone. Brand breakouts available. | Quote is custom. |
| Station vs average | Both exist. Station-level is the native database; MSA/state averages are a **custom retail feed** product. | Need to confirm how the MSA average is weighted (station count vs volume). |
| Coverage | US + Canada; also Mexico/global via other SKUs. Daily coverage claimed **~90% of stations**, **>95% within 48 hours**. ~1 in 4 US stations send prices directly. Clients named: Google, Waze, **AAA**, connected vehicles. | “Trusted by AAA” means AAA’s published daily prices are generally understood to be OPIS-sourced; AAA’s own site is still not our license. |
| Update frequency | Real-time / throughout the day for retailer tools (PricePro); custom feeds **daily, weekly, monthly, or quarterly**. OPISNAVX: **>2 million** station price updates/day; APIs, bulk files, geographic requests. | For MVP, ask for one stamped daily metro average, not PricePro. |
| Historical depth | **Retail history dates back to 2007.** Retail DataHouse: ~140k stations, 38k+ geographies, history from **Jan 2007** to **as current as 8 days prior** (that product is a delayed online database, not a live feed). Spot/rack history is longer in some markets (OPIS says 30+ years **in some markets**, not specifically retail). | Confirm whether a **live** custom MSA feed can be backfilled to 2007, or only DataHouse with an 8-day lag. |
| API | **Rack API** is a documented REST/JSON product (history from 2016 for rack). Retail: OPISNAVX “various APIs”; ICE Developer Portal says OPIS data can be delivered via ICE API and bulk files. Custom retail feeds: email/FTP/Excel also advertised. | Retail REST API for metro averages is **not** a self-serve public product. **Contact sales.** |
| Bulk download | Yes: FTP, Excel, CSV, custom files, ICE bulk. DataHouse: xls/csv. | Confirm historical bulk for two MSAs. |
| List price | **Not public** for retail metro/station. Enterprise / contact sales. A Washington State DES OPIS **wholesale rack** order form (Jul 2025–Jun 2028) shows ~$29k / $36k / $42k over three years for a **small rack-location package** — that is **not** a retail quote and must not be used as one. | **Contact sales.** |
| Trial / startup | OPIS states nearly every product is available for a **free trial or demo**. Custom retail: call 888.301.2645 or email the OPIS Retail Fuel Quality Team. | Ask for a **paid prototype SOW** (two MSAs, history dump, 12 months of daily updates), not a verbal demo of PricePro. |
| Website ToS | [opis.com/terms-of-use](https://www.opis.com/terms-of-use/): site content is **personal, non-commercial**; no copy, store, or derivative works without **prior written consent**; scraping/bots prohibited. | The website is not the product license. |
| Contract terms (published T&Cs) | OPIS Services Terms (Oct 2024) default to **Internal Use only** (“employees… internal business purposes”; “not licensed for external use” unless the SOW says otherwise). Default license **forbids** using deliverables or **data derived from them** for text/data mining or **developing, training, tuning, or operating ML/AI models**. External distribution of deliverables needs prior written approval. Client indemnifies OPIS if derived findings are shared with third parties. | **This default contract cannot support Gas Wait.** The SOW must carve out training and consumer-facing derived signals. |
| Consumer redistribution | **OPISNAVX** is the SKU that already licenses station prices into cars and phones (claimed 30M cars / 100M phone users). That is the closest public analog to showing prices in an app. It is almost certainly a different, heavier license than a research feed. | Do not assume a DataHouse or custom-average SOW includes OPISNAVX rights. |

OPIS is the stronger **historical metro-average** candidate because they publicly sell MSA custom feeds and advertise retail history to 2007.

### 3.2 PDI Technologies / GasBuddy

GasBuddy LLC is a PDI company (acquired 2021). Consumer app and B2B data are different legal objects.

| Topic | Public information | Gap |
| --- | --- | --- |
| Daily retail gasoline | **PDI Data Services — Retail Fuel Prices:** “real-time and historical fuel prices by **site, grade, or aggregated by location and time**.” Aimed at financial firms, government, “others.” | Aggregation methods and metro definitions are not specified on the public page. |
| Station vs average | Both are advertised (site-level **or** aggregated). Native GasBuddy product is **station-level**, often **user-reported**, with timestamps of last update. | Crowdsourced timestamps are irregular; a daily metro average would need PDI to compute it. |
| Coverage | GasBuddy: **150,000+** stations, US and Canada. PDI: “thousands of fuel retail locations… across North America” plus site-attribute feed for all fuel-selling US/Canada locations. | Ask how sparse early history is outside large metros. |
| Update frequency | PDI: “daily or intra-day data feeds.” GasBuddy consumer app is continuous / user-driven. Delivery includes **Snowflake**. | Ask for a vendor-built **daily metro close**, not raw ping-level reports. |
| Historical depth | University of Chicago Booth **Kilts Center** GasBuddy extract (PDI): **2018 through 2024**, station name/address/coords, regular/mid/premium/diesel, last-update timestamp, amenities, CSV. GasBuddy public charts advertise **up to 10 years** of area averages for **display on gasbuddy.com**, not as a bulk research dump. | Commercial history depth **contact sales**. Kilts is **not** a startup license (UChicago only; consulting prohibited; data must be destroyed at end; papers may use only “limited excerpts”). |
| API | No public self-serve GasBuddy price API for third-party apps. PDI: secure feeds / Snowflake. A GasBuddy GitHub maintainer has stated consumer APIs are **not allowed without a business agreement**. | **Contact sales.** Do not use unofficial GraphQL/REST wrappers. |
| Bulk | PDI: multiple formats including Snowflake. Kilts academic files are CSV. | Confirm a two-MSA historical extract. |
| List price | **Not public.** A 2023 GitHub comment from a GasBuddy engineer said commercial access is “a big number.” That is **not a quote** and should not be treated as one. | **Contact sales.** |
| Trial / startup | PDI Data Services and GasBuddy Business Pages: web forms (“a member of our team will be reaching out”). Kilts is academic-only. No public “startup tier.” | Ask explicitly for a **research / prototype** SOW. Possible, not promised. |
| Consumer ToS | [gasbuddy.com/disclaimer/usa](https://www.gasbuddy.com/disclaimer/usa) (effective 2026-04-24): no commercial use of GasBuddy Content on other sites; no **creating a database** by systematically downloading; no **forwarding data** without written consent; no scraping/bots; no commercial exploitation of the Site. Legal: legal@pditechnologies.com. | Scraping GasBuddy or wrapping the app is a hard no. |
| Licensing for ML / app | No public PDI master terms analogous to the OPIS PDF were found. Kilts policy is the only detailed public license, and it **forbids** non-academic use. | Assume nothing. Put training, storage, and derived consumer signals in the SOW. |

PDI/GasBuddy is the stronger **station-level / real-time consumer-app** candidate. It is a weaker fit for a **small metro-average history dump** unless they will sell that slice without a nationwide station firehose.

### 3.3 Direct comparison (what matters for Gas Wait)

| Question | OPIS | PDI / GasBuddy |
| --- | --- | --- |
| Metro daily averages as a first-class product | **Yes** (custom MSA feeds) | Advertised as “aggregated by location and time”; confirm MSA |
| Station-level | Yes (~150k NA) | Yes (150k+); crowdsourced + retailer |
| Public retail history floor | **2007** | Academic extract **2018–2024**; commercial unknown |
| Named as AAA / maps / cars | AAA, Google, Waze, OEMs (OPISNAVX) | Consumer app with 100M+ downloads (marketing figure) |
| Documented ML ban in default T&Cs | **Yes — must override in SOW** | Not found; still must grant in SOW |
| Documented consumer redistribution SKU | **OPISNAVX** | GasBuddy *is* the consumer app; third-party redistribution is a sales conversation |
| Self-serve price | No | No |
| Usable without a contract | **No** (site ToS + scrape ban) | **No** (ToS + scrape ban) |
| Best first ask | Custom daily **MSA averages** + history + ML/derived-use SOW | Same slice via Data Services; walk away if they will only sell nationwide station firehose |

**Neither website, app, or AAA page is a substitute for a contract.**

---

## 4. Questions to ask sales

Send the same packet to OPIS Retail and PDI Data Services. Ask them to answer in writing.

**Product**

1. Can you deliver a **daily regular-gasoline average** for CBSA 35620 (New York) and CBSA 31080 (Los Angeles), one row per metro per calendar day?
2. How is the average constructed (simple mean of reporting stations, volume-weighted, brand-weighted, trimmed)?
3. Is formulation (conventional / RFG / CARB) mixed or separable?
4. What is the **earliest complete daily date** for each metro?
5. What is the daily snapshot time, timezone, and when is that day’s number considered final?
6. Missing days: do you skip, carry forward, or interpolate? (We will not accept silent interpolation.)
7. Delivery: historical bulk file now, then daily incremental (S3/FTP/SFTP/Snowflake/API)? Schema sample?
8. Can we start with **two metros only**, regular grade only, no diesel, no station IDs?

**Commercial**

9. Price for: (i) history dump, (ii) 12 months of daily updates, (iii) optional year-2 nationwide metro expansion.
10. Is there a **prototype / startup / academic-adjacent** SKU?
11. Minimum term and minimum geography (do you force nationwide)?
12. Who else licenses metro averages for **non-retailer** use (fintech, auto, consumer app)?

**If they push station-level**

13. What is the delta price vs metro averages for the same two metros?
14. Typical daily non-null rate per station in those metros?
15. Can we **not** display station prices and still license station data for training only?

---

## 5. Licensing questions

These are go/no-go. Get them as SOW language, not email vibes.

1. **Internal vs external use.** Default OPIS T&Cs are Internal Use only. We need external use of **derived recommendations**, even if raw prices stay off the screen.
2. **Machine learning.** Default OPIS T&Cs forbid training/tuning/operating ML/AI on the deliverables **or data derived from them**. That clause must be deleted or carved out. Ask PDI the same, in writing.
3. **Storage.** May we retain historical observations for the term? After termination, may we keep (a) trained model weights, (b) evaluation reports that do not reproduce the price series?
4. **Derived outputs.** May we show consumers WAIT / FILL UP / NO CLEAR SIGNAL computed from the data, without showing the vendor’s price, without attributing the number as an OPIS/PDI print?
5. **Attribution.** Required disclaimer copy? Prohibition on using OPIS/GasBuddy/PDI marks in the app?
6. **Redistribution.** We will not resell the feed. Confirm that showing a derived ternary signal is not “redistribution of Deliverables.”
7. **If we later show a metro average in-app**, is that OPISNAVX / a display license, and what is the separate fee?
8. **Training on a subset / distillation.** May we train on NYC+LA and later apply the model in a third metro if we then buy that metro’s feed?
9. **Indemnity.** OPIS T&Cs make the client indemnify OPIS if derived findings are shared with third parties. That is incompatible with an app unless narrowed.
10. **Audit / non-compete.** Any restriction on competing with GasBuddy’s consumer app or OPISNAVX?
11. **Assignment / startup.** Can a Delaware C-corp / unincorporated project sign? Personal-use website ToS is not enough.
12. **Kill switch.** If the contract ends, can the app keep serving a model trained during the licensed window, without the live feed?

Do not sign a standard Internal-Use + no-ML OPIS SOW and “figure out the app later.”

---

## 6. Recommended MVP geography

**Buy: New York metro and Los Angeles metro.**

| Metro | Why |
| --- | --- |
| New York–Newark–Jersey City | Best EIA pairing we have: NYC / PADD 1B weekly retail actually tracks **NY Harbor**. City EIA MAE was in the same band as national. Conventional Harbor market. |
| Los Angeles–Long Beach–Anaheim | Only CARB/RBOB pairing. Weekly LA was noisier, but **spec matching was real** (RBOB d5 corr 0.53 vs Harbor 0.37). If 1–3 day pass-through exists anywhere, it should show up here. |

**Do not buy first:** Houston / PADD 3 (local Gulf spot underperformed Harbor as a weekly predictor), PADD 5, nationwide, “all US MSAs,” Seattle, Miami.

**Do not substitute:** EIA weekly NYC / LA. Those remain the wrong clock for a 72-hour product. They are a useful **auxiliary** label, not the purchase.

One metro is enough to **fail**. Two metros (Harbor + CARB) are the minimum to **believe** a pass. Nationwide is how we waste the budget.

---

## 7. Recommended historical depth

| Depth | What it buys | Verdict |
| --- | --- | --- |
| < 3 years daily | ~750 weekdays; one regime; weak chronological test | **No-go** |
| 5 years (e.g. 2021–2026) | Misses COVID and 2022 spike | Bare minimum if that is all they will sell |
| **8 years (2018–now)** | Aligns with Kilts GasBuddy window; includes 2020 and 2022; ~2,000–2,900 daily rows per metro | **Ask as the default** |
| Back to **Jan 2007** | OPIS’s public retail-history floor; 2008 included | **Take it if the price delta is small** |

We do not need 30 years of retail. We need a chronological train/test split that is not just 2023–2025.

---

## 8. Recommended sampling frequency

| Frequency | Use | For this MVP |
| --- | --- | --- |
| Weekly | We already have EIA | Insufficient |
| **One timestamped snapshot per calendar day** | Native 1-day and 3-day labels | **Buy this** |
| 2–4 fixed intraday snapshots | Same-day updates of the same 3-day target | Nice-to-have on a later SOW |
| Tick / every change / station ping | Station app, not a direction product | Do not buy |

Business-day vs calendar-day: accept the vendor’s calendar, **do not interpolate weekends**. If Saturday/Sunday are missing, the 3-day horizon is still defined on observed stamps.

---

## 9. Recommended target definition

Do not buy data until this is written into the experiment design (later). The purchase should make these labels possible:

**Primary target (product-shaped):**

```
y_{m,t} = metro_regular_price(m, t+H) − metro_regular_price(m, t)
```

- `m` ∈ {NYC metro, LA metro}
- `H` ∈ {1 calendar day, 3 calendar days}
- Units: $/gal
- Prediction time: the snapshot timestamp for day `t` (no later information)

**Decision map (frozen, not tuned on test):** reuse δ ∈ {$0.03, $0.04, $0.05} from the weekly work as a starting dead zone; if typical 3-day metro moves are smaller, freeze a new δ on **training** data only.

**Secondary (diagnostic only):** 7-day metro Δ, to compare with EIA weekly city prints. Not the shippable target.

**Not the target:** min price over the next 3 days; cheapest station in the metro; “will NY Harbor go up.”

Station-level is unnecessary to define this target. A metro average **is** the target.

---

## 10. Go / no-go criteria for purchasing

### Go (all must be true)

1. Written license allows **storage**, **model training**, and **consumer-facing derived WAIT/FILL/NO SIGNAL**.
2. Default **Internal Use only** and **no-ML** clauses are removed or carved out in the SOW.
3. Daily **metro** regular series for **NYC and LA**, not a verbal “we can probably aggregate.”
4. At least **5 years** of daily history, preferably **2018–present**, with documented snapshot times and no silent interpolation.
5. Bulk historical delivery we can use offline (not only a GUI).
6. Price and term are a **prototype box** (two metros, one grade, 12 months) — not a forced nationwide station or OPISNAVX display deal.
7. We can keep **model weights** if the feed is later cancelled.

### No-go

1. License is website ToS, GasBuddy consumer app, AAA page, or “internal analytics only.”
2. ML/training is prohibited and they will not amend.
3. We may train but **may not show any derived recommendation** to a user.
4. History is weekly, shorter than 3 years, or current-only.
5. The only SKU is nationwide station-level or connected-car redistribution.
6. Vendor interpolates missing days without flags.
7. Quote exists only as “call us” with no written SOW after a serious ask — treat as not available yet; do not scrape instead.

### After purchase, scientific go/no-go (not a reason to skip the license)

The dataset is worth **renewing / expanding** only if, on a chronological holdout:

- A model using **lagged metro retail + already-owned EIA wholesale** beats a **metro momentum baseline** on 1-day and/or 3-day direction by a margin that is not noise; **and**
- Dead-zone coverage is usable (not 90% NO CLEAR SIGNAL at 3¢); **and**
- NYC and LA do not tell opposite stories for reasons we cannot explain.

If metro daily labels are **not** more predictable at 1–3 days than weekly EIA was at 7 days, **stop**. Do not “fix it” with station-level data. That would be buying granularity to hide a missing short-horizon effect.

---

## Three acquisition strategies

### Option A — Metro-level daily retail (recommended)

| | |
| --- | --- |
| Problem it solves | 1- and 3-day **metro pump** direction; the actual MVP question. |
| Predictive usefulness | **Highest expected value per dollar.** Matches how a user is located. Aligns with EIA weekly city results without pretending weekly is daily. Wholesale hubs can still be features. |
| Data volume | Tiny: 2 metros × ~2,000–4,700 days × 1 grade ≈ **4k–10k rows**. |
| Integration | Low: one table, as-of join like the weekly builder. |
| Licensing | Still hard (ML + derived app use), but the **smallest** surface area to negotiate. No station PII, no map display. |
| Cost | **Contact sales.** Should be the cheapest honest SKU if they will unbundle. |
| Supports intended app? | **Yes**, for WAIT/FILL by metro. Not for “this Shell is 4¢ cheaper.” |

### Option B — State-level daily retail

| | |
| --- | --- |
| Problem it solves | 1–3 day **state average** direction. |
| Predictive usefulness | **Weaker than metro.** New York State ≠ NYC Harbor market; California is closer to a CARB product but still mixes LA and the interior. Our weekly work already showed PADD averages can look better than cities because they are smoother, which is the opposite of a local product. |
| Data volume | Still tiny (2 states × daily). |
| Integration | Low. |
| Licensing | Similar legal issues, slightly easier optics than station-level. |
| Cost | **Contact sales.** Possibly similar to two MSAs; not worth it if MSA is available. |
| Supports intended app? | **Poorly.** A user in Manhattan does not want “New York State.” Only a fallback if they refuse MSA and will sell NY + CA state. |

### Option C — Station-level daily retail

| | |
| --- | --- |
| Problem it solves | Station price nowcasting, cheapest-nearby, routing, GasBuddy-class UX. |
| Predictive usefulness | **Not needed to prove 1–3 day direction.** Station noise is large relative to a 3¢ dead zone. Useful **after** metro labels work, to measure how often the metro signal is wrong for a typical station. |
| Data volume | High: thousands of stations × daily × years → **millions of rows** per metro; nationwide is an order of magnitude more. |
| Integration | High: entity matching, missingness, irregular timestamps, leakage if using same-day competitor prints carelessly. |
| Licensing | **Hardest.** Displaying station prices is OPISNAVX / GasBuddy-app territory. Training-only station data may still be priced as the firehose. |
| Cost | **Contact sales.** Anecdote that commercial GasBuddy access is “a big number” is unofficial. Expect this to be the expensive option. |
| Supports intended app? | Supports a **different** app than the stated MVP. Buying it now is scope creep. |

**Station-level is not necessary for the MVP.** Metro averages are.

---

## Minimum dataset to request (one paragraph)

Ask OPIS first, PDI second, for a **12-month prototype plus history**: daily regular-gasoline **MSA averages** for **New York and Los Angeles**, one timestamped snapshot per day, history from **at least 2018 (2007 if cheap)**, bulk file + daily update, **no station IDs**, with a written license to **store, train, and show derived WAIT / FILL UP / NO CLEAR SIGNAL to end users**. Do not buy rack, nationwide coverage, or OPISNAVX display rights on the first PO. Do not scrape. If both vendors refuse the ML/derived-use carve-out, **do not purchase**.

### Contacts (public)

- OPIS retail custom feeds: 888.301.2645; Retail Fuel Quality Team via [opis.com custom retail](https://www.opis.com/product/pricing/retail-fuel-prices/customized-retail-fuel-prices-margins/). Rack/API CS: energycs@opisnet.com.
- PDI Data Services: [pditechnologies.com/gain-insights/insights-analytics/pdi-data-services](https://pditechnologies.com/gain-insights/insights-analytics/pdi-data-services/). Legal: legal@pditechnologies.com.

---

## Sources

- OPIS custom retail feeds; retail fuel prices; gasoline products; oil price history; Retail DataHouse; OPISNAVX; website Terms of Use; Services Terms and Conditions (Oct 2024); ICE OPIS catalog; WA DES OPIS subscription order form (rack, not retail).
- PDI Data Services catalog; GasBuddy Terms of Service (USA); Kilts Center PDI/GasBuddy academic dataset page.
- Gas Wait experiments: `reports/first_model_results.md`, `reports/regional_experiment.md` (unchanged by this note).
