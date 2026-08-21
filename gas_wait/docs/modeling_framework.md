# Gas Wait modeling framework

Reusable infrastructure for feature engineering, point-in-time joins, walk-forward
backtesting, consumer signals, and theoretical economics. Designed so **licensed daily
retail data** can plug in later without rewriting experiment code.

## Architecture

```
data/processed/*.csv
        │
        ▼
┌───────────────────┐     ┌────────────────────┐
│ features.py       │     │ point_in_time.py   │
│ (grouped calcs)   │────▶│ publication clocks │
└───────────────────┘     └─────────┬──────────┘
                                    │
        ┌───────────────────────────┼───────────────────────────┐
        ▼                           ▼                           ▼
 targets.py                   weekly_pipeline.py            config.py
 (horizons)                   (EIA weekly orchestrator)   (feature groups)
        │                           │
        └─────────────┬─────────────┘
                      ▼
               backtest.py ──▶ signals.py ──▶ economics.py
```

| Module | Role |
| --- | --- |
| `config.py` | Named feature groups and column resolution |
| `features.py` | Modular time-series feature functions |
| `point_in_time.py` | Publication rules and leakage assertions |
| `targets.py` | Target horizon interface (weekly implemented; daily stubbed) |
| `backtest.py` | Holdout and walk-forward evaluation |
| `signals.py` | WAIT / FILL UP / NO CLEAR SIGNAL |
| `economics.py` | Theoretical tank-fill benchmark |
| `weekly_pipeline.py` | Exp01 weekly EIA dataset builder (read-only validation) |

## Feature groups

Experiments select groups by name:

```python
from modeling.config import resolve_feature_columns

feature_cols = resolve_feature_columns([
    "RETAIL_MOMENTUM",
    "CRUDE",
    "WHOLESALE",
])
```

| Group | Examples | Source frequency |
| --- | --- | --- |
| `RETAIL_MOMENTUM` | `retail_d7`, `retail_d14` | Weekly EIA retail |
| `CRUDE` | `wti_d1`, `wti_d3`, `wti_d5`, `wti_vol_20` | Daily WTI |
| `WHOLESALE` | `nyh_d1`, `nyh_d5`, `gc_d1`, `gc_d5`, vol | Daily NYH / Gulf |
| `SPREADS` | `crack_nyh_d5` | Derived (42×gas − crude Δ) |
| `INVENTORY` | `inv_wow`, `inv_seasonal_z` | Weekly WPSR (release-lagged) |
| `SEASONALITY` | `sin_doy`, `cos_doy`, `is_summer` | Prediction calendar |

Extended columns (e.g. `wti_d7`, `retail_chg_p4`, `inv_yoy`) are available when
`resolve_feature_columns(..., extended=True)`.

Each function in `features.py` documents source data, lag, and leakage risk.

## Point-in-time rules

Prediction clock for the weekly EIA experiment: **Tuesday 12:00 p.m. ET** after
Monday retail *T*.

| Data | Economic date | Public availability |
| --- | --- | --- |
| Weekly retail | Monday *T* | Tuesday noon ET |
| Daily spot | Session *D* | Next weekday noon ET |
| Gasoline stocks | Week-ending Friday *F* | Following Wed noon ET (Thu on some holiday weeks) |

`point_in_time.py` provides:

- `filter_known_at()` — drop rows not yet public
- `attach_public_availability()` — add publication timestamps
- `assert_feature_timestamps()` — `feature_ts <= prediction_ts`

**No interpolation** of missing sessions. **No forward-fill** across publication boundaries.

## Target interface

```python
from modeling.targets import TargetKind, TargetSpec, build_target_column

spec = TargetSpec(kind=TargetKind.NEXT_WEEK_RETAIL_CHANGE, frequency="weekly")
labeled = build_target_column(retail_df, spec)
```

Implemented:

- `next_week_retail_change` — retail(Monday T+7) − retail(Monday T)

Not implemented (require daily retail):

- `24h`, `48h`, `72h`, `7d` horizons — raise `NotImplementedError` with an explicit message.

## Backtesting

```python
from modeling.backtest import BacktestConfig, TrainMode, run_holdout_backtest

config = BacktestConfig(
    train_mode=TrainMode.HOLDOUT,
    train_fraction=0.70,
    feature_groups=["RETAIL_MOMENTUM", "CRUDE"],
)
result = run_holdout_backtest(dataset, model, config)
```

Supports:

- Chronological holdout split
- Expanding-window walk-forward
- Rolling-window walk-forward
- Configurable retrain frequency (`yearly`, `each_row`, integer stride)
- Configurable feature groups and signal threshold

Returns predictions, errors, directional correctness, and signals.

## Signal generation

Default research threshold δ = **$0.03/gal** (not optimized on test):

| Condition | Signal |
| --- | --- |
| predicted_change ≤ −δ | WAIT |
| predicted_change ≥ +δ | FILL UP |
| otherwise | NO CLEAR SIGNAL |

```python
from modeling.signals import generate_signal

generate_signal(-0.04).to_dict()
# {"signal": "WAIT", "predicted_change": -0.04, "threshold": 0.03, ...}
```

## Economic evaluation

`economics.py` computes **theoretical** outcomes assuming a fixed tank size
(default **15 gallons**) at the observed average price series. This is a research
benchmark — not a claim about real consumer behavior.

Metrics include WAIT savings vs fill-now, signal counts, silent percentage, and
cumulative theoretical savings.

## Plugging in daily retail (future)

When licensed daily MSA retail arrives:

1. Load daily retail CSV with `observation_date` and `retail_price`.
2. Set `TargetSpec(frequency="daily", kind=TargetKind.H72)` once data exists.
3. Add `add_retail_momentum_daily()` (session changes on daily rows).
4. Join with `PublicationRule` for the vendor's release lag.
5. Reuse `backtest.py`, `signals.py`, and `economics.py` unchanged.

Do **not** upsample weekly EIA retail to fake daily targets.

## Validation

`tests/modeling/framework_test_*.py` rebuilds the weekly dataset via
`weekly_pipeline.build_weekly_eia_dataset()` and compares feature columns to the
saved `data/processed/weekly_model_dataset.csv` from exp01. It does not retrain models
or overwrite experiment artifacts.

See also `docs/modeling_design.md` for leakage inventory and product context.
