# Personal MVP — weekly Gas Wait CLI

## How to run

From `gas_wait/`:

```bash
PYTHONPATH=src python -m gas_wait_cli
```

Signal history:

```bash
PYTHONPATH=src python -m gas_wait_cli --history
```

JSON output (for scripts):

```bash
PYTHONPATH=src python -m gas_wait_cli --json
```

## What it does

`gas_wait_cli` is a **local command-line tool** for personal daily use. Each run:

1. Loads the latest EIA processed datasets (no live API calls).
2. Builds the current **leakage-safe** weekly feature row (Tuesday-noon prediction clock).
3. Fits the frozen **exp01 ridge_full** specification (Ridge α=1.0, StandardScaler, 17 features) on all historical rows with known targets.
4. Predicts the **next-week national retail change** ($/gal).
5. Converts the prediction to **WAIT**, **FILL UP**, or **NO CLEAR SIGNAL** using the frozen **±$0.03** threshold.
6. Appends one row to `data/processed/signal_history.csv` (deduplicated by `prediction_date`).

## What it does NOT do

- **Not a 72-hour product** — horizon is ~7 days (next EIA weekly retail print).
- **Not daily retail** — no 24h/48h/72h targets; weekly EIA retail only.
- **Not local** — U.S. national average, not station or MSA pricing.
- **Not a new model** — same ridge_full spec as exp01 MODEL 4; no threshold tuning.
- **No web app, API, database, or cloud** — runs locally only.

## Data freshness

The CLI documents and applies these rules (`inference.py`):

| Rule | Threshold | Behavior |
| --- | --- | --- |
| Prediction clock | Tuesday 12:00 p.m. ET | **Fail** if latest retail is not yet public |
| Maximum retail age | **8 calendar days** | **Fail** if latest retail Monday is older |
| Missing weekly print | **>7 days** behind expected Monday | **Fail** if processed retail lags expected update |
| Behind expected Monday | any lag, within limits above | **Warning** printed; signal still shown |

Warnings look like:

```
WARNING: Latest retail print is from YYYY-MM-DD (N day(s) behind the expected Monday YYYY-MM-DD). Signal may be stale.
```

Hard failures always print:

```
Unable to generate today's signal because required data is unavailable.
```

## Interpreting signals

| Signal | Meaning (research threshold δ = $0.03/gal) |
| --- | --- |
| **WAIT** | Model predicts national retail may **fall** more than 3¢ over the next week |
| **FILL UP** | Model predicts national retail may **rise** more than 3¢ over the next week |
| **NO CLEAR SIGNAL** | Predicted move is inside the ±3¢ dead zone |

These are **research signals** from a weekly national model. They are not fill-timing guarantees.

## Why the horizon is weekly

EIA publishes U.S. retail gasoline **once per week** (Monday series). The validated experiment predicts:

`retail(Monday T+7) − retail(Monday T)`

That is inherently a **~7-day** national forecast. A true “should I fill in the next 72 hours?” product needs **licensed daily retail** data, which this MVP deliberately does not fake.

## Signal history

File: `data/processed/signal_history.csv`

Columns:

| Column | Description |
| --- | --- |
| `timestamp` | When the CLI recorded the signal (ET) |
| `prediction_date` | Tuesday prediction date for Monday retail *T* |
| `predicted_change` | Model output ($/gal) |
| `signal` | WAIT / FILL UP / NO CLEAR SIGNAL |
| `model_version` | e.g. `ridge_full_alpha1.0` |
| `latest_data_date` | Latest retail/spot/inventory date used |

Duplicate runs on the same `prediction_date` do **not** append twice. Older entries are preserved; new weeks append in chronological order. Only signal metadata is stored — no raw EIA rows.

The `--history` flag shows signal counts only. **Outcomes are not scored** until each target week completes (no premature evaluation).

## Safety and honesty

```
Unable to generate today's signal because required data is unavailable.
```

Output always includes:

- `Horizon: ~7 days`
- `National weekly signal — not local station pricing.`
- `Last available data: YYYY-MM-DD`

It never claims tomorrow’s price, station-level accuracy, or daily retail predictions.

## Architecture

```
processed EIA CSVs
       ↓
weekly_pipeline (latest row + history)
       ↓
inference.py (ridge_full fit + predict + group explanations)
       ↓
gas_wait_cli.py (format + signal_history.csv)
```

See also [modeling_framework.md](modeling_framework.md) and [modeling_design.md](modeling_design.md).
