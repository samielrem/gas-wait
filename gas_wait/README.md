# Gas Wait

Gasoline-price prediction project. The long-term goal is a system that tells a user:

- **WAIT**: gas prices are likely to decrease soon
- **FILL UP**: gas prices are likely to increase soon
- **NO CLEAR SIGNAL**: prediction is uncertain

This repository currently contains the data science project setup and the first EIA data pipeline. The mobile app and ML model are not built yet.

## Folder structure

- `data/raw/`: original API responses as downloaded from EIA
- `data/processed/`: cleaned CSV files with standardized metadata columns
- `notebooks/`: exploratory analysis and experiment notebooks
- `src/`: reusable Python source code
- `src/data/`: EIA client, dataset definitions, fetch/build scripts
- `models/`: saved trained models and related artifacts (empty for now)
- `tests/`: unit tests

## Data we are collecting

The first pipeline uses the [EIA Open Data API v2](https://www.eia.gov/opendata/documentation.php) as the primary authoritative source. All series below are **weekly** and **U.S. national** (or U.S. benchmark) to keep the initial pipeline simple and aligned on one native frequency.

| Dataset | EIA route | Why it matters |
| --- | --- | --- |
| U.S. regular gasoline retail price | `petroleum/pri/gnd/data` | Direct consumer-facing price signal; this is the outcome we ultimately want to forecast. |
| WTI crude oil spot price | `petroleum/pri/spt/data` | Crude oil is the largest component of retail gasoline prices; upstream cost pressure often precedes pump-price changes. |
| U.S. total gasoline stocks | `petroleum/stoc/wstk/data` | Inventory builds suggest oversupply (bearish for prices); draws suggest tight supply (bullish for prices). |

Planned later categories, not yet implemented:

- Wholesale / RBOB gasoline price
- Refinery utilization
- Gasoline production
- Product supplied / demand proxy

## Important limitation: weekly retail prices

EIA retail gasoline prices are published **weekly**, not daily. That means this source alone is **not sufficient** for a true 3-day consumer recommendation such as “fill up in the next 72 hours.” Before building that product signal, we will need a higher-frequency retail price source and a clear rule for how weekly authoritative data combines with faster but noisier signals.

We are **not** upsampling weekly data to daily frequency in this step. Each dataset keeps its native EIA frequency.

## Setup

1. Create a virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r gas_wait/requirements.txt
```

2. Register for a free EIA API key: https://www.eia.gov/opendata/register.php

3. Copy the example env file and add your key:

```bash
cp .env.example .env
# edit .env and set EIA_API_KEY
export EIA_API_KEY=your_key_here
```

## Run the pipeline

Fetch datasets from EIA and print a report:

```bash
cd gas_wait
PYTHONPATH=src python -m data.build_dataset --fetch
```

Report only (after data has already been downloaded):

```bash
cd gas_wait
PYTHONPATH=src python -m data.build_dataset
```

Fetch without reporting via the fetch module directly:

```bash
cd gas_wait
PYTHONPATH=src python -m data.fetch_datasets
```

## Tests

```bash
cd gas_wait
PYTHONPATH=src python -m unittest discover -s tests -p "test_*.py"
```

This discovers EIA, leakage, regional, and modeling-framework tests (via
``tests/test_modeling_framework_suite.py``).

## Modeling framework

Reusable modules under `src/modeling/` support feature groups, point-in-time joins,
walk-forward backtesting, WAIT/FILL UP signals, and theoretical economics. See
[docs/modeling_framework.md](docs/modeling_framework.md).

Quick example — resolve feature columns for an experiment:

```python
from modeling.config import resolve_feature_columns

cols = resolve_feature_columns(["RETAIL_MOMENTUM", "CRUDE", "WHOLESALE"])
```

The weekly EIA pipeline in `weekly_pipeline.py` reproduces exp01 features for
validation only; it does not overwrite saved experiment results.

## Personal MVP (weekly CLI)

Local weekly signal for personal use (not the 72-hour product):

```bash
cd gas_wait
PYTHONPATH=src python -m gas_wait_cli
PYTHONPATH=src python -m gas_wait_cli --history
PYTHONPATH=src python -m gas_wait_cli --json
```

See [docs/personal_mvp.md](docs/personal_mvp.md).

## Environment check

```bash
python gas_wait/src/verify_imports.py
```
