# Gas Wait

Gas Wait is a research project that asks a practical question: **is the U.S.
average retail gasoline price more likely to rise or fall over the next week?**
It combines weekly consumer prices with daily crude and wholesale markets,
gasoline inventories, and seasonality, then translates the forecast into:

- **WAIT** — predicted week-ahead change is at or below −$0.03/gal
- **FILL UP** — predicted week-ahead change is at or above +$0.03/gal
- **NO CLEAR SIGNAL** — prediction is inside the ±$0.03/gal dead zone

> **Current scope:** a weekly prediction for the **U.S. national average** and
> a local command-line research MVP. It does not predict tomorrow's price or
> the price at a local station.

## Results

The final national Ridge model achieved **77.3% out-of-sample directional
accuracy predicting week-ahead U.S. gasoline price movements** on a
**525-week chronological holdout**.

In plain language: **the model correctly predicted whether the U.S. average
gasoline price would rise or fall approximately 3 out of 4 times on unseen
historical weeks.**

| Holdout metric | Ridge model | Momentum baseline |
| --- | ---: | ---: |
| Directional accuracy | **77.3%** | 68.1% |
| Mean absolute error | **2.86¢/gal** | 3.95¢/gal |
| Evaluation period | 525 unseen weeks | Same 525 weeks |

The expanding yearly walk-forward evaluation produced **79.1% directional
accuracy** for the same full Ridge specification.

These are historical research results, not a guarantee of future accuracy or
savings. The holdout begins at the fixed chronological cutoff
`2016-07-19`; thresholds were not optimized on the test period. See
[`reports/first_model_results.md`](reports/first_model_results.md) for the full
methodology and metrics.

## How the current model works

The target is:

```text
U.S. regular retail price on Monday T+7
minus
U.S. regular retail price on Monday T
```

The prediction clock is **Tuesday at 12:00 p.m. Eastern**, after Monday's
weekly retail observation. A standardized Ridge regression (`alpha=1.0`) uses
17 frozen features:

- recent weekly retail momentum
- daily WTI crude changes and volatility
- NY Harbor and Gulf Coast wholesale gasoline changes and volatility
- a wholesale/crude crack-spread feature
- publication-lagged gasoline inventories
- calendar seasonality

Point-in-time joins prevent future observations, same-day market closes, and
unpublished inventory reports from entering a feature row. Missing market
sessions are not interpolated. See
[`docs/modeling_design.md`](docs/modeling_design.md) and
[`docs/modeling_framework.md`](docs/modeling_framework.md).

## Data sources

All model data comes from the
[U.S. Energy Information Administration Open Data API v2](https://www.eia.gov/opendata/documentation.php):

| Data | Native frequency | Role |
| --- | --- | --- |
| U.S. regular retail gasoline | Weekly | Forecast target and momentum |
| WTI crude spot | Daily | Upstream market signal |
| NY Harbor regular gasoline spot | Daily | Wholesale gasoline signal |
| Gulf Coast regular gasoline spot | Daily | Wholesale gasoline signal |
| U.S. total gasoline stocks | Weekly | Supply and seasonal context |

Regional research additionally uses EIA weekly city, state, and PADD retail
series plus the nearest available EIA wholesale hub. No AAA, GasBuddy, OPIS,
PDI, scraped, synthetic, or licensed commercial data is included.

EIA information products are generally U.S. government public-domain
materials; attribution and the EIA API terms still apply. Source metadata is
tracked, while downloaded raw responses and generated CSV datasets are
excluded from Git so the repository stays small and reproducible.

## Reproduce the project

### 1. Clone and create an environment

```bash
git clone https://github.com/samielrem/gas-wait.git
cd gas-wait
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r gas_wait/requirements.txt
```

### 2. Configure an EIA API key

Register for a free key at
[eia.gov/opendata/register.php](https://www.eia.gov/opendata/register.php).
Never commit the key.

```bash
cp .env.example .env
# Edit .env and replace the placeholder, then load it into your shell:
set -a
source .env
set +a
```

### 3. Download EIA data

From `gas_wait/`:

```bash
cd gas_wait
PYTHONPATH=src python -m data.build_dataset --fetch
```

This recreates ignored files under `data/raw/` and `data/processed/`. To print
the local dataset report without downloading again:

```bash
PYTHONPATH=src python -m data.build_dataset
```

### 4. Run the automated tests

After downloading the required EIA datasets:

```bash
PYTHONPATH=src python -m unittest discover -s tests -p "test_*.py"
```

The public-release audit ran **62 passing automated tests**. The personal MVP
milestone originally had 59; JSON-output and raw-credential-redaction tests
were added afterward.
Tests make no live API calls.

### 5. Reproduce the research artifacts

The checked-in reports and notebooks preserve the original experiment
outputs. The commands below regenerate derived CSVs, figures, and reports, so
run them only when you intentionally want to reproduce those artifacts:

```bash
# National exp01
PYTHONPATH=src python -m modeling.evaluate

# Discover/fetch regional EIA retail and run the regional experiment
PYTHONPATH=src python -m modeling.discover_regional_retail
PYTHONPATH=src python -m modeling.run_regional_experiment
```

The notebooks in `notebooks/` include saved research outputs and use
repository-relative paths. To inspect them interactively, install Jupyter
Lab in the active environment and launch it from `gas_wait/`:

```bash
python -m pip install jupyterlab
python -m jupyter lab notebooks/
```

## Run the personal CLI

After downloading the EIA data, from `gas_wait/`:

```bash
PYTHONPATH=src python -m gas_wait_cli
PYTHONPATH=src python -m gas_wait_cli --history
PYTHONPATH=src python -m gas_wait_cli --json
```

The CLI refits the frozen national Ridge specification on historical rows
with known targets and generates the latest **~7-day national weekly**
signal. It displays data freshness and writes only signal metadata to the
ignored local file `data/processed/signal_history.csv`. See
[`docs/personal_mvp.md`](docs/personal_mvp.md).

## Current product status and limitations

### Current

- U.S. national average retail gasoline
- Monday-to-Monday (~7-day) target
- Tuesday-noon point-in-time prediction clock
- research backtests and a personal local CLI
- EIA latest-vintage data, not archived original-release vintages

### Future direction

- local or metro-level retail forecasts
- daily targets and 24–72 hour decisions
- licensed daily retail observations and explicit vendor publication clocks

The major missing ingredient is legitimate **daily local retail gasoline
data**. Weekly EIA retail is not upsampled or treated as a daily target.
Until licensed daily retail exists, Gas Wait cannot honestly claim to predict
tomorrow's local station price.

## Repository structure

```text
gas_wait/
├── data/        # tracked source metadata; ignored downloaded/generated data
├── docs/        # modeling, data, and personal MVP documentation
├── models/      # placeholder for local model artifacts
├── notebooks/   # reproducible research notebooks with saved outputs
├── reports/     # frozen experiment findings and figures
├── src/         # data pipeline, modeling framework, and CLI
└── tests/       # unit, leakage, framework, regional, and CLI tests
```

## Additional research

- [`reports/first_model_results.md`](reports/first_model_results.md) — national
  holdout and walk-forward results
- [`reports/regional_experiment.md`](reports/regional_experiment.md) —
  geographic wholesale-hub experiment
- [`docs/commercial_data_requirements.md`](docs/commercial_data_requirements.md)
  — requirements for future licensed daily retail data

## License

The project source code is available under the
[MIT License](../LICENSE). EIA data remains subject to its source attribution
and API terms.
