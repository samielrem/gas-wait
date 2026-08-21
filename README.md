# Gas Wait

Gas Wait is a leakage-aware gasoline-price forecasting research project. The
current system predicts whether the **U.S. national average retail gasoline
price** will rise or fall over roughly the next week and exposes the result
through a small local CLI.

> This is a weekly national research MVP—not a prediction of tomorrow's local
> station price. A future 24–72 hour local product requires licensed daily
> retail data.

## Headline result

The final Ridge model achieved **77.3% out-of-sample directional accuracy
predicting week-ahead U.S. gasoline price movements** on a 525-week
chronological holdout.

That means the model correctly predicted whether the U.S. average gasoline
price would rise or fall approximately 3 out of 4 times on unseen historical
weeks.

- Ridge holdout directional accuracy: **77.3%**
- Momentum baseline directional accuracy: **68.1%**
- Ridge holdout MAE: **2.86¢/gal**
- Momentum baseline MAE: **3.95¢/gal**
- Ridge walk-forward directional accuracy: **79.1%**
- Automated tests: **62 passing**

These are historical research results, not guaranteed future accuracy or
savings. Full methodology:
[`gas_wait/reports/first_model_results.md`](gas_wait/reports/first_model_results.md)

## What is included

- EIA Open Data API pipeline for weekly retail, daily WTI and wholesale
  gasoline, and weekly inventories
- point-in-time feature engineering and leakage assertions
- national and regional historical experiments
- reusable walk-forward backtesting framework
- WAIT / FILL UP / NO CLEAR SIGNAL decision engine
- personal weekly command-line MVP
- documented limitations and future daily-retail requirements

## Quick start

```bash
git clone https://github.com/samielrem/gas-wait.git
cd gas-wait
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r gas_wait/requirements.txt

cp .env.example .env
# Add your free EIA API key, then:
set -a
source .env
set +a

cd gas_wait
PYTHONPATH=src python -m data.build_dataset --fetch
PYTHONPATH=src python -m unittest discover -s tests -p "test_*.py"
PYTHONPATH=src python -m gas_wait_cli
```

Obtain an API key from
[EIA Open Data](https://www.eia.gov/opendata/register.php). Downloaded raw
responses, processed datasets, API keys, and local signal history are ignored
by Git.

## Start here

- [Detailed project README](gas_wait/README.md)
- [Modeling design](gas_wait/docs/modeling_design.md)
- [Reusable framework](gas_wait/docs/modeling_framework.md)
- [Personal CLI](gas_wait/docs/personal_mvp.md)
- [National results](gas_wait/reports/first_model_results.md)
- [Regional experiment](gas_wait/reports/regional_experiment.md)
- [Future commercial-data requirements](gas_wait/docs/commercial_data_requirements.md)

## Data and license

Model inputs come from the U.S. Energy Information Administration. EIA
government information products are generally public domain; attribution and
API terms still apply. No scraped or licensed AAA, GasBuddy, OPIS, or PDI data
is included.

The project source code is available under the [MIT License](LICENSE).
