#!/usr/bin/env python3
"""Create and execute notebooks/02_first_model.ipynb."""

from __future__ import annotations

import base64
import io
import json
import shutil
import sys
from contextlib import redirect_stdout
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = ROOT / "notebooks"
FIGURES = NOTEBOOKS / "figures"
REPORT_FIGURES = ROOT / "reports" / "figures"
NB_PATH = NOTEBOOKS / "02_first_model.ipynb"

sys.path.insert(0, str(ROOT / "src"))


def md(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": _lines(source)}


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": _lines(source),
    }


def _lines(source: str) -> list[str]:
    text = source.strip("\n") + "\n"
    parts = text.split("\n")
    return [line + "\n" for line in parts[:-1]] + ([parts[-1] + "\n"] if parts[-1] else [])


CELLS = [
    md(
        """# 02 — First model (weekly retail Δ)

Research backtest, not production. Ridge only. No extra downloads. No LA RBOB.

**Target:** `retail_price(Monday T+7) − retail_price(Monday T)` in $/gal.

**Prediction clock:** Tuesday 12:00 p.m. ET after Monday T.

**Question:** Does lagged daily WTI / NY Harbor / Gulf Coast improve one-week-ahead U.S. regular retail change versus a retail-momentum baseline?

Full write-up: `reports/first_model_results.md`."""
    ),
    code(
        """from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import pandas as pd

ROOT = Path.cwd()
if (ROOT / "src" / "modeling").exists():
    pass
elif (ROOT / "gas_wait" / "src" / "modeling").exists():
    ROOT = ROOT / "gas_wait"
elif (ROOT / "notebooks").exists() and (ROOT.parent / "src" / "modeling").exists():
    ROOT = ROOT.parent

sys.path.insert(0, str(ROOT / "src"))

from modeling.build_features import ALL_MODEL_FEATURES, _assert_no_leakage, build_weekly_model_dataset
from modeling.evaluate import (
    DECISION_THRESHOLDS,
    MODEL_LABELS,
    decision_metrics,
    regression_metrics,
)
from modeling.train_baseline import chronological_split, fit_predict_holdout, walk_forward_all

result = build_weekly_model_dataset()
df = result.frame
dropped = result.dropped
print(f"kept={len(df)} dropped={len(dropped)}")
print(df["prediction_date"].min().date(), "→", df["prediction_date"].max().date())
df.head()"""
    ),
    md("## Leakage checks (must pass before training)"),
    code(
        """_assert_no_leakage(df)

checks = pd.DataFrame(
    {
        "check": [
            "spot timestamp <= Monday T",
            "spot timestamp < prediction Tuesday",
            "inventory Friday lag >= 10 days",
            "inventory release <= prediction_ts",
            "target is exactly Monday T+7",
            "no LA RBOB columns",
            "features have no NA",
        ],
        "pass": [
            df["spot_feature_timestamp"].le(df["retail_monday"]).all(),
            df["spot_feature_timestamp"].lt(df["prediction_date"]).all(),
            ((df["prediction_date"] - df["inventory_feature_timestamp"]).dt.days >= 10).all(),
            df["inventory_release_ts_utc"].le(df["prediction_ts_utc"]).all(),
            (df["target_timestamp"] - df["retail_monday"]).dt.days.eq(7).all(),
            not any("rbob" in c.lower() or "la_" in c.lower() for c in df.columns),
            not df[ALL_MODEL_FEATURES].isna().any().any(),
        ],
    }
)
checks"""
    ),
    md("## Dropped rows"),
    code(
        """dropped["reason_group"] = dropped["reason"].str.split(":").str[0]
dropped["reason_group"].value_counts().rename("count").to_frame()"""
    ),
    md(
        """## Train / test

Chronological 70/30. No shuffling. Ridge alpha is frozen at 1.0."""
    ),
    code(
        """split = chronological_split(df)
holdout = fit_predict_holdout(split)
walk = walk_forward_all(df)
print("train", split.train["prediction_date"].min().date(), "→", split.train["prediction_date"].max().date(), len(split.train))
print("test ", split.test["prediction_date"].min().date(), "→", split.test["prediction_date"].max().date(), len(split.test))"""
    ),
    md("## Holdout regression"),
    code(
        """rows = []
for name, label in MODEL_LABELS.items():
    m = regression_metrics(split.test["target"], holdout[name].predictions)
    m["model"] = label
    rows.append(m)
pd.DataFrame(rows).set_index("model")[["n", "mae", "rmse", "r2", "corr", "directional_accuracy"]]"""
    ),
    md("## Walk-forward regression"),
    code(
        """actual = df.set_index("prediction_date")["target"]
rows = []
for name, label in MODEL_LABELS.items():
    aligned = pd.concat([actual, walk[name].rename("pred")], axis=1).dropna()
    m = regression_metrics(aligned["target"], aligned["pred"])
    m["model"] = label
    rows.append(m)
pd.DataFrame(rows).set_index("model")[["n", "mae", "rmse", "r2", "corr", "directional_accuracy"]]"""
    ),
    md("## Decision metrics (δ = $0.03 / $0.04 / $0.05)"),
    code(
        """dec_rows = []
for delta in DECISION_THRESHOLDS:
    for name, label in MODEL_LABELS.items():
        d = decision_metrics(split.test["target"], holdout[name].predictions, delta)
        d["model"] = label
        dec_rows.append(d)
pd.DataFrame(dec_rows).set_index(["delta", "model"])[
    [
        "n_wait",
        "n_fill",
        "n_none",
        "pct_no_signal",
        "pct_correct_among_signals",
        "mean_actual_after_wait",
        "mean_actual_after_fill",
        "mean_savings_per_fill_usd",
    ]
]"""
    ),
    md("## Charts"),
    code(
        """from matplotlib import image as mpimg
import matplotlib.pyplot as plt

fig_dir = ROOT / "reports" / "figures"
for name in [
    "holdout_actual_vs_predicted.png",
    "holdout_residuals.png",
    "holdout_prediction_distribution.png",
    "holdout_cumulative_savings.png",
    "holdout_mae_by_year.png",
]:
    path = fig_dir / name
    print(path.relative_to(ROOT))
    img = mpimg.imread(path)
    plt.figure(figsize=(10, 4.4))
    plt.imshow(img)
    plt.axis("off")
    plt.title(name)
    plt.close()"""
    ),
    md(
        """## Honest takeaways

1. **Ridge beats momentum** on MAE and direction (MODEL 4 holdout MAE 2.86¢ vs 3.95¢; direction 77% vs 68%).
2. **Daily wholesale/crude features are the lift.** MODEL 3 vs MODEL 2 is the large jump.
3. **Inventory + seasonality add almost nothing** once markets are in (~0.07¢ MAE).
4. **Economic benefit is small in dollars:** ~$0.16 per 15-gallon fill vs always-fill-now on WAIT calls only. FILL quality is better than momentum, but that does not show up in the always-fill P&L by construction.
5. **~57% of weeks are silent** at a 3¢ dead zone.
6. This is **not** a 3-day local product. It is evidence that hub spots lead *national weekly* pumps, which is a reason to buy licensed daily retail — not a reason to ship an app on EIA Mondays."""
    ),
]


def build_notebook() -> dict:
    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
        "cells": CELLS,
    }


def _run_cell_source(source: str, namespace: dict, filename: str) -> None:
    import ast

    tree = ast.parse(source, filename=filename)
    if not tree.body:
        return
    last = tree.body[-1]
    if isinstance(last, ast.Expr):
        preamble = ast.Module(body=tree.body[:-1], type_ignores=[])
        exec(compile(preamble, filename, "exec"), namespace)
        value = eval(compile(ast.Expression(last.value), filename, "eval"), namespace)
        if value is None:
            return
        if hasattr(value, "to_string"):
            print(value.to_string())
        else:
            print(value)
        return
    exec(compile(source, filename, "exec"), namespace)


def execute(nb: dict) -> dict:
    namespace: dict = {}
    count = 0
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        count += 1
        cell["execution_count"] = count
        source = "".join(cell["source"])
        buf = io.StringIO()
        with redirect_stdout(buf):
            _run_cell_source(source, namespace, f"cell_{count}")
        text = buf.getvalue()
        outputs = []
        if text:
            outputs.append(
                {
                    "output_type": "stream",
                    "name": "stdout",
                    "text": text if text.endswith("\n") else text + "\n",
                }
            )
        if "holdout_actual_vs_predicted.png" in source:
            for name in [
                "holdout_actual_vs_predicted.png",
                "holdout_residuals.png",
                "holdout_prediction_distribution.png",
                "holdout_cumulative_savings.png",
                "holdout_mae_by_year.png",
            ]:
                path = REPORT_FIGURES / name
                if path.exists():
                    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
                    outputs.append(
                        {
                            "output_type": "display_data",
                            "data": {"image/png": encoded, "text/plain": f"<Figure {name}>"},
                            "metadata": {},
                        }
                    )
        cell["outputs"] = outputs
    return nb


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    if REPORT_FIGURES.exists():
        for png in REPORT_FIGURES.glob("holdout_*.png"):
            shutil.copy2(png, FIGURES / png.name)
    nb = build_notebook()
    execute(nb)
    NB_PATH.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    print(f"Wrote executed notebook to {NB_PATH}")


if __name__ == "__main__":
    main()
