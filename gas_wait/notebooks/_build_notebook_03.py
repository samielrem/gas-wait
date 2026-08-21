#!/usr/bin/env python3
"""Create and execute notebooks/03_regional_experiment.ipynb."""

from __future__ import annotations

import base64
import io
import json
import shutil
from contextlib import redirect_stdout
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = ROOT / "notebooks"
FIGURES = NOTEBOOKS / "figures"
REPORT_FIGURES = ROOT / "reports" / "figures"
NB_PATH = NOTEBOOKS / "03_regional_experiment.ipynb"


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
        """# 03 — Regional retail ↔ wholesale experiment

Does matching EIA weekly retail geography to the relevant daily gasoline hub beat the national weekly model?

**Not production. No trees. No holdout tuning.** Ridge α = 1.0 and δ = $0.03 are frozen from exp01. Holdout cutoff is the same Tuesday: 2016-07-19.

Full write-up: `reports/regional_experiment.md`. The first-model report was not modified."""
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

from modeling.regional_config import SKIPPED_GEOGRAPHIES, VIABLE_REGIONS, NATIONAL_HOLDOUT_CUTOFF

coverage = pd.read_csv(ROOT / "data" / "processed" / "eia_weekly_regular_retail_coverage.csv")
print("geographies", len(coverage))
print("holdout cutoff", NATIONAL_HOLDOUT_CUTOFF)
coverage.sort_values(["start", "duoarea"])"""
    ),
    md("## Viable maps vs skipped geographies"),
    code(
        """maps = pd.DataFrame(
    [
        {
            "region": r.label,
            "retail_duoarea": r.retail.facets["duoarea"][0],
            "matched_hub": r.matched_hub.geographic_area,
            "mismatch_control": r.mismatched_hub.geographic_area,
        }
        for r in VIABLE_REGIONS
    ]
)
skipped = pd.DataFrame(list(SKIPPED_GEOGRAPHIES))
maps"""
    ),
    code("""skipped"""),
    md("## Holdout metrics (precomputed regional datasets)"),
    code(
        """from modeling.evaluate import PRIMARY_DELTA, decision_metrics, regression_metrics
from modeling.run_regional_experiment import MODEL_LABELS, split_at_cutoff, fit_region
from modeling.build_regional_features import build_regional_dataset

rows = []
for region in VIABLE_REGIONS:
    built = build_regional_dataset(region)
    split = split_at_cutoff(built.frame)
    holdout = fit_region(split)
    y = split.test["target"]
    matched = regression_metrics(y, holdout["ridge_matched"].predictions)
    mom = regression_metrics(y, holdout["momentum_baseline"].predictions)
    mis = regression_metrics(y, holdout["ridge_mismatched"].predictions)
    dec = decision_metrics(y, holdout["ridge_matched"].predictions, PRIMARY_DELTA)
    rows.append(
        {
            "region": region.label,
            "n_test": int(matched["n"]),
            "mom_mae": mom["mae"],
            "matched_mae": matched["mae"],
            "mismatch_mae": mis["mae"],
            "mom_dir": mom["directional_accuracy"],
            "matched_dir": matched["directional_accuracy"],
            "matched_r2": matched["r2"],
            "wait": int(dec["n_wait"]),
            "fill": int(dec["n_fill"]),
            "pct_silent": dec["pct_no_signal"],
        }
    )
pd.DataFrame(rows).set_index("region")"""
    ),
    md("## Charts"),
    code(
        """from matplotlib import image as mpimg
import matplotlib.pyplot as plt

fig_dir = ROOT / "reports" / "figures"
for name in [
    "regional_mae_by_region.png",
    "regional_diracc_by_region.png",
    "regional_passthrough_corr.png",
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

1. EIA has 29 weekly regular retail geographies. Only six had a defensible match to NY Harbor, Gulf Coast, or LA RBOB.
2. **PADD 1B → NY Harbor** is the only matched model that beats national MAE (2.73¢ vs 2.93¢). Direction is ~the same (77.6% vs 77.5%).
3. **California** has the highest direction (79.3%), but momentum was already 77%. Markets barely add.
4. Matching **raises pass-through correlation** on the coasts where the spec differs (Harbor vs Gulf for the Northeast; RBOB vs Harbor for CA/LA).
5. Matching **fails on the Gulf**: NY Harbor tracks Houston/PADD 3 weekly retail as well or better than Gulf Coast spot. Do not force that map.
6. City EIA series are often *noisier* than the U.S. average. This is not a 3-day local product.
7. First city MVP: **New York City + NY Harbor**. Second: **Los Angeles + LA RBOB**. Not PADD 3 or PADD 5."""
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
        if "regional_mae_by_region.png" in source:
            for name in [
                "regional_mae_by_region.png",
                "regional_diracc_by_region.png",
                "regional_passthrough_corr.png",
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
        for png in REPORT_FIGURES.glob("regional_*.png"):
            shutil.copy2(png, FIGURES / png.name)
    nb = build_notebook()
    execute(nb)
    NB_PATH.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    print(f"Wrote executed notebook to {NB_PATH}")


if __name__ == "__main__":
    main()
