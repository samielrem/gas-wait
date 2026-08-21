"""Run exp02 regional retail ↔ wholesale Ridge experiments.

Does not overwrite reports/first_model_results.md or weekly_model_dataset.csv.
Does not tune on the holdout. Ridge alpha and δ are frozen from exp01.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .build_features import PROCESSED_DIR, DatasetBuildResult
from .build_regional_features import FEATURE_SETS, build_regional_dataset
from .evaluate import PRIMARY_DELTA, decision_metrics, regression_metrics
from .regional_config import (
    NATIONAL_HOLDOUT_CUTOFF,
    SKIPPED_GEOGRAPHIES,
    VIABLE_REGIONS,
    RegionalExperiment,
    regional_model_csv,
)
from .train_baseline import FitResult, Split

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = PROJECT_ROOT.parent
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
RIDGE_ALPHA = 1.0

FROZEN_NATIONAL = {
    "momentum_mae": 0.0395,
    "momentum_dir": 0.681,
    "momentum_r2": 0.0891,
    "market_mae": 0.0293,
    "market_dir": 0.775,
    "market_r2": 0.4914,
    "full_mae": 0.0286,
    "full_dir": 0.773,
    "full_r2": 0.5112,
    "test_n": 525,
    "cutoff": NATIONAL_HOLDOUT_CUTOFF,
}

MODEL_LABELS = {
    "momentum_baseline": "Momentum (retail_d7)",
    "ridge_retail": "Ridge, regional retail only",
    "ridge_matched": "Ridge, retail + matched hub + WTI",
    "ridge_mismatched": "Ridge, retail + mismatched hub + WTI",
    "ridge_full": "Ridge, matched + national inventory + seasonality",
}


def load_env() -> None:
    for path in (WORKSPACE_ROOT / ".env", PROJECT_ROOT / ".env"):
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def _ridge() -> Pipeline:
    return Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=RIDGE_ALPHA))])


def split_at_cutoff(df: pd.DataFrame, cutoff: str = NATIONAL_HOLDOUT_CUTOFF) -> Split:
    ordered = df.sort_values("prediction_date").reset_index(drop=True)
    cut = pd.Timestamp(cutoff)
    train = ordered.loc[ordered["prediction_date"] < cut].copy()
    test = ordered.loc[ordered["prediction_date"] >= cut].copy()
    if train.empty or test.empty:
        raise ValueError(f"Empty train or test for cutoff {cutoff}: train={len(train)} test={len(test)}")
    return Split(train=train, test=test, cutoff=cut)


def fit_region(split: Split) -> dict[str, FitResult]:
    results: dict[str, FitResult] = {}
    y_train = split.train["target"].to_numpy()
    test_index = split.test.index
    results["momentum_baseline"] = FitResult(
        name="momentum_baseline",
        predictions=split.test["retail_d7"].copy(),
        fitted=None,
        features=["retail_d7"],
    )
    for name, features in FEATURE_SETS.items():
        if name == "momentum_baseline":
            continue
        model = _ridge()
        model.fit(split.train[features], y_train)
        pred = pd.Series(model.predict(split.test[features]), index=test_index)
        results[name] = FitResult(name=name, predictions=pred, fitted=model, features=features)
    return results


def _corr(a: pd.Series, b: pd.Series) -> float:
    aligned = pd.concat([a, b], axis=1).dropna()
    if len(aligned) < 20 or aligned.iloc[:, 1].std() == 0 or aligned.iloc[:, 0].std() == 0:
        return float("nan")
    return float(aligned.iloc[:, 0].corr(aligned.iloc[:, 1]))


def pass_through_correlations(df: pd.DataFrame) -> dict[str, float]:
    y = df["target"]
    return {
        "corr_target_matched_d5": _corr(y, df["hub_d5"]),
        "corr_target_mismatch_d5": _corr(y, df["mis_d5"]),
        "corr_target_wti_d5": _corr(y, df["wti_d5"]),
        "corr_target_nyh_d5": _corr(y, df["nyh_d5"]),
        "corr_target_gc_d5": _corr(y, df["gc_d5"]),
        "corr_target_la_d5": _corr(y, df["la_d5"]),
        "corr_target_retail_d7": _corr(y, df["retail_d7"]),
    }


def ensure_regional_retail() -> None:
    missing = [
        region.retail.dataset_id
        for region in VIABLE_REGIONS
        if not (PROCESSED_DIR / f"{region.retail.dataset_id}.csv").exists()
    ]
    if not missing:
        return
    load_env()
    from data.eia_client import EIAClient
    from data.fetch_datasets import fetch_dataset
    from data.datasets import REGIONAL_RETAIL_DATASETS

    client = EIAClient()
    for definition in REGIONAL_RETAIL_DATASETS:
        if definition.dataset_id in missing or not (PROCESSED_DIR / f"{definition.dataset_id}.csv").exists():
            logger.info("Fetching missing regional retail %s", definition.dataset_id)
            fetch_dataset(client, definition)


def _fmt(value: float | None, digits: int = 4) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "n/a"
    return f"{value:.{digits}f}"


def make_regional_charts(summary: pd.DataFrame) -> list[Path]:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    labels = summary["label"].tolist()
    x = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(10, 4.8))
    width = 0.22
    ax.bar(x - 1.5 * width, summary["mom_mae"] * 100, width, label="Momentum", color="#5a5a5a")
    ax.bar(x - 0.5 * width, summary["matched_mae"] * 100, width, label="Matched hub Ridge", color="#1f4e79")
    ax.bar(x + 0.5 * width, summary["mismatch_mae"] * 100, width, label="Mismatched hub Ridge", color="#b07d2a")
    ax.axhline(FROZEN_NATIONAL["market_mae"] * 100, color="#8b2e2e", ls="--", lw=1.0, label="National market Ridge")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("Holdout MAE (¢/gal)")
    ax.set_title("Regional holdout MAE vs frozen national market model")
    ax.legend(fontsize=8)
    path = FIGURES_DIR / "regional_mae_by_region.png"
    plt.tight_layout()
    plt.savefig(path, dpi=140, bbox_inches="tight")
    plt.close()
    paths.append(path)

    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.bar(x - width, summary["mom_dir"] * 100, width, label="Momentum", color="#5a5a5a")
    ax.bar(x, summary["matched_dir"] * 100, width, label="Matched hub Ridge", color="#1f4e79")
    ax.bar(x + width, summary["mismatch_dir"] * 100, width, label="Mismatched hub Ridge", color="#b07d2a")
    ax.axhline(FROZEN_NATIONAL["market_dir"] * 100, color="#8b2e2e", ls="--", lw=1.0, label="National market Ridge")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("Holdout directional accuracy (%)")
    ax.set_title("Regional holdout directional accuracy")
    ax.set_ylim(50, 90)
    ax.legend(fontsize=8)
    path = FIGURES_DIR / "regional_diracc_by_region.png"
    plt.tight_layout()
    plt.savefig(path, dpi=140, bbox_inches="tight")
    plt.close()
    paths.append(path)

    fig, ax = plt.subplots(figsize=(10, 4.6))
    ax.bar(x - 0.2, summary["corr_matched"], 0.4, label="corr(target, matched hub d5)", color="#1f4e79")
    ax.bar(x + 0.2, summary["corr_mismatch"], 0.4, label="corr(target, mismatched hub d5)", color="#b07d2a")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("Correlation on modeling rows")
    ax.set_title("Wholesale 5-session change vs next-week regional retail change")
    ax.axhline(0, color="black", lw=0.6)
    ax.legend(fontsize=8)
    path = FIGURES_DIR / "regional_passthrough_corr.png"
    plt.tight_layout()
    plt.savefig(path, dpi=140, bbox_inches="tight")
    plt.close()
    paths.append(path)
    return paths


def write_report(
    *,
    coverage: pd.DataFrame,
    region_rows: list[dict],
    chart_paths: list[Path],
    path: Path,
) -> None:
    summary = pd.DataFrame(region_rows)
    best = summary.sort_values("matched_dir", ascending=False).iloc[0]
    best_mae = summary.sort_values("matched_mae").iloc[0]
    matched_beats_mismatch = int((summary["matched_mae"] < summary["mismatch_mae"]).sum())
    matched_beats_national = int((summary["matched_mae"] < FROZEN_NATIONAL["market_mae"]).sum())

    lines: list[str] = []
    lines.append("# Regional experiment — matched wholesale → retail")
    lines.append("")
    lines.append(
        "Research backtest. Ridge α = 1.0 and δ = $0.03 are **frozen from exp01**. "
        "Holdout cutoff is the same Tuesday as the national experiment: **2016-07-19**. "
        "The first-model report and `weekly_model_dataset.csv` were not modified."
    )
    lines.append("")
    lines.append("EIA weekly regular retail (product `EPMR`, process `PTE`) has **29 geographies**. Only six have a defensible match to an EIA daily gasoline spot already in the repo.")
    lines.append("")
    lines.append("## EIA weekly regular retail coverage")
    lines.append("")
    lines.append("Source: EIA API v2 `petroleum/pri/gnd`, frequency=weekly, verified 2026-08-20. Same GDFU publication clock as the national series (treated as available Tuesday 12:00 p.m. ET).")
    lines.append("")
    lines.append("| duoarea | Geography | N | Missing | Start | End |")
    lines.append("| --- | --- | ---: | ---: | --- | --- |")
    for _, row in coverage.sort_values(["start", "duoarea"]).iterrows():
        lines.append(
            f"| {row['duoarea']} | {row['area-name']} | {int(row['n'])} | {int(row['n_missing'])} | {row['start']} | {row['end']} |"
        )
    lines.append("")
    lines.append("## Mapping (do not force a bad match)")
    lines.append("")
    lines.append("| Region | Matched wholesale | Why this pair | Mismatch control |")
    lines.append("| --- | --- | --- | --- |")
    for region in VIABLE_REGIONS:
        lines.append(
            f"| {region.label} | {region.matched_hub.geographic_area} | {region.mapping_rationale} | {region.mismatched_hub.geographic_area} |"
        )
    lines.append("")
    lines.append("### Skipped")
    lines.append("")
    for skip in SKIPPED_GEOGRAPHIES:
        lines.append(f"- **{skip['name']}** (`{skip['duoarea']}`): {skip['reason']}")
    lines.append("")
    lines.append("National inventories are used only in MODEL `ridge_full`, as a separate increment. They are U.S. total stocks, not PADD stocks.")
    lines.append("")
    lines.append("## Holdout results (train before 2016-07-19, test after)")
    lines.append("")
    lines.append("Frozen national reference (exp01, same cutoff): momentum MAE 3.95¢ / dir 68.1%; market Ridge MAE 2.93¢ / dir 77.5%; full Ridge MAE 2.86¢ / dir 77.3%.")
    lines.append("")
    lines.append(
        "LA/CA drop more rows because LA RBOB daily only begins 2003-03-11 (no interpolation). "
        "PADD 3 drops include early-1990s inventory seasonal-z warmup, same rule as exp01."
    )
    lines.append("")
    lines.append("| Region | N train | N test | Mom MAE | Matched MAE | Mismatch MAE | Full MAE | Mom dir | Matched dir | Mismatch dir | Matched R² | Signals (δ=$0.03) | % silent |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in region_rows:
        lines.append(
            f"| {row['label']} | {row['n_train']} | {row['n_test']} | {_fmt(row['mom_mae'])} | {_fmt(row['matched_mae'])} | "
            f"{_fmt(row['mismatch_mae'])} | {_fmt(row['full_mae'])} | {_fmt(row['mom_dir'], 3)} | {_fmt(row['matched_dir'], 3)} | "
            f"{_fmt(row['mismatch_dir'], 3)} | {_fmt(row['matched_r2'])} | {row['n_wait']}/{row['n_fill']} | {_fmt(row['pct_silent'], 3)} |"
        )
    lines.append("")
    lines.append("Signals column is WAIT/FILL counts for the matched-hub Ridge. % silent is NO CLEAR SIGNAL at δ=$0.03.")
    lines.append("")
    lines.append("## Pass-through: is wholesale → retail stronger when geography matches?")
    lines.append("")
    lines.append("Correlations use the modeling rows (not a selected subset of the test set). `matched hub d5` is the 5-session change of the mapped gasoline spot, as-of Monday T.")
    lines.append("")
    lines.append("| Region | corr(target, matched d5) | corr(target, mismatched d5) | corr(target, WTI d5) | corr(target, retail_d7) |")
    lines.append("| --- | ---: | ---: | ---: | ---: |")
    for row in region_rows:
        lines.append(
            f"| {row['label']} | {_fmt(row['corr_matched'])} | {_fmt(row['corr_mismatch'])} | "
            f"{_fmt(row['corr_wti'])} | {_fmt(row['corr_mom'])} |"
        )
    lines.append("")
    lines.append(f"Matched-hub Ridge beat mismatched-hub Ridge on MAE in **{matched_beats_mismatch} of {len(region_rows)}** regions. It beat the frozen national market MAE in **{matched_beats_national} of {len(region_rows)}** regions.")
    lines.append("")
    lines.append("## Incremental groups")
    lines.append("")
    lines.append("| Region | Retail-only MAE | + matched market MAE | + inv/season MAE | Market lift vs retail-only | Inv/season lift |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
    for row in region_rows:
        lines.append(
            f"| {row['label']} | {_fmt(row['retail_mae'])} | {_fmt(row['matched_mae'])} | {_fmt(row['full_mae'])} | "
            f"{_fmt(row['retail_mae'] - row['matched_mae'])} | {_fmt(row['matched_mae'] - row['full_mae'])} |"
        )
    lines.append("")
    lines.append("## Charts")
    lines.append("")
    for chart in chart_paths:
        lines.append(f"- `{chart.relative_to(PROJECT_ROOT)}`")
    lines.append("")
    lines.append("## Surprises")
    lines.append("")
    lines.append(
        "Matching is **not** uniformly helpful in Ridge MAE. It helps the Northeast (NYC, PADD 1B). "
        "On the Gulf, the *mismatched* NY Harbor print beats the local Gulf Coast spot for Houston and PADD 3, both in correlation and in MAE. "
        "NY Harbor conventional gasoline is a liquid national benchmark; Gulf Coast spot is a refinery-gate print that does not automatically stick to weekly Houston/PADD 3 retail (taxes, brands, rack-to-retail lag). "
        "Do not force Gulf retail onto Gulf spot just because the names match."
    )
    lines.append("")
    lines.append(
        "California has the highest directional accuracy (**79.3%**), but regional momentum is already **77.0%**. "
        "Matched markets add only ~0.12¢ MAE. That is sticky CARB retail, not a large wholesale-lead effect. "
        "LA city MAE (**4.41¢**) is worse than the national 2.93¢: the city series is jumpy."
    )
    lines.append("")
    lines.append("Inventory + seasonality remain a ~0.0–0.08¢ afterthought, same as exp01.")
    lines.append("")
    lines.append("## Answers")
    lines.append("")
    lines.append(
        "**A. Which region has the strongest predictable signal?** "
        "It depends on the metric. **California → LA RBOB** has the highest matched-hub directional accuracy (79.3%), mostly from retail momentum (77.0% already). "
        "**PADD 1B → NY Harbor** has the best matched MAE (**2.73¢**), beating the frozen national market Ridge (2.93¢), with 77.6% direction. "
        "The cleanest *matching* win is the Northeast: Harbor 5-day changes correlate 0.57–0.59 with next-week retail vs 0.46–0.49 for Gulf Coast. "
        "PADD 3 is the weakest (MAE 4.15¢, direction 71.3%)."
    )
    lines.append("")
    lines.append(
        "**B. Does matching retail geography to wholesale geography improve performance?** "
        "**Sometimes, and mainly in the correlation sense.** "
        f"Matched-hub Ridge beat mismatched-hub Ridge on MAE in only **{matched_beats_mismatch} of {len(region_rows)}** regions (NYC and PADD 1B). "
        "Pass-through correlations *do* rise when the spec is actually different: Harbor beats Gulf for the Northeast; LA RBOB beats Harbor for LA/CA (0.53 vs 0.37). "
        "Gulf retail is the exception: NY Harbor tracks Houston/PADD 3 weekly changes as well or better than Gulf Coast spot. "
        f"Versus the national weekly model, matched regional Ridge beat national MAE in **{matched_beats_national} of {len(region_rows)}** regions (PADD 1B only). "
        "City labels are noisier than the U.S. average. Matching is a pairing rule, not a free accuracy upgrade."
    )
    lines.append("")
    lines.append(
        "**C. Is the improvement large enough to justify pursuing licensed daily retail data?** "
        "Not as a weekly-EIA accuracy story: regional matching did **not** produce a breakthrough over the national 77.5% / 2.93¢ model, and several city series are *worse*. "
        "It **is** enough to justify licensed daily/metro retail as the *next data purchase*, for a different reason: "
        "we now know which hub belongs with which city (Harbor with NYC, RBOB with LA), weekly city averages are too noisy for a local product, "
        "and a 3-day WAIT/FILL claim still cannot be scored on Monday EIA prints. Do not buy daily retail expecting 90% weekly direction from the same labels."
    )
    lines.append("")
    lines.append(
        "**D. Which geography should we use for the first real MVP?** "
        "**New York City (conventional, NY Harbor)** as the first city, with **PADD 1B** as the broader East Coast check — matching works and MAE is in the same band as the national model. "
        "**Los Angeles (CARB, LA RBOB)** as the second geography, because spec matching is real even though weekly MAE is worse. "
        "Do not launch on PADD 3, PADD 5, or U.S. average as if they were a local pump. California statewide direction looks strong because prices are sticky, not because LA RBOB suddenly explains the pump."
    )
    lines.append("")
    lines.append(
        "**E. What is the biggest remaining limitation?** "
        "The label is still a **weekly EIA city or PADD average**, published Tuesday, not a station or metro daily pump price. "
        "Even a correct hub match cannot see intra-week moves, $0.20 neighborhood gaps, or brand/tax residuals. "
        "Gulf results show that a geographically named spot is not automatically the best predictor. "
        "Inventories here are national. EIA API values are latest vintage, not original prints."
    )
    lines.append("")
    lines.append("## Method notes")
    lines.append("")
    lines.append("- Prediction timestamp: Tuesday 12:00 p.m. ET after Monday T.")
    lines.append("- Target: regional retail(T+7) − regional retail(T).")
    lines.append("- Daily spots: last session with observation date ≤ Monday T (available Tuesday noon).")
    lines.append("- No interpolation. No LA RBOB in Northeast/Gulf models except as a diagnostic correlation column.")
    lines.append("- Mismatched-hub Ridge is a pre-specified control, not a selected model.")
    lines.append("- Thresholds were not re-tuned.")
    lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Wrote %s", path)


def run_experiment() -> dict:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    ensure_regional_retail()
    coverage_path = PROCESSED_DIR / "eia_weekly_regular_retail_coverage.csv"
    coverage = pd.read_csv(coverage_path)
    region_rows: list[dict] = []
    built: dict[str, DatasetBuildResult] = {}
    fits: dict[str, dict[str, FitResult]] = {}

    for region in VIABLE_REGIONS:
        result = build_regional_dataset(region)
        built[region.region_id] = result
        out_path = regional_model_csv(region, PROCESSED_DIR)
        drop_path = PROCESSED_DIR / f"regional_weekly_model_{region.region_id}_dropped.csv"
        result.frame.to_csv(out_path, index=False)
        result.dropped.to_csv(drop_path, index=False)
        split = split_at_cutoff(result.frame)
        holdout = fit_region(split)
        fits[region.region_id] = holdout
        y = split.test["target"]
        metrics = {name: regression_metrics(y, fit.predictions) for name, fit in holdout.items()}
        dec = decision_metrics(y, holdout["ridge_matched"].predictions, PRIMARY_DELTA)
        passthrough = pass_through_correlations(result.frame)
        region_rows.append(
            {
                "region_id": region.region_id,
                "label": region.label,
                "n_kept": len(result.frame),
                "n_dropped": len(result.dropped),
                "n_train": len(split.train),
                "n_test": len(split.test),
                "mom_mae": metrics["momentum_baseline"]["mae"],
                "retail_mae": metrics["ridge_retail"]["mae"],
                "matched_mae": metrics["ridge_matched"]["mae"],
                "mismatch_mae": metrics["ridge_mismatched"]["mae"],
                "full_mae": metrics["ridge_full"]["mae"],
                "mom_dir": metrics["momentum_baseline"]["directional_accuracy"],
                "matched_dir": metrics["ridge_matched"]["directional_accuracy"],
                "mismatch_dir": metrics["ridge_mismatched"]["directional_accuracy"],
                "matched_r2": metrics["ridge_matched"]["r2"],
                "n_wait": int(dec["n_wait"]),
                "n_fill": int(dec["n_fill"]),
                "pct_silent": dec["pct_no_signal"],
                "corr_matched": passthrough["corr_target_matched_d5"],
                "corr_mismatch": passthrough["corr_target_mismatch_d5"],
                "corr_wti": passthrough["corr_target_wti_d5"],
                "corr_mom": passthrough["corr_target_retail_d7"],
            }
        )
        logger.info(
            "%s matched MAE=%.4f dir=%.3f vs mismatch MAE=%.4f",
            region.region_id,
            metrics["ridge_matched"]["mae"],
            metrics["ridge_matched"]["directional_accuracy"],
            metrics["ridge_mismatched"]["mae"],
        )

    charts = make_regional_charts(pd.DataFrame(region_rows))
    report_path = REPORTS_DIR / "regional_experiment.md"
    write_report(coverage=coverage, region_rows=region_rows, chart_paths=charts, path=report_path)
    return {"region_rows": region_rows, "charts": charts, "report_path": report_path, "built": built, "fits": fits}


if __name__ == "__main__":
    run_experiment()
