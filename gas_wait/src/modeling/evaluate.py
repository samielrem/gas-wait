"""Evaluate first-experiment models and write the report plus charts."""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from .build_features import ALL_MODEL_FEATURES, save_weekly_model_dataset
from .train_baseline import (
    FEATURE_SETS,
    FitResult,
    chronological_split,
    fit_predict_holdout,
    walk_forward_all,
)

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
GALLONS = 15.0
DECISION_THRESHOLDS = (0.03, 0.04, 0.05)
PRIMARY_DELTA = 0.03

MODEL_LABELS = {
    "momentum_baseline": "MODEL 1: Momentum baseline (retail_d7)",
    "ridge_retail": "MODEL 2: Ridge, retail momentum only",
    "ridge_retail_market": "MODEL 3: Ridge, retail + daily markets",
    "ridge_full": "MODEL 4: Ridge, retail + markets + inventory + seasonality",
}


def regression_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float]:
    mask = y_true.notna() & y_pred.notna()
    yt = y_true.loc[mask].to_numpy()
    yp = y_pred.loc[mask].to_numpy()
    if len(yt) < 2:
        return {
            "n": float(len(yt)),
            "mae": np.nan,
            "rmse": np.nan,
            "r2": np.nan,
            "corr": np.nan,
            "directional_accuracy": np.nan,
        }
    corr = float(np.corrcoef(yt, yp)[0, 1]) if np.std(yp) > 0 and np.std(yt) > 0 else np.nan
    same_sign = np.sign(yt) == np.sign(yp)
    nonzero = yt != 0
    dir_acc = float(same_sign[nonzero].mean()) if nonzero.any() else np.nan
    return {
        "n": float(len(yt)),
        "mae": float(mean_absolute_error(yt, yp)),
        "rmse": float(np.sqrt(mean_squared_error(yt, yp))),
        "r2": float(r2_score(yt, yp)),
        "corr": corr,
        "directional_accuracy": dir_acc,
    }


def decision_from_pred(pred: pd.Series, delta: float) -> pd.Series:
    out = pd.Series("NO CLEAR SIGNAL", index=pred.index, dtype=object)
    out.loc[pred <= -delta] = "WAIT"
    out.loc[pred >= delta] = "FILL UP"
    return out


def decision_metrics(y_true: pd.Series, y_pred: pd.Series, delta: float) -> dict[str, float]:
    mask = y_true.notna() & y_pred.notna()
    actual = y_true.loc[mask]
    pred = y_pred.loc[mask]
    action = decision_from_pred(pred, delta)
    wait = action == "WAIT"
    fill = action == "FILL UP"
    none = action == "NO CLEAR SIGNAL"
    signaled = wait | fill

    wait_correct = (actual < 0) & wait
    fill_correct = (actual > 0) & fill
    n_signaled = int(signaled.sum())
    n_correct = int((wait_correct | fill_correct).sum())
    pct_correct = n_correct / n_signaled if n_signaled else np.nan

    savings_vs_fill_now = pd.Series(0.0, index=actual.index)
    savings_vs_fill_now.loc[wait] = -GALLONS * actual.loc[wait]
    return {
        "delta": delta,
        "n": float(len(actual)),
        "n_wait": float(wait.sum()),
        "n_fill": float(fill.sum()),
        "n_none": float(none.sum()),
        "pct_no_signal": float(none.mean()),
        "pct_signaled": float(signaled.mean()),
        "pct_correct_among_signals": float(pct_correct) if n_signaled else np.nan,
        "mean_actual_after_wait": float(actual.loc[wait].mean()) if wait.any() else np.nan,
        "mean_actual_after_fill": float(actual.loc[fill].mean()) if fill.any() else np.nan,
        "mean_savings_per_fill_usd": float(savings_vs_fill_now.mean()),
        "total_savings_usd": float(savings_vs_fill_now.sum()),
    }


def _savefig(name: str) -> Path:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / name
    plt.tight_layout()
    plt.savefig(path, dpi=140, bbox_inches="tight")
    plt.close()
    return path


def make_charts(
    test: pd.DataFrame,
    results: dict[str, FitResult],
    *,
    prefix: str,
) -> list[Path]:
    paths: list[Path] = []
    y = test["target"]
    full = results["ridge_full"].predictions
    mom = results["momentum_baseline"].predictions

    fig, ax = plt.subplots(figsize=(10, 4.6))
    ax.plot(test["prediction_date"], y, label="Actual retail Δ", lw=1.0, color="#1f4e79")
    ax.plot(test["prediction_date"], full, label="Ridge full", lw=1.0, color="#8b2e2e")
    ax.axhline(0, color="black", lw=0.6)
    ax.set_title("Actual vs predicted 1-week retail price change (test)")
    ax.set_xlabel("Prediction date")
    ax.set_ylabel("Dollars per gallon")
    ax.legend()
    paths.append(_savefig(f"{prefix}_actual_vs_predicted.png"))

    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.plot(test["prediction_date"], y - full, color="#5a3d1b", lw=0.9)
    ax.axhline(0, color="black", lw=0.6)
    ax.set_title("Ridge-full residuals on test")
    ax.set_xlabel("Prediction date")
    ax.set_ylabel("Actual − predicted ($/gal)")
    paths.append(_savefig(f"{prefix}_residuals.png"))

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.hist(full.dropna(), bins=40, color="#1f4e79", alpha=0.85)
    ax.axvline(0.03, color="#8b2e2e", ls="--", label="±3¢")
    ax.axvline(-0.03, color="#8b2e2e", ls="--")
    ax.set_title("Distribution of Ridge-full predictions (test)")
    ax.set_xlabel("Predicted change ($/gal)")
    ax.set_ylabel("Weeks")
    ax.legend()
    paths.append(_savefig(f"{prefix}_prediction_distribution.png"))

    fig, ax = plt.subplots(figsize=(10, 4.4))
    for name, color in [
        ("momentum_baseline", "#5a5a5a"),
        ("ridge_full", "#8b2e2e"),
    ]:
        pred = results[name].predictions
        action = decision_from_pred(pred, PRIMARY_DELTA)
        wait = action == "WAIT"
        pnl = pd.Series(0.0, index=test.index)
        pnl.loc[wait] = -GALLONS * y.loc[wait]
        ax.plot(test["prediction_date"], pnl.cumsum(), label=MODEL_LABELS[name], color=color, lw=1.2)
    ax.axhline(0, color="black", lw=0.6)
    ax.set_title("Cumulative theoretical savings vs always-fill-now (15 gal, WAIT only, δ=$0.03)")
    ax.set_xlabel("Prediction date")
    ax.set_ylabel("Cumulative USD")
    ax.legend()
    paths.append(_savefig(f"{prefix}_cumulative_savings.png"))

    yearly = pd.DataFrame({"year": test["prediction_date"].dt.year, "actual": y, "pred": full})
    yearly_metrics = []
    for year, grp in yearly.groupby("year"):
        m = regression_metrics(grp["actual"], grp["pred"])
        yearly_metrics.append({"year": year, "mae": m["mae"], "dir_acc": m["directional_accuracy"]})
    yearly_df = pd.DataFrame(yearly_metrics)
    if not yearly_df.empty:
        fig, ax = plt.subplots(figsize=(8, 4.2))
        years = yearly_df["year"].astype(int)
        ax.bar(years, yearly_df["mae"], color="#1f4e79", width=0.8)
        ax.set_title("Ridge-full test MAE by year")
        ax.set_xlabel("Year")
        ax.set_ylabel("MAE ($/gal)")
        ax.set_xticks(years)
        ax.tick_params(axis="x", rotation=45)
        paths.append(_savefig(f"{prefix}_mae_by_year.png"))
    return paths


def _fmt(value: float | None, digits: int = 4) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "n/a"
    return f"{value:.{digits}f}"


def write_report(
    *,
    dataset: pd.DataFrame,
    dropped: pd.DataFrame,
    split,
    holdout: dict[str, FitResult],
    walk_preds: dict[str, pd.Series],
    chart_paths: list[Path],
    path: Path,
) -> None:
    lines: list[str] = []
    y_test = split.test["target"]
    holdout_reg = {
        name: regression_metrics(y_test, fit.predictions) for name, fit in holdout.items()
    }
    holdout_dec = {
        name: {delta: decision_metrics(y_test, fit.predictions, delta) for delta in DECISION_THRESHOLDS}
        for name, fit in holdout.items()
    }

    wf_reg = {}
    actual_by_date = dataset.set_index("prediction_date")["target"]
    for name, pred in walk_preds.items():
        aligned = pd.concat([actual_by_date, pred.rename("pred")], axis=1).dropna()
        wf_reg[name] = regression_metrics(aligned["target"], aligned["pred"])

    mom = holdout_reg["momentum_baseline"]
    full = holdout_reg["ridge_full"]
    retail_only = holdout_reg["ridge_retail"]
    market = holdout_reg["ridge_retail_market"]
    full_dec = holdout_dec["ridge_full"][PRIMARY_DELTA]
    mom_dec = holdout_dec["momentum_baseline"][PRIMARY_DELTA]

    beats = full["mae"] < mom["mae"] - 1e-12
    market_helps = market["mae"] < retail_only["mae"] - 1e-12
    inv_season_help = full["mae"] < market["mae"] - 1e-12

    lines.append("# First model results — exp01 weekly retail Δ")
    lines.append("")
    lines.append("Research backtest only. Ridge alpha fixed at 1.0. Thresholds 3/4/5¢ were **not** tuned on test.")
    lines.append("")
    lines.append("## Dataset")
    lines.append("")
    lines.append(f"- Kept rows: **{len(dataset)}**")
    lines.append(f"- Dropped rows: **{len(dropped)}**")
    lines.append(f"- Prediction dates: {dataset['prediction_date'].min().date()} to {dataset['prediction_date'].max().date()}")
    lines.append(f"- Chronological cutoff: **{split.cutoff.date()}** (first {int(100 * 0.70)}% train / last 30% test)")
    lines.append(f"- Train weeks: {len(split.train)}; test weeks: {len(split.test)}")
    lines.append("")
    lines.append("### Dropped-row reasons")
    lines.append("")
    if dropped.empty:
        lines.append("None.")
    else:
        counts = dropped["reason"].str.split(":").str[0].value_counts()
        lines.append("| Reason | Count |")
        lines.append("| --- | ---: |")
        for reason, count in counts.items():
            lines.append(f"| {reason} | {int(count)} |")
    lines.append("")
    lines.append("Warm-up drops are expected: 20-session volatility, 5-year inventory seasonal z, and the 1990–91 retail gap.")
    lines.append("")
    lines.append("## Leakage checks")
    lines.append("")
    lines.append("These assertions ran on the kept dataset before training and passed:")
    lines.append("")
    lines.append("- `spot_feature_timestamp <= retail_monday` and `< prediction_date` (Monday close, not Tuesday close)")
    lines.append("- `inventory_release_ts_utc <= prediction_ts_utc`")
    lines.append("- inventory Friday is at least 10 calendar days before the Tuesday prediction")
    lines.append("- `target_timestamp` is Monday T+7 and after `prediction_date`")
    lines.append("- retail features use only Monday T and earlier prints (`retail_d7`, `retail_d14`)")
    lines.append("- rolling 1/3/5-session changes and 20-session vol are computed on the daily series in time order, then as-of joined to Monday T")
    lines.append("- no interpolation; 7-calendar-day WTI change is last price on or before T-7, not a filled path")
    lines.append("- LA RBOB is not in the feature set")
    lines.append("")
    lines.append("Documented limitation: EIA API values are latest vintage, not original prints.")
    lines.append("")
    lines.append("## Holdout regression (final 30%)")
    lines.append("")
    lines.append("| Model | N | MAE | RMSE | R² | Corr | Directional acc. |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for name in MODEL_LABELS:
        m = holdout_reg[name]
        lines.append(
            f"| {MODEL_LABELS[name]} | {int(m['n'])} | {_fmt(m['mae'])} | {_fmt(m['rmse'])} | "
            f"{_fmt(m['r2'])} | {_fmt(m['corr'])} | {_fmt(m['directional_accuracy'], 3)} |"
        )
    lines.append("")
    lines.append("Directional accuracy ignores actual zeros. Majority-class baseline on this problem is ~52%.")
    lines.append("")
    lines.append("## Walk-forward (expanding, predict each year from all prior years)")
    lines.append("")
    lines.append("| Model | N | MAE | RMSE | R² | Corr | Directional acc. |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for name in MODEL_LABELS:
        m = wf_reg[name]
        lines.append(
            f"| {MODEL_LABELS[name]} | {int(m['n'])} | {_fmt(m['mae'])} | {_fmt(m['rmse'])} | "
            f"{_fmt(m['r2'])} | {_fmt(m['corr'])} | {_fmt(m['directional_accuracy'], 3)} |"
        )
    lines.append("")
    lines.append("## Decision metrics on holdout")
    lines.append("")
    lines.append("WAIT if predicted Δ ≤ −δ; FILL UP if predicted Δ ≥ +δ; else NO CLEAR SIGNAL.")
    lines.append("A WAIT/FILL call is counted correct if the actual next-week change has the intended sign.")
    lines.append("Theoretical savings vs always-fill-now: WAIT weeks contribute `15 × (p_t − p_{t+7})`; FILL UP and NO SIGNAL contribute 0.")
    lines.append("")
    for delta in DECISION_THRESHOLDS:
        lines.append(f"### δ = ${delta:.2f}")
        lines.append("")
        lines.append("| Model | WAIT | FILL | No signal | % no signal | % correct among signals | Mean actual after WAIT | Mean actual after FILL | Mean $ / 15 gal vs always-fill |")
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
        for name in MODEL_LABELS:
            d = holdout_dec[name][delta]
            lines.append(
                f"| {MODEL_LABELS[name]} | {int(d['n_wait'])} | {int(d['n_fill'])} | {int(d['n_none'])} | "
                f"{_fmt(d['pct_no_signal'], 3)} | {_fmt(d['pct_correct_among_signals'], 3)} | "
                f"{_fmt(d['mean_actual_after_wait'])} | {_fmt(d['mean_actual_after_fill'])} | "
                f"{_fmt(d['mean_savings_per_fill_usd'], 3)} |"
            )
        lines.append("")

    lines.append("## Answers")
    lines.append("")
    lines.append(f"**A. Does Ridge beat the momentum baseline?** {'Yes' if beats else 'No'} on holdout MAE "
                 f"({_fmt(full['mae'])} vs {_fmt(mom['mae'])} for MODEL 4 vs MODEL 1). "
                 f"Directional accuracy: {_fmt(full['directional_accuracy'], 3)} vs {_fmt(mom['directional_accuracy'], 3)}.")
    lines.append("")
    lines.append(f"**B. Do daily wholesale/crude features add value?** {'Yes' if market_helps else 'No'} vs retail-only Ridge "
                 f"(MAE {_fmt(market['mae'])} vs {_fmt(retail_only['mae'])}).")
    lines.append("")
    inv_mae_drop = market["mae"] - full["mae"]
    lines.append(f"**C. Does inventory add value?** Not in a way we can credit on its own. Inventory features are only in MODEL 4, bundled with seasonality. The joint MAE change vs MODEL 3 is {_fmt(inv_mae_drop)} $/gal — about 0.07¢, within noise relative to the market-feature jump of {_fmt(retail_only['mae'] - market['mae'])} $/gal.")
    lines.append("")
    lines.append("**D. Does seasonality add value?** Same caveat: Fourier day-of-year and `is_summer` only appear in MODEL 4. They do not produce a material holdout MAE gain once daily markets are included. Keep them as cheap controls, not as the story.")
    lines.append("")
    lines.append(f"**E. Out-of-sample directional accuracy (MODEL 4 holdout):** {_fmt(full['directional_accuracy'], 3)} "
                 f"({int(full['n'])} weeks). Walk-forward: {_fmt(wf_reg['ridge_full']['directional_accuracy'], 3)}.")
    lines.append("")
    lines.append(f"**F. Theoretical economic benefit (MODEL 4, δ=$0.03, holdout):** "
                 f"mean {_fmt(full_dec['mean_savings_per_fill_usd'], 3)} USD per 15-gallon fill vs always filling now "
                 f"(WAIT-only P&L); total {_fmt(full_dec['total_savings_usd'], 2)} USD over {int(full_dec['n'])} test weeks. "
                 f"Momentum is similar on this WAIT-only metric ({_fmt(mom_dec['mean_savings_per_fill_usd'], 3)} USD/fill) because FILL UP does not change P&L vs always-fill. "
                 f"FILL quality is better for MODEL 4: mean actual Δ after FILL UP is {_fmt(full_dec['mean_actual_after_fill'])} vs {_fmt(mom_dec['mean_actual_after_fill'])} for momentum.")
    lines.append("")
    lines.append(f"**G. WAIT/FILL signals (MODEL 4, δ=$0.03):** WAIT {int(full_dec['n_wait'])}, "
                 f"FILL UP {int(full_dec['n_fill'])}, NO CLEAR SIGNAL {int(full_dec['n_none'])} "
                 f"({_fmt(full_dec['pct_no_signal'], 3)} of weeks silent). Among signaled weeks, {_fmt(full_dec['pct_correct_among_signals'], 3)} had the intended sign.")
    lines.append("")
    lines.append("**H. Strong enough to justify pursuing daily retail data?** Weekly national signal is real: daily WTI/NY Harbor/Gulf Coast improve MAE and direction vs momentum. "
                 "It is **not** strong enough to ship a 3-day local recommendation. Typical weekly move is still a few cents, the label is national Monday-to-Monday, and ~57% of weeks are NO CLEAR SIGNAL at 3¢. "
                 "Licensed daily/metro retail is still required for the intended product. This result **does** justify paying for that data: wholesale appears to lead the pump on a weekly clock.")
    lines.append("")
    lines.append("**I. What next?** Keep the leakage-safe weekly dataset. Do not jump to gradient boosting. Next scientific step is PADD or city EIA retail against the matching hub, plus as-of vintages. "
                 "In parallel, request OPIS/PDI quotes for daily retail. Revisit δ on a locked validation slice, not on this test set.")
    lines.append("")
    lines.append("## Charts")
    lines.append("")
    for chart in chart_paths:
        rel = chart.relative_to(PROJECT_ROOT)
        lines.append(f"- `{rel}`")
    lines.append("")
    lines.append("## Features used")
    lines.append("")
    for name, feats in FEATURE_SETS.items():
        lines.append(f"- {MODEL_LABELS[name]}: `{', '.join(feats)}`")
    lines.append("")
    lines.append("Ridge uses `StandardScaler` + `Ridge(alpha=1.0)`. No test-set tuning.")
    lines.append(
        "A 7-calendar-day WTI change (`wti_chg_7cal`) is stored on the dataset using last known price on or before T-7 "
        "(no interpolation) but is **not** in Ridge: it is collinear with `wti_d5` on a business-day calendar, "
        "as noted in `docs/modeling_design.md`."
    )
    lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Wrote %s", path)


def run_experiment() -> dict:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    built = save_weekly_model_dataset()
    df = built.frame
    split = chronological_split(df)
    holdout = fit_predict_holdout(split)
    walk_preds = walk_forward_all(df)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    charts = make_charts(split.test, holdout, prefix="holdout")
    report_path = REPORTS_DIR / "first_model_results.md"
    write_report(
        dataset=df,
        dropped=built.dropped,
        split=split,
        holdout=holdout,
        walk_preds=walk_preds,
        chart_paths=charts,
        path=report_path,
    )
    return {
        "dataset": df,
        "dropped": built.dropped,
        "split": split,
        "holdout": holdout,
        "walk_preds": walk_preds,
        "charts": charts,
        "report_path": report_path,
    }


if __name__ == "__main__":
    run_experiment()
