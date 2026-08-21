"""Execute Gas Wait EDA for notebook 01_data_exploration.

This script is the analysis engine used to run the notebook cells
successfully without adding Jupyter as a project dependency.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
FIGURES = Path(__file__).resolve().parent / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)
PROCESSED.mkdir(parents=True, exist_ok=True)

plt.rcParams.update(
    {
        "figure.figsize": (11, 4.8),
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 10,
    }
)

CHANGE_HORIZONS = (1, 2, 4, 8)
LAGS = (0, 1, 2, 3, 4, 6, 8)
FUTURE_HORIZONS = (1, 2, 3)

HISTORICAL_EVENTS = [
    ("1990-08-02", "1991-02-28", "Gulf War / Kuwait invasion"),
    ("2005-08-25", "2005-10-15", "Hurricane Katrina"),
    ("2008-06-01", "2009-03-31", "2008 oil spike and financial crisis"),
    ("2011-01-15", "2011-06-30", "Arab Spring / Libya disruption"),
    ("2014-06-01", "2016-02-29", "2014-16 oil glut and price collapse"),
    ("2020-03-01", "2020-05-31", "COVID-19 demand collapse"),
    ("2022-02-24", "2022-07-31", "Russia-Ukraine war price spike"),
]


def load_series(filename: str, value_name: str) -> pd.DataFrame:
    df = pd.read_csv(PROCESSED / filename, parse_dates=["observation_date"])
    df = df[["observation_date", "value", "units", "series-description", "frequency", "geographic_area"]].copy()
    df = df.sort_values("observation_date").reset_index(drop=True)
    df = df.rename(columns={"value": value_name})
    df["weekday"] = df["observation_date"].dt.day_name()
    df["gap_days"] = df["observation_date"].diff().dt.days
    return df


def coverage_row(name: str, df: pd.DataFrame, value_col: str) -> dict:
    dates = df["observation_date"]
    gaps = df["gap_days"].dropna()
    return {
        "dataset": name,
        "observations": int(len(df)),
        "non_null_values": int(df[value_col].notna().sum()),
        "missing_values": int(df[value_col].isna().sum()),
        "start": dates.min().date().isoformat(),
        "end": dates.max().date().isoformat(),
        "frequency_claimed": df["frequency"].iloc[0],
        "weekday": ", ".join(f"{k} ({v})" for k, v in df["weekday"].value_counts().items()),
        "median_gap_days": float(gaps.median()) if len(gaps) else None,
        "max_gap_days": float(gaps.max()) if len(gaps) else None,
        "irregular_gaps": int((gaps != 7).sum()) if len(gaps) else 0,
        "units_in_file": df["units"].iloc[0],
        "series_description": df["series-description"].iloc[0],
        "geographic_area": df["geographic_area"].iloc[0],
        "min": float(df[value_col].min(skipna=True)),
        "median": float(df[value_col].median(skipna=True)),
        "max": float(df[value_col].max(skipna=True)),
    }


def corr_with_p(x: pd.Series, y: pd.Series) -> dict:
    paired = pd.concat([x, y], axis=1).dropna()
    if len(paired) < 10:
        return {"n": int(len(paired)), "pearson": np.nan, "pearson_p": np.nan, "spearman": np.nan, "spearman_p": np.nan}
    pearson = stats.pearsonr(paired.iloc[:, 0], paired.iloc[:, 1])
    spearman = stats.spearmanr(paired.iloc[:, 0], paired.iloc[:, 1])
    return {
        "n": int(len(paired)),
        "pearson": float(pearson.statistic),
        "pearson_p": float(pearson.pvalue),
        "spearman": float(spearman.statistic),
        "spearman_p": float(spearman.pvalue),
    }


def annotate_event(date: pd.Timestamp) -> str:
    labels = []
    for start, end, label in HISTORICAL_EVENTS:
        if pd.Timestamp(start) <= date <= pd.Timestamp(end):
            labels.append(label)
    return "; ".join(labels) if labels else ""


def savefig(name: str) -> Path:
    path = FIGURES / name
    plt.tight_layout()
    plt.savefig(path, dpi=140, bbox_inches="tight")
    plt.close()
    return path


def run() -> dict:
    retail = load_series("retail_gasoline_regular_us_weekly.csv", "retail_price")
    wti = load_series("wti_crude_spot_us_weekly.csv", "wti_price")
    stocks = load_series("gasoline_inventories_us_weekly.csv", "gasoline_stocks")

    coverage = pd.DataFrame(
        [
            coverage_row("U.S. regular gasoline retail price", retail, "retail_price"),
            coverage_row("WTI crude oil spot price", wti, "wti_price"),
            coverage_row("U.S. total gasoline stocks", stocks, "gasoline_stocks"),
        ]
    )
    coverage.to_csv(PROCESSED / "eda_coverage_summary.csv", index=False)

    missing = retail.loc[retail["retail_price"].isna(), ["observation_date"]].copy()
    missing["dataset"] = "U.S. regular gasoline retail price"
    missing["issue"] = "Null retail price; EIA paused weekly retail gasoline collection"
    missing.to_csv(PROCESSED / "eda_missing_observations.csv", index=False)

    overlap_exact = {
        "retail_and_wti_exact_dates": int(len(set(retail.observation_date) & set(wti.observation_date))),
        "retail_and_stocks_exact_dates": int(len(set(retail.observation_date) & set(stocks.observation_date))),
        "wti_and_stocks_exact_dates": int(len(set(wti.observation_date) & set(stocks.observation_date))),
    }

    wti_for_join = wti.rename(columns={"observation_date": "wti_date"})[["wti_date", "wti_price"]]
    stocks_for_join = stocks.rename(columns={"observation_date": "stocks_date"})[["stocks_date", "gasoline_stocks"]]
    panel = pd.merge_asof(
        retail.sort_values("observation_date"),
        wti_for_join.sort_values("wti_date"),
        left_on="observation_date",
        right_on="wti_date",
        direction="backward",
        tolerance=pd.Timedelta("6 days"),
    )
    panel = pd.merge_asof(
        panel.sort_values("observation_date"),
        stocks_for_join.sort_values("stocks_date"),
        left_on="observation_date",
        right_on="stocks_date",
        direction="backward",
        tolerance=pd.Timedelta("6 days"),
    )
    panel["days_since_wti"] = (panel["observation_date"] - panel["wti_date"]).dt.days
    panel["days_since_stocks"] = (panel["observation_date"] - panel["stocks_date"]).dt.days
    panel["gasoline_stocks_mmbbl"] = panel["gasoline_stocks"] / 1000.0
    panel = panel.dropna(subset=["retail_price"]).copy()

    alignment = pd.DataFrame(
        [
            {
                "item": "retail weekday",
                "value": "Monday (EIA Gasoline and Diesel Fuel Update)",
            },
            {
                "item": "wti weekday",
                "value": "Friday (weekly WTI series ending Friday)",
            },
            {
                "item": "stocks weekday",
                "value": "Friday (Weekly Petroleum Status Report week ending)",
            },
            {
                "item": "exact calendar-date overlap, retail vs WTI",
                "value": overlap_exact["retail_and_wti_exact_dates"],
            },
            {
                "item": "exact calendar-date overlap, retail vs stocks",
                "value": overlap_exact["retail_and_stocks_exact_dates"],
            },
            {
                "item": "join method",
                "value": "merge_asof backward, tolerance 6 days (no future Friday leaked onto Monday)",
            },
            {
                "item": "typical days WTI leads retail observation",
                "value": float(panel["days_since_wti"].median()),
            },
            {
                "item": "typical days stocks lead retail observation",
                "value": float(panel["days_since_stocks"].median()),
            },
            {
                "item": "aligned rows with retail, WTI, and stocks",
                "value": int(panel[["retail_price", "wti_price", "gasoline_stocks"]].dropna().shape[0]),
            },
            {
                "item": "stocks units correction",
                "value": "File metadata says million barrels; EIA series description is thousand barrels. Analysis uses thousand barrels and a mmbbl companion column.",
            },
        ]
    )
    alignment.to_csv(PROCESSED / "eda_alignment_summary.csv", index=False)

    for h in CHANGE_HORIZONS:
        panel[f"retail_chg_{h}w"] = panel["retail_price"].diff(h)
        panel[f"retail_pct_{h}w"] = panel["retail_price"].pct_change(h)
        panel[f"wti_chg_{h}w"] = panel["wti_price"].diff(h)
        panel[f"wti_pct_{h}w"] = panel["wti_price"].pct_change(h)
        panel[f"stocks_chg_{h}w"] = panel["gasoline_stocks"].diff(h)
        panel[f"stocks_pct_{h}w"] = panel["gasoline_stocks"].pct_change(h)

    for h in FUTURE_HORIZONS:
        panel[f"retail_fwd_chg_{h}w"] = panel["retail_price"].shift(-h) - panel["retail_price"]
        panel[f"retail_fwd_pct_{h}w"] = panel["retail_price"].shift(-h) / panel["retail_price"] - 1
        panel[f"retail_fwd_down_{h}w"] = (panel["retail_price"].shift(-h) < panel["retail_price"]).astype(float)

    future_prices = pd.concat(
        [panel["retail_price"].shift(-h) for h in FUTURE_HORIZONS],
        axis=1,
    )
    panel["retail_min_next_1to3w"] = future_prices.min(axis=1)
    panel["wait_1to3w_saves"] = panel["retail_min_next_1to3w"] < panel["retail_price"]
    panel["wait_1to3w_savings"] = panel["retail_price"] - panel["retail_min_next_1to3w"]
    panel["wait_1w_saves"] = panel["retail_price"].shift(-1) < panel["retail_price"]
    panel["wait_1w_savings"] = panel["retail_price"] - panel["retail_price"].shift(-1)

    panel["month"] = panel["observation_date"].dt.month
    panel["quarter"] = panel["observation_date"].dt.quarter
    panel["year"] = panel["observation_date"].dt.year
    panel["retail_yoy_pct"] = panel["retail_price"].pct_change(52)
    month_stock_median = panel.groupby("month")["gasoline_stocks"].transform("median")
    panel["stocks_seasonal_dev"] = panel["gasoline_stocks"] - month_stock_median

    change_rows = []
    for series, prefix in [
        ("retail gasoline", "retail"),
        ("WTI", "wti"),
        ("gasoline stocks", "stocks"),
    ]:
        for h in CHANGE_HORIZONS:
            s = panel[f"{prefix}_chg_{h}w"]
            p = panel[f"{prefix}_pct_{h}w"]
            change_rows.append(
                {
                    "series": series,
                    "horizon_weeks": h,
                    "n": int(s.notna().sum()),
                    "mean_change": float(s.mean()),
                    "median_change": float(s.median()),
                    "std_change": float(s.std()),
                    "mean_pct": float(p.mean()),
                    "median_pct": float(p.median()),
                    "std_pct": float(p.std()),
                    "pct_negative": float((s < 0).mean()),
                    "pct_positive": float((s > 0).mean()),
                }
            )
    change_stats = pd.DataFrame(change_rows)
    change_stats.to_csv(PROCESSED / "eda_change_stats.csv", index=False)

    corr_rows = []
    pairs = [
        ("retail_price", "wti_price", "retail level vs WTI level"),
        ("retail_price", "gasoline_stocks", "retail level vs inventory level"),
        ("retail_chg_1w", "wti_chg_1w", "1-week retail change vs 1-week WTI change"),
        ("retail_pct_1w", "wti_pct_1w", "1-week retail pct vs 1-week WTI pct"),
        ("retail_chg_2w", "wti_chg_2w", "2-week retail change vs 2-week WTI change"),
        ("retail_pct_2w", "wti_pct_2w", "2-week retail pct vs 2-week WTI pct"),
        ("retail_chg_4w", "wti_chg_4w", "4-week retail change vs 4-week WTI change"),
        ("retail_pct_4w", "wti_pct_4w", "4-week retail pct vs 4-week WTI pct"),
        ("retail_chg_8w", "wti_chg_8w", "8-week retail change vs 8-week WTI change"),
        ("retail_pct_8w", "wti_pct_8w", "8-week retail pct vs 8-week WTI pct"),
        ("retail_chg_1w", "stocks_chg_1w", "1-week retail change vs 1-week inventory change"),
        ("retail_pct_1w", "stocks_pct_1w", "1-week retail pct vs 1-week inventory pct"),
        ("retail_chg_4w", "stocks_chg_4w", "4-week retail change vs 4-week inventory change"),
        ("retail_pct_4w", "stocks_pct_4w", "4-week retail pct vs 4-week inventory pct"),
        ("retail_chg_4w", "stocks_seasonal_dev", "4-week retail change vs seasonal inventory deviation"),
        ("retail_chg_1w", "retail_chg_1w", "retail 1-week change autocorrelation placeholder"),
    ]
    for x, y, label in pairs:
        if x == y:
            result = corr_with_p(panel[x], panel[x].shift(1))
            label = "retail 1-week change autocorrelation (lag 1)"
        else:
            result = corr_with_p(panel[x], panel[y])
        corr_rows.append({"relationship": label, "x": x, "y": y, **result})
    correlations = pd.DataFrame(corr_rows)
    correlations.to_csv(PROCESSED / "eda_correlations.csv", index=False)

    lag_rows = []
    for predictor, pred_label in [
        ("wti_chg_1w", "WTI 1-week change"),
        ("wti_pct_1w", "WTI 1-week pct change"),
        ("stocks_chg_1w", "inventory 1-week change"),
        ("stocks_pct_1w", "inventory 1-week pct change"),
        ("stocks_seasonal_dev", "inventory seasonal deviation"),
        ("retail_chg_1w", "retail 1-week change (momentum)"),
    ]:
        for lag in LAGS:
            result = corr_with_p(panel[predictor].shift(lag), panel["retail_chg_1w"])
            lag_rows.append(
                {
                    "predictor": pred_label,
                    "predictor_col": predictor,
                    "lag_weeks": lag,
                    "target": "retail 1-week change at t",
                    "interpretation": "predictor at t-lag vs retail change ending at t",
                    **result,
                }
            )
            for h in FUTURE_HORIZONS:
                result_fwd = corr_with_p(panel[predictor].shift(lag), panel[f"retail_fwd_chg_{h}w"])
                lag_rows.append(
                    {
                        "predictor": pred_label,
                        "predictor_col": predictor,
                        "lag_weeks": lag,
                        "target": f"retail change {h} week(s) ahead",
                        "interpretation": "predictor at t-lag vs future retail change; lag 0 is a valid forecast feature",
                        **result_fwd,
                    }
                )
    lagged = pd.DataFrame(lag_rows)
    lagged.to_csv(PROCESSED / "eda_lagged_correlations.csv", index=False)

    seasonal_month = (
        panel.groupby("month")
        .agg(
            retail_mean=("retail_price", "mean"),
            retail_median=("retail_price", "median"),
            retail_1w_mean=("retail_chg_1w", "mean"),
            retail_1w_pct_negative=("retail_chg_1w", lambda s: float((s < 0).mean())),
            stocks_mean=("gasoline_stocks", "mean"),
            n=("retail_price", "size"),
        )
        .reset_index()
    )
    seasonal_month.to_csv(PROCESSED / "eda_seasonality_monthly.csv", index=False)

    seasonal_quarter = (
        panel.groupby("quarter")
        .agg(
            retail_mean=("retail_price", "mean"),
            retail_1w_mean=("retail_chg_1w", "mean"),
            retail_1w_pct_negative=("retail_chg_1w", lambda s: float((s < 0).mean())),
            n=("retail_price", "size"),
        )
        .reset_index()
    )
    seasonal_quarter.to_csv(PROCESSED / "eda_seasonality_quarterly.csv", index=False)

    z = panel["retail_pct_1w"].abs()
    threshold = z.quantile(0.99)
    unusual = panel.loc[z >= threshold, ["observation_date", "retail_price", "retail_chg_1w", "retail_pct_1w", "wti_price", "wti_pct_1w", "gasoline_stocks"]].copy()
    unusual["historical_event"] = unusual["observation_date"].map(annotate_event)
    unusual = unusual.sort_values("retail_pct_1w", key=lambda s: s.abs(), ascending=False)
    unusual.to_csv(PROCESSED / "eda_unusual_movements.csv", index=False)

    def directional_accuracy(pred: pd.Series, actual: pd.Series) -> float:
        paired = pd.concat([pred, actual], axis=1).dropna()
        if paired.empty:
            return np.nan
        return float(np.sign(paired.iloc[:, 0]).eq(np.sign(paired.iloc[:, 1])).mean())

    target_rows = []
    for h in FUTURE_HORIZONS:
        actual = panel[f"retail_fwd_chg_{h}w"]
        down = panel[f"retail_fwd_down_{h}w"]
        target_rows.append(
            {
                "target": f"TARGET {'ABC'[h-1]}: retail change {h} week(s) ahead",
                "n": int(actual.notna().sum()),
                "mean_change_usd": float(actual.mean()),
                "median_change_usd": float(actual.median()),
                "std_change_usd": float(actual.std()),
                "mean_abs_change_usd": float(actual.abs().mean()),
                "share_price_lower": float(down.mean()),
                "share_save_at_least_5c": float((actual <= -0.05).mean()),
                "share_save_at_least_10c": float((actual <= -0.10).mean()),
                "corr_wti_chg_1w": corr_with_p(panel["wti_chg_1w"], actual)["pearson"],
                "corr_wti_pct_1w": corr_with_p(panel["wti_pct_1w"], actual)["pearson"],
                "corr_stocks_chg_1w": corr_with_p(panel["stocks_chg_1w"], actual)["pearson"],
                "corr_retail_momentum": corr_with_p(panel["retail_chg_1w"], actual)["pearson"],
                "sign_acc_wti_chg": directional_accuracy(panel["wti_chg_1w"], actual),
                "sign_acc_momentum": directional_accuracy(panel["retail_chg_1w"], actual),
                "majority_baseline": float(max(down.mean(), 1 - down.mean())),
            }
        )

    down_1to3 = panel["wait_1to3w_saves"].astype(float)
    savings = panel["wait_1to3w_savings"]
    target_rows.append(
        {
            "target": "TARGET D: cheaper fill exists within 1-3 weeks",
            "n": int(down_1to3.notna().sum()),
            "mean_change_usd": float((-savings).mean()),
            "median_change_usd": float((-savings).median()),
            "std_change_usd": float(savings.std()),
            "mean_abs_change_usd": float(savings.mean()),
            "share_price_lower": float(down_1to3.mean()),
            "share_save_at_least_5c": float((savings >= 0.05).mean()),
            "share_save_at_least_10c": float((savings >= 0.10).mean()),
            "corr_wti_chg_1w": corr_with_p(panel["wti_chg_1w"], -savings)["pearson"],
            "corr_wti_pct_1w": corr_with_p(panel["wti_pct_1w"], -savings)["pearson"],
            "corr_stocks_chg_1w": corr_with_p(panel["stocks_chg_1w"], -savings)["pearson"],
            "corr_retail_momentum": corr_with_p(panel["retail_chg_1w"], -savings)["pearson"],
            "sign_acc_wti_chg": directional_accuracy(panel["wti_chg_1w"], panel["retail_fwd_chg_1w"]),
            "sign_acc_momentum": directional_accuracy(panel["retail_chg_1w"], panel["retail_fwd_chg_1w"]),
            "majority_baseline": float(max(down_1to3.mean(), 1 - down_1to3.mean())),
        }
    )
    wait1 = panel["wait_1w_savings"]
    target_rows.append(
        {
            "target": "TARGET E: dollars saved by waiting 1 week",
            "n": int(wait1.notna().sum()),
            "mean_change_usd": float((-wait1).mean()),
            "median_change_usd": float((-wait1).median()),
            "std_change_usd": float(wait1.std()),
            "mean_abs_change_usd": float(wait1.abs().mean()),
            "share_price_lower": float(panel["wait_1w_saves"].mean()),
            "share_save_at_least_5c": float((wait1 >= 0.05).mean()),
            "share_save_at_least_10c": float((wait1 >= 0.10).mean()),
            "corr_wti_chg_1w": corr_with_p(panel["wti_chg_1w"], -wait1)["pearson"],
            "corr_wti_pct_1w": corr_with_p(panel["wti_pct_1w"], -wait1)["pearson"],
            "corr_stocks_chg_1w": corr_with_p(panel["stocks_chg_1w"], -wait1)["pearson"],
            "corr_retail_momentum": corr_with_p(panel["retail_chg_1w"], -wait1)["pearson"],
            "sign_acc_wti_chg": directional_accuracy(-panel["wti_chg_1w"], wait1),
            "sign_acc_momentum": directional_accuracy(-panel["retail_chg_1w"], wait1),
            "majority_baseline": float(max(panel["wait_1w_saves"].mean(), 1 - panel["wait_1w_saves"].mean())),
        }
    )
    targets = pd.DataFrame(target_rows)
    targets.to_csv(PROCESSED / "eda_target_diagnostics.csv", index=False)

    keep_cols = [
        "observation_date",
        "retail_price",
        "wti_date",
        "wti_price",
        "stocks_date",
        "gasoline_stocks",
        "gasoline_stocks_mmbbl",
        "days_since_wti",
        "days_since_stocks",
        "month",
        "quarter",
        "year",
        "stocks_seasonal_dev",
        "retail_yoy_pct",
        "wait_1w_saves",
        "wait_1w_savings",
        "wait_1to3w_saves",
        "wait_1to3w_savings",
    ]
    change_cols = [c for c in panel.columns if c.startswith(("retail_chg_", "retail_pct_", "wti_chg_", "wti_pct_", "stocks_chg_", "stocks_pct_", "retail_fwd_"))]
    panel.loc[:, keep_cols + change_cols].to_csv(PROCESSED / "eda_weekly_panel.csv", index=False)

    fig, ax = plt.subplots()
    ax.plot(panel["observation_date"], panel["retail_price"], color="#1f4e79", lw=1.2)
    ax.set_title("U.S. Regular Gasoline Retail Price")
    ax.set_xlabel("Observation date (Monday)")
    ax.set_ylabel("Dollars per gallon")
    ax.text(0.0, -0.22, "Source: EIA API v2 petroleum/pri/gnd · 1990-08-20 to 2026-08-17", transform=ax.transAxes, fontsize=8)
    savefig("retail_price_over_time.png")

    fig, ax = plt.subplots()
    ax.plot(wti["observation_date"], wti["wti_price"], color="#8b2e2e", lw=1.2)
    ax.set_title("WTI Crude Oil Spot Price")
    ax.set_xlabel("Observation date (Friday)")
    ax.set_ylabel("Dollars per barrel")
    ax.text(0.0, -0.22, "Source: EIA API v2 petroleum/pri/spt · 1986-01-03 to 2026-08-14", transform=ax.transAxes, fontsize=8)
    savefig("wti_price_over_time.png")

    fig, ax = plt.subplots()
    ax.plot(stocks["observation_date"], stocks["gasoline_stocks"] / 1000.0, color="#2f6b3c", lw=1.2)
    ax.set_title("U.S. Total Gasoline Stocks")
    ax.set_xlabel("Observation date (Friday)")
    ax.set_ylabel("Million barrels")
    ax.text(0.0, -0.22, "Source: EIA API v2 petroleum/stoc/wstk · values stored as thousand barrels", transform=ax.transAxes, fontsize=8)
    savefig("gasoline_stocks_over_time.png")

    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.plot(panel["observation_date"], panel["retail_price"], color="#1f4e79", lw=1.2, label="Retail gasoline")
    ax2.plot(panel["observation_date"], panel["wti_price"], color="#8b2e2e", lw=1.0, alpha=0.85, label="WTI")
    ax1.set_title("Retail Gasoline vs WTI")
    ax1.set_xlabel("Retail observation date (Monday)")
    ax1.set_ylabel("Retail price (dollars per gallon)")
    ax2.set_ylabel("WTI (dollars per barrel)")
    ax2.spines.top.set_visible(False)
    lines = ax1.get_lines() + ax2.get_lines()
    ax1.legend(lines, [line.get_label() for line in lines], loc="upper left")
    ax1.text(0.0, -0.22, "WTI is as-of joined from the Friday on or before each Monday retail date", transform=ax1.transAxes, fontsize=8)
    savefig("retail_vs_wti_levels.png")

    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    axes[0].plot(panel["observation_date"], panel["retail_chg_4w"], color="#1f4e79", lw=0.9)
    axes[0].axhline(0, color="black", lw=0.6)
    axes[0].set_ylabel("Retail 4-week change ($/gal)")
    axes[0].set_title("Rolling 4-week changes")
    axes[1].plot(panel["observation_date"], panel["wti_chg_4w"], color="#8b2e2e", lw=0.9)
    axes[1].axhline(0, color="black", lw=0.6)
    axes[1].set_ylabel("WTI 4-week change ($/bbl)")
    axes[2].plot(panel["observation_date"], panel["stocks_chg_4w"] / 1000.0, color="#2f6b3c", lw=0.9)
    axes[2].axhline(0, color="black", lw=0.6)
    axes[2].set_ylabel("Stocks 4-week change (mmbbl)")
    axes[2].set_xlabel("Date")
    savefig("rolling_4w_changes.png")

    lag_plot = lagged[
        (lagged["predictor"].isin(["WTI 1-week change", "inventory 1-week change", "retail 1-week change (momentum)"]))
        & (lagged["target"] == "retail 1-week change at t")
    ]
    fig, ax = plt.subplots()
    for pred, color in [
        ("WTI 1-week change", "#8b2e2e"),
        ("inventory 1-week change", "#2f6b3c"),
        ("retail 1-week change (momentum)", "#1f4e79"),
    ]:
        sub = lag_plot[lag_plot["predictor"] == pred]
        ax.plot(sub["lag_weeks"], sub["pearson"], marker="o", label=pred, color=color)
    ax.axhline(0, color="black", lw=0.6)
    ax.set_title("Lagged correlation with contemporaneous 1-week retail price change")
    ax.set_xlabel("Lag in weeks (positive = predictor precedes retail change)")
    ax.set_ylabel("Pearson correlation")
    ax.legend()
    savefig("lagged_correlations.png")

    fig, ax = plt.subplots()
    ax.boxplot(
        [panel.loc[panel["month"] == m, "retail_price"].dropna() for m in range(1, 13)],
        tick_labels=list("JFMAMJJASOND"),
    )
    ax.set_title("Seasonal pattern in retail gasoline prices")
    ax.set_xlabel("Month")
    ax.set_ylabel("Dollars per gallon")
    ax.text(0.0, -0.22, "Boxes pool all years. Levels mix trend and seasonality; use monthly mean changes for the seasonal cycle.", transform=ax.transAxes, fontsize=8)
    savefig("seasonal_retail_boxplot.png")

    fig, ax = plt.subplots()
    ax.bar(seasonal_month["month"], seasonal_month["retail_1w_mean"], color="#1f4e79")
    ax.axhline(0, color="black", lw=0.6)
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(list("JFMAMJJASOND"))
    ax.set_title("Average 1-week retail gasoline price change by month")
    ax.set_xlabel("Month")
    ax.set_ylabel("Mean weekly change (dollars per gallon)")
    savefig("seasonal_weekly_change_by_month.png")

    forecast_lag = lagged[
        (lagged["predictor"] == "WTI 1-week change")
        & (lagged["lag_weeks"] == 0)
        & (lagged["target"].str.contains("ahead"))
    ]

    best_lead = (
        lagged[
            (lagged["predictor"] == "WTI 1-week change")
            & (lagged["target"] == "retail 1-week change at t")
        ]
        .assign(abs_corr=lambda d: d["pearson"].abs())
        .sort_values("abs_corr", ascending=False)
        .iloc[0]
    )

    findings = {
        "coverage": coverage.to_dict(orient="records"),
        "missing_retail_dates": missing["observation_date"].dt.date.astype(str).tolist(),
        "overlap_exact": overlap_exact,
        "aligned_rows": int(len(panel)),
        "typical_wti_lead_days": float(panel["days_since_wti"].median()),
        "best_wti_lag_for_same_week_retail_change": {
            "lag_weeks": int(best_lead["lag_weeks"]),
            "pearson": float(best_lead["pearson"]),
        },
        "level_corr_retail_wti": float(correlations.loc[correlations["x"].eq("retail_price") & correlations["y"].eq("wti_price"), "pearson"].iloc[0]),
        "chg1_corr_retail_wti": float(correlations.loc[correlations["relationship"].eq("1-week retail change vs 1-week WTI change"), "pearson"].iloc[0]),
        "chg1_corr_retail_stocks": float(correlations.loc[correlations["relationship"].eq("1-week retail change vs 1-week inventory change"), "pearson"].iloc[0]),
        "retail_ac1": float(correlations.loc[correlations["relationship"].str.contains("autocorrelation"), "pearson"].iloc[0]),
        "forecast_corr_wti_vs_future": forecast_lag[["target", "pearson", "n"]].to_dict(orient="records"),
        "target_diagnostics": targets.to_dict(orient="records"),
        "unusual_count": int(len(unusual)),
        "unusual_with_known_event": int((unusual["historical_event"] != "").sum()),
        "share_wait_1w_saves": float(panel["wait_1w_saves"].mean()),
        "share_wait_1to3w_saves": float(panel["wait_1to3w_saves"].mean()),
        "mean_abs_1w_retail_change": float(panel["retail_chg_1w"].abs().mean()),
    }
    with open(PROCESSED / "eda_findings.json", "w", encoding="utf-8") as handle:
        json.dump(findings, handle, indent=2)

    print("EDA complete")
    print(json.dumps({k: findings[k] for k in ["aligned_rows", "typical_wti_lead_days", "level_corr_retail_wti", "chg1_corr_retail_wti", "chg1_corr_retail_stocks", "retail_ac1", "share_wait_1w_saves", "mean_abs_1w_retail_change"]}, indent=2))
    print(targets[["target", "mean_abs_change_usd", "share_price_lower", "corr_wti_chg_1w", "corr_retail_momentum", "sign_acc_wti_chg", "majority_baseline"]].to_string(index=False))
    print("\nWTI lag table vs retail 1w change:")
    print(
        lagged[
            (lagged["predictor"] == "WTI 1-week change") & (lagged["target"] == "retail 1-week change at t")
        ][["lag_weeks", "pearson", "pearson_p", "n"]].to_string(index=False)
    )
    print("\nInventory lag table vs retail 1w change:")
    print(
        lagged[
            (lagged["predictor"] == "inventory 1-week change") & (lagged["target"] == "retail 1-week change at t")
        ][["lag_weeks", "pearson", "pearson_p", "n"]].to_string(index=False)
    )
    print("\nForecast: WTI chg at t vs future retail")
    print(forecast_lag[["target", "pearson", "pearson_p", "n"]].to_string(index=False))
    return findings


if __name__ == "__main__":
    run()
