"""Production inference for the personal weekly Gas Wait MVP.

Uses the frozen exp01 ``ridge_full`` specification (Ridge alpha=1.0, StandardScaler,
``ALL_MODEL_FEATURES``). Refits on historical rows with known targets each run —
same specification as exp01 MODEL 4, no hyperparameter search.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from modeling.config import DEFAULT_SIGNAL_THRESHOLD
from modeling.signals import SignalResult, generate_signal
from modeling.train_baseline import FEATURE_SETS, RIDGE_ALPHA
from modeling.weekly_pipeline import PROCESSED_DIR, build_weekly_eia_dataset

EASTERN = ZoneInfo("America/New_York")
MODEL_NAME = "Ridge"
MODEL_ID = "ridge_full"
MODEL_VERSION = f"ridge_full_alpha{RIDGE_ALPHA}"
FEATURES = FEATURE_SETS["ridge_full"]

# Documented freshness rules (see docs/personal_mvp.md).
# Hard fail when retail Monday is older than this many calendar days.
RETAIL_MAX_AGE_DAYS = 8
# Hard fail when retail is more than this many days behind the expected Monday print.
RETAIL_MISSING_WEEK_DAYS = 7

GROUP_FEATURE_MAP: dict[str, tuple[str, ...]] = {
    "Retail momentum": ("retail_d7", "retail_d14"),
    "WTI": ("wti_d1", "wti_d3", "wti_d5", "wti_vol_20"),
    "Wholesale gasoline": ("nyh_d1", "nyh_d5", "nyh_vol_20", "gc_d1", "gc_d5"),
    "Crack spread": ("crack_nyh_d5",),
    "Inventory": ("inv_wow", "inv_seasonal_z"),
    "Seasonality": ("sin_doy", "cos_doy", "is_summer"),
}

GROUP_VALUE_HINTS: dict[str, tuple[tuple[str, str, str], ...]] = {
    "Retail momentum": (
        ("retail_d7", "negative", "Recent retail momentum is negative"),
        ("retail_d7", "positive", "Recent retail momentum is positive"),
    ),
    "WTI": (
        ("wti_d5", "negative", "WTI is falling"),
        ("wti_d5", "positive", "WTI is rising"),
    ),
    "Wholesale gasoline": (
        ("nyh_d5", "negative", "Wholesale gasoline is falling"),
        ("nyh_d5", "positive", "Wholesale gasoline is rising"),
    ),
    "Crack spread": (
        ("crack_nyh_d5", "negative", "Crack spread is narrowing"),
        ("crack_nyh_d5", "positive", "Crack spread is widening"),
    ),
    "Inventory": (
        ("inv_wow", "positive", "Gasoline inventories rose week-over-week"),
        ("inv_wow", "negative", "Gasoline inventories fell week-over-week"),
    ),
}


@dataclass(frozen=True)
class SignalForecast:
    prediction_date: pd.Timestamp
    retail_monday: pd.Timestamp
    predicted_change: float
    signal: SignalResult
    latest_data_date: pd.Timestamp
    spot_feature_timestamp: pd.Timestamp
    inventory_feature_timestamp: pd.Timestamp
    feature_row: pd.Series
    explanations: tuple[str, ...]
    data_warning: str | None = None


@dataclass(frozen=True)
class DataFreshness:
    retail_monday: pd.Timestamp
    spot_date: pd.Timestamp
    inventory_date: pd.Timestamp
    prediction_ts_et: pd.Timestamp
    is_stale: bool
    stale_reason: str | None
    is_warning: bool = False
    warning_message: str | None = None


def _freshness_warning(retail_monday: pd.Timestamp, now_et: datetime) -> str | None:
    """Warn when retail is usable but older than the expected latest Monday."""
    expected_monday = _latest_monday(pd.Timestamp(now_et.date()))
    if retail_monday >= expected_monday:
        return None
    days_behind = (expected_monday - retail_monday).days
    return (
        f"WARNING: Latest retail print is from {retail_monday.strftime('%Y-%m-%d')} "
        f"({days_behind} day(s) behind the expected Monday {expected_monday.strftime('%Y-%m-%d')}). "
        "Signal may be stale."
    )


def _ridge_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=RIDGE_ALPHA)),
        ]
    )


def _latest_monday(on_or_before: pd.Timestamp) -> pd.Timestamp:
    d = on_or_before.normalize()
    return d - pd.Timedelta(days=d.dayofweek)


def build_latest_signal_row(processed_dir: Path = PROCESSED_DIR) -> pd.DataFrame:
    """Return the most recent complete feature row (next-week target not required)."""
    rows = build_weekly_eia_dataset(processed_dir, require_target=False)
    if rows.empty:
        raise ValueError("Unable to generate today's signal because required data is unavailable.")
    latest = rows.sort_values("prediction_date").tail(1)
    missing = [c for c in FEATURES if latest[c].isna().any()]
    if missing:
        raise ValueError(
            "Unable to generate today's signal because required data is unavailable. "
            f"Missing features: {', '.join(missing)}"
        )
    return latest


def check_data_freshness(row: pd.DataFrame, *, now: datetime | None = None) -> DataFreshness:
    """Detect unavailable, stale, or unexpectedly old retail data."""
    r = row.iloc[0]
    now_et = now or datetime.now(EASTERN)
    if now_et.tzinfo is None:
        now_et = now_et.replace(tzinfo=EASTERN)
    else:
        now_et = now_et.astimezone(EASTERN)

    pred_ts = pd.Timestamp(r["prediction_ts_utc"]).tz_convert(EASTERN)
    retail_monday = pd.Timestamp(r["retail_monday"])
    spot_date = pd.Timestamp(r["spot_feature_timestamp"])
    inv_date = pd.Timestamp(r["inventory_feature_timestamp"])
    base = dict(
        retail_monday=retail_monday,
        spot_date=spot_date,
        inventory_date=inv_date,
        prediction_ts_et=pred_ts,
    )

    if now_et < pred_ts:
        return DataFreshness(
            **base,
            is_stale=True,
            stale_reason="Unable to generate today's signal because required data is unavailable.",
        )

    days_since_retail = (now_et.date() - retail_monday.date()).days
    if days_since_retail > RETAIL_MAX_AGE_DAYS:
        return DataFreshness(
            **base,
            is_stale=True,
            stale_reason=(
                "Unable to generate today's signal because required data is unavailable. "
                f"Latest retail is {days_since_retail} days old (limit: {RETAIL_MAX_AGE_DAYS} days)."
            ),
        )

    expected_monday = _latest_monday(pd.Timestamp(now_et.date()))
    if (expected_monday - retail_monday).days > RETAIL_MISSING_WEEK_DAYS:
        return DataFreshness(
            **base,
            is_stale=True,
            stale_reason=(
                "Unable to generate today's signal because required data is unavailable. "
                f"Expected a retail print within {RETAIL_MISSING_WEEK_DAYS} days of "
                f"{expected_monday.strftime('%Y-%m-%d')}."
            ),
        )

    warning = _freshness_warning(retail_monday, now_et)
    return DataFreshness(
        **base,
        is_stale=False,
        stale_reason=None,
        is_warning=warning is not None,
        warning_message=warning,
    )


def _group_contributions(model: Pipeline, row: pd.Series) -> dict[str, float]:
    scaler: StandardScaler = model.named_steps["scaler"]
    ridge: Ridge = model.named_steps["ridge"]
    x = row[FEATURES].to_frame().T
    x_scaled = scaler.transform(x)[0]
    coefs = ridge.coef_
    contribs = x_scaled * coefs
    out: dict[str, float] = {}
    for group, cols in GROUP_FEATURE_MAP.items():
        idx = [FEATURES.index(c) for c in cols if c in FEATURES]
        out[group] = float(contribs[idx].sum()) if idx else 0.0
    return out


def explain_prediction(model: Pipeline, row: pd.Series, *, max_bullets: int = 3) -> tuple[str, ...]:
    """Short group-level context aligned with model direction (not causal)."""
    group_contrib = _group_contributions(model, row)
    total = sum(abs(v) for v in group_contrib.values()) or 1.0
    ranked = sorted(group_contrib.items(), key=lambda kv: abs(kv[1]), reverse=True)
    bullets: list[str] = []
    for group, contrib in ranked:
        if abs(contrib) / total < 0.08:
            continue
        hints = GROUP_VALUE_HINTS.get(group, ())
        added = False
        for col, direction, text in hints:
            if col not in row.index or pd.isna(row[col]):
                continue
            val = float(row[col])
            if direction == "negative" and val < 0 and contrib < 0:
                bullets.append(text)
                added = True
                break
            if direction == "positive" and val > 0 and contrib > 0:
                bullets.append(text)
                added = True
                break
        if not added and abs(contrib) / total >= 0.12:
            if contrib > 0:
                bullets.append(f"{group} inputs lean toward higher prices")
            else:
                bullets.append(f"{group} inputs lean toward lower prices")
        if len(bullets) >= max_bullets:
            break
    return tuple(bullets)


def fit_ridge_full(training: pd.DataFrame) -> Pipeline:
    """Fit frozen ridge_full spec on rows with known targets."""
    train = training.dropna(subset=["target"])
    if len(train) < 100:
        raise ValueError("Insufficient training history for ridge_full.")
    model = _ridge_pipeline()
    model.fit(train[FEATURES], train["target"].to_numpy())
    return model


def generate_weekly_signal(
    processed_dir: Path = PROCESSED_DIR,
    *,
    now: datetime | None = None,
) -> SignalForecast:
    """Build latest features, fit ridge_full on history, return current signal."""
    history = build_weekly_eia_dataset(processed_dir, require_target=True)
    if history.empty:
        raise ValueError("Unable to generate today's signal because required data is unavailable.")

    latest_row = build_latest_signal_row(processed_dir)
    freshness = check_data_freshness(latest_row, now=now)
    if freshness.is_stale:
        raise ValueError(freshness.stale_reason or "Unable to generate today's signal because required data is unavailable.")

    model = fit_ridge_full(history)
    pred = float(model.predict(latest_row.loc[:, FEATURES])[0])
    sig = generate_signal(pred, threshold=DEFAULT_SIGNAL_THRESHOLD)
    r = latest_row.iloc[0]
    explanations = explain_prediction(model, r)

    latest_data = max(
        pd.Timestamp(r["retail_monday"]),
        pd.Timestamp(r["spot_feature_timestamp"]),
        pd.Timestamp(r["inventory_feature_timestamp"]),
    )

    return SignalForecast(
        prediction_date=pd.Timestamp(r["prediction_date"]),
        retail_monday=pd.Timestamp(r["retail_monday"]),
        predicted_change=pred,
        signal=sig,
        latest_data_date=latest_data,
        spot_feature_timestamp=pd.Timestamp(r["spot_feature_timestamp"]),
        inventory_feature_timestamp=pd.Timestamp(r["inventory_feature_timestamp"]),
        feature_row=r,
        explanations=explanations,
        data_warning=freshness.warning_message,
    )
