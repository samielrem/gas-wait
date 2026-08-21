"""Point-in-time weekly feature builder for exp01.

Prediction clock: Tuesday 12:00 p.m. America/New_York after Monday retail T.
Target: retail(T+7) - retail(T). Features never use information after prediction_ts.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
EASTERN = ZoneInfo("America/New_York")

RETAIL_PATH = PROCESSED_DIR / "retail_gasoline_regular_us_weekly.csv"
WTI_PATH = PROCESSED_DIR / "wti_crude_spot_us_daily.csv"
NYH_PATH = PROCESSED_DIR / "ny_harbor_gasoline_regular_spot_daily.csv"
GC_PATH = PROCESSED_DIR / "gulf_coast_gasoline_regular_spot_daily.csv"
INV_PATH = PROCESSED_DIR / "gasoline_inventories_us_weekly.csv"

RETAIL_FEATURES = ["retail_d7", "retail_d14"]
MARKET_FEATURES = [
    "wti_d1",
    "wti_d3",
    "wti_d5",
    "wti_vol_20",
    "nyh_d1",
    "nyh_d5",
    "nyh_vol_20",
    "gc_d1",
    "gc_d5",
    "crack_nyh_d5",
]
INVENTORY_FEATURES = ["inv_wow", "inv_seasonal_z"]
SEASONALITY_FEATURES = ["sin_doy", "cos_doy", "is_summer"]
ALL_MODEL_FEATURES = RETAIL_FEATURES + MARKET_FEATURES + INVENTORY_FEATURES + SEASONALITY_FEATURES

TIMESTAMP_COLS = [
    "prediction_date",
    "prediction_ts_utc",
    "target",
    "target_timestamp",
    "retail_monday",
    "retail_feature_timestamp",
    "spot_feature_timestamp",
    "inventory_feature_timestamp",
    "inventory_release_ts_utc",
]


@dataclass
class DatasetBuildResult:
    frame: pd.DataFrame
    dropped: pd.DataFrame


def _load_series(path: Path, value_name: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["observation_date"])
    out = df[["observation_date", "value"]].rename(columns={"value": value_name})
    out = out.sort_values("observation_date").drop_duplicates("observation_date")
    return out.reset_index(drop=True)


def _us_federal_holidays(start: pd.Timestamp, end: pd.Timestamp) -> pd.DatetimeIndex:
    from pandas.tseries.holiday import USFederalHolidayCalendar

    cal = USFederalHolidayCalendar()
    return cal.holidays(start=start.normalize(), end=end.normalize())


def _at_noon_et(day: pd.Timestamp) -> pd.Timestamp:
    naive = pd.Timestamp(year=day.year, month=day.month, day=day.day, hour=12)
    return naive.tz_localize(EASTERN)


def _to_utc(ts: pd.Series | pd.Timestamp) -> pd.Series | pd.Timestamp:
    if isinstance(ts, pd.Timestamp):
        if ts.tzinfo is None:
            ts = ts.tz_localize(EASTERN)
        return ts.tz_convert("UTC")
    if ts.dt.tz is None:
        ts = ts.dt.tz_localize(EASTERN)
    return ts.dt.tz_convert("UTC")


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> pd.Timestamp:
    first = pd.Timestamp(year=year, month=month, day=1)
    offset = (weekday - first.dayofweek) % 7
    return first + pd.Timedelta(days=offset + 7 * (n - 1))


def _last_monday(year: int, month: int) -> pd.Timestamp:
    last = pd.Timestamp(year=year, month=month, day=1) + pd.offsets.MonthEnd(0)
    return last - pd.Timedelta(days=(last.dayofweek - 0) % 7)


def _holiday_travel_week(prediction_date: pd.Timestamp) -> bool:
    year = prediction_date.year
    memorial = _last_monday(year, 5)
    labor = _nth_weekday(year, 9, 0, 1)
    july4 = pd.Timestamp(year=year, month=7, day=4)
    july4_week_monday = july4 - pd.Timedelta(days=july4.dayofweek)
    thanksgiving = _nth_weekday(year, 11, 3, 4)
    windows = [
        (memorial, memorial + pd.Timedelta(days=6)),
        (labor, labor + pd.Timedelta(days=6)),
        (july4_week_monday, july4_week_monday + pd.Timedelta(days=6)),
        (thanksgiving - pd.Timedelta(days=thanksgiving.dayofweek), thanksgiving + pd.Timedelta(days=3)),
    ]
    for start, end in windows:
        if start <= prediction_date <= end:
            return True
    return False


def _is_summer(prediction_date: pd.Timestamp) -> bool:
    memorial = _last_monday(prediction_date.year, 5)
    labor = _nth_weekday(prediction_date.year, 9, 0, 1)
    return memorial <= prediction_date <= labor


def _add_session_features(df: pd.DataFrame, prefix: str, value_col: str) -> pd.DataFrame:
    out = df.copy()
    price = out[value_col]
    out[f"{prefix}_d1"] = price.diff(1)
    out[f"{prefix}_d3"] = price.diff(3)
    out[f"{prefix}_d5"] = price.diff(5)
    out[f"{prefix}_vol_20"] = out[f"{prefix}_d1"].rolling(window=20, min_periods=20).std()
    lagged = pd.DataFrame(
        {
            "observation_date": out["observation_date"],
            "asof": out["observation_date"] - pd.Timedelta(days=7),
        }
    )
    hist = out[["observation_date", value_col]].rename(
        columns={"observation_date": "hist_date", value_col: "hist_price"}
    )
    lagged = pd.merge_asof(
        lagged.sort_values("asof"),
        hist.sort_values("hist_date"),
        left_on="asof",
        right_on="hist_date",
        direction="backward",
    ).set_index("observation_date")
    out[f"{prefix}_chg_7cal"] = price.to_numpy() - lagged.loc[out["observation_date"], "hist_price"].to_numpy()
    return out


def _wpsr_release_ts(friday: pd.Timestamp, holidays: pd.DatetimeIndex) -> pd.Timestamp:
    """Stocks for week-ending Friday become public the following Wednesday noon ET.

    If the Monday of the release week or that Wednesday is a U.S. federal holiday,
    delay to Thursday noon (EIA holiday-week practice).
    """
    wednesday = friday + pd.Timedelta(days=5)
    monday = friday + pd.Timedelta(days=3)
    release_day = wednesday
    holiday_days = set(pd.DatetimeIndex(holidays).normalize())
    if monday.normalize() in holiday_days or wednesday.normalize() in holiday_days:
        release_day = friday + pd.Timedelta(days=6)
    return _at_noon_et(release_day)


def _asof_daily(mondays: pd.Series, daily: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    left = pd.DataFrame({"retail_monday": mondays}).sort_values("retail_monday")
    right = daily[["observation_date", *cols]].sort_values("observation_date")
    merged = pd.merge_asof(
        left,
        right,
        left_on="retail_monday",
        right_on="observation_date",
        direction="backward",
    )
    return merged.rename(columns={"observation_date": "spot_feature_timestamp"})


def build_weekly_model_dataset(
    processed_dir: Path = PROCESSED_DIR,
) -> DatasetBuildResult:
    """Build one row per retail Monday with point-in-time features."""
    retail = _load_series(processed_dir / RETAIL_PATH.name, "retail_price")
    wti = _load_series(processed_dir / WTI_PATH.name, "wti")
    nyh = _load_series(processed_dir / NYH_PATH.name, "nyh")
    gc = _load_series(processed_dir / GC_PATH.name, "gc")
    inv = _load_series(processed_dir / INV_PATH.name, "stocks")

    retail_valid = retail.dropna(subset=["retail_price"]).copy()
    retail_valid["next_date"] = retail_valid["observation_date"].shift(-1)
    retail_valid["next_price"] = retail_valid["retail_price"].shift(-1)
    retail_valid["gap_days"] = (retail_valid["next_date"] - retail_valid["observation_date"]).dt.days

    drop_records: list[dict] = []

    for _, row in retail.loc[retail["retail_price"].isna()].iterrows():
        drop_records.append(
            {
                "retail_monday": row["observation_date"],
                "reason": "missing_retail_at_t",
            }
        )

    candidates = retail_valid.copy()
    bad_next = candidates["gap_days"].ne(7) | candidates["next_price"].isna()
    for _, row in candidates.loc[bad_next].iterrows():
        drop_records.append(
            {
                "retail_monday": row["observation_date"],
                "reason": "missing_or_nonweekly_retail_at_tplus7",
            }
        )
    candidates = candidates.loc[~bad_next].copy()

    candidates["retail_monday"] = candidates["observation_date"]
    candidates["prediction_date"] = candidates["retail_monday"] + pd.Timedelta(days=1)
    candidates["prediction_ts_et"] = candidates["prediction_date"].map(_at_noon_et)
    candidates["prediction_ts_utc"] = _to_utc(candidates["prediction_ts_et"])
    candidates["target_timestamp"] = candidates["next_date"]
    candidates["target"] = candidates["next_price"] - candidates["retail_price"]
    candidates["retail_last"] = candidates["retail_price"]
    candidates["retail_feature_timestamp"] = candidates["retail_monday"]
    candidates["retail_d7"] = candidates["retail_price"] - candidates["retail_price"].shift(1)
    prev2 = candidates["retail_price"].shift(2)
    prev1_date = candidates["retail_monday"].shift(1)
    prev2_date = candidates["retail_monday"].shift(2)
    week_ok = (
        (candidates["retail_monday"] - prev1_date).dt.days.eq(7)
        & (candidates["retail_monday"] - prev2_date).dt.days.eq(14)
    )
    candidates["retail_d14"] = np.where(week_ok, candidates["retail_price"] - prev2, np.nan)

    wti_f = _add_session_features(wti, "wti", "wti")
    nyh_f = _add_session_features(nyh, "nyh", "nyh")
    gc_f = _add_session_features(gc, "gc", "gc")

    wti_join = _asof_daily(
        candidates["retail_monday"],
        wti_f,
        ["wti", "wti_d1", "wti_d3", "wti_d5", "wti_vol_20", "wti_chg_7cal"],
    )
    nyh_join = _asof_daily(
        candidates["retail_monday"],
        nyh_f,
        ["nyh", "nyh_d1", "nyh_d3", "nyh_d5", "nyh_vol_20"],
    )
    gc_join = _asof_daily(
        candidates["retail_monday"],
        gc_f,
        ["gc", "gc_d1", "gc_d3", "gc_d5"],
    )

    frame = candidates.merge(wti_join, on="retail_monday", how="left")
    nyh_join = nyh_join.rename(columns={"spot_feature_timestamp": "nyh_timestamp"})
    gc_join = gc_join.rename(columns={"spot_feature_timestamp": "gc_timestamp"})
    frame = frame.merge(nyh_join, on="retail_monday", how="left")
    frame = frame.merge(gc_join, on="retail_monday", how="left")

    spot_latest = frame[["spot_feature_timestamp", "nyh_timestamp", "gc_timestamp"]].max(axis=1)
    frame["spot_feature_timestamp"] = spot_latest
    frame["crack_nyh_d5"] = 42.0 * frame["nyh_d5"] - frame["wti_d5"]

    holidays = _us_federal_holidays(
        inv["observation_date"].min() - pd.Timedelta(days=14),
        candidates["prediction_date"].max() + pd.Timedelta(days=14),
    )
    inv = inv.dropna(subset=["stocks"]).copy()
    inv["inventory_feature_timestamp"] = inv["observation_date"]
    inv["inventory_release_ts_et"] = inv["observation_date"].map(lambda d: _wpsr_release_ts(d, holidays))
    inv["inventory_release_ts_utc"] = _to_utc(inv["inventory_release_ts_et"])
    inv["inv_wow"] = inv["stocks"].diff(1)
    inv["weekofyear"] = inv["observation_date"].dt.isocalendar().week.astype(int)
    inv["year"] = inv["observation_date"].dt.year
    zscores: list[float] = []
    for idx, row in inv.iterrows():
        hist = inv.loc[
            (inv["weekofyear"] == row["weekofyear"])
            & (inv["year"] < row["year"])
            & (inv["year"] >= row["year"] - 5)
        ]
        if len(hist) < 3 or hist["stocks"].std(ddof=0) == 0:
            zscores.append(np.nan)
        else:
            zscores.append((row["stocks"] - hist["stocks"].mean()) / hist["stocks"].std(ddof=0))
    inv["inv_seasonal_z"] = zscores

    left = frame[["retail_monday", "prediction_ts_utc"]].sort_values("prediction_ts_utc")
    right = inv[
        [
            "inventory_release_ts_utc",
            "inventory_feature_timestamp",
            "inv_wow",
            "inv_seasonal_z",
            "stocks",
        ]
    ].sort_values("inventory_release_ts_utc")
    inv_join = pd.merge_asof(
        left,
        right,
        left_on="prediction_ts_utc",
        right_on="inventory_release_ts_utc",
        direction="backward",
    )
    frame = frame.drop(columns=[c for c in ["inventory_feature_timestamp", "inventory_release_ts_utc"] if c in frame.columns])
    frame = frame.merge(
        inv_join[
            [
                "retail_monday",
                "inventory_feature_timestamp",
                "inventory_release_ts_utc",
                "inv_wow",
                "inv_seasonal_z",
            ]
        ],
        on="retail_monday",
        how="left",
    )

    doy = frame["prediction_date"].dt.dayofyear
    frame["sin_doy"] = np.sin(2 * np.pi * doy / 365.25)
    frame["cos_doy"] = np.cos(2 * np.pi * doy / 365.25)
    frame["is_summer"] = frame["prediction_date"].map(_is_summer).astype(int)
    frame["is_holiday_travel_week"] = frame["prediction_date"].map(_holiday_travel_week).astype(int)
    frame["month"] = frame["prediction_date"].dt.month

    required = ALL_MODEL_FEATURES + ["target", "spot_feature_timestamp", "inventory_feature_timestamp"]
    missing_mask = frame[required].isna().any(axis=1)
    for _, row in frame.loc[missing_mask].iterrows():
        missing_cols = [c for c in required if pd.isna(row[c])]
        drop_records.append(
            {
                "retail_monday": row["retail_monday"],
                "reason": "missing_required_features:" + ",".join(missing_cols[:8]),
            }
        )
    kept = frame.loc[~missing_mask].copy()

    dropped = pd.DataFrame(drop_records)
    _assert_no_leakage(kept)
    kept = kept[TIMESTAMP_COLS + ALL_MODEL_FEATURES + ["retail_last", "wti_chg_7cal", "month", "is_holiday_travel_week"]]
    kept = kept.sort_values("prediction_date").reset_index(drop=True)
    logger.info("Modeling rows kept=%s dropped=%s", len(kept), len(dropped))
    return DatasetBuildResult(frame=kept, dropped=dropped)


def _assert_no_leakage(df: pd.DataFrame) -> None:
    if df.empty:
        raise ValueError("Modeling dataset is empty after filters.")

    pred_dates = df["prediction_date"].dt.tz_localize(None)
    if not df["spot_feature_timestamp"].le(df["retail_monday"]).all():
        bad = df.loc[df["spot_feature_timestamp"] > df["retail_monday"]]
        raise AssertionError(f"Spot feature after Monday T: {bad[['retail_monday', 'spot_feature_timestamp']].head()}")

    if not df["spot_feature_timestamp"].lt(pred_dates).all():
        raise AssertionError("Spot feature timestamp is not strictly before prediction_date.")

    if not df["retail_feature_timestamp"].eq(df["retail_monday"]).all():
        raise AssertionError("Retail feature timestamp must equal Monday T.")

    if not df["target_timestamp"].gt(df["retail_monday"]).all():
        raise AssertionError("Target timestamp is not after Monday T.")

    if not (df["target_timestamp"] - df["retail_monday"]).dt.days.eq(7).all():
        raise AssertionError("Target is not exactly 7 days after Monday T.")

    inv_lag_days = (pred_dates - df["inventory_feature_timestamp"]).dt.days
    if (inv_lag_days < 10).any():
        bad = df.loc[inv_lag_days < 10, ["prediction_date", "inventory_feature_timestamp"]]
        raise AssertionError(f"Inventory Friday is too close to Tuesday prediction:\n{bad.head()}")

    release_ok = df["inventory_release_ts_utc"].le(df["prediction_ts_utc"])
    if not release_ok.all():
        raise AssertionError("Inventory release is after prediction timestamp.")

    if (df["target_timestamp"] <= pred_dates).any():
        raise AssertionError("Target timestamp is not after prediction_date.")


def save_weekly_model_dataset(
    processed_dir: Path = PROCESSED_DIR,
) -> DatasetBuildResult:
    result = build_weekly_model_dataset(processed_dir)
    out_path = processed_dir / "weekly_model_dataset.csv"
    drop_path = processed_dir / "weekly_model_dataset_dropped.csv"
    result.frame.to_csv(out_path, index=False)
    result.dropped.to_csv(drop_path, index=False)
    logger.info("Wrote %s and %s", out_path, drop_path)
    return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    save_weekly_model_dataset()
