"""Point-in-time filtering and publication clocks for Gas Wait features.

Conservative rules follow ``docs/modeling_design.md``:

- Daily spots: session *D* available next weekday 12:00 p.m. ET.
- Weekly retail: Monday *T* available Tuesday 12:00 p.m. ET.
- Inventories: week-ending Friday *F* available following Wednesday noon ET
  (Thursday on some holiday weeks).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable
from zoneinfo import ZoneInfo

import pandas as pd

EASTERN = ZoneInfo("America/New_York")


class PublicationRule(str, Enum):
    """How ``public_available_ts`` is derived from ``observation_date``."""

    DAILY_SPOT_NEXT_WEEKDAY_NOON_ET = "daily_spot_next_weekday_noon_et"
    WEEKLY_RETAIL_TUESDAY_NOON_ET = "weekly_retail_tuesday_noon_et"
    WPSR_WEDNESDAY_NOON_ET = "wpsr_wednesday_noon_et"
    KNOWN_AT_PREDICTION = "known_at_prediction"
    CUSTOM = "custom"


@dataclass(frozen=True)
class TimestampedSeries:
    """Observations with economic date and public availability."""

    frame: pd.DataFrame
    observation_col: str
    value_cols: tuple[str, ...]
    rule: PublicationRule
    availability_col: str = "public_available_ts"


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


def _us_federal_holidays(start: pd.Timestamp, end: pd.Timestamp) -> pd.DatetimeIndex:
    from pandas.tseries.holiday import USFederalHolidayCalendar

    cal = USFederalHolidayCalendar()
    return cal.holidays(start=start.normalize(), end=end.normalize())


def daily_spot_public_ts(session_date: pd.Timestamp) -> pd.Timestamp:
    """Next weekday noon ET after a daily spot session (conservative)."""
    nxt = session_date + pd.offsets.BDay(1)
    return _at_noon_et(nxt)


def weekly_retail_public_ts(monday_date: pd.Timestamp) -> pd.Timestamp:
    """Tuesday noon ET after Monday retail print."""
    tuesday = monday_date + pd.Timedelta(days=1)
    return _at_noon_et(tuesday)


def wpsr_release_ts(friday: pd.Timestamp, holidays: pd.DatetimeIndex | None = None) -> pd.Timestamp:
    """WPSR stocks for week-ending Friday become public the following Wednesday noon ET."""
    if holidays is None:
        holidays = _us_federal_holidays(friday - pd.Timedelta(days=14), friday + pd.Timedelta(days=14))
    wednesday = friday + pd.Timedelta(days=5)
    monday = friday + pd.Timedelta(days=3)
    release_day = wednesday
    holiday_days = set(pd.DatetimeIndex(holidays).normalize())
    if monday.normalize() in holiday_days or wednesday.normalize() in holiday_days:
        release_day = friday + pd.Timedelta(days=6)
    return _at_noon_et(release_day)


def attach_public_availability(
    df: pd.DataFrame,
    rule: PublicationRule,
    *,
    observation_col: str = "observation_date",
    out_col: str = "public_available_ts",
    holidays: pd.DatetimeIndex | None = None,
    custom_fn: Callable[[pd.Timestamp], pd.Timestamp] | None = None,
) -> pd.DataFrame:
    """Add ``public_available_ts`` column from observation dates."""
    out = df.copy()
    if rule is PublicationRule.DAILY_SPOT_NEXT_WEEKDAY_NOON_ET:
        out[out_col] = out[observation_col].map(daily_spot_public_ts)
    elif rule is PublicationRule.WEEKLY_RETAIL_TUESDAY_NOON_ET:
        out[out_col] = out[observation_col].map(weekly_retail_public_ts)
    elif rule is PublicationRule.WPSR_WEDNESDAY_NOON_ET:
        out[out_col] = out[observation_col].map(lambda d: wpsr_release_ts(d, holidays))
    elif rule is PublicationRule.KNOWN_AT_PREDICTION:
        out[out_col] = out[observation_col]
    elif rule is PublicationRule.CUSTOM:
        if custom_fn is None:
            raise ValueError("CUSTOM rule requires custom_fn")
        out[out_col] = out[observation_col].map(custom_fn)
    else:
        raise ValueError(f"Unknown rule: {rule}")
    out[f"{out_col}_utc"] = _to_utc(out[out_col])
    return out


def filter_known_at(
    source: pd.DataFrame,
    prediction_timestamp: pd.Timestamp,
    *,
    availability_col: str = "public_available_ts",
    observation_col: str = "observation_date",
) -> pd.DataFrame:
    """Return rows whose public availability is on or before ``prediction_timestamp``."""
    pred = prediction_timestamp
    if pred.tzinfo is None:
        pred = pred.tz_localize(EASTERN)
    avail = source[availability_col]
    if hasattr(avail.dt, "tz") and avail.dt.tz is None:
        avail = avail.dt.tz_localize(EASTERN)
    mask = avail <= pred
    known = source.loc[mask].copy()
    if known.empty:
        return known
    return known.sort_values(observation_col)


def asof_at_prediction(
    left_dates: pd.Series,
    right: pd.DataFrame,
    prediction_timestamps: pd.Series,
    *,
    left_on: str = "observation_date",
    value_cols: list[str],
    availability_col: str = "public_available_ts",
    out_obs_col: str = "feature_observation_date",
) -> pd.DataFrame:
    """As-of join using only rows public by each row's ``prediction_timestamp``.

    For each prediction time, filters ``right`` to publicly available rows, then
    merge_asof on observation date (backward).
    """
    left = pd.DataFrame({"left_date": left_dates, "prediction_ts": prediction_timestamps})
    pieces: list[pd.DataFrame] = []
    for idx, row in left.iterrows():
        pred_ts = row["prediction_ts"]
        known = filter_known_at(right, pred_ts, availability_col=availability_col, observation_col=left_on)
        if known.empty:
            continue
        one = pd.DataFrame({"left_date": [row["left_date"]]})
        merged = pd.merge_asof(
            one.sort_values("left_date"),
            known[[left_on, *value_cols]].sort_values(left_on),
            left_on="left_date",
            right_on=left_on,
            direction="backward",
        )
        merged["prediction_ts"] = pred_ts
        merged[out_obs_col] = merged[left_on]
        pieces.append(merged)
    if not pieces:
        return pd.DataFrame(columns=["left_date", left_on, *value_cols, "prediction_ts", out_obs_col])
    return pd.concat(pieces, ignore_index=True)


def asof_on_dates(
    left_dates: pd.Series,
    right: pd.DataFrame,
    value_cols: list[str],
    *,
    right_date_col: str = "observation_date",
    out_obs_col: str = "feature_observation_date",
) -> pd.DataFrame:
    """Backward as-of join on calendar dates (no per-row publication filter).

    Use when the left date is already on or before the prediction clock and
    publication lag is satisfied by construction (exp01: Monday T before Tuesday noon).
    """
    left = pd.DataFrame({"left_date": left_dates}).sort_values("left_date")
    right = right.sort_values(right_date_col)
    merged = pd.merge_asof(
        left,
        right[[right_date_col, *value_cols]].sort_values(right_date_col),
        left_on="left_date",
        right_on=right_date_col,
        direction="backward",
    )
    merged = merged.rename(columns={right_date_col: out_obs_col})
    return merged


def merge_asof_before_prediction(
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    left_date_col: str,
    right_date_col: str = "observation_date",
    prediction_ts_col: str = "prediction_ts_utc",
    right_avail_col: str = "public_available_ts_utc",
    suffix: str = "",
) -> pd.DataFrame:
    """Merge-asof on dates after filtering ``right`` to rows public by prediction time."""
    rcols = [c for c in right.columns if c not in (right_date_col, right_avail_col)]
    right_cols = [right_date_col, *rcols]
    known_right = right[right_cols + [right_avail_col]].copy()
    merged_rows: list[pd.DataFrame] = []
    for pred_ts, grp in left.groupby(prediction_ts_col, sort=False):
        known = known_right.loc[known_right[right_avail_col] <= pred_ts]
        part = pd.merge_asof(
            grp.sort_values(left_date_col),
            known.sort_values(right_date_col),
            left_on=left_date_col,
            right_on=right_date_col,
            direction="backward",
            suffixes=("", suffix),
        )
        merged_rows.append(part)
    return pd.concat(merged_rows, ignore_index=True).sort_values(left_date_col)


def assert_feature_timestamps(
    df: pd.DataFrame,
    prediction_ts_col: str,
    feature_ts_map: dict[str, str],
    *,
    strict: bool = True,
) -> None:
    """Assert every feature source timestamp is <= prediction timestamp.

    Parameters
    ----------
    feature_ts_map:
        Maps logical name -> column name holding the feature's observation/economic date.
    strict:
        If True, require ``feature_ts <= prediction_ts``; if False, allow equality only
        for calendar features marked KNOWN_AT_PREDICTION.
    """
    if df.empty:
        raise ValueError("Cannot assert leakage on empty frame.")
    pred = pd.to_datetime(df[prediction_ts_col])
    if pred.dt.tz is not None:
        pred = pred.dt.tz_convert("UTC").dt.tz_localize(None)
    for name, col in feature_ts_map.items():
        if col not in df.columns:
            raise AssertionError(f"Missing timestamp column {col} for {name}")
        fts = pd.to_datetime(df[col])
        if fts.dt.tz is not None:
            fts = fts.dt.tz_convert("UTC").dt.tz_localize(None)
        pred_s = pred.dt.tz_localize(None) if pred.dt.tz is not None else pred
        if strict:
            ok = fts <= pred_s
        else:
            ok = fts <= pred_s
        if not ok.all():
            bad = df.loc[~ok, [prediction_ts_col, col]].head()
            raise AssertionError(f"Feature timestamp leak ({name}): {bad}")


def assert_no_future_observations(
    df: pd.DataFrame,
    prediction_date_col: str,
    observation_cols: list[str],
) -> None:
    """Assert economic observation dates are not after the prediction calendar date."""
    pred = pd.to_datetime(df[prediction_date_col])
    for col in observation_cols:
        obs = pd.to_datetime(df[col])
        if not (obs <= pred).all():
            raise AssertionError(f"{col} has observations after {prediction_date_col}")
