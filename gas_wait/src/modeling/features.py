"""Reusable time-series feature functions for Gas Wait.

Each function documents source frequency, lag assumptions, and leakage risk.
Rolling statistics use only past rows in time order — never future observations.
No interpolation of missing market sessions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _sorted_copy(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    return df.sort_values(date_col).drop_duplicates(date_col).reset_index(drop=True)


def session_changes(
    df: pd.DataFrame,
    value_col: str,
    periods: tuple[int, ...],
    prefix: str,
    *,
    date_col: str = "observation_date",
) -> pd.DataFrame:
    """Business-day (row) price changes on a daily series.

    Source: daily spot/retail series sorted by ``date_col``.
    Frequency: one row per completed session (gaps stay gaps).
    Lag: change at row *t* uses prices at rows *t* and *t-k* only.
    Leakage risk: low if joined with as-of <= prediction time; do not use
    same-day close before its publication lag.
    """
    out = _sorted_copy(df, date_col)
    price = out[value_col]
    for k in periods:
        out[f"{prefix}_d{k}"] = price.diff(k)
    return out


def calendar_day_change(
    df: pd.DataFrame,
    value_col: str,
    calendar_days: int,
    prefix: str,
    *,
    date_col: str = "observation_date",
    out_col: str | None = None,
) -> pd.DataFrame:
    """Calendar-day change using last observation on or before *t - calendar_days*.

    Source: daily series.
    Frequency: daily (irregular sessions allowed).
    Lag: uses only prices with observation_date <= t - calendar_days.
    Leakage risk: safe when the as-of join respects publication timing.
    No interpolation between sessions.
    """
    out = _sorted_copy(df, date_col)
    col = out_col or f"{prefix}_chg_{calendar_days}cal"
    lagged = pd.DataFrame(
        {
            date_col: out[date_col],
            "asof": out[date_col] - pd.Timedelta(days=calendar_days),
        }
    )
    hist = out[[date_col, value_col]].rename(columns={date_col: "hist_date", value_col: "hist_price"})
    merged = pd.merge_asof(
        lagged.sort_values("asof"),
        hist.sort_values("hist_date"),
        left_on="asof",
        right_on="hist_date",
        direction="backward",
    ).set_index(date_col)
    out[col] = out[value_col].to_numpy() - merged.loc[out[date_col], "hist_price"].to_numpy()
    return out


def rolling_volatility(
    df: pd.DataFrame,
    change_col: str,
    window: int,
    prefix: str,
    *,
    date_col: str = "observation_date",
) -> pd.DataFrame:
    """Rolling std of a change column over ``window`` prior sessions.

    Source: 1-session change series on daily data.
    Lag: window includes current row and *window-1* prior rows only.
    Leakage risk: compute on full history then as-of join; do not slice after rolling.
    """
    out = df.copy()
    out[f"{prefix}_vol_{window}"] = out[change_col].rolling(window=window, min_periods=window).std()
    return out


def moving_average_distance(
    df: pd.DataFrame,
    value_col: str,
    window: int,
    prefix: str,
    *,
    date_col: str = "observation_date",
) -> pd.DataFrame:
    """Level minus rolling mean (not percent — avoids near-zero blowups).

    Source: daily price level.
    Lag: MA uses current and prior *window-1* sessions.
    Leakage risk: same as rolling_volatility.
    """
    out = df.copy()
    ma = out[value_col].rolling(window=window, min_periods=window).mean()
    out[f"{prefix}_ma_dist_{window}"] = out[value_col] - ma
    return out


def short_vs_long_momentum(
    df: pd.DataFrame,
    value_col: str,
    short_period: int,
    long_period: int,
    prefix: str,
    *,
    date_col: str = "observation_date",
) -> pd.DataFrame:
    """Short-session change minus long-session change on the same series.

    Source: daily prices.
    Lag: both diffs use only past rows.
    """
    out = df.copy()
    short = out[value_col].diff(short_period)
    long = out[value_col].diff(long_period)
    out[f"{prefix}_mom_{short_period}_{long_period}"] = short - long
    return out


def add_crude_features(
    df: pd.DataFrame,
    value_col: str = "wti",
    *,
    date_col: str = "observation_date",
    prefix: str = "wti",
    include_extended: bool = False,
) -> pd.DataFrame:
    """WTI crude features (daily sessions).

    Source: EIA daily WTI spot ($/bbl).
    """
    out = session_changes(df, value_col, (1, 3, 5), prefix, date_col=date_col)
    if include_extended:
        out = session_changes(out, value_col, (7,), prefix, date_col=date_col)
    else:
        out = calendar_day_change(out, value_col, 7, prefix, date_col=date_col, out_col=f"{prefix}_chg_7cal")
    out = rolling_volatility(out, f"{prefix}_d1", 20, prefix, date_col=date_col)
    if include_extended:
        out = moving_average_distance(out, value_col, 20, prefix, date_col=date_col)
        out = short_vs_long_momentum(out, value_col, 5, 20, prefix, date_col=date_col)
    return out


def add_wholesale_features(
    df: pd.DataFrame,
    value_col: str,
    prefix: str,
    *,
    date_col: str = "observation_date",
    include_extended: bool = False,
) -> pd.DataFrame:
    """NY Harbor or Gulf Coast daily gasoline spot features ($/gal).

    Source: EIA daily conventional regular spot.
    """
    periods = (1, 3, 5, 7) if include_extended else (1, 5)
    out = session_changes(df, value_col, periods, prefix, date_col=date_col)
    out = rolling_volatility(out, f"{prefix}_d1", 20, prefix, date_col=date_col)
    if include_extended:
        out = moving_average_distance(out, value_col, 20, prefix, date_col=date_col)
    return out


def add_spread_features(
    merged_daily: pd.DataFrame,
    *,
    wti_d5_col: str = "wti_d5",
    nyh_d5_col: str = "nyh_d5",
    nyh_level_col: str | None = "nyh",
    gc_level_col: str | None = "gc",
) -> pd.DataFrame:
    """Economically meaningful spreads on aligned daily rows.

    - ``crack_nyh_d5`` = 42 * NYH 5-session change - WTI 5-session change
    - ``spread_nyh_gc`` = NYH level - GC level (optional)
    """
    out = merged_daily.copy()
    out["crack_nyh_d5"] = 42.0 * out[nyh_d5_col] - out[wti_d5_col]
    if nyh_level_col and gc_level_col and nyh_level_col in out.columns and gc_level_col in out.columns:
        out["spread_nyh_gc"] = out[nyh_level_col] - out[gc_level_col]
    return out


def add_retail_momentum_weekly(
    df: pd.DataFrame,
    price_col: str,
    date_col: str = "observation_date",
    *,
    prefix: str = "retail",
) -> pd.DataFrame:
    """Weekly retail momentum (EIA Monday series).

    Source: EIA weekly retail ($/gal).
    Frequency: weekly (expect 7-day gaps).
    ``retail_d7`` = 1-week change; ``retail_d14`` = 2-week change with gap check;
    ``retail_chg_p4`` = 4-week change with gap check.
    Lag: uses only prior Monday prints.
    """
    out = _sorted_copy(df, date_col)
    out[f"{prefix}_d7"] = out[price_col].diff(1)
    prev1_date = out[date_col].shift(1)
    prev2_date = out[date_col].shift(2)
    prev4_date = out[date_col].shift(4)
    week_ok_2 = (out[date_col] - prev1_date).dt.days.eq(7) & (out[date_col] - prev2_date).dt.days.eq(14)
    week_ok_4 = (out[date_col] - prev4_date).dt.days.eq(28)
    out[f"{prefix}_d14"] = np.where(week_ok_2, out[price_col] - out[price_col].shift(2), np.nan)
    out[f"{prefix}_chg_p4"] = np.where(week_ok_4, out[price_col] - out[price_col].shift(4), np.nan)
    return out


def add_inventory_level_features(
    inv_df: pd.DataFrame,
    *,
    date_col: str = "observation_date",
    value_col: str = "stocks",
    seasonal_years: int = 5,
) -> pd.DataFrame:
    """Inventory changes and seasonal z-score on weekly WPSR stocks.

    Source: EIA weekly gasoline stocks (thousand barrels).
    Frequency: week-ending Friday; **must** be joined by release timestamp, not Friday date alone.
    ``inv_yoy`` requires 52 prior weekly rows.
    ``inv_seasonal_z`` uses only calendar weeks from years strictly before the current year.
    """
    out = _sorted_copy(inv_df, date_col)
    out["inv_wow"] = out[value_col].diff(1)
    out["inv_yoy"] = out[value_col].diff(52)
    out["weekofyear"] = out[date_col].dt.isocalendar().week.astype(int)
    out["year"] = out[date_col].dt.year
    zscores: list[float] = []
    for _, row in out.iterrows():
        hist = out.loc[
            (out["weekofyear"] == row["weekofyear"])
            & (out["year"] < row["year"])
            & (out["year"] >= row["year"] - seasonal_years)
        ]
        if len(hist) < 3 or hist[value_col].std(ddof=0) == 0:
            zscores.append(np.nan)
        else:
            zscores.append((row[value_col] - hist[value_col].mean()) / hist[value_col].std(ddof=0))
    out["inv_seasonal_z"] = zscores
    return out


def add_seasonality_features(
    prediction_dates: pd.Series,
    *,
    include_weekofyear: bool = False,
) -> pd.DataFrame:
    """Calendar seasonality from prediction date (known at prediction time).

    Source: prediction clock only — no price data.
    Leakage risk: none if ``prediction_dates`` are the model cutoff instants.
    """
    pred = pd.to_datetime(prediction_dates)
    doy = pred.dt.dayofyear
    frame = pd.DataFrame(
        {
            "prediction_date": pred,
            "month": pred.dt.month,
            "sin_doy": np.sin(2 * np.pi * doy / 365.25),
            "cos_doy": np.cos(2 * np.pi * doy / 365.25),
        }
    )
    if include_weekofyear:
        frame["weekofyear"] = pred.dt.isocalendar().week.astype(int)
    frame["is_summer"] = pred.map(_is_summer_us).astype(int)
    return frame


def _is_summer_us(prediction_date: pd.Timestamp) -> bool:
    memorial = _last_monday(prediction_date.year, 5)
    labor = _nth_weekday(prediction_date.year, 9, 0, 1)
    return memorial <= prediction_date <= labor


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> pd.Timestamp:
    first = pd.Timestamp(year=year, month=month, day=1)
    offset = (weekday - first.dayofweek) % 7
    return first + pd.Timedelta(days=offset + 7 * (n - 1))


def _last_monday(year: int, month: int) -> pd.Timestamp:
    last = pd.Timestamp(year=year, month=month, day=1) + pd.offsets.MonthEnd(0)
    return last - pd.Timedelta(days=(last.dayofweek - 0) % 7)
