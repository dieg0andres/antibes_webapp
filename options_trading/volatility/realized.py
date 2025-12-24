"""Realized volatility estimators and orchestration utilities.

Supports OHLC-based estimators (close-to-close, Parkinson, Garman–Klass,
Rogers–Satchell, Yang–Zhang) and intraday realized variance. All inputs are
normalized to tz-aware UTC indices, prices are assumed pre-adjusted and
positive, and window sizes are in bars/sessions. Annualization uses
sqrt(bars_per_year) based on the provided TradingCalendar.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np
import pandas as pd

from .calendar import (
    EQUITIES_RTH,
    TradingCalendar,
    annualization_factor as calc_annualization_factor,
    bars_per_year as calc_bars_per_year,
)
from .transforms import (
    aggregate_to_sessions,
    apply_adjustment_factor,
    candles_to_df,
    ensure_tz_aware_utc_index,
    infer_bar_minutes,
    validate_ohlc,
)

RealizedMethod = Literal[
    "close_to_close", "parkinson", "garman_klass", "rogers_satchell", "yang_zhang", "realized_variance"
]


@dataclass
class RealizedVolResult:
    """Container for realized volatility outputs.

    Attributes:
        rv: Annualized volatility series (if annualize=True).
        variance: Per-bar/session variance series (not annualized).
        method: Estimator name used.
        window: Rolling window length in bars/sessions.
        annualization: Multiplier applied to convert sigma_bar to sigma_ann.
        metadata: Auxiliary info (calendar, bar_minutes, ddof, etc.).
        skew: Optional rolling skew of close-to-close returns.
        excess_kurtosis: Optional rolling excess kurtosis of returns.
    """

    rv: pd.Series
    variance: pd.Series
    method: RealizedMethod
    window: int
    annualization: float
    metadata: dict
    skew: Optional[pd.Series] = None
    excess_kurtosis: Optional[pd.Series] = None


def _rolling_variance_from_returns(
    returns: pd.Series, window: int, ddof: int, assume_mean_zero: bool
) -> pd.Series:
    """Rolling variance of log returns with configurable mean assumption.

    close-to-close formula: r_t = ln(C_t / C_{t-1}); var = sum((r - mu)^2)/(N-ddof)
    where mu=0 when assume_mean_zero=True else sample mean.
    """
    if window <= ddof:
        raise ValueError("window must be greater than ddof")

    if assume_mean_zero:
        return returns.pow(2).rolling(window=window, min_periods=window).sum() / (window - ddof)
    return returns.rolling(window=window, min_periods=window).var(ddof=ddof)


def _rolling_moments(
    returns: pd.Series, window: int, assume_mean_zero: bool
) -> tuple[pd.Series, pd.Series]:
    """Rolling skew and excess kurtosis of close-to-close returns."""
    def _skew(arr: np.ndarray) -> float:
        r = pd.Series(arr)
        mu = 0.0 if assume_mean_zero else r.mean()
        centered = r - mu
        std = centered.std(ddof=0)
        if std == 0 or pd.isna(std):
            return math.nan
        z = centered / std
        return float((z**3).mean())

    def _kurt(arr: np.ndarray) -> float:
        r = pd.Series(arr)
        mu = 0.0 if assume_mean_zero else r.mean()
        centered = r - mu
        std = centered.std(ddof=0)
        if std == 0 or pd.isna(std):
            return math.nan
        z = centered / std
        return float((z**4).mean() - 3.0)

    roller = returns.rolling(window=window, min_periods=window)
    return roller.apply(_skew, raw=True), roller.apply(_kurt, raw=True)


def _choose_aggregation(aggregation: str, bar_minutes: float) -> str:
    """Resolve aggregation mode, treating daily-or-faster bars as session-level by default."""
    if aggregation not in {"auto", "none", "session"}:
        raise ValueError("aggregation must be one of {'auto', 'none', 'session'}")
    if aggregation == "auto":
        return "session" if bar_minutes <= 24 * 60 else "none"
    return aggregation


def _compute_close_to_close_variance(
    df: pd.DataFrame, window: int, ddof: int, assume_mean_zero: bool
) -> pd.Series:
    """Close-to-close variance: r_t = ln(C_t/C_{t-1}); rolling var over window."""
    returns = np.log(df["close"] / df["close"].shift(1))
    return _rolling_variance_from_returns(returns, window, ddof, assume_mean_zero)


def _compute_parkinson_variance(df: pd.DataFrame, window: int) -> pd.Series:
    """Parkinson variance: v_t = (ln(H/L))^2 / (4 ln 2); rolling mean over window."""
    v_t = (np.log(df["high"] / df["low"]) ** 2) / (4.0 * math.log(2.0))
    return v_t.rolling(window=window, min_periods=window).mean()


def _compute_garman_klass_variance(df: pd.DataFrame, window: int) -> pd.Series:
    """Garman–Klass variance using range and open-close drift adjustment."""
    log_hl = np.log(df["high"] / df["low"])
    log_co = np.log(df["close"] / df["open"])
    v_t = 0.5 * (log_hl**2) - (2 * math.log(2.0) - 1.0) * (log_co**2)
    return v_t.rolling(window=window, min_periods=window).mean()


def _compute_rogers_satchell_variance(df: pd.DataFrame, window: int) -> pd.Series:
    """Rogers–Satchell variance capturing open-relative directional moves."""
    log_ho = np.log(df["high"] / df["open"])
    log_lo = np.log(df["low"] / df["open"])
    log_hc = np.log(df["high"] / df["close"])
    log_lc = np.log(df["low"] / df["close"])
    v_t = log_ho * log_hc + log_lo * log_lc
    return v_t.rolling(window=window, min_periods=window).mean()


def _compute_yang_zhang_variance(
    df: pd.DataFrame, window: int, ddof: int, assume_mean_zero: bool, k_override: float | None = None
) -> pd.Series:
    """Yang–Zhang variance combining overnight, open-close, and RS components.

    Components:
        r_o = ln(O_t / C_{t-1})   (overnight)
        r_c = ln(C_t / O_t)       (open-to-close)
        RS  = ln(H/O)*ln(H/C) + ln(L/O)*ln(L/C)
    yz = var(r_o) + k*var(r_c) + (1-k)*mean(RS), with k per Yang–Zhang weighting.
    """
    prev_close = df["close"].shift(1)
    r_o = np.log(df["open"] / prev_close)
    r_c = np.log(df["close"] / df["open"])
    v_rs = np.log(df["high"] / df["open"]) * np.log(df["high"] / df["close"]) + np.log(df["low"] / df["open"]) * np.log(
        df["low"] / df["close"]
    )

    sigma_o2 = _rolling_variance_from_returns(r_o, window, ddof, assume_mean_zero)
    sigma_c2 = _rolling_variance_from_returns(r_c, window, ddof, assume_mean_zero)
    sigma_rs2 = v_rs.rolling(window=window, min_periods=window).mean()

    if k_override is not None:
        k = k_override
    else:
        k = 0.34 / (1.34 + (window + 1) / (window - 1))

    return sigma_o2 + k * sigma_c2 + (1 - k) * sigma_rs2


def _realized_variance_intraday(
    df: pd.DataFrame,
    session_df: pd.DataFrame,
    window: int,
    session_spec,
    include_overnight: bool,
) -> pd.Series:
    """Per-session realized variance from intraday path plus optional overnight gaps.

    For each session: rv = (ln(C0/O0))^2 + sum_{i>0}(ln(C_i/C_{i-1}))^2
    plus ln(O_t / C_{t-1})^2 when include_overnight=True.
    """
    session_keys = df.index.map(session_spec.session_key)
    session_close_index = session_df.index
    session_dates = [session_spec.session_key(ts) for ts in session_close_index]
    close_map = dict(zip(session_dates, session_df["close"]))

    rv_records = []
    grouped = df.copy()
    grouped["_session_key"] = session_keys

    session_order = sorted(grouped["_session_key"].unique())
    prev_close_lookup = close_map

    for i, session_date in enumerate(session_order):
        chunk = grouped[grouped["_session_key"] == session_date].sort_index()
        open0 = float(chunk["open"].iloc[0])
        close0 = float(chunk["close"].iloc[0])
        rv_value = math.log(close0 / open0) ** 2

        if len(chunk) > 1:
            intraday_returns = np.log(chunk["close"] / chunk["close"].shift(1)).dropna()
            rv_value += float((intraday_returns**2).sum())

        if include_overnight and i > 0:
            prev_date = session_order[i - 1]
            prev_close = prev_close_lookup.get(prev_date)
            if prev_close is not None:
                overnight_return = math.log(open0 / float(prev_close))
                rv_value += overnight_return**2

        rv_records.append((session_date, rv_value))

    rv_index = [session_spec.session_close_timestamp(date) for date, _ in rv_records]
    rv_series = pd.Series([v for _, v in rv_records], index=pd.DatetimeIndex(rv_index, tz="UTC"))
    return rv_series.rolling(window=window, min_periods=window).mean()


def realized_vol(
    data: pd.DataFrame | dict | list,
    method: RealizedMethod = "yang_zhang",
    window: int = 20,
    *,
    calendar: TradingCalendar = EQUITIES_RTH,
    aggregation: str = "auto",
    annualize: bool = True,
    assume_mean_zero: bool = True,
    ddof: int | None = None,
    adjustment_factor: pd.Series | None = None,
    include_moments: bool = False,
    strict: bool = True,
    bars_per_day_override: float | None = None,
    k_override: float | None = None,
    include_overnight: bool = True,
) -> RealizedVolResult:
    """Compute realized volatility (annualized) from OHLC data.

    Window is a rolling span in bars/sessions. Variance is per-bar; rv =
    sqrt(variance) * sqrt(bars_per_year) when annualize=True. Inputs are assumed
    adjusted and positive. Aggregation auto-defaults to session-level for
    intraday/daily bars so session closes serve as index (includes overnight).
    The realized_variance method uses full intraday paths (open->first close,
    subsequent close-to-close, plus overnight when enabled).

    Args:
        data: OHLC DataFrame or candle payload.
        method: Estimator name.
        window: Rolling window length in bars/sessions.
        calendar: Trading calendar for sessions and annualization.
        aggregation: "auto"|"none"|"session" for pre-estimation aggregation.
        annualize: Apply sqrt(bars_per_year) scaling when True.
        assume_mean_zero: If True, treat mean return as zero for variance.
        ddof: Degrees of freedom override; defaults depend on assume_mean_zero.
        adjustment_factor: Optional Series to multiply OHLC.
        include_moments: Compute skew and excess kurtosis of cc returns.
        strict: Enforce timestamp regularity and OHLC validity.
        bars_per_day_override: Optional bars/day for annualization when not session indexed.
        k_override: Override Yang–Zhang k weight.
        include_overnight: Include overnight gap in realized_variance.

    Returns:
        RealizedVolResult with variance (per-bar), rv (annualized if enabled),
        and optional higher moments.

    Raises:
        ValueError: On invalid inputs, unsupported method, or failed validations.

    Method vs aggregation semantics (important):

        - `window` is always measured in BARS of the series being rolled over.
        With session-level output, one bar ≈ one trading session/day. With intraday/no aggregation,
        one bar ≈ one intraday candle (e.g., 15 minutes).

    Aggregation modes:
        - aggregation="session": intraday candles are aggregated into one OHLCV bar per trading session
        using the calendar’s SessionSpec. The aggregated bar is indexed at the session close timestamp
        (in session tz) converted to UTC. This produces a session-level (daily-like) series.
        - aggregation="none": use the input candles as-is (no session aggregation). Output stays at the
        input frequency (intraday stays intraday).
        - aggregation="auto": for bar sizes <= 1 day, defaults to "session" (fund-style RV). Otherwise "none".

    When to use which:
        - For “fund RV” / options IV comparisons (daily/session risk): prefer session-level output
        (aggregation="session" or auto). This includes overnight effects naturally at the session level.
        - For intraday monitoring or short-horizon features: use aggregation="none" with methods that are
        meaningful on intraday bars (typically close-to-close).

    Estimator expectations:
        - OHLC range-based estimators (Parkinson, Garman–Klass, Rogers–Satchell) and Yang–Zhang are
        conceptually intended for one bar per session/day. Therefore they should generally be used with
        session-level bars (aggregation="session"/auto). Passing intraday data with session aggregation is
        valid, but it will be reduced to session OHLC and will not exploit the intraday return path beyond
        session open/high/low/close.
        - Yang–Zhang should NOT be applied directly on raw intraday bars with aggregation="none" because its
        “overnight” and open/close components are defined per session/day.
        - close_to_close can be used either on session bars (daily/session close-to-close) or on intraday bars
        (intraday close-to-close). If used on intraday bars, it measures intraday-only risk unless your feed
        includes overnight bars.

    Path-aware estimator:
        - realized_variance computes per-session realized variance using ALL intraday returns within each
        session (open->first close plus subsequent close-to-close returns), and can optionally include an
        explicit overnight gap term (prev session close -> current session open) via include_overnight=True.
        The output is session-indexed regardless of aggregation settings.

    Overnight handling:
        - include_overnight only affects method="realized_variance". Other methods incorporate overnight risk
        only via session-level bars (e.g., close-to-close across session closes) or via Yang–Zhang’s own
        overnight component when computed on session bars.
    """
    if window <= 1:
        raise ValueError("window must be greater than 1")
    if ddof is None:
        effective_ddof = 0 if assume_mean_zero else 1
    else:
        effective_ddof = ddof

    df = candles_to_df(data, strict=strict)
    df = ensure_tz_aware_utc_index(df)
    df = apply_adjustment_factor(df, adjustment_factor)
    validate_ohlc(df, strict=strict)

    bar_minutes = infer_bar_minutes(df.index, session=calendar.session, strict=strict)
    agg_mode = _choose_aggregation(aggregation, bar_minutes)

    session_df = None
    if agg_mode == "session" or method == "realized_variance":
        session_df = aggregate_to_sessions(df, calendar.session, strict=strict)

    data_for_estimators = session_df if agg_mode == "session" else df

    if method in {"parkinson", "garman_klass", "rogers_satchell", "yang_zhang"} and data_for_estimators is None:
        data_for_estimators = session_df

    final_index_is_session = agg_mode == "session" or method == "realized_variance"
    if final_index_is_session:
        bars_year = calendar.days_per_year
    else:
        bars_year = calc_bars_per_year(bar_minutes, calendar, bars_per_day_override)
    ann_factor = math.sqrt(bars_year) if annualize else 1.0

    if method == "close_to_close":
        variance = _compute_close_to_close_variance(data_for_estimators, window, effective_ddof, assume_mean_zero)
    elif method == "parkinson":
        variance = _compute_parkinson_variance(data_for_estimators, window)
    elif method == "garman_klass":
        variance = _compute_garman_klass_variance(data_for_estimators, window)
    elif method == "rogers_satchell":
        variance = _compute_rogers_satchell_variance(data_for_estimators, window)
    elif method == "yang_zhang":
        variance = _compute_yang_zhang_variance(
            data_for_estimators, window, effective_ddof, assume_mean_zero, k_override
        )
    elif method == "realized_variance":
        if session_df is None:
            session_df = aggregate_to_sessions(df, calendar.session, strict=strict)
        variance = _realized_variance_intraday(df, session_df, window, calendar.session, include_overnight)
        data_for_estimators = session_df
    else:
        raise ValueError(f"Unsupported method {method}")

    rv = variance.pow(0.5) * ann_factor

    skew_series: Optional[pd.Series] = None
    kurt_series: Optional[pd.Series] = None
    if include_moments:
        moments_close = data_for_estimators["close"]
        cc_returns = np.log(moments_close / moments_close.shift(1))
        skew_series, kurt_series = _rolling_moments(cc_returns, window, assume_mean_zero)

    metadata = {
        "bar_minutes_inferred": bar_minutes,
        "bars_per_year": bars_year,
        "calendar": calendar.name,
        "session": calendar.session.name,
        "aggregation": "session" if final_index_is_session else agg_mode,
        "ddof": effective_ddof,
        "assume_mean_zero": assume_mean_zero,
        "annualize": annualize,
        "annualization_factor": ann_factor,
        "include_overnight": include_overnight,
        "k_override": k_override,
        "bars_per_day_override": bars_per_day_override,
    }

    return RealizedVolResult(
        rv=rv,
        variance=variance,
        method=method,
        window=window,
        annualization=ann_factor,
        metadata=metadata,
        skew=skew_series,
        excess_kurtosis=kurt_series,
    )

