"""Trading calendar helpers for session-aware realized volatility calculations.

This module defines session specifications (timezone-aware trading days that may
cross midnight), trading calendar presets, and annualization helpers. All
timestamps are normalized to tz-aware UTC when returned. Prices are expected to
be pre-adjusted (splits/dividends) and positive when used alongside the
volatility estimators.
"""

from __future__ import annotations

import datetime as dt
import math
from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class SessionSpec:
    """Describe a trading session window in a specific timezone.

    Attributes:
        name: Identifier for the session preset.
        tz: IANA timezone string.
        start_time: Local session start time.
        end_time: Local session end time (may be earlier than start_time for
            cross-midnight sessions).
    """

    name: str
    tz: str
    start_time: dt.time
    end_time: dt.time

    @property
    def crosses_midnight(self) -> bool:
        return self.end_time < self.start_time

    def _as_session_tz(self, ts: pd.Timestamp) -> pd.Timestamp:
        if ts.tzinfo is None:
            localized = ts.tz_localize("UTC")
        else:
            localized = ts
        return localized.tz_convert(self.tz)

    def session_key(self, ts: pd.Timestamp) -> dt.date:
        """Return the session date key for a UTC timestamp."""
        local_ts = self._as_session_tz(ts)
        if not self.crosses_midnight:
            return local_ts.date()
        if local_ts.timetz() >= self.start_time:
            return local_ts.date()
        return (local_ts - pd.Timedelta(days=1)).date()

    def session_close_timestamp(self, session_date: dt.date) -> pd.Timestamp:
        """Return the UTC timestamp of the session close for a session date."""
        close_date = session_date
        if self.crosses_midnight:
            close_date = session_date + dt.timedelta(days=1)
        close_dt = dt.datetime.combine(close_date, self.end_time)
        close_ts = pd.Timestamp(close_dt, tz=self.tz)
        return close_ts.tz_convert("UTC")


@dataclass(frozen=True)
class TradingCalendar:
    """Metadata for trading-time annualization and session mapping."""

    name: str
    minutes_per_day: float
    days_per_year: float
    session: SessionSpec


EQUITIES_RTH = TradingCalendar(
    name="EQUITIES_RTH",
    minutes_per_day=390,
    days_per_year=252,
    session=SessionSpec(
        name="EQUITIES_RTH",
        tz="America/New_York",
        start_time=dt.time(hour=9, minute=30),
        end_time=dt.time(hour=16, minute=0),
    ),
)

EQUITIES_24H = TradingCalendar(
    name="EQUITIES_24H",
    minutes_per_day=1440,
    days_per_year=365,
    session=SessionSpec(
        name="EQUITIES_24H",
        tz="UTC",
        start_time=dt.time(hour=0, minute=0, second=0),
        end_time=dt.time(hour=23, minute=59, second=59),
    ),
)

FUTURES_ES_GLOBEX = TradingCalendar(
    name="FUTURES_ES_GLOBEX",
    minutes_per_day=1380,
    days_per_year=252,
    session=SessionSpec(
        name="FUTURES_ES_GLOBEX",
        tz="America/New_York",
        start_time=dt.time(hour=18, minute=0),
        end_time=dt.time(hour=17, minute=0),
    ),
)


def bars_per_year(bar_minutes: float, calendar: TradingCalendar, bars_per_day_override: float | None = None) -> float:
    """Compute bars per year given bar size and calendar assumptions.

    Args:
        bar_minutes: Bar size in minutes.
        calendar: Trading calendar preset.
        bars_per_day_override: Optional override for bars per day (e.g., custom
            aggregation).

    Returns:
        Bars per year as float.

    Raises:
        ValueError: If inputs are non-positive.
    """
    if bar_minutes <= 0:
        raise ValueError("bar_minutes must be positive")
    if bars_per_day_override is not None and bars_per_day_override <= 0:
        raise ValueError("bars_per_day_override must be positive")

    bars_per_day = bars_per_day_override if bars_per_day_override is not None else calendar.minutes_per_day / bar_minutes
    return bars_per_day * calendar.days_per_year


def annualization_factor(
    bar_minutes: float, calendar: TradingCalendar, bars_per_day_override: float | None = None
) -> float:
    """Return sqrt(bars_per_year) for volatility annualization."""
    return math.sqrt(bars_per_year(bar_minutes, calendar, bars_per_day_override))

