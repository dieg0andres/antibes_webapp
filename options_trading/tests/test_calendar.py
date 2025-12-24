"""Calendar utilities tests to ensure session keys and annualization behave."""

import datetime as dt

import pandas as pd

from options_trading.volatility.calendar import (
    EQUITIES_RTH,
    FUTURES_ES_GLOBEX,
    TradingCalendar,
    bars_per_year,
)


def test_session_key_standard_hours():
    # Ensure RTH session key maps intraday timestamp to same-date session.
    ts = pd.Timestamp("2024-01-02 15:00:00Z")  # 10:00 ET
    session_date = EQUITIES_RTH.session.session_key(ts)
    assert session_date == dt.date(2024, 1, 2)


def test_session_key_cross_midnight():
    # Cross-midnight sessions should attribute early-morning times to prior date.
    ts = pd.Timestamp("2024-01-03 02:00:00Z")  # 21:00 ET previous day
    session_date = FUTURES_ES_GLOBEX.session.session_key(ts)
    assert session_date == dt.date(2024, 1, 2)


def test_session_close_timestamp_cross_midnight():
    # Session close timestamp should land on following calendar day for cross-midnight.
    close_ts = FUTURES_ES_GLOBEX.session.session_close_timestamp(dt.date(2024, 1, 2))
    assert close_ts == pd.Timestamp("2024-01-03 22:00:00Z")


def test_bars_per_year_equities_rth():
    # Annualization helper matches minutes_per_day / bar_minutes * days_per_year.
    bar_minutes = 30
    expected = (EQUITIES_RTH.minutes_per_day / bar_minutes) * EQUITIES_RTH.days_per_year
    assert bars_per_year(bar_minutes, EQUITIES_RTH) == expected

