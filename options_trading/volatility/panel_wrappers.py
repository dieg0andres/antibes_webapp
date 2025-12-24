"""Thin wrappers for constructing volatility panels with injected loaders.

These helpers remain provider-agnostic; callers supply loader callables to fetch
session and intraday candles, and the wrapper delegates to the core panel
builder. No Django imports or settings are used here.
"""

from __future__ import annotations

from typing import Callable

import pandas as pd

from .calendar import EQUITIES_RTH, TradingCalendar
from .panel import VolatilityStatePanelResult, volatility_state_panel


def volatility_state_panel_for_ticker(
    *,
    ticker: str,
    load_session_candles: Callable[..., pd.DataFrame | dict | list],
    load_intraday_candles: Callable[..., pd.DataFrame | dict | list] | None = None,
    calendar: TradingCalendar = EQUITIES_RTH,
    session_days: int = 500,
    intraday_days: int = 200,
    intraday_freq: str = "15min",
    **panel_kwargs,
) -> VolatilityStatePanelResult:
    """
    Fetch candles via injected loaders and build a volatility state panel.

    Parameters
    ----------
    ticker : str
        Symbol to fetch.
    load_session_candles : Callable
        Callable returning session/daily candles (DataFrame or payload) when
        invoked like `load_session_candles(ticker=..., days=session_days, frequency="daily")`.
    load_intraday_candles : Callable | None
        Optional callable returning intraday candles when invoked like
        `load_intraday_candles(ticker=..., days=intraday_days, frequency=intraday_freq)`.
        If None, intraday data is skipped and realized_variance is omitted.
    calendar : TradingCalendar
        Calendar preset for session mapping and annualization.
    session_days : int
        Lookback passed to the session loader.
    intraday_days : int
        Lookback passed to the intraday loader.
    intraday_freq : str
        Frequency hint passed to the intraday loader (e.g., "15min").
    **panel_kwargs :
        Forwarded to `volatility_state_panel` (e.g., windows, flags).

    Returns
    -------
    VolatilityStatePanelResult
        The assembled panel and metadata.
    """

    session_payload = load_session_candles(ticker=ticker, days=session_days, frequency="daily")
    intraday_payload = (
        load_intraday_candles(ticker=ticker, days=intraday_days, frequency=intraday_freq)
        if load_intraday_candles is not None
        else None
    )

    return volatility_state_panel(
        session_data=session_payload,
        intraday_data=intraday_payload,
        calendar=calendar,
        **panel_kwargs,
    )

