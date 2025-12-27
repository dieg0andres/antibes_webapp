"""IV/RV panel composition for ATM implied vol vs realized vol state.

This module keeps IV calculation (priced from options) separate from realized
volatility state (computed from returns), then combines them into a snapshot
with derived IV/RV metrics (VRP, ratios, variance spreads). It intentionally
avoids Django dependencies and focuses on pure-Python composition.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from .calendar import EQUITIES_RTH, TradingCalendar
from .iv_calculation import IVXATMResult, ivx_atm
from .panel import VolatilityStatePanelResult, volatility_state_panel


@dataclass
class IVRVPanelResult:
    symbol: str | None
    as_of: pd.Timestamp | None
    target_dte_days: float

    rv_panel: VolatilityStatePanelResult
    iv_result: IVXATMResult

    rv20: float | None
    rv20_source: str | None
    ivx: float | None
    vrp: float | None
    iv_over_rv: float | None
    var_vrp: float | None

    flags: dict
    notes: list[str]


def iv_rv_panel(
    *,
    session_data: pd.DataFrame | dict | list,
    intraday_data: pd.DataFrame | dict | list | None,
    option_chain: dict | pd.DataFrame,
    calendar: TradingCalendar = EQUITIES_RTH,
    target_dte_days: float = 30.0,
    rv_reference: Literal["primary20", "yz20"] = "primary20",
    panel_kwargs: dict | None = None,
    iv_kwargs: dict | None = None,
) -> IVRVPanelResult:
    """
    Compute IV/RV panel snapshot and attach full diagnostics.

    Derived metrics:
      vrp       = IVX - RV20
      iv_over_rv = IVX / RV20 (if RV20 > 0)
      var_vrp   = IVX^2 - RV20^2

    Notes on rv_reference:
      - primary20 may use realized_variance when available (more data) but can
        switch sources; source reported in rv20_source.
      - yz20 is stable, computed from session OHLC (Yang–Zhang).
    """

    panel_kwargs = panel_kwargs or {}
    iv_kwargs = iv_kwargs or {}

    rv_panel = volatility_state_panel(
        session_data=session_data,
        intraday_data=intraday_data,
        calendar=calendar,
        **panel_kwargs,
    )

    iv_result = ivx_atm(
        option_chain,
        target_dte_days=target_dte_days,
        **iv_kwargs,
    )

    # RV20 selection
    if rv_reference == "primary20":
        rv20 = rv_panel.latest.get("vol_primary_20") if rv_panel.latest is not None else None
        rv20_source = rv_panel.latest.get("primary_src_20") if rv_panel.latest is not None else None
    elif rv_reference == "yz20":
        rv20 = rv_panel.latest.get("vol_yz_20") if rv_panel.latest is not None else None
        rv20_source = None
    else:
        raise ValueError("rv_reference must be one of {'primary20','yz20'}")

    # Normalize RV20
    if rv20 is not None and isinstance(rv20, pd.Series):
        rv20 = float(rv20)
    if rv20 is not None and (pd.isna(rv20) or np.isinf(rv20)):
        rv20 = None

    # Normalize IVX
    ivx = iv_result.ivx_calc
    if ivx is not None and (pd.isna(ivx) or np.isinf(ivx)):
        ivx = None

    vrp = iv_over_rv = var_vrp = None
    if ivx is not None and rv20 is not None:
        vrp = ivx - rv20
        iv_over_rv = ivx / rv20 if rv20 > 0 else None
        var_vrp = ivx * ivx - rv20 * rv20

    notes: list[str] = []
    flags = {
        "IVX_AVAILABLE": ivx is not None,
        "RV20_AVAILABLE": rv20 is not None,
        "RV_REFERENCE": rv_reference,
        "IV_FLAGS": iv_result.flags,
    }
    if ivx is None:
        notes.append("IVX_UNAVAILABLE")
    if rv20 is None:
        notes.append("RV20_UNAVAILABLE")
    if iv_result.notes:
        notes.extend(iv_result.notes)

    symbol = iv_result.symbol or rv_panel.settings.get("symbol")

    return IVRVPanelResult(
        symbol=symbol,
        as_of=iv_result.as_of,
        target_dte_days=target_dte_days,
        rv_panel=rv_panel,
        iv_result=iv_result,
        rv20=rv20,
        rv20_source=rv20_source,
        ivx=ivx,
        vrp=vrp,
        iv_over_rv=iv_over_rv,
        var_vrp=var_vrp,
        flags=flags,
        notes=notes,
    )

