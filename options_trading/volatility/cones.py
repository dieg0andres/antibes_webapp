# options_trading/volatility/cones.py
"""
Volatility Cones (VC) — practitioner-first implementation.

This module builds *realized-volatility cones* and the key UI-friendly statistics
required to overlay an implied-volatility term structure on top of those cones.

Conceptual definition (trader view)
-----------------------------------
For each horizon h (in *trading days*), we compute a historical distribution of
realized volatility measured over rolling h-day windows. A *volatility cone* is
the collection of those distributions across multiple horizons, typically shown
as percentile bands vs horizon.

Given an implied volatility term structure (IV vs maturity), a practitioner uses
the cone to answer:
- "Is implied vol rich/cheap relative to what realized vol typically does at
  that horizon?"
- "Where does today's realized vol sit within its own historical distribution?"
  (rv_today_percentile_rank)

Design choices in this implementation
-------------------------------------
1) Inputs:
   - We accept *session-level* OHLC time series for a ticker.
   - We compute realized-vol series internally using the shared estimators in
     realized.py, rather than requiring the caller to pass precomputed RV series.

2) RV methods supported (as requested):
   - CTC_OVERNIGHT: Close-to-close realized volatility (session close returns).
     Overnight risk is inherently included because consecutive closes span the
     overnight gap.
   - YZ: Yang–Zhang OHLC estimator.

3) Horizons and lookbacks:
   - Horizons (trading days): e.g. [10, 21, 42, 63, 126, 252]
   - Lookbacks (years): e.g. [2, 5, 10]
   - Lookback sampling is performed as the *last N RV observations* where
     N = lookback_years * sessions_per_year, rather than slicing a calendar
     date range. This avoids off-by-one issues due to timestamp conventions
     (midnight vs session close) and keeps counts stable.

4) Overlap/autocorrelation:
   - v1 accepts overlapping windows (rolling stride = 1 session).
   - We record this in metadata as overlap_method="overlap_unadjusted".
     (This is a cone-of-quantiles; we are not applying Hodges–Tompkins style
     adjustments which address different bias objects.)

5) Output:
   - A pandas "table" (MultiIndex) useful for debugging or ad-hoc graphing.
   - A JSON-ready "jsonable" dict intended to be stored in VolatilitySnapshot.cone.

Important unit consistency
--------------------------
- Cone horizons are expressed in TRADING DAYS.
- Implied-vol term structure typically uses calendar DTE in pricing; the overlay
  must map calendar DTE to approximate trading-day horizons *outside this module*.
  (We store trading-day horizons here only.)

Expected input formatting
-------------------------
- session_ohlc must represent *session bars* (daily bars) for the underlying.
- Index should be increasing and ideally tz-aware UTC.
- Columns required: open, high, low, close (lowercase).

This module is pure Python and Django-agnostic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Sequence

import pandas as pd


# -----------------------------
# Public types / result objects
# -----------------------------

RVMethod = Literal["CTC_OVERNIGHT", "YZ"]
OverlapMethod = Literal["overlap_unadjusted"]
ConeFlag = Literal["OK", "INSUFFICIENT_HISTORY", "NO_RV_TODAY"]


@dataclass(frozen=True)
class VolatilityConeResult:
    """
    Container for volatility cone outputs.

    Attributes:
        asof_utc:
            ISO8601 timestamp (UTC) representing the "as-of" time for the cone.
            Typically the timestamp of the last session bar.
        symbol:
            Ticker symbol.
        horizons_days:
            Horizons (trading days) used as cone "x-axis".
        percentiles:
            Percentiles (0-1) computed per horizon.
        lookbacks_years_requested:
            Lookbacks requested by the caller.
        lookbacks_years_used:
            Lookbacks that are possible given raw session history length.
            (Note: per-horizon feasibility still depends on horizon length.)
        overlap_method:
            How rolling windows were sampled (v1: overlap_unadjusted).
        sessions_per_year:
            Used to translate years -> sessions.
        rv_methods:
            RV methods computed.
        table:
            MultiIndex DataFrame with rows:
                (rv_method, lookback_years, horizon_days)
            and columns:
                count, min, max, p{percentile}, rv_today, rv_today_percentile_rank, flag
        jsonable:
            JSON-ready dict matching the persisted schema in VolatilitySnapshot.cone.
        notes:
            Human-readable notes, suitable for debugging/audit.
    """

    asof_utc: str
    symbol: str
    horizons_days: tuple[int, ...]
    percentiles: tuple[float, ...]
    lookbacks_years_requested: tuple[int, ...]
    lookbacks_years_used: tuple[int, ...]
    overlap_method: OverlapMethod
    sessions_per_year: int
    rv_methods: tuple[RVMethod, ...]
    table: pd.DataFrame
    jsonable: dict
    notes: list[str]


# -----------------------------
# Core API
# -----------------------------

def compute_volatility_cone(
    *,
    symbol: str,
    session_ohlc: pd.DataFrame | dict | list,
    asof_utc: str | None = None,
    rv_methods: Sequence[RVMethod] = ("CTC_OVERNIGHT", "YZ"),
    horizons_days: Sequence[int] = (10, 21, 42, 63, 126, 252),
    lookback_years: Sequence[int] = (2, 5, 10),
    percentiles: Sequence[float] = (0.10, 0.25, 0.50, 0.75, 0.90),
    sessions_per_year: int = 252,
    overlap_method: OverlapMethod = "overlap_unadjusted",
    strict: bool = True,
    min_samples: int = 30,
    calendar: Any | None = None,
) -> VolatilityConeResult:
    """
    Compute realized-volatility cones for the given session OHLC history.

    Key behavior:
      - Computes RV(h) internally for each rv_method and horizon.
      - For each lookback (years) and horizon, forms the distribution from the
        last N RV observations up to "asof", where N = years * sessions_per_year.
      - Computes rv_today and its percentile rank within that distribution.

    Returns:
        VolatilityConeResult with:
          - table: MultiIndex DataFrame for debugging/graphing
          - jsonable: JSON-ready structure suitable for persistence
    """
    _validate_inputs(horizons_days, lookback_years, percentiles, sessions_per_year, min_samples)

    realized_vol, EQUITIES_RTH, candles_to_df = _lazy_imports()
    cal = calendar if calendar is not None else EQUITIES_RTH

    session_df = _to_session_df(session_ohlc, candles_to_df=candles_to_df, strict=strict)
    session_df = session_df.sort_index()

    if asof_utc is None:
        asof_utc = _ts_to_iso_utc(session_df.index[-1])

    # Precompute RV series for each method and horizon.
    horizons = tuple(int(h) for h in horizons_days)
    pct = tuple(float(p) for p in percentiles)
    methods = tuple(rv_methods)

    rv_series: dict[RVMethod, dict[int, pd.Series]] = {m: {} for m in methods}
    notes: list[str] = []

    for m in methods:
        realized_method = _map_rv_method_to_realized(m)
        for h in horizons:
            res = realized_vol(
                session_df,
                method=realized_method,
                window=h,
                calendar=cal,
                aggregation="session",
                annualize=True,
                strict=strict,
            )
            rv_series[m][h] = res.rv
        notes.append(f"Computed RV series for {m} using realized.py estimators.")

    # Determine which lookbacks are possible from raw session history length.
    lookbacks_req = tuple(int(y) for y in lookback_years)
    lookbacks_used = tuple(y for y in lookbacks_req if len(session_df) >= y * sessions_per_year)
    if not lookbacks_used:
        notes.append("Insufficient session history for all requested lookbacks; cone may be mostly empty.")

    rows: list[dict] = []
    jsonable = _init_cone_json(
        asof_utc=asof_utc,
        symbol=symbol,
        sessions_per_year=sessions_per_year,
        overlap_method=overlap_method,
        horizons=horizons,
        percentiles=pct,
        lookbacks_requested=lookbacks_req,
        lookbacks_used=lookbacks_used,
        rv_methods=methods,
    )

    # Use session-date for alignment (midnight vs session-close timestamp mismatches).
    asof_date = session_df.index[-1].normalize()

    for m in methods:
        for y in lookbacks_used:
            lookback_sessions = y * sessions_per_year

            for h in horizons:
                # Feasibility check: ensure enough underlying sessions exist to generate
                # at least lookback_sessions RV observations for horizon h.
                required_sessions = lookback_sessions + (h - 1)
                if len(session_df) < required_sessions:
                    entry = _empty_entry(flag="INSUFFICIENT_HISTORY")
                    rows.append(_row_dict(m, y, h, entry, pct))
                    _set_json_entry(jsonable, m, y, h, entry)
                    continue

                rv = rv_series[m][h]
                rv_dates = rv.index.normalize()

                # RV values up to "asof" (by session date).
                rv_upto = rv.loc[rv_dates <= asof_date].dropna()

                if len(rv_upto) < lookback_sessions or len(rv_upto) < min_samples:
                    entry = _empty_entry(flag="INSUFFICIENT_HISTORY", count=int(len(rv_upto)))
                    rows.append(_row_dict(m, y, h, entry, pct))
                    _set_json_entry(jsonable, m, y, h, entry)
                    continue

                # Sample used for cone distribution: last N RV observations.
                s = rv_upto.iloc[-lookback_sessions:]

                # "Today's RV": last RV at or before asof_date.
                # (If rv_upto is non-empty, this is defined.)
                if len(rv_upto) == 0:
                    entry = _empty_entry(flag="NO_RV_TODAY")
                    rows.append(_row_dict(m, y, h, entry, pct))
                    _set_json_entry(jsonable, m, y, h, entry)
                    continue

                rv_today_val = float(rv_upto.iloc[-1])

                q = s.quantile(list(pct))
                p_dict = {f"{p:g}": float(q.loc[p]) for p in pct}

                entry = {
                    "count": int(len(s)),  # should equal lookback_sessions
                    "min": float(s.min()),
                    "max": float(s.max()),
                    "p": p_dict,
                    "rv_today": rv_today_val,
                    "rv_today_percentile_rank": float((s <= rv_today_val).mean()),
                    "flag": "OK",
                }

                rows.append(_row_dict(m, y, h, entry, pct))
                _set_json_entry(jsonable, m, y, h, entry)

    table = _build_table(rows, pct)

    return VolatilityConeResult(
        asof_utc=asof_utc,
        symbol=symbol,
        horizons_days=horizons,
        percentiles=pct,
        lookbacks_years_requested=lookbacks_req,
        lookbacks_years_used=lookbacks_used,
        overlap_method=overlap_method,
        sessions_per_year=sessions_per_year,
        rv_methods=methods,
        table=table,
        jsonable=jsonable,
        notes=notes,
    )


# -----------------------------
# Internal helpers
# -----------------------------

def _validate_inputs(
    horizons_days: Sequence[int],
    lookback_years: Sequence[int],
    percentiles: Sequence[float],
    sessions_per_year: int,
    min_samples: int,
) -> None:
    if sessions_per_year <= 0:
        raise ValueError("sessions_per_year must be positive.")
    if min_samples <= 0:
        raise ValueError("min_samples must be positive.")

    horizons = [int(h) for h in horizons_days]
    if not horizons or any(h <= 1 for h in horizons):
        raise ValueError("horizons_days must contain integers > 1.")
    if sorted(horizons) != horizons:
        raise ValueError("horizons_days must be sorted ascending for stable UI behavior.")

    lbs = [int(y) for y in lookback_years]
    if not lbs or any(y <= 0 for y in lbs):
        raise ValueError("lookback_years must contain positive integers.")
    if sorted(lbs) != lbs:
        raise ValueError("lookback_years must be sorted ascending.")

    pcts = [float(p) for p in percentiles]
    if not pcts or any((p <= 0.0 or p >= 1.0) for p in pcts):
        raise ValueError("percentiles must be floats in the open interval (0,1).")
    if len(set(pcts)) != len(pcts):
        raise ValueError("percentiles must be unique.")
    if sorted(pcts) != pcts:
        raise ValueError("percentiles must be sorted ascending.")


def _lazy_imports():
    from .realized import realized_vol  # type: ignore
    from .calendar import EQUITIES_RTH  # type: ignore
    try:
        from .transforms import candles_to_df  # type: ignore
    except Exception:  # pragma: no cover
        candles_to_df = None
    return realized_vol, EQUITIES_RTH, candles_to_df


def _to_session_df(
    data: pd.DataFrame | dict | list,
    *,
    candles_to_df: Any | None,
    strict: bool,
) -> pd.DataFrame:
    if candles_to_df is not None:
        df = candles_to_df(data, strict=strict)
        if isinstance(df.index, pd.DatetimeIndex) and df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        elif isinstance(df.index, pd.DatetimeIndex):
            df.index = df.index.tz_convert("UTC")
        return df

    if not isinstance(data, pd.DataFrame):
        raise ValueError(
            "session_ohlc must be a pandas DataFrame when transforms.candles_to_df is unavailable."
        )

    df = data.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("session_ohlc DataFrame must have a DatetimeIndex.")
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")

    for c in ("open", "high", "low", "close"):
        if c not in df.columns:
            raise ValueError(f"session_ohlc is missing required column '{c}'.")
    return df[["open", "high", "low", "close"]]


def _ts_to_iso_utc(ts: pd.Timestamp) -> str:
    ts = pd.Timestamp(ts)
    if ts.tz is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts.strftime("%Y-%m-%dT%H:%M:%SZ")


def _map_rv_method_to_realized(method: RVMethod) -> str:
    if method == "YZ":
        return "yang_zhang"
    if method == "CTC_OVERNIGHT":
        return "close_to_close"
    raise ValueError(f"Unsupported RVMethod: {method}")


def _init_cone_json(
    *,
    asof_utc: str,
    symbol: str,
    sessions_per_year: int,
    overlap_method: OverlapMethod,
    horizons: tuple[int, ...],
    percentiles: tuple[float, ...],
    lookbacks_requested: tuple[int, ...],
    lookbacks_used: tuple[int, ...],
    rv_methods: tuple[RVMethod, ...],
) -> dict:
    return {
        "asof_utc": asof_utc,
        "symbol": symbol,
        "sessions_per_year": sessions_per_year,
        "overlap_method": overlap_method,
        "horizons_days": list(horizons),
        "percentiles": list(percentiles),
        "lookbacks_years_requested": list(lookbacks_requested),
        "lookbacks_years_used": list(lookbacks_used),
        "rv_methods": {m: {"lookbacks": {}} for m in rv_methods},
        "notes": [
            "cones computed from session OHLC",
            "rolling windows overlap accepted; quantiles are unadjusted",
            "lookback sampling uses last N RV observations (N = years*sessions_per_year) to keep counts stable",
            "RV alignment performed by session date (normalized timestamps)",
        ],
    }


def _set_json_entry(jsonable: dict, rv_method: RVMethod, lookback_years: int, horizon_days: int, entry: dict) -> None:
    mnode = jsonable["rv_methods"][rv_method]["lookbacks"]
    ykey = str(int(lookback_years))
    hkey = str(int(horizon_days))
    if ykey not in mnode:
        mnode[ykey] = {}
    mnode[ykey][hkey] = entry


def _empty_entry(*, flag: ConeFlag, count: int = 0) -> dict:
    return {
        "count": int(count),
        "min": None,
        "max": None,
        "p": None,
        "rv_today": None,
        "rv_today_percentile_rank": None,
        "flag": flag,
    }


def _row_dict(rv_method: RVMethod, lookback_years: int, horizon_days: int, entry: dict, percentiles: tuple[float, ...]) -> dict:
    row = {
        "rv_method": rv_method,
        "lookback_years": int(lookback_years),
        "horizon_days": int(horizon_days),
        "count": entry["count"],
        "min": entry["min"],
        "max": entry["max"],
        "rv_today": entry["rv_today"],
        "rv_today_percentile_rank": entry["rv_today_percentile_rank"],
        "flag": entry["flag"],
    }
    if entry.get("p"):
        for p in percentiles:
            row[f"p{p:g}"] = entry["p"].get(f"{p:g}")
    else:
        for p in percentiles:
            row[f"p{p:g}"] = None
    return row


def _build_table(rows: list[dict], percentiles: tuple[float, ...]) -> pd.DataFrame:
    if not rows:
        cols = ["count", "min", "max"] + [f"p{p:g}" for p in percentiles] + [
            "rv_today",
            "rv_today_percentile_rank",
            "flag",
        ]
        return pd.DataFrame(columns=cols)

    df = pd.DataFrame(rows)
    df.set_index(["rv_method", "lookback_years", "horizon_days"], inplace=True)
    ordered_cols = ["count", "min", "max"] + [f"p{p:g}" for p in percentiles] + [
        "rv_today",
        "rv_today_percentile_rank",
        "flag",
    ]
    return df[ordered_cols].sort_index()
