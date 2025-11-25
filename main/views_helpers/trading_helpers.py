from __future__ import annotations

import pickle
from typing import List

import pandas as pd
from django.conf import settings
from django.core.cache import cache


def _stale_context(message: str) -> dict:
    return {
        "is_stale": True,
        "open_pnl": 0.0,
        "pnl_class": "neutral",
        "open_positions": [],
        "error_message": message,
    }


def _load_cached_dataframe():
    cache_key = getattr(settings, "TRADING_LOG_CACHE_KEY", None)
    if not cache_key:
        return None, "Trading log cache key is not configured."

    payload = cache.get(cache_key)
    if not payload:
        return None, "Trading log data is not available yet. Run the update task."

    try:
        df = pickle.loads(payload)
    except Exception:
        return None, "Cached trading log data could not be read. Please refresh the log."

    if not isinstance(df, pd.DataFrame):
        return None, "Cached trading log data is invalid."

    return df, None


def _compute_status_pnl(df: pd.DataFrame, status_value: str) -> float:
    if df.empty:
        return 0.0

    required = ("Price", "Qty", "Multiplier", "STATUS")
    if any(column not in df.columns for column in required):
        return 0.0

    mask = df["STATUS"].astype(str).str.strip().eq(status_value)
    rows = df.loc[mask]
    if rows.empty:
        return 0.0

    prices = pd.to_numeric(rows["Price"], errors="coerce")
    qtys = pd.to_numeric(rows["Qty"], errors="coerce").abs()
    multipliers = pd.to_numeric(rows["Multiplier"], errors="coerce")

    pnl_series = prices * multipliers * qtys
    pnl = pnl_series.sum(skipna=True)
    return float(pnl) if pd.notna(pnl) else 0.0


def _build_positions_preview(df: pd.DataFrame, limit: int = 6) -> List[dict]:
    if df.empty or "STATUS" not in df.columns:
        return []

    mask = df["STATUS"].astype(str).str.strip().eq("Open")
    open_rows = df.loc[mask]
    if open_rows.empty:
        return []

    preview_rows = open_rows.head(limit)
    preview = []
    for _, row in preview_rows.iterrows():
        preview.append(
            {
                "ticker": str(row.get("TICKER", "")).strip(),
                "qty": row.get("Qty", ""),
                "price": row.get("Price", ""),
                "order_action": row.get("OrderAction", ""),
            }
        )
    return preview


def _compute_total_fees(df: pd.DataFrame) -> float:
    if df.empty or "Fees" not in df.columns:
        return 0.0
    fees = pd.to_numeric(df["Fees"], errors="coerce")
    total = fees.sum(skipna=True)
    return float(total) if pd.notna(total) else 0.0


def build_trading_dashboard_context() -> dict:
    df, error = _load_cached_dataframe()
    if df is None:
        return _stale_context(error or "Trading data is unavailable.")

    open_pnl = _compute_status_pnl(df, "Open")
    realized_pnl = _compute_status_pnl(df, "Closed")
    total_fees = _compute_total_fees(df)
    positions_preview = _build_positions_preview(df)
    pnl_class = "positive" if open_pnl >= 0 else "negative"
    realized_class = "positive" if realized_pnl >= 0 else "negative"

    return {
        "is_stale": False,
        "open_pnl": open_pnl,
        "pnl_class": pnl_class,
        "realized_pnl": realized_pnl,
        "realized_class": realized_class,
        "total_fees": total_fees,
        "open_positions": positions_preview,
        "error_message": None,
    }

