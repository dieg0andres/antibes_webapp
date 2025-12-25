"""Transforms and validations for realized volatility inputs.

Responsibilities:
- Normalize candle-like inputs to tz-aware UTC OHLC DataFrames.
- Apply optional adjustment factors (assumed prices are already adjusted).
- Infer bar size with session-aware strictness (only within-session spacing is
  validated; cross-session gaps are expected).
- Aggregate intraday bars to session OHLC indexed by session close converted to UTC.
- Validate OHLC integrity (positive prices, range constraints).
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .calendar import SessionSpec

LOG_PATH = Path("logs") / "options_trading_volatility.log"


def _get_dq_logger() -> logging.Logger:
    """Return a file-logging data-quality logger (idempotent)."""
    logger = logging.getLogger("options_trading.data_quality")
    target = LOG_PATH.resolve()
    keep_first = True
    for h in list(logger.handlers):
        if isinstance(h, logging.FileHandler) and Path(getattr(h, "baseFilename", "")).resolve() == target:
            if keep_first:
                keep_first = False
            else:
                logger.removeHandler(h)
    existing = [
        h for h in logger.handlers if isinstance(h, logging.FileHandler) and Path(getattr(h, "baseFilename", "")).resolve() == target
    ]
    if not existing:
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(target, mode="a", encoding="utf-8")
        fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logger.addHandler(fh)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


def _log_block(logger: logging.Logger, level: int, header: str, lines: list[str]) -> None:
    """Log a header and indented detail lines so each line is timestamped."""
    logger.log(level, header)
    for line in lines:
        logger.log(level, f"  {line}")


def _format_bad_rows(df: pd.DataFrame, mask: pd.Series, n: int = 5) -> list[str]:
    bad = df.loc[mask, ["open", "high", "low", "close"]].head(n)
    lines = []
    for ts, row in bad.iterrows():
        lines.append(
            f"ts={ts} open={row['open']} high={row['high']} low={row['low']} close={row['close']} "
            f"(open-high={row['open']-row['high']:.4g}, low-open={row['low']-row['open']:.4g})"
        )
    more = mask.sum() - len(bad)
    if more > 0:
        lines.append(f"... plus {more} more rows")
    return lines


def _format_bad_deltas(
    index: pd.DatetimeIndex,
    same_session_mask: np.ndarray | pd.Series,
    minutes: pd.Series,
    bad_mask: np.ndarray | pd.Series,
    n: int = 5,
) -> list[str]:
    same = np.asarray(same_session_mask, dtype=bool)
    within_pos = np.flatnonzero(same)
    if within_pos.size == 0:
        return ["(no within-session deltas to validate)"]

    bad_within = np.asarray(getattr(bad_mask, "to_numpy", lambda: bad_mask)(), dtype=bool)
    if bad_within.size != within_pos.size:
        return [f"(unable to align bad mask: bad_within={bad_within.size}, within_pos={within_pos.size})"]

    bad_pos = within_pos[bad_within]
    mins = minutes.to_numpy()

    lines = []
    for k in bad_pos[:n]:
        prev_ts = index[k]
        curr_ts = index[k + 1]
        lines.append(f"prev={prev_ts} curr={curr_ts} delta_min={float(mins[k]):.4g}")
    if bad_pos.size > n:
        lines.append(f"... plus {bad_pos.size - n} more deltas")
    return lines


def ensure_tz_aware_utc_index(df: pd.DataFrame, tz: str = "UTC") -> pd.DataFrame:
    """Return copy with tz-aware UTC index.

    Args:
        df: DataFrame with DatetimeIndex.
        tz: Timezone used to localize naive indices before converting to UTC.

    Returns:
        Copy of df with tz-aware UTC index.

    Raises:
        ValueError: If index is not a DatetimeIndex.
    """
    idx = df.index
    if not isinstance(idx, pd.DatetimeIndex):
        raise ValueError("DataFrame index must be a DatetimeIndex")

    if idx.tz is None:
        localized = idx.tz_localize(tz)
    else:
        localized = idx
    df = df.copy()
    df.index = localized.tz_convert("UTC")
    return df


def candles_to_df(candles_or_payload: Any, tz: str = "UTC", strict: bool = True) -> pd.DataFrame:
    """Normalize candle payload to OHLC DataFrame with UTC index.

    Accepts DataFrame, Schwab-style dict with "candles", or list of candle dicts.
    Prices are expected positive and already adjusted unless an external factor
    is provided later.

    Returns:
        OHLC DataFrame with tz-aware UTC index.

    Raises:
        ValueError: For missing required fields, non-datetime inputs, or
            duplicate timestamps when strict=True.
    """
    if isinstance(candles_or_payload, pd.DataFrame):
        df = candles_or_payload.copy()
    else:
        payload = candles_or_payload
        if isinstance(payload, dict):
            if "candles" not in payload:
                raise ValueError("dict payload must contain 'candles'")
            payload = payload["candles"]
        if isinstance(payload, list):
            df = pd.DataFrame(payload)
        else:
            raise ValueError("Unsupported candles payload type; expected DataFrame, dict, or list")

    if "datetime" in df.columns:
        dt_col = df["datetime"]
        idx = pd.to_datetime(dt_col, unit="ms", utc=True)
        df = df.drop(columns=["datetime"])
    else:
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("Input must include a 'datetime' column or a DatetimeIndex")
        idx = pd.to_datetime(df.index, utc=True)

    df = df.copy()
    df.index = idx
    df.sort_index(inplace=True)

    if df.index.has_duplicates:
        if strict:
            raise ValueError("Duplicate timestamps found")
        df = df[~df.index.duplicated(keep="last")]

    required = {"open", "high", "low", "close"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required OHLC columns: {missing}")

    for col in ["open", "high", "low", "close"]:
        df[col] = df[col].astype(float)

    if "volume" in df.columns:
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce")

    return ensure_tz_aware_utc_index(df, tz=tz)


def apply_adjustment_factor(df: pd.DataFrame, adjustment_factor: pd.Series) -> pd.DataFrame:
    """Multiply OHLC by an aligned adjustment factor Series.

    Returns:
        Adjusted DataFrame.

    Raises:
        ValueError: If alignment introduces missing values.
    """
    if adjustment_factor is None:
        return df

    aligned = adjustment_factor.reindex(df.index)
    if aligned.isna().any():
        raise ValueError("adjustment_factor index must align to data index without missing values")

    df = df.copy()
    for col in ["open", "high", "low", "close"]:
        df[col] = df[col] * aligned
    return df


def infer_bar_minutes(
    index: pd.DatetimeIndex, *, session: SessionSpec | None = None, strict: bool = True, tolerance: float = 0.02
) -> float:
    """Infer bar size (minutes) and validate within-session regularity.

    When a session is supplied, strictness is checked only within a session.
    Cross-session gaps (overnight/weekend) are ignored. strict=False emits a
    warning on within-session irregularities instead of raising.

    Returns:
        Inferred bar size in minutes.

    Raises:
        ValueError: If bar size cannot be inferred or strict validation fails.
    """
    if len(index) < 2:
        raise ValueError("Need at least two timestamps to infer bar size")

    deltas = index.to_series().diff().dropna()
    minutes = deltas.dt.total_seconds() / 60.0
    mode = minutes.mode().iloc[0]

    if session is None:
        if strict:
            deviations = (minutes - mode).abs()
            if not (deviations <= tolerance * mode).all():
                raise ValueError("Irregular timestamp spacing detected beyond tolerance")
        return float(mode)

    session_keys = index.map(session.session_key)
    same_session = session_keys[1:] == session_keys[:-1]
    within_minutes = minutes.iloc[same_session]
    if len(within_minutes) > 0:
        mode = pd.Series(within_minutes).mode().iloc[0]
        deviations = (within_minutes - mode).abs()
        bad_mask = deviations > tolerance * mode
        if bad_mask.any():
            bad_count = int(bad_mask.sum())
            total = len(within_minutes)
            log_hint = f"(see {LOG_PATH.as_posix()})"
            header = (
                f"INTRADAY_IRREGULAR_SPACING strict={strict} mode_minutes={mode:.4g} "
                f"tol={tolerance:.4g} bad_deltas={bad_count}/{total} {log_hint}"
            )
            lines = _format_bad_deltas(index, same_session, minutes, bad_mask, n=5)
            if strict:
                _log_block(_get_dq_logger(), logging.ERROR, header, lines)
                raise ValueError(header)
            warnings.warn(header, UserWarning)
            _log_block(_get_dq_logger(), logging.WARNING, header, lines)

    return float(mode)


def aggregate_to_sessions(df: pd.DataFrame, session: SessionSpec, strict: bool = True) -> pd.DataFrame:
    """Aggregate intraday bars to one OHLC per session, indexed at session close (UTC).

    Raises:
        ValueError: If input index is not datetime or duplicate session indices appear in strict mode.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("aggregate_to_sessions requires a DatetimeIndex")

    key_series = df.index.map(session.session_key)
    grouped = df.copy()
    grouped["_session_key"] = key_series

    aggs = []
    for session_date, chunk in grouped.groupby("_session_key", sort=True):
        chunk = chunk.sort_index()
        record = {
            "open": chunk["open"].iloc[0],
            "high": chunk["high"].max(),
            "low": chunk["low"].min(),
            "close": chunk["close"].iloc[-1],
        }
        if "volume" in chunk.columns:
            record["volume"] = chunk["volume"].sum()

        aggs.append((session_date, record))

    agg_df = pd.DataFrame([r for _, r in aggs])
    agg_index = [session.session_close_timestamp(date) for date, _ in aggs]
    agg_df.index = pd.DatetimeIndex(agg_index, tz="UTC")
    agg_df.sort_index(inplace=True)

    if agg_df.index.has_duplicates:
        if strict:
            raise ValueError("Duplicate session indices after aggregation")
        agg_df = agg_df[~agg_df.index.duplicated(keep="last")]

    return agg_df


def validate_ohlc(df: pd.DataFrame, strict: bool = True) -> None:
    """Validate positivity and range consistency for OHLC data.

    Raises:
        ValueError: On missing columns, non-positive prices, or range violations.
    """
    required = {"open", "high", "low", "close"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required OHLC columns: {missing}")

    ohlc = df[["open", "high", "low", "close"]]
    if (ohlc <= 0).any().any():
        raise ValueError("OHLC values must be positive for log returns")

    if (df["high"] < df["low"]).any():
        raise ValueError("Found high < low")

    open_outside = (df["open"] < df["low"]) | (df["open"] > df["high"])
    close_outside = (df["close"] < df["low"]) | (df["close"] > df["high"])

    if strict:
        if open_outside.any():
            header = f"Found open outside [low, high] for {int(open_outside.sum())} rows."
            detail_lines = _format_bad_rows(df, open_outside)
            _log_block(_get_dq_logger(), logging.ERROR, header, detail_lines)
            raise ValueError(header + "\n" + "\n".join(detail_lines))

        if close_outside.any():
            header = f"Found close outside [low, high] for {int(close_outside.sum())} rows."
            detail_lines = _format_bad_rows(df, close_outside)
            _log_block(_get_dq_logger(), logging.ERROR, header, detail_lines)
            raise ValueError(header + "\n" + "\n".join(detail_lines))
    else:
        if open_outside.any() or close_outside.any():
            log_hint = f"(see {LOG_PATH.as_posix()})"
            msg = (
                f"OHLC_INCONSISTENCY strict=False total_rows={len(df)} "
                f"open_outside={int(open_outside.sum())} close_outside={int(close_outside.sum())} "
                f"{log_hint}"
            )
            warnings.warn(msg, UserWarning)
            detail_lines: list[str] = []
            if open_outside.any():
                detail_lines.append("examples_open:")
                detail_lines.extend(_format_bad_rows(df, open_outside))
            if close_outside.any():
                detail_lines.append("examples_close:")
                detail_lines.extend(_format_bad_rows(df, close_outside))
            _log_block(_get_dq_logger(), logging.WARNING, msg, detail_lines)