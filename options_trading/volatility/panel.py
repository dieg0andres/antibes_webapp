"""Session-level volatility state panel construction.

This module assembles a per-session panel of volatility features (annualized
vol, heat spreads, z-scores, ranks/percentiles, tail frequencies, and a light
regime/correlation view). It prefers realized variance when available on shorter
windows and falls back to Yang–Zhang otherwise. Outputs are indexed by session
close timestamp (UTC) for consistent alignment with downstream analytics and
dashboards.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from .calendar import EQUITIES_RTH, TradingCalendar
from .realized import realized_vol
from .transforms import (
    aggregate_to_sessions,
    apply_adjustment_factor,
    candles_to_df,
    validate_ohlc,
)


@dataclass
class VolatilityStatePanelResult:
    """Container for the volatility state panel."""

    series: pd.DataFrame
    latest: pd.Series
    settings: dict
    coverage: dict


def _safe_log(s: pd.Series) -> pd.Series:
    """Return log(s) with safety: values <= 0 become NaN."""
    s = s.where(s > 0)
    return np.log(s)


def _rolling_rank_pct(series: pd.Series, window: int) -> tuple[pd.Series, pd.Series]:
    """Return rolling rank and percentile over a moving window.

    rank_t = (x_t - min_window) / (max_window - min_window); NaN if denom == 0
    pct_t  = mean(x_window <= x_t) over the window (inclusive)
    """

    def _rank(arr: np.ndarray) -> float:
        s = pd.Series(arr)
        cur = s.iloc[-1]
        lo = s.min()
        hi = s.max()
        denom = hi - lo
        if denom == 0 or math.isnan(denom):
            return math.nan
        return float((cur - lo) / denom)

    def _pct(arr: np.ndarray) -> float:
        s = pd.Series(arr)
        cur = s.iloc[-1]
        return float((s <= cur).mean())

    roller = series.rolling(window=window, min_periods=window)
    return roller.apply(_rank, raw=True), roller.apply(_pct, raw=True)


def _coverage_for_columns(df: pd.DataFrame, cols: Iterable[str]) -> dict:
    summary = {}
    for col in cols:
        if col not in df.columns:
            continue
        s = df[col]
        non_nan = int(s.count())
        first_valid = s.first_valid_index()
        last_valid = s.last_valid_index()
        summary[col] = {
            "non_nan_count": non_nan,
            "first_valid_timestamp": first_valid,
            "last_valid_timestamp": last_valid,
        }
    return summary


def volatility_state_panel(
    *,
    session_data: pd.DataFrame | dict | list,
    intraday_data: pd.DataFrame | dict | list | None = None,
    calendar: TradingCalendar = EQUITIES_RTH,
    rv_windows: tuple[int, ...] = (10, 20, 60, 120),
    primary_max_window: int = 120,
    include_realized_variance: bool = True,
    include_overnight: bool = True,
    annualize: bool = True,
    strict: bool = True,
    adjustment_factor: pd.Series | None = None,
    heat_pairs: tuple[tuple[int, int], ...] = ((20, 120), (10, 60)),
    z_lookback: int = 252,
    z_windows: tuple[int, ...] = (20,),
    rank_lookback: int = 252,
    rank_windows: tuple[int, ...] = (20,),
    include_rank_percentile: bool = True,
    tail_sigma_window: int = 60,
    tail_lookback: int = 252,
    corr_window: int = 120,
    corr_vol_window: int = 20,
    include_conditional_betas: bool = True,
    conditional_beta_min_count: int = 20,
    include_regime: bool = True,
    trend_ma_window: int = 200,
) -> VolatilityStatePanelResult:
    """
    Build a session-level volatility state panel.

    Parameters
    ----------
    session_data : DataFrame | dict | list
        Session or daily candles. Will be aggregated to session closes via
        `aggregate_to_sessions`; index becomes session close UTC.
    intraday_data : DataFrame | dict | list | None
        Optional intraday candles for realized variance; ignored if None or
        include_realized_variance=False.
    calendar : TradingCalendar
        Trading calendar preset (session hours + annualization inputs).
    rv_windows : tuple[int, ...]
        Rolling window lengths (bars/sessions) for realized vol.
    primary_max_window : int
        For windows <= this threshold, prefer realized_variance when present;
        otherwise fall back to Yang–Zhang. Longer windows always use YZ.
    include_realized_variance : bool
        Compute realized_variance (intraday path) when intraday_data provided.
    include_overnight : bool
        Whether overnight gap contributes to realized_variance.
    annualize : bool
        Annualize vol outputs via sqrt(bars_per_year).
    strict : bool
        Passed through to validators/ingestion; enforces timestamp regularity and
        OHLC range checks.
    adjustment_factor : pd.Series | None
        Optional multiplicative factor applied to OHLC before computations.
    heat_pairs : tuple[tuple[int, int], ...]
        Pairs of windows (a,b) to compute heat = ln(vol_a) - ln(vol_b).
    z_lookback : int
        Lookback for z-score of log vol (YZ only).
    z_windows : tuple[int, ...]
        Windows (subset of rv_windows) to compute z-scores on YZ vol.
    rank_lookback : int
        Lookback for rank/percentile features (YZ only).
    rank_windows : tuple[int, ...]
        Windows (subset of rv_windows) for rank/percentile.
    include_rank_percentile : bool
        Whether to compute rank/percentile features.
    tail_sigma_window : int
        Window for baseline sigma (ret_log std, ddof=0).
    tail_lookback : int
        Window for tail hit frequency.
    corr_window : int
        Rolling window for ret vs dlogvol correlation.
    corr_vol_window : int
        Vol window (must be in rv_windows) used for dlogvol in correlation.
    include_conditional_betas : bool
        Whether to compute conditional betas (neg/pos returns) on ret vs dlogvol.
    conditional_beta_min_count : int
        Minimum subset size within the rolling window to report beta; clamped to
        corr_window internally.
    include_regime : bool
        Whether to compute simple SMA-based bull/bear indicator.
    trend_ma_window : int
        SMA window for regime detection.

    Returns
    -------
    VolatilityStatePanelResult
        series : DataFrame indexed by session close UTC with all features
        latest : last row of the series
        settings : dict of configuration used
        coverage : dict summarizing data availability per key series

    Raises
    ------
    ValueError
        On window validation errors, missing required columns, or invalid
        aggregation/overlap choices.

    Notes
    -----
    - Primary selection rule: for W <= primary_max_window, use realized variance
      when available; otherwise fall back to Yang–Zhang. For W > primary_max_window,
      always use Yang–Zhang.
    - Z-scores and rank/percentile are computed on Yang–Zhang only for stability
      and longer history (does not depend on intraday coverage).
    - NaNs indicate warmup, insufficient history, or missing intraday coverage.
    - Formulas:
        * ret_log_t = ln(C_t / C_{t-1})
        * heat(a,b) = ln(vol_a) - ln(vol_b)
        * zlogvol = (ln(vol_t) - mean_ln_vol) / std_ln_vol over lookback
        * rv_rank = (vol_t - min_window) / (max_window - min_window)
        * rv_pct = mean(vol_window <= vol_t) over lookback
        * tail2/3 freq = rolling mean of 1{|ret| > k*sigma_ref}
        * corr = rolling corr(ret_log, dlogvol)
        * dlogvol_t = ln(vol_yz_vw,t) - ln(vol_yz_vw,t-1)
        * beta_S = cov_S(ret_log, dlogvol) / var_S(ret_log) for subset S (ret<0 or ret>0),
          with intercept: cov = sxy - (sx*sy)/n, var = sx2 - (sx*sx)/n; counts reported as
          n_ret_neg_cw / n_ret_pos_cw; beta set to NaN when count < min_count or var<=0.
    """

    rv_windows_set = set(rv_windows)
    if not set(z_windows).issubset(rv_windows_set):
        raise ValueError("z_windows must be a subset of rv_windows")
    if include_rank_percentile and not set(rank_windows).issubset(rv_windows_set):
        raise ValueError("rank_windows must be a subset of rv_windows")
    if corr_vol_window not in rv_windows_set:
        raise ValueError("corr_vol_window must be in rv_windows")
    if conditional_beta_min_count < 2:
        raise ValueError("conditional_beta_min_count must be at least 2")
    eff_min_beta = min(conditional_beta_min_count, corr_window)

    # Ingest session-level data
    session_df = candles_to_df(session_data, strict=strict)
    session_df = apply_adjustment_factor(session_df, adjustment_factor)
    validate_ohlc(session_df, strict=strict)
    session_df = aggregate_to_sessions(session_df, calendar.session, strict=strict)

    panel = pd.DataFrame(index=session_df.index)
    panel["close"] = session_df["close"]
    panel["ret_log"] = np.log(panel["close"] / panel["close"].shift(1))

    # Yang–Zhang vols (always computed)
    vol_yz_cols = []
    for w in rv_windows:
        yz_res = realized_vol(
            session_df,
            method="yang_zhang",
            window=w,
            calendar=calendar,
            aggregation="session",
            annualize=annualize,
            assume_mean_zero=True,
            strict=strict,
            adjustment_factor=None,
        )
        col = f"vol_yz_{w}"
        panel[col] = yz_res.rv
        vol_yz_cols.append(col)

    # Realized variance vols (optional, needs intraday)
    vol_rvar_cols = []
    if include_realized_variance and intraday_data is not None:
        intraday_df = candles_to_df(intraday_data, strict=strict)
        intraday_df = apply_adjustment_factor(intraday_df, adjustment_factor)
        validate_ohlc(intraday_df, strict=strict)
        for w in rv_windows:
            rvar_res = realized_vol(
                intraday_df,
                method="realized_variance",
                window=w,
                calendar=calendar,
                annualize=annualize,
                include_overnight=include_overnight,
                strict=strict,
                adjustment_factor=None,
            )
            col = f"vol_rvar_{w}"
            panel[col] = rvar_res.rv  # index already session closes
            vol_rvar_cols.append(col)

    # Primary selection
    primary_cols = []
    primary_src_cols = []
    for w in rv_windows:
        yz_col = f"vol_yz_{w}"
        rvar_col = f"vol_rvar_{w}"
        primary_col = f"vol_primary_{w}"
        src_col = f"primary_src_{w}"
        use_rvar = (w <= primary_max_window) and (rvar_col in panel.columns)
        if use_rvar:
            chosen = panel[rvar_col]
            fallback = panel[yz_col]
            panel[primary_col] = chosen.where(~chosen.isna(), fallback)
            panel[src_col] = np.where(chosen.notna(), "rvar", "yz")
        else:
            panel[primary_col] = panel[yz_col]
            panel[src_col] = "yz"
        primary_cols.append(primary_col)
        primary_src_cols.append(src_col)

    # Heat features
    heat_cols = []
    for a, b in heat_pairs:
        if a not in rv_windows_set or b not in rv_windows_set:
            raise ValueError(f"Heat pair ({a},{b}) must be in rv_windows")
        col_yz = f"heat_yz_{a}_{b}"
        col_p = f"heat_primary_{a}_{b}"
        panel[col_yz] = _safe_log(panel[f"vol_yz_{a}"]) - _safe_log(panel[f"vol_yz_{b}"])
        panel[col_p] = _safe_log(panel[f"vol_primary_{a}"]) - _safe_log(panel[f"vol_primary_{b}"])
        heat_cols.extend([col_yz, col_p])

    # Z-scores on YZ
    z_cols = []
    for w in z_windows:
        x = _safe_log(panel[f"vol_yz_{w}"])
        mu = x.rolling(window=z_lookback, min_periods=z_lookback).mean()
        sd = x.rolling(window=z_lookback, min_periods=z_lookback).std(ddof=0)
        sd = sd.replace(0.0, np.nan)
        z = (x - mu) / sd
        col = f"zlogvol_yz_{w}_{z_lookback}"
        panel[col] = z
        z_cols.append(col)

    # Rank / Percentile on YZ
    rank_cols = []
    pct_cols = []
    if include_rank_percentile:
        for w in rank_windows:
            s = panel[f"vol_yz_{w}"]
            r, p = _rolling_rank_pct(s, window=rank_lookback)
            rank_col = f"rv_rank_yz_{w}_{rank_lookback}"
            pct_col = f"rv_pct_yz_{w}_{rank_lookback}"
            panel[rank_col] = r
            panel[pct_col] = p
            rank_cols.append(rank_col)
            pct_cols.append(pct_col)

    # Tail features
    tail_cols = []
    sigma_ref_col = f"sigma_ref_{tail_sigma_window}"
    sigma_ref = panel["ret_log"].rolling(window=tail_sigma_window, min_periods=tail_sigma_window).std(ddof=0)
    panel[sigma_ref_col] = sigma_ref
    tail2 = panel["ret_log"].abs() > 2 * sigma_ref
    tail3 = panel["ret_log"].abs() > 3 * sigma_ref
    p_tail2_col = f"p_tail2_{tail_lookback}"
    p_tail3_col = f"p_tail3_{tail_lookback}"
    panel[p_tail2_col] = tail2.rolling(window=tail_lookback, min_periods=tail_lookback).mean()
    panel[p_tail3_col] = tail3.rolling(window=tail_lookback, min_periods=tail_lookback).mean()
    tail_cols.extend([sigma_ref_col, p_tail2_col, p_tail3_col])

    # Leverage effect proxy: corr(ret_log, dlogvol) on YZ
    leverage_cols = []
    dlogvol_col = f"dlogvol_yz_{corr_vol_window}"
    corr_col = f"corr_ret_dlogvol_yz{corr_vol_window}_{corr_window}"
    dlogvol = _safe_log(panel[f"vol_yz_{corr_vol_window}"]).diff()
    panel[dlogvol_col] = dlogvol
    panel[corr_col] = (
        panel["ret_log"].rolling(window=corr_window).corr(dlogvol)
    )
    leverage_cols.extend([dlogvol_col, corr_col])

    # Conditional betas (with intercept) on ret vs dlogvol
    cond_beta_cols = []
    if include_conditional_betas:
        x = panel["ret_log"]
        y = panel[dlogvol_col]
        valid = x.notna() & y.notna()
        neg = valid & (x < 0)
        pos = valid & (x > 0)

        def _rolling_beta(mask: pd.Series) -> tuple[pd.Series, pd.Series]:
            n = mask.astype(int).rolling(corr_window, min_periods=corr_window).sum()
            sx = x.where(mask, 0.0).rolling(corr_window, min_periods=corr_window).sum()
            sy = y.where(mask, 0.0).rolling(corr_window, min_periods=corr_window).sum()
            sx2 = (x.where(mask, 0.0) ** 2).rolling(corr_window, min_periods=corr_window).sum()
            sxy = (x.where(mask, 0.0) * y.where(mask, 0.0)).rolling(corr_window, min_periods=corr_window).sum()
            cov = sxy - (sx * sy) / n
            var = sx2 - (sx * sx) / n
            beta = cov / var
            beta = beta.where((n >= eff_min_beta) & (var > 0))
            return beta, n

        beta_neg, n_neg = _rolling_beta(neg)
        beta_pos, n_pos = _rolling_beta(pos)

        beta_neg_col = f"beta_ret_dlogvol_yz{corr_vol_window}_neg_{corr_window}"
        beta_pos_col = f"beta_ret_dlogvol_yz{corr_vol_window}_pos_{corr_window}"
        n_neg_col = f"n_ret_neg_{corr_window}"
        n_pos_col = f"n_ret_pos_{corr_window}"

        panel[beta_neg_col] = beta_neg
        panel[beta_pos_col] = beta_pos
        panel[n_neg_col] = n_neg
        panel[n_pos_col] = n_pos

        cond_beta_cols.extend([beta_neg_col, beta_pos_col, n_neg_col, n_pos_col])

    # Regime (simple SMA)
    regime_cols = []
    if include_regime:
        sma_col = f"sma_{trend_ma_window}"
        bull_col = f"is_bull_{trend_ma_window}"
        panel[sma_col] = panel["close"].rolling(window=trend_ma_window, min_periods=trend_ma_window).mean()
        panel[bull_col] = panel["close"] > panel[sma_col]
        regime_cols.extend([sma_col, bull_col])

    settings = {
        "calendar": calendar.name,
        "rv_windows": rv_windows,
        "primary_max_window": primary_max_window,
        "include_realized_variance": include_realized_variance,
        "include_overnight": include_overnight,
        "annualize": annualize,
        "strict": strict,
        "heat_pairs": heat_pairs,
        "z_lookback": z_lookback,
        "z_windows": z_windows,
        "rank_lookback": rank_lookback,
        "rank_windows": rank_windows,
        "include_rank_percentile": include_rank_percentile,
        "tail_sigma_window": tail_sigma_window,
        "tail_lookback": tail_lookback,
        "corr_window": corr_window,
        "corr_vol_window": corr_vol_window,
        "include_conditional_betas": include_conditional_betas,
        "conditional_beta_min_count": conditional_beta_min_count,
        "include_regime": include_regime,
        "trend_ma_window": trend_ma_window,
        "adjustment_factor_provided": adjustment_factor is not None,
        "adjustment_factor_name": getattr(adjustment_factor, "name", None) if adjustment_factor is not None else None,
    }

    coverage_cols = (
        vol_yz_cols
        + vol_rvar_cols
        + primary_cols
        + heat_cols
        + z_cols
        + rank_cols
        + pct_cols
        + tail_cols
        + leverage_cols
        + cond_beta_cols
        + regime_cols
    )
    coverage = _coverage_for_columns(panel, coverage_cols)

    latest = panel.iloc[-1] if not panel.empty else pd.Series(dtype=float)

    return VolatilityStatePanelResult(series=panel, latest=latest, settings=settings, coverage=coverage)

