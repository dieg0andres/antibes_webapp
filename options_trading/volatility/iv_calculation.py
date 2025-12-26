"""ATM implied volatility (IVX) calculation with Schwab chain normalization.

This module computes constant-maturity ATM implied volatility (annualized
decimal, e.g., 0.18 = 18%) by inverting Black–Scholes on option quotes. It
selects the ATM strike nearest to the forward F = S*exp((r-q)T), averages call
and put IV when available, and linearly interpolates in total variance
(TV = sigma^2 * T) when the target maturity lies between expiries.

Vendor IVs (if provided) are optional and used for comparison only.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np
import pandas as pd

from options_trading.pricing.black_scholes import BSParams, implied_vol


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class ATMOptionIV:
    right: str
    expiry: str
    dte_days: int
    strike: float
    bid: float | None
    ask: float | None
    mark: float | None
    mid: float | None
    price_used: float | None
    iv_calc: float | None
    iv_vendor: float | None
    vendor_iv_raw: float | None
    spread: float | None
    notes: list[str]


@dataclass
class ExpiryATMIV:
    expiry: str
    dte_days: int
    T_years: float
    forward: float
    atm_strike: float
    call: Optional[ATMOptionIV]
    put: Optional[ATMOptionIV]
    iv_calc_atm: float | None
    iv_vendor_atm: float | None
    notes: list[str]


@dataclass
class IVXATMResult:
    symbol: str | None
    as_of: pd.Timestamp | None
    underlying_price: float
    interest_rate: float
    dividend_yield: float
    target_dte_days: float
    target_T_years: float
    day_count: int
    iv_source: str
    price_source: str
    atm_method: str
    interpolate: str
    ivx_calc: float | None
    ivx_vendor: float | None
    ivx_diff: float | None
    expiry_lower: Optional[ExpiryATMIV]
    expiry_upper: Optional[ExpiryATMIV]
    tv_lower: float | None
    tv_upper: float | None
    tv_target: float | None
    flags: dict
    notes: list[str]


# ---------------------------------------------------------------------------
# Normalization helpers
# ---------------------------------------------------------------------------


def _to_decimal_rate(x: float | None) -> float | None:
    if x is None:
        return None
    return x / 100.0 if x > 1.0 else x


def normalize_schwab_chain(chain: dict) -> tuple[pd.DataFrame, dict]:
    """
    Convert Schwab option chain JSON dict into a normalized quotes DataFrame plus metadata.

    Output DataFrame columns:
      - expiry (str "YYYY-MM-DD")
      - dte (int)
      - right ("call" or "put")
      - strike (float)
      - bid (float|nan)
      - ask (float|nan)
      - mark (float|nan)
      - vendor_iv (float|nan)  # annualized decimal if present

    Metadata dict keys:
      - symbol (str|None)
      - underlying_price (float)
      - interest_rate (float, decimal)
      - dividend_yield (float, decimal)
      - as_of (pd.Timestamp|None)
    """

    def _clean_vendor_iv(v):
        if v is None:
            return np.nan
        try:
            v = float(v)
        except (TypeError, ValueError):
            return np.nan
        if not math.isfinite(v) or v <= 0:
            return np.nan
        # Schwab vendor IV is often in percent, so convert to decimal
        return v / 100.0 if v > 1.0 else v

    rows = []
    for right_key in ("callExpDateMap", "putExpDateMap"):
        mapping = chain.get(right_key) or {}
        right = "call" if right_key.startswith("call") else "put"
        for exp_key, strike_map in mapping.items():
            # exp_key format: "YYYY-MM-DD:DTE"
            try:
                expiry_str, dte_str = exp_key.split(":")
                dte_val = int(dte_str)
            except Exception:
                continue
            for strike_key, options_list in strike_map.items():
                if not options_list:
                    continue
                opt = options_list[0]
                bid = opt.get("bid")
                ask = opt.get("ask")
                mark = opt.get("mark")
                strike = float(opt.get("strikePrice")) if opt.get("strikePrice") is not None else float(strike_key)
                vendor_iv = _clean_vendor_iv(opt.get("volatility"))
                rows.append(
                    {
                        "expiry": expiry_str,
                        "dte": dte_val,
                        "right": right,
                        "strike": strike,
                        "bid": bid if bid is not None else np.nan,
                        "ask": ask if ask is not None else np.nan,
                        "mark": mark if mark is not None else np.nan,
                        "vendor_iv": vendor_iv,
                    }
                )

    df = pd.DataFrame(rows)
    meta = {
        "symbol": chain.get("symbol"),
        "underlying_price": chain.get("underlyingPrice"),
        "interest_rate": _to_decimal_rate(chain.get("interestRate")),
        "dividend_yield": _to_decimal_rate(chain.get("dividendYield")),
        "as_of": pd.to_datetime(chain.get("asOfDate"), utc=True) if chain.get("asOfDate") else None,
    }
    return df, meta


# ---------------------------------------------------------------------------
# Core calculations
# ---------------------------------------------------------------------------


def _mid_price(bid: float | None, ask: float | None) -> float | None:
    if bid is None or ask is None:
        return None
    if not math.isfinite(bid) or not math.isfinite(ask):
        return None
    if bid <= 0 or ask <= 0 or ask < bid:
        return None
    return 0.5 * (bid + ask)


def _price_for_inversion(row: pd.Series, price_source: Literal["mid_then_mark", "mark"]) -> float | None:
    if price_source == "mark":
        return row["mark"] if row.get("mark", 0) and row["mark"] > 0 else None
    mid = _mid_price(row.get("bid"), row.get("ask"))
    if mid is not None:
        return mid
    return row["mark"] if row.get("mark", 0) and row["mark"] > 0 else None


def _forward_price(S: float, r: float, q: float, T: float) -> float:
    return float(S * math.exp((r - q) * T))


def _choose_atm_strike(df_expiry: pd.DataFrame, forward: float) -> float:
    strikes = df_expiry["strike"].unique()
    idx = np.argmin(np.abs(strikes - forward))
    return float(strikes[idx])


def _invert_iv(
    price: float | None,
    S: float,
    K: float,
    T: float,
    r: float,
    q: float,
    right: Literal["call", "put"],
    sigma_low: float,
    sigma_high: float,
    tol: float,
    max_iter: int,
) -> tuple[float | None, list[str]]:
    if price is None or not math.isfinite(price) or price <= 0:
        return None, ["no valid price"]
    try:
        params = BSParams(S=S, K=K, T=T, sigma=0.2, type=right, r=r, q=q)
        iv_val = implied_vol(
            target_price=price,
            params=params,
            tol=tol,
            max_iter=max_iter,
            sigma_low=sigma_low,
            sigma_high=sigma_high,
        )
        return iv_val, []
    except ValueError as e:
        return None, [str(e)]


def _extract_option_row(df: pd.DataFrame, expiry: str, strike: float, right: str) -> pd.Series | None:
    subset = df[(df["expiry"] == expiry) & (df["strike"] == strike) & (df["right"] == right)]
    if subset.empty:
        return None
    if {"bid", "ask"}.issubset(subset.columns):
        subset = subset.assign(spread=subset["ask"] - subset["bid"])
        subset = subset.sort_values(["spread", "strike"]).head(1)
    return subset.iloc[0]


def _atm_iv_for_expiry(
    df: pd.DataFrame,
    expiry: str,
    dte_days: int,
    S: float,
    r: float,
    q: float,
    price_source: Literal["mid_then_mark", "mark"],
    sigma_low: float,
    sigma_high: float,
    tol: float,
    max_iter: int,
    atm_method: Literal["closest_forward"],
    day_count: int,
) -> ExpiryATMIV:
    T = dte_days / float(day_count)
    forward = _forward_price(S, r, q, T)
    df_expiry = df[df["expiry"] == expiry]
    atm_strike = _choose_atm_strike(df_expiry, forward)

    notes: list[str] = []
    call_iv = put_iv = call_vendor = put_vendor = None
    call_row = _extract_option_row(df, expiry, atm_strike, "call")
    put_row = _extract_option_row(df, expiry, atm_strike, "put")

    def _build(row: pd.Series | None, right: str) -> Optional[ATMOptionIV]:
        if row is None:
            return None
        price_used = _price_for_inversion(row, price_source)
        iv_calc, iv_notes = _invert_iv(price_used, S, atm_strike, T, r, q, right, sigma_low, sigma_high, tol, max_iter)
        iv_vendor = row.get("vendor_iv") if "vendor_iv" in row else None
        vendor_raw = row.get("vendor_iv")
        spread = None
        if "bid" in row and "ask" in row and row["bid"] > 0 and row["ask"] > 0:
            spread = row["ask"] - row["bid"]
        return ATMOptionIV(
            right=right,
            expiry=expiry,
            dte_days=dte_days,
            strike=atm_strike,
            bid=row.get("bid"),
            ask=row.get("ask"),
            mark=row.get("mark"),
            mid=_mid_price(row.get("bid"), row.get("ask")),
            price_used=price_used,
            iv_calc=iv_calc,
            iv_vendor=iv_vendor,
            vendor_iv_raw=vendor_raw,
            spread=spread,
            notes=iv_notes,
        )

    call_obj = _build(call_row, "call")
    put_obj = _build(put_row, "put")

    iv_list = [x.iv_calc for x in (call_obj, put_obj) if x and x.iv_calc is not None]
    iv_vendor_list = [x.iv_vendor for x in (call_obj, put_obj) if x and x.iv_vendor is not None]

    iv_calc_atm = float(np.mean(iv_list)) if iv_list else None
    iv_vendor_atm = float(np.mean(iv_vendor_list)) if iv_vendor_list else None

    return ExpiryATMIV(
        expiry=expiry,
        dte_days=dte_days,
        T_years=T,
        forward=forward,
        atm_strike=atm_strike,
        call=call_obj,
        put=put_obj,
        iv_calc_atm=iv_calc_atm,
        iv_vendor_atm=iv_vendor_atm,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def ivx_atm(
    chain: dict | pd.DataFrame,
    *,
    target_dte_days: float = 30.0,
    underlying_price: float | None = None,
    interest_rate: float | None = None,
    dividend_yield: float | None = None,
    symbol: str | None = None,
    as_of: pd.Timestamp | None = None,
    iv_source: Literal["invert_mid_then_mark", "vendor"] = "invert_mid_then_mark",
    compare_vendor: bool = True,
    vendor_mismatch_threshold: float = 0.05,
    atm_method: Literal["closest_forward"] = "closest_forward",
    price_source: Literal["mid_then_mark", "mark"] = "mid_then_mark",
    day_count: int = 365,
    interpolate: Literal["total_variance"] = "total_variance",
    sigma_low: float = 1e-6,
    sigma_high: float = 5.0,
    tol: float = 1e-6,
    max_iter: int = 100,
) -> IVXATMResult:
    """
    Compute constant-maturity ATM implied volatility IVX at target_dte_days.

    Steps:
    - Normalize chain (Schwab dict) or use provided DataFrame.
    - For each expiry, choose ATM strike nearest forward F = S*exp((r-q)T).
    - Invert BS IV for call/put using price_source (mid then mark by default).
    - ATM IV per expiry = mean of available call/put IVs.
    - If target maturity lies between expiries, interpolate linearly in total variance:
        TV = sigma^2 * T, TV* = TV1 + (T*-T1)/(T2-T1)*(TV2-TV1), IVX = sqrt(TV*/T*).

    IV values are annualized decimals (0.18 = 18%).
    """

    if isinstance(chain, pd.DataFrame):
        df = chain.copy()
        meta = {
            "symbol": symbol or df.attrs.get("symbol"),
            "underlying_price": underlying_price if underlying_price is not None else df.attrs.get("underlying_price"),
            "interest_rate": _to_decimal_rate(interest_rate if interest_rate is not None else df.attrs.get("interest_rate")),
            "dividend_yield": _to_decimal_rate(dividend_yield if dividend_yield is not None else df.attrs.get("dividend_yield")),
            "as_of": as_of or df.attrs.get("as_of"),
        }
    else:
        df, meta_raw = normalize_schwab_chain(chain)
        meta = {
            "symbol": symbol or meta_raw.get("symbol"),
            "underlying_price": underlying_price if underlying_price is not None else meta_raw.get("underlying_price"),
            "interest_rate": _to_decimal_rate(interest_rate if interest_rate is not None else meta_raw.get("interest_rate")),
            "dividend_yield": _to_decimal_rate(dividend_yield if dividend_yield is not None else meta_raw.get("dividend_yield")),
            "as_of": as_of or meta_raw.get("as_of"),
        }

    missing = [k for k in ("underlying_price", "interest_rate", "dividend_yield") if meta.get(k) is None]
    if missing:
        raise ValueError(
            f"Missing required metadata fields: {missing}. Provide explicitly or via chain/df.attrs."
        )

    S = float(meta["underlying_price"])
    r = float(meta["interest_rate"])
    q = float(meta["dividend_yield"])
    symbol_out = meta.get("symbol")

    target_T = target_dte_days / day_count
    notes: list[str] = []

    # Prepare expiries
    if df.empty:
        raise ValueError("Empty option chain")
    if "dte" not in df.columns or "expiry" not in df.columns:
        raise ValueError("Chain/DataFrame must include 'expiry' and 'dte'")

    expiries = (
        df[["expiry", "dte"]].drop_duplicates().sort_values("dte").assign(T=lambda x: x["dte"] / day_count)
    )

    expiry_objs: list[ExpiryATMIV] = []
    for _, row in expiries.iterrows():
        expiry_objs.append(
            _atm_iv_for_expiry(
                df=df,
                expiry=row["expiry"],
                dte_days=int(row["dte"]),
                S=S,
                r=r,
                q=q,
                price_source=price_source,
                sigma_low=sigma_low,
                sigma_high=sigma_high,
                tol=tol,
                max_iter=max_iter,
                atm_method=atm_method,
                day_count=day_count,
            )
        )

    # Find bracketing expiries
    expiry_objs = sorted(expiry_objs, key=lambda e: e.T_years)
    lower = upper = None
    for e in expiry_objs:
        if abs(e.T_years - target_T) < 1e-9:
            lower = upper = e
            break
        if e.T_years <= target_T:
            lower = e
        if e.T_years >= target_T and upper is None:
            upper = e
            break

    # Handle extrapolation edges
    flags = {"EXACT_MATCH": False, "EXTRAPOLATED": False, "VENDOR_MISMATCH": False}
    if lower is not None and upper is not None and lower is upper:
        flags["EXACT_MATCH"] = True
    elif lower is None:
        lower = upper
        flags["EXTRAPOLATED"] = True
    elif upper is None:
        upper = lower
        flags["EXTRAPOLATED"] = True

    def _ivx_from_two(e1: ExpiryATMIV, e2: ExpiryATMIV) -> tuple[float | None, float | None, float | None]:
        if e1 is None or e2 is None:
            return None, None, None
        iv1 = e1.iv_calc_atm if iv_source == "invert_mid_then_mark" else e1.iv_vendor_atm
        iv2 = e2.iv_calc_atm if iv_source == "invert_mid_then_mark" else e2.iv_vendor_atm
        if iv1 is None or iv2 is None:
            return None, None, None
        tv1 = iv1 * iv1 * e1.T_years
        tv2 = iv2 * iv2 * e2.T_years
        if e1.T_years == e2.T_years:
            tv_target = tv1
        else:
            tv_target = tv1 + (target_T - e1.T_years) / (e2.T_years - e1.T_years) * (tv2 - tv1)
        ivx = math.sqrt(tv_target / target_T) if tv_target is not None else None
        return ivx, tv1, tv2

    ivx_calc = ivx_vendor = ivx_diff = None
    tv_lower = tv_upper = tv_target = None

    if flags["EXTRAPOLATED"]:
        nearest = lower or upper
        if nearest is not None:
            iv_sel = nearest.iv_calc_atm if iv_source == "invert_mid_then_mark" else nearest.iv_vendor_atm
            if iv_sel is not None:
                ivx_calc = iv_sel
                tv_lower = tv_upper = iv_sel * iv_sel * nearest.T_years
                tv_target = iv_sel * iv_sel * target_T
        if compare_vendor and nearest is not None:
            vendor_iv = nearest.iv_vendor_atm
            if vendor_iv is not None:
                ivx_vendor = vendor_iv
                if ivx_calc is not None:
                    ivx_diff = ivx_calc - ivx_vendor
                    if abs(ivx_diff) > vendor_mismatch_threshold:
                        flags["VENDOR_MISMATCH"] = True
    else:
        if lower is not None and upper is not None:
            ivx_calc, tv_lower, tv_upper = _ivx_from_two(lower, upper)
            tv_target = None
            if tv_lower is not None and tv_upper is not None and lower is not upper:
                tv_target = tv_lower + (target_T - lower.T_years) / (upper.T_years - lower.T_years) * (tv_upper - tv_lower)
            elif tv_lower is not None and lower is upper:
                tv_target = tv_lower

        if compare_vendor:
            vendor_lower = lower.iv_vendor_atm if lower else None
            vendor_upper = upper.iv_vendor_atm if upper else None
            if vendor_lower is not None and vendor_upper is not None:
                tv1 = vendor_lower * vendor_lower * lower.T_years
                tv2 = vendor_upper * vendor_upper * upper.T_years
                if lower is upper or abs(upper.T_years - lower.T_years) < 1e-12:
                    tv_star = tv1
                else:
                    tv_star = tv1 + (target_T - lower.T_years) / (upper.T_years - lower.T_years) * (tv2 - tv1)
                ivx_vendor = math.sqrt(tv_star / target_T)
                if ivx_calc is not None and ivx_vendor is not None:
                    ivx_diff = ivx_calc - ivx_vendor
                    if abs(ivx_diff) > vendor_mismatch_threshold:
                        flags["VENDOR_MISMATCH"] = True

    result = IVXATMResult(
        symbol=symbol_out,
        as_of=meta.get("as_of"),
        underlying_price=S,
        interest_rate=r,
        dividend_yield=q,
        target_dte_days=target_dte_days,
        target_T_years=target_T,
        day_count=day_count,
        iv_source=iv_source,
        price_source=price_source,
        atm_method=atm_method,
        interpolate=interpolate,
        ivx_calc=ivx_calc,
        ivx_vendor=ivx_vendor,
        ivx_diff=ivx_diff,
        expiry_lower=lower,
        expiry_upper=upper,
        tv_lower=tv_lower,
        tv_upper=tv_upper,
        tv_target=tv_target,
        flags=flags,
        notes=notes,
    )
    return result

