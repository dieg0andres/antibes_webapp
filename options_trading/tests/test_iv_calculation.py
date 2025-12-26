# options_trading/tests/test_iv_calculation.py
"""
Tests for options_trading.volatility.iv_calculation

We test two complementary dimensions:

1) Pure synthetic tests:
   - Prove the math and control-flow is correct in a fully controlled setting.
   - We generate option quotes from Black–Scholes with a known sigma, then check
     that inversion + variance interpolation recover sigma.
   - We explicitly test:
       * exact-match vs bracketing interpolation vs extrapolation (no rescaling!)
       * mid vs mark price source behavior
       * vendor mismatch flagging

2) Snapshot-based tests (Schwab JSON fixtures):
   - Prove our parser/normalizer is robust against real-world Schwab quirks.
   - Prove ivx_atm runs end-to-end on multiple real symbols without crashing.
   - We keep assertions "soft" on numeric values because these are real market
     snapshots (prices, strikes, spreads, etc. can vary widely).
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from options_trading.volatility.iv_calculation import ivx_atm, normalize_schwab_chain

# Synthetic pricing uses your Black–Scholes implementation.
from options_trading.pricing.black_scholes import BSParams, bs_price


# --------------------------------------------------------------------------------------
# Fixture helpers (snapshot tests)
# --------------------------------------------------------------------------------------

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures" / "option_chains"


def _load_chain_json(filename: str) -> dict:
    path = FIXTURES_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Missing fixture file: {path}")
    return json.loads(path.read_text())


def _assert_reasonable_iv(iv: float) -> None:
    """
    Basic sanity: IV is an annualized decimal. We just ensure it is finite and positive.
    Upper bound is generous to avoid false failures on weird products.
    """
    assert iv is not None
    assert math.isfinite(iv)
    assert iv > 0.0
    assert iv < 5.0  # 500% annual vol would be extreme; this is a loose guardrail.


# --------------------------------------------------------------------------------------
# Synthetic data generator
# --------------------------------------------------------------------------------------

def _make_synthetic_quotes_df(
    *,
    S: float,
    r: float,
    q: float,
    sigma_true: float,
    dtes: tuple[int, int],
    day_count: int = 365,
    price_mode: str = "mid",  # "mid" or "mark"
    vendor_iv: float | None = None,
) -> pd.DataFrame:
    """
    Create a minimal normalized quotes DataFrame consistent with iv_calculation expectations.

    The module expects a normalized schema:
      expiry (YYYY-MM-DD string), dte (int), right ("call"/"put"), strike (float),
      bid/ask/mark floats, vendor_iv optional.

    We generate two expiries (dtes[0], dtes[1]) and quotes for call+put at the ATM strike.
    """
    rows: list[dict[str, Any]] = []
    # Use dummy expiry strings; only dte is used to compute T.
    expiries = [("2026-01-01", dtes[0]), ("2026-02-01", dtes[1])]

    for expiry_str, dte in expiries:
        T = dte / float(day_count)
        F = S * math.exp((r - q) * T)

        # Choose three strikes around the forward to exercise "closest strike" logic.
        k_atm = float(round(F, 2))
        strikes = [k_atm - 1.0, k_atm, k_atm + 1.0]

        # Only provide quotes for the actual ATM strike (k_atm) for simplicity.
        K = k_atm

        for right in ("call", "put"):
            params = BSParams(S=S, K=K, T=T, sigma=sigma_true, type=right, r=r, q=q)
            theo = float(bs_price(params))  # per-share price

            if price_mode == "mid":
                bid = max(0.0001, theo - 0.01)
                ask = theo + 0.01
                mark = theo
            elif price_mode == "mark":
                # Force mid invalid (bid/ask <= 0), but keep mark usable.
                bid = 0.0
                ask = 0.0
                mark = theo
            else:
                raise ValueError("price_mode must be 'mid' or 'mark'")

            rows.append(
                {
                    "expiry": expiry_str,
                    "dte": int(dte),
                    "right": right,
                    "strike": float(K),
                    "bid": float(bid),
                    "ask": float(ask),
                    "mark": float(mark),
                    "vendor_iv": float(vendor_iv) if vendor_iv is not None else np.nan,
                }
            )

        # Add some non-ATM strikes (no quotes) just to ensure unique() includes more strikes
        # without affecting selection. This is optional but helps test strike choice stability.
        for extra_k in strikes:
            if extra_k == k_atm:
                continue
            # add a placeholder row with NaN prices so it doesn't get selected as quote
            rows.append(
                {
                    "expiry": expiry_str,
                    "dte": int(dte),
                    "right": "call",
                    "strike": float(extra_k),
                    "bid": np.nan,
                    "ask": np.nan,
                    "mark": np.nan,
                    "vendor_iv": np.nan,
                }
            )
            rows.append(
                {
                    "expiry": expiry_str,
                    "dte": int(dte),
                    "right": "put",
                    "strike": float(extra_k),
                    "bid": np.nan,
                    "ask": np.nan,
                    "mark": np.nan,
                    "vendor_iv": np.nan,
                }
            )

    df = pd.DataFrame(rows)

    # Provide metadata via df.attrs to exercise "df.attrs fallback".
    df.attrs["underlying_price"] = S
    df.attrs["interest_rate"] = r
    df.attrs["dividend_yield"] = q
    df.attrs["symbol"] = "SYN"
    df.attrs["as_of"] = pd.Timestamp("2026-01-01", tz="UTC")

    return df


# --------------------------------------------------------------------------------------
# Synthetic tests (math correctness)
# --------------------------------------------------------------------------------------

def test_ivx_atm_synthetic_exact_match_recovers_sigma():
    """
    If there is an expiry with DTE exactly equal to target_dte_days,
    we expect EXACT_MATCH=True and IVX ~= sigma_true.
    """
    sigma_true = 0.20
    df = _make_synthetic_quotes_df(
        S=100.0, r=0.02, q=0.00, sigma_true=sigma_true,
        dtes=(30, 60),
        vendor_iv=0.20,
    )

    res = ivx_atm(df, target_dte_days=30, compare_vendor=True, vendor_mismatch_threshold=0.05)

    assert res.flags["EXACT_MATCH"] is True
    assert res.flags["EXTRAPOLATED"] is False
    _assert_reasonable_iv(res.ivx_calc)
    assert res.ivx_calc == pytest.approx(sigma_true, abs=5e-4)


def test_ivx_atm_synthetic_variance_interpolation_recovers_sigma_constant_vol():
    """
    With constant sigma across expiries, total variance interpolation should recover sigma.
    """
    sigma_true = 0.25
    df = _make_synthetic_quotes_df(
        S=100.0, r=0.01, q=0.00, sigma_true=sigma_true,
        dtes=(20, 40),
        vendor_iv=0.25,
    )

    res = ivx_atm(df, target_dte_days=30, compare_vendor=True, vendor_mismatch_threshold=0.05)

    assert res.flags["EXTRAPOLATED"] is False
    _assert_reasonable_iv(res.ivx_calc)
    assert res.ivx_calc == pytest.approx(sigma_true, abs=8e-4)


def test_ivx_atm_synthetic_extrapolation_no_rescaling_low():
    """
    Critical: in extrapolation, we should NOT rescale IV by sqrt(T_nearest / T_target).
    We should return nearest expiry IV as-is.
    """
    sigma_true = 0.30
    df = _make_synthetic_quotes_df(
        S=100.0, r=0.00, q=0.00, sigma_true=sigma_true,
        dtes=(20, 40),
        vendor_iv=0.30,
    )

    # Below min dte => extrapolate to nearest (20d)
    res = ivx_atm(df, target_dte_days=10, compare_vendor=False)

    assert res.flags["EXTRAPOLATED"] is True
    _assert_reasonable_iv(res.ivx_calc)
    assert res.ivx_calc == pytest.approx(sigma_true, abs=1e-3)


def test_ivx_atm_synthetic_extrapolation_no_rescaling_high():
    """
    Same test, above max dte => extrapolate to nearest (40d).
    """
    sigma_true = 0.30
    df = _make_synthetic_quotes_df(
        S=100.0, r=0.00, q=0.00, sigma_true=sigma_true,
        dtes=(20, 40),
        vendor_iv=0.30,
    )

    res = ivx_atm(df, target_dte_days=90, compare_vendor=False)

    assert res.flags["EXTRAPOLATED"] is True
    _assert_reasonable_iv(res.ivx_calc)
    assert res.ivx_calc == pytest.approx(sigma_true, abs=1e-3)


def test_ivx_atm_synthetic_mark_fallback_works():
    """
    If bid/ask are invalid but mark is present, price_source mid_then_mark should use mark.
    """
    sigma_true = 0.22
    df = _make_synthetic_quotes_df(
        S=100.0, r=0.01, q=0.00, sigma_true=sigma_true,
        dtes=(20, 40),
        price_mode="mark",
        vendor_iv=0.22,
    )

    res = ivx_atm(df, target_dte_days=30, price_source="mid_then_mark", compare_vendor=False)

    _assert_reasonable_iv(res.ivx_calc)
    assert res.ivx_calc == pytest.approx(sigma_true, abs=8e-4)


def test_ivx_atm_synthetic_vendor_mismatch_flag():
    """
    If vendor IV differs materially from calculated IV, VENDOR_MISMATCH should trigger.
    """
    sigma_true = 0.20
    vendor_sigma = 0.35  # big difference
    df = _make_synthetic_quotes_df(
        S=100.0, r=0.01, q=0.00, sigma_true=sigma_true,
        dtes=(20, 40),
        vendor_iv=vendor_sigma,
    )

    res = ivx_atm(
        df,
        target_dte_days=30,
        compare_vendor=True,
        vendor_mismatch_threshold=0.05,
    )

    _assert_reasonable_iv(res.ivx_calc)
    assert res.ivx_vendor is not None
    assert math.isfinite(res.ivx_vendor)
    assert res.flags["VENDOR_MISMATCH"] is True
    assert res.ivx_diff == pytest.approx(sigma_true - vendor_sigma, abs=2e-3)


def test_ivx_atm_dataframe_missing_metadata_raises():
    """
    If chain is a DataFrame and no explicit args or df.attrs provide S/r/q, we must raise.
    """
    df = pd.DataFrame(
        [{"expiry": "2026-01-01", "dte": 30, "right": "call", "strike": 100.0, "bid": 1.0, "ask": 1.1, "mark": 1.05}]
    )
    # Ensure attrs missing
    df.attrs.clear()

    with pytest.raises(ValueError) as exc:
        ivx_atm(df, target_dte_days=30)
    msg = str(exc.value)
    assert "Missing required metadata fields" in msg


# --------------------------------------------------------------------------------------
# Snapshot tests (fixtures)
# --------------------------------------------------------------------------------------

@pytest.mark.parametrize(
    "filename",
    [
        "schwab_chain_SPY.json",
        "schwab_chain_QQQ.json",
        "schwab_chain_GLD.json",
        "schwab_chain_XOM.json",
    ],
)
def test_normalize_schwab_chain_outputs_schema_and_metadata(filename: str):
    """
    Validate normalize_schwab_chain returns a DataFrame with expected columns and
    metadata with required fields.
    """
    chain = _load_chain_json(filename)
    df, meta = normalize_schwab_chain(chain)

    # Schema checks
    expected_cols = {"expiry", "dte", "right", "strike", "bid", "ask", "mark", "vendor_iv"}
    assert expected_cols.issubset(df.columns)
    assert not df.empty

    # Metadata checks
    assert "underlying_price" in meta and meta["underlying_price"] is not None
    assert "interest_rate" in meta and meta["interest_rate"] is not None
    assert "dividend_yield" in meta and meta["dividend_yield"] is not None

    # Rates should be decimals (not percent) after normalization
    assert 0.0 <= float(meta["interest_rate"]) <= 1.0
    assert 0.0 <= float(meta["dividend_yield"]) <= 1.0

    # Right values should be normalized
    assert set(df["right"].dropna().unique()).issubset({"call", "put"})


@pytest.mark.parametrize(
    "filename",
    [
        "schwab_chain_SPY.json",
        "schwab_chain_QQQ.json",
        "schwab_chain_GLD.json",
        "schwab_chain_XOM.json",
    ],
)
def test_normalize_schwab_chain_vendor_iv_is_positive_or_nan(filename: str):
    """
    Vendor IV should be either NaN or a positive decimal.
    This confirms we filtered sentinel/negative values (like -999) and normalized percent->decimal.
    """
    chain = _load_chain_json(filename)
    df, _ = normalize_schwab_chain(chain)

    v = df["vendor_iv"]
    v_non_nan = v.dropna()
    if not v_non_nan.empty:
        assert (v_non_nan > 0).all()
        # sanity: vendor IV should not be absurdly huge for liquid underlyings
        assert (v_non_nan < 5.0).all()


@pytest.mark.parametrize(
    "filename",
    [
        "schwab_chain_SPY.json",
        "schwab_chain_QQQ.json",
        "schwab_chain_GLD.json",
        "schwab_chain_XOM.json",
    ],
)
def test_ivx_atm_runs_end_to_end_on_snapshots(filename: str):
    """
    Smoke test: ivx_atm should run on real Schwab snapshots without crashing.
    We do not assert exact numeric values (these are market snapshots),
    but we do assert ivx_calc is finite and positive for a typical target maturity.
    """
    chain = _load_chain_json(filename)

    # compare_vendor=False here keeps the test robust even if vendor IV is missing at ATM strikes.
    # Vendor behavior is tested comprehensively in synthetic tests above.
    res = ivx_atm(
        chain,
        target_dte_days=30.0,
        compare_vendor=False,
        price_source="mid_then_mark",
        iv_source="invert_mid_then_mark",
    )

    assert res is not None
    assert res.expiry_lower is not None or res.expiry_upper is not None

    _assert_reasonable_iv(res.ivx_calc)

    # sanity flags structure
    assert isinstance(res.flags, dict)
    assert "EXTRAPOLATED" in res.flags
    assert "EXACT_MATCH" in res.flags
    assert "VENDOR_MISMATCH" in res.flags


@pytest.mark.parametrize(
    "filename",
    [
        "schwab_chain_SPY.json",
        "schwab_chain_QQQ.json",
        "schwab_chain_GLD.json",
        "schwab_chain_XOM.json",
    ],
)
def test_dict_input_and_df_input_agree_for_snapshots(filename: str):
    """
    If we normalize a Schwab dict to DataFrame + attrs, ivx_atm should produce
    essentially the same ivx_calc as dict input (same algorithm, same quotes).
    """
    chain = _load_chain_json(filename)

    # dict path
    res_dict = ivx_atm(chain, target_dte_days=30.0, compare_vendor=False)

    # df path
    df, meta = normalize_schwab_chain(chain)
    df.attrs["symbol"] = meta.get("symbol")
    df.attrs["underlying_price"] = float(meta["underlying_price"])
    df.attrs["interest_rate"] = float(meta["interest_rate"])
    df.attrs["dividend_yield"] = float(meta["dividend_yield"])
    df.attrs["as_of"] = meta.get("as_of")

    res_df = ivx_atm(df, target_dte_days=30.0, compare_vendor=False)

    # Both should be finite and close (they should match exactly unless minor floating differences)
    _assert_reasonable_iv(res_dict.ivx_calc)
    _assert_reasonable_iv(res_df.ivx_calc)
    assert res_df.ivx_calc == pytest.approx(res_dict.ivx_calc, rel=1e-10, abs=1e-10)

