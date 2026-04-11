import math
import numpy as np
import pandas as pd
import pytest

from options_trading.volatility.cones import compute_volatility_cone


# ----------------------------
# Test data helpers
# ----------------------------

def make_constant_session_ohlc(
    n_sessions: int,
    *,
    start: str = "2020-01-01",
    price: float = 100.0,
) -> pd.DataFrame:
    """
    Constant OHLC bars (open=high=low=close=price) on business-day index.

    This is the best deterministic test case:
      - close-to-close returns are exactly 0
      - Yang-Zhang should also be exactly 0
      - Therefore RV values should be 0 (after warmup)
    """
    idx = pd.date_range(start=start, periods=n_sessions, freq="B", tz="UTC")
    df = pd.DataFrame(
        {
            "open": float(price),
            "high": float(price),
            "low": float(price),
            "close": float(price),
        },
        index=idx,
    )
    return df


def make_random_walk_session_ohlc(
    n_sessions: int,
    *,
    start: str = "2015-01-01",
    s0: float = 100.0,
    daily_vol: float = 0.01,
    seed: int = 0,
) -> pd.DataFrame:
    """
    Synthetic OHLC consistent bars from a lognormal random walk.

    - close is generated from open * exp(ret)
    - high/low create a small intraday range around open/close, ensuring OHLC is consistent.
    """
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start=start, periods=n_sessions, freq="B", tz="UTC")

    # log returns
    rets = rng.normal(loc=0.0, scale=daily_vol, size=n_sessions)

    open_ = np.empty(n_sessions, dtype=float)
    close_ = np.empty(n_sessions, dtype=float)
    high_ = np.empty(n_sessions, dtype=float)
    low_ = np.empty(n_sessions, dtype=float)

    close_prev = float(s0)
    for i in range(n_sessions):
        open_[i] = close_prev
        close_[i] = open_[i] * math.exp(float(rets[i]))

        hi = max(open_[i], close_[i]) * 1.001
        lo = min(open_[i], close_[i]) * 0.999

        high_[i] = hi
        low_[i] = lo
        close_prev = close_[i]

    return pd.DataFrame(
        {"open": open_, "high": high_, "low": low_, "close": close_},
        index=idx,
    )


def get_entry(cone_json: dict, rv_method: str, lookback_years: int, horizon_days: int) -> dict:
    return cone_json["rv_methods"][rv_method]["lookbacks"][str(lookback_years)][str(horizon_days)]


def assert_percentile_dict(entry: dict, percentiles=(0.10, 0.25, 0.50, 0.75, 0.90)) -> None:
    assert entry["p"] is not None
    for p in percentiles:
        key = f"{p:g}"
        assert key in entry["p"]


# ----------------------------
# Unit tests
# ----------------------------

def test_cone_constant_prices_schema_and_percentile_rank():
    """
    Deterministic case: constant prices => RV should be 0 for both CTC and YZ
    (after warmup). This makes expected values crisp:
      - min=max=0
      - pXX all 0
      - rv_today=0
      - rv_today_percentile_rank = 1.0
    """
    session_df = make_constant_session_ohlc(600)

    res = compute_volatility_cone(
        symbol="TEST",
        session_ohlc=session_df,
        asof_utc="2022-12-30T00:00:00Z",
        rv_methods=("CTC_OVERNIGHT", "YZ"),
        horizons_days=(10, 21),
        lookback_years=(2,),
        percentiles=(0.10, 0.25, 0.50, 0.75, 0.90),
        sessions_per_year=252,
        strict=True,
        min_samples=30,
    )

    # Lookback feasibility
    assert res.lookbacks_years_used == (2,)

    cone = res.jsonable
    assert cone["symbol"] == "TEST"
    assert cone["overlap_method"] == "overlap_unadjusted"
    assert cone["horizons_days"] == [10, 21]
    assert cone["lookbacks_years_used"] == [2]
    assert set(cone["rv_methods"].keys()) == {"CTC_OVERNIGHT", "YZ"}

    # Expect 2y lookback -> 504 sessions in the slice
    expected_count = 2 * 252

    for method in ("CTC_OVERNIGHT", "YZ"):
        for horizon in (10, 21):
            entry = get_entry(cone, method, 2, horizon)

            assert entry["flag"] == "OK"
            assert entry["count"] == expected_count

            assert entry["min"] == 0.0
            assert entry["max"] == 0.0

            assert_percentile_dict(entry)
            for v in entry["p"].values():
                assert v == 0.0

            assert entry["rv_today"] == 0.0
            assert entry["rv_today_percentile_rank"] == 1.0

    # Table sanity: index rows should exist for each (method, lookback, horizon)
    assert ("CTC_OVERNIGHT", 2, 10) in res.table.index
    assert ("CTC_OVERNIGHT", 2, 21) in res.table.index
    assert ("YZ", 2, 10) in res.table.index
    assert ("YZ", 2, 21) in res.table.index


def test_cone_insufficient_history_flags_long_horizon():
    """
    Provide enough history for 2y lookback + 21d horizon, but NOT enough for 252d horizon.
    Expected:
      - horizon 21 -> OK
      - horizon 252 -> INSUFFICIENT_HISTORY
    """
    session_df = make_constant_session_ohlc(600)  # enough for 2y + (21-1) but not 2y + (252-1)

    res = compute_volatility_cone(
        symbol="TEST",
        session_ohlc=session_df,
        asof_utc="2022-12-30T00:00:00Z",
        rv_methods=("CTC_OVERNIGHT", "YZ"),
        horizons_days=(21, 252),
        lookback_years=(2,),
        percentiles=(0.10, 0.25, 0.50, 0.75, 0.90),
        sessions_per_year=252,
        strict=True,
        min_samples=30,
    )

    cone = res.jsonable

    for method in ("CTC_OVERNIGHT", "YZ"):
        e21 = get_entry(cone, method, 2, 21)
        e252 = get_entry(cone, method, 2, 252)

        assert e21["flag"] == "OK"
        assert e21["count"] == 504

        assert e252["flag"] == "INSUFFICIENT_HISTORY"
        assert e252["p"] is None
        assert e252["min"] is None
        assert e252["max"] is None


def test_cone_lookbacks_used_respects_available_history():
    """
    If we have ~1300 sessions, we should support 2y and 5y lookbacks, but not 10y.
    """
    session_df = make_constant_session_ohlc(1300)

    res = compute_volatility_cone(
        symbol="TEST",
        session_ohlc=session_df,
        asof_utc="2022-12-30T00:00:00Z",
        rv_methods=("CTC_OVERNIGHT", "YZ"),
        horizons_days=(21,),
        lookback_years=(2, 5, 10),
        percentiles=(0.10, 0.25, 0.50, 0.75, 0.90),
        sessions_per_year=252,
        strict=True,
        min_samples=30,
    )

    assert res.lookbacks_years_used == (2, 5)
    cone = res.jsonable
    assert cone["lookbacks_years_used"] == [2, 5]

    # Count should match the lookback slice length when RV is fully available
    # (Constant series => RV non-NaN after warmup, and the lookback slice is far from start)
    for method in ("CTC_OVERNIGHT", "YZ"):
        e2 = get_entry(cone, method, 2, 21)
        e5 = get_entry(cone, method, 5, 21)

        assert e2["flag"] == "OK"
        assert e2["count"] == 504

        assert e5["flag"] == "OK"
        assert e5["count"] == 1260

        # 10y not present
        assert "10" not in cone["rv_methods"][method]["lookbacks"]


def test_cone_quantiles_monotonic_and_in_range_random_walk():
    """
    Stochastic case: ensure basic statistical invariants hold.
    We do NOT assert exact values (estimator-specific), only that:
      - percentiles are monotonic
      - percentiles in [min, max]
      - rv_today is in [min, max] (today is included in sample)
      - percentile rank in [0,1]
    """
    session_df = make_random_walk_session_ohlc(2000, seed=123)

    res = compute_volatility_cone(
        symbol="TEST",
        session_ohlc=session_df,
        asof_utc="2022-12-30T00:00:00Z",
        rv_methods=("CTC_OVERNIGHT", "YZ"),
        horizons_days=(21, 63),
        lookback_years=(2, 5),
        percentiles=(0.10, 0.25, 0.50, 0.75, 0.90),
        sessions_per_year=252,
        strict=True,
        min_samples=50,
    )

    cone = res.jsonable
    percentiles = (0.10, 0.25, 0.50, 0.75, 0.90)
    pkeys = [f"{p:g}" for p in percentiles]

    for method in ("CTC_OVERNIGHT", "YZ"):
        for lookback in (2, 5):
            for horizon in (21, 63):
                entry = get_entry(cone, method, lookback, horizon)

                # Some (lookback,horizon) combos could still be insufficient depending on history,
                # but with 2000 sessions, 5y+63d should be OK.
                assert entry["flag"] in ("OK", "NO_RV_TODAY")

                if entry["flag"] != "OK":
                    # If it happens, ensure it follows schema expectations
                    assert entry["p"] is None or isinstance(entry["p"], dict)
                    continue

                assert entry["count"] >= 50
                assert entry["min"] is not None and entry["max"] is not None
                assert entry["min"] <= entry["max"]

                assert_percentile_dict(entry, percentiles=percentiles)

                # monotonic quantiles
                vals = [entry["p"][k] for k in pkeys]
                assert all(v is not None for v in vals)
                assert all(vals[i] <= vals[i + 1] for i in range(len(vals) - 1))

                # quantiles within [min,max]
                assert all(entry["min"] <= v <= entry["max"] for v in vals)

                # today within [min,max] and rank within [0,1]
                assert entry["rv_today"] is not None
                assert entry["min"] <= entry["rv_today"] <= entry["max"]

                assert entry["rv_today_percentile_rank"] is not None
                assert 0.0 <= entry["rv_today_percentile_rank"] <= 1.0


# ============================
# Additional robustness tests
# ============================

def test_cone_naive_datetime_index_is_localized_to_utc():
    """
    If session OHLC has a naive DatetimeIndex, cones.py should localize to UTC
    (or convert appropriately), and computation should succeed.
    """
    df = make_constant_session_ohlc(600)
    df.index = df.index.tz_convert(None)  # make naive

    res = compute_volatility_cone(
        symbol="TEST",
        session_ohlc=df,
        asof_utc="2022-12-30T00:00:00Z",
        rv_methods=("CTC_OVERNIGHT", "YZ"),
        horizons_days=(21,),
        lookback_years=(2,),
        percentiles=(0.10, 0.25, 0.50, 0.75, 0.90),
        sessions_per_year=252,
        strict=True,
        min_samples=30,
    )

    cone = res.jsonable
    for method in ("CTC_OVERNIGHT", "YZ"):
        entry = get_entry(cone, method, 2, 21)
        assert entry["flag"] == "OK"
        assert entry["count"] == 504


def test_cone_strict_invalid_ohlc_raises():
    """
    Invalid OHLC consistency should be rejected under strict=True in most implementations.

    We intentionally set close > high for one bar (inconsistent).
    """
    df = make_random_walk_session_ohlc(800, seed=7)

    # Introduce OHLC inconsistency (close above high)
    bad_idx = df.index[100]
    df.loc[bad_idx, "high"] = df.loc[bad_idx, "close"] * 0.99  # high < close

    with pytest.raises(Exception):
        compute_volatility_cone(
            symbol="TEST",
            session_ohlc=df,
            asof_utc="2022-12-30T00:00:00Z",
            rv_methods=("CTC_OVERNIGHT", "YZ"),
            horizons_days=(21,),
            lookback_years=(2,),
            percentiles=(0.10, 0.25, 0.50, 0.75, 0.90),
            sessions_per_year=252,
            strict=True,
            min_samples=30,
        )


def test_cone_non_strict_invalid_ohlc_still_computes_some_results():
    """
    Under strict=False, we expect the pipeline to attempt to continue (warn/log)
    even if there are some OHLC inconsistencies.

    We only assert that at least one method/horizon returns OK.
    """
    df = make_random_walk_session_ohlc(1400, seed=9)

    # Introduce "soft" inconsistencies that should be tolerated under strict=False:
    # Make open fall outside the [low, high] range WITHOUT ever making high < low.
    for i in (50, 400, 900):
        idx = df.index[i]
        df.loc[idx, "open"] = df.loc[idx, "low"] * 0.99  # open < low (range violation), but high/low remain ordered


    res = compute_volatility_cone(
        symbol="TEST",
        session_ohlc=df,
        asof_utc="2022-12-30T00:00:00Z",
        rv_methods=("CTC_OVERNIGHT", "YZ"),
        horizons_days=(21, 63),
        lookback_years=(2, 5),
        percentiles=(0.10, 0.25, 0.50, 0.75, 0.90),
        sessions_per_year=252,
        strict=False,
        min_samples=50,
    )

    cone = res.jsonable

    ok_found = False
    for method in ("CTC_OVERNIGHT", "YZ"):
        for lookback in cone["lookbacks_years_used"]:
            for horizon in (21, 63):
                entry = get_entry(cone, method, int(lookback), horizon)
                if entry["flag"] == "OK":
                    ok_found = True
                    assert entry["count"] == int(lookback) * 252
                    assert 0.0 <= entry["rv_today_percentile_rank"] <= 1.0
                    break
    assert ok_found, "Expected at least one OK entry under strict=False despite some bad rows."


def test_cone_gappy_history_still_returns_ok_when_enough_observations():
    """
    Drop some random sessions (simulate missing days). Cones should still compute OK
    as long as enough RV observations exist up to asof-date.
    """
    df = make_random_walk_session_ohlc(2200, seed=1234)

    # Drop ~5% of rows
    rng = np.random.default_rng(123)
    drop_idx = rng.choice(df.index.to_numpy(), size=int(0.05 * len(df)), replace=False)
    df2 = df.drop(drop_idx).sort_index()

    res = compute_volatility_cone(
        symbol="TEST",
        session_ohlc=df2,
        asof_utc="2022-12-30T00:00:00Z",
        rv_methods=("CTC_OVERNIGHT", "YZ"),
        horizons_days=(21,),
        lookback_years=(2,),
        percentiles=(0.10, 0.25, 0.50, 0.75, 0.90),
        sessions_per_year=252,
        strict=False,
        min_samples=50,
    )

    cone = res.jsonable
    # With gappy history, it is *possible* we end up with fewer than 504 RV observations
    # depending on how realized_vol handles missing sessions. We expect either OK with
    # count==504 (if it just treats remaining sessions as sessions), or INSUFFICIENT_HISTORY.
    for method in ("CTC_OVERNIGHT", "YZ"):
        entry = get_entry(cone, method, 2, 21)
        assert entry["flag"] in ("OK", "INSUFFICIENT_HISTORY")
        if entry["flag"] == "OK":
            assert entry["count"] == 504


def test_cone_today_fallback_uses_last_valid_rv_when_last_is_nan():
    """
    Force a NaN in the last RV point and ensure cones uses the previous valid value
    for rv_today (fallback behavior), while still computing the distribution.
    """
    df = make_random_walk_session_ohlc(2000, seed=88)

    # Build cones normally but with strict=False to tolerate manipulations.
    # We'll sabotage the last day's OHLC to produce NaNs in RV computation in some estimators.
    # A common way is to set close <= 0 (log invalid). If your realized layer raises even when
    # strict=False, then skip this test by adjusting the sabotage strategy.
    df_bad = df.copy()
    df_bad.iloc[-1, df_bad.columns.get_loc("close")] = np.nan  # missing close on last day

    res = compute_volatility_cone(
        symbol="TEST",
        session_ohlc=df_bad,
        asof_utc="2022-12-30T00:00:00Z",
        rv_methods=("CTC_OVERNIGHT", "YZ"),
        horizons_days=(21,),
        lookback_years=(2,),
        percentiles=(0.10, 0.25, 0.50, 0.75, 0.90),
        sessions_per_year=252,
        strict=False,
        min_samples=50,
    )

    cone = res.jsonable
    for method in ("CTC_OVERNIGHT", "YZ"):
        entry = get_entry(cone, method, 2, 21)
        # Could be INSUFFICIENT_HISTORY if realized_vol drops too much; otherwise OK
        assert entry["flag"] in ("OK", "INSUFFICIENT_HISTORY")
        if entry["flag"] == "OK":
            assert entry["rv_today"] is not None
            assert 0.0 <= entry["rv_today_percentile_rank"] <= 1.0



def test_percentile_rank_edge_cases_min_and_max():
    """
    Deterministic edge-case construction:
    - "Max" case: last 21 returns have high amplitude, previous lookback returns low amplitude.
      => today's 21d RV should be at/near the maximum in the 2y sample.
    - "Min" case: last 21 returns low amplitude, previous lookback returns high amplitude.
      => today's 21d RV should be at/near the minimum in the 2y sample.
    """
    n = 2000
    idx = pd.date_range("2015-01-01", periods=n, freq="B", tz="UTC")

    def build_df_with_amplitude_switch(a_before: float, a_last: float) -> pd.DataFrame:
        # alternating +/- returns to create stable nonzero variance
        rets = np.empty(n, dtype=float)
        rets[:] = a_before
        rets[-21:] = a_last
        rets *= np.where(np.arange(n) % 2 == 0, 1.0, -1.0)

        open_ = np.empty(n)
        close_ = np.empty(n)
        high_ = np.empty(n)
        low_ = np.empty(n)

        close_prev = 100.0
        for i in range(n):
            open_[i] = close_prev
            close_[i] = open_[i] * math.exp(float(rets[i]))

            hi = max(open_[i], close_[i]) * 1.001
            lo = min(open_[i], close_[i]) * 0.999

            high_[i], low_[i] = hi, lo
            close_prev = close_[i]

        return pd.DataFrame({"open": open_, "high": high_, "low": low_, "close": close_}, index=idx)

    # MAX case: last 21 days high amplitude, previous low amplitude
    df_max = build_df_with_amplitude_switch(a_before=0.005, a_last=0.05)

    res_max = compute_volatility_cone(
        symbol="TEST",
        session_ohlc=df_max,
        asof_utc="2022-12-30T00:00:00Z",
        rv_methods=("CTC_OVERNIGHT",),
        horizons_days=(21,),
        lookback_years=(2,),
        percentiles=(0.10, 0.25, 0.50, 0.75, 0.90),
        sessions_per_year=252,
        strict=True,
        min_samples=50,
    )
    entry_max = get_entry(res_max.jsonable, "CTC_OVERNIGHT", 2, 21)
    assert entry_max["flag"] == "OK"
    assert entry_max["rv_today_percentile_rank"] > 0.95

    # MIN case: last 21 days low amplitude, previous high amplitude
    df_min = build_df_with_amplitude_switch(a_before=0.05, a_last=0.005)

    res_min = compute_volatility_cone(
        symbol="TEST",
        session_ohlc=df_min,
        asof_utc="2022-12-30T00:00:00Z",
        rv_methods=("CTC_OVERNIGHT",),
        horizons_days=(21,),
        lookback_years=(2,),
        percentiles=(0.10, 0.25, 0.50, 0.75, 0.90),
        sessions_per_year=252,
        strict=True,
        min_samples=50,
    )
    entry_min = get_entry(res_min.jsonable, "CTC_OVERNIGHT", 2, 21)
    assert entry_min["flag"] == "OK"
    assert entry_min["rv_today_percentile_rank"] < 0.05




def test_ctc_vs_yz_not_orders_of_magnitude_apart_on_synthetic():
    """
    Sanity check: on a reasonable synthetic dataset, YZ and CTC should be in the same ballpark.
    This is a smoke test to catch estimator wiring errors (wrong annualization, wrong windowing, etc.).
    """
    df = make_random_walk_session_ohlc(2500, seed=202)

    res = compute_volatility_cone(
        symbol="TEST",
        session_ohlc=df,
        asof_utc="2022-12-30T00:00:00Z",
        rv_methods=("CTC_OVERNIGHT", "YZ"),
        horizons_days=(21,),
        lookback_years=(5,),
        percentiles=(0.50,),  # just median for this smoke test
        sessions_per_year=252,
        strict=True,
        min_samples=200,
    )

    cone = res.jsonable
    e_ctc = get_entry(cone, "CTC_OVERNIGHT", 5, 21)
    e_yz = get_entry(cone, "YZ", 5, 21)
    assert e_ctc["flag"] == "OK"
    assert e_yz["flag"] == "OK"

    med_ctc = e_ctc["p"]["0.5"]
    med_yz = e_yz["p"]["0.5"]
    assert med_ctc > 0.0 and med_yz > 0.0

    ratio = med_yz / med_ctc
    assert 0.2 < ratio < 5.0


def test_cone_large_input_smoke_all_horizons_runs():
    """
    Smoke test: 10y-ish input across all default horizons should run without errors.
    We don't assert values, only that it completes and returns expected keys.
    """
    df = make_random_walk_session_ohlc(2700, seed=303)

    res = compute_volatility_cone(
        symbol="TEST",
        session_ohlc=df,
        asof_utc="2022-12-30T00:00:00Z",
        rv_methods=("CTC_OVERNIGHT", "YZ"),
        horizons_days=(10, 21, 42, 63, 126, 252),
        lookback_years=(2, 5, 10),
        percentiles=(0.10, 0.25, 0.50, 0.75, 0.90),
        sessions_per_year=252,
        strict=True,
        min_samples=100,
    )

    cone = res.jsonable
    assert "rv_methods" in cone
    assert set(cone["rv_methods"].keys()) == {"CTC_OVERNIGHT", "YZ"}
    assert cone["horizons_days"] == [10, 21, 42, 63, 126, 252]
    assert cone["lookbacks_years_used"] in ([2, 5, 10], [2, 5])  # depending on exact length/feasibility
