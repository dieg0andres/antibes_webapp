"""Tests for realized volatility estimators and session-aware behaviors."""

import math

import numpy as np
import pandas as pd
import pytest

from options_trading.volatility import EQUITIES_RTH, realized_vol


def _make_df(start: str, periods: int, freq: str, price: float = 100.0) -> pd.DataFrame:
    idx = pd.date_range(start=start, periods=periods, freq=freq, tz="UTC")
    return pd.DataFrame(
        {"open": price, "high": price, "low": price, "close": price, "volume": 1},
        index=idx,
    )


def test_constant_prices_all_methods_zero():
    # Flat prices should yield zero variance after warmup for all estimators.
    df = _make_df("2024-01-01", 30, "1D", price=100.0)
    methods = ["close_to_close", "parkinson", "garman_klass", "rogers_satchell", "yang_zhang", "realized_variance"]
    for method in methods:
        res = realized_vol(df, method=method, window=5, annualize=False)
        assert (res.variance.dropna() == 0).all()


def test_scale_invariance():
    # Scaling prices should not change variance estimates.
    idx = pd.date_range("2024-01-01", periods=40, freq="1D", tz="UTC")
    prices = pd.Series(np.linspace(100, 120, len(idx)), index=idx)
    df = pd.DataFrame(
        {
            "open": prices * 0.99,
            "high": prices * 1.01,
            "low": prices * 0.98,
            "close": prices,
        }
    )
    res_base = realized_vol(df, method="yang_zhang", window=10, annualize=False)
    df_scaled = df * 10.0
    res_scaled = realized_vol(df_scaled, method="yang_zhang", window=10, annualize=False)
    assert np.allclose(res_base.variance.dropna(), res_scaled.variance.dropna())


def test_range_methods_zero_when_no_range():
    # Range-based estimators should be zero when H=L=O=C.
    df = _make_df("2024-01-01", 25, "1D", price=50.0)
    for method in ["parkinson", "garman_klass", "rogers_satchell"]:
        res = realized_vol(df, method=method, window=5, annualize=False)
        assert (res.variance.dropna() == 0).all()


def test_irregular_timestamps_strict_raises():
    # Within-session irregular spacing should raise in strict mode.
    idx = pd.DatetimeIndex(
        [
            pd.Timestamp("2024-01-01 14:30:00Z"),
            pd.Timestamp("2024-01-01 14:45:00Z"),
            pd.Timestamp("2024-01-01 15:30:00Z"),  # irregular jump
        ]
    )
    df = pd.DataFrame({"open": 100, "high": 101, "low": 99, "close": 100.5}, index=idx)
    with pytest.raises(ValueError):
        realized_vol(df, method="close_to_close", window=2, strict=True)


def test_irregular_timestamps_strict_false_warns():
    # strict=False should warn but continue on within-session irregular spacing.
    idx = pd.DatetimeIndex(
        [
            pd.Timestamp("2024-01-01 14:30:00Z"),
            pd.Timestamp("2024-01-01 14:45:00Z"),
            pd.Timestamp("2024-01-01 15:30:00Z"),  # irregular within session
        ]
    )
    df = pd.DataFrame({"open": 100, "high": 101, "low": 99, "close": 100.5}, index=idx)
    with pytest.warns(UserWarning):
        realized_vol(df, method="close_to_close", window=2, strict=False)


def test_ddof_defaults_depend_on_assume_mean_zero():
    # ddof should default to 0 when assume_mean_zero and 1 otherwise.
    closes = pd.Series([100.0, 105.0, 103.0, 106.0], index=pd.date_range("2024-01-01", periods=4, freq="1D", tz="UTC"))
    df = pd.DataFrame({"open": closes, "high": closes, "low": closes, "close": closes})

    returns = np.log(closes / closes.shift(1)).dropna()
    window = 3
    var_mean_zero = (returns.pow(2).rolling(window, min_periods=window).sum() / window).iloc[-1]
    var_not_mean_zero = returns.rolling(window, min_periods=window).var(ddof=1).iloc[-1]

    res_mz = realized_vol(df, method="close_to_close", window=window, assume_mean_zero=True, annualize=False)
    res_not_mz = realized_vol(df, method="close_to_close", window=window, assume_mean_zero=False, annualize=False)

    assert math.isclose(res_mz.variance.iloc[-1], var_mean_zero)
    assert math.isclose(res_not_mz.variance.iloc[-1], var_not_mean_zero)


def test_business_day_gaps_do_not_raise_strict():
    # Business-day gaps (weekends) should not trigger strict spacing errors.
    idx = pd.date_range("2024-01-01", periods=20, freq="B", tz="UTC")
    df = pd.DataFrame({"open": 100, "high": 100, "low": 100, "close": 100}, index=idx)
    res = realized_vol(df, method="close_to_close", window=5, annualize=False, strict=True)
    assert (res.variance.dropna() == 0).all()


def test_session_aggregation_reduces_to_session_bars_and_sets_close_index():
    # Aggregation should produce one bar per session indexed at session close UTC.
    candles = [
        {"datetime": pd.Timestamp("2024-01-02 14:30:00Z").value // 10**6, "open": 100, "high": 101, "low": 99, "close": 100},
        {"datetime": pd.Timestamp("2024-01-02 19:30:00Z").value // 10**6, "open": 100, "high": 102, "low": 98, "close": 101},
        {"datetime": pd.Timestamp("2024-01-03 14:30:00Z").value // 10**6, "open": 102, "high": 103, "low": 101, "close": 103},
        {"datetime": pd.Timestamp("2024-01-03 19:30:00Z").value // 10**6, "open": 103, "high": 104, "low": 102, "close": 104},
    ]
    df = pd.DataFrame(candles)
    res = realized_vol(df, method="yang_zhang", window=2, aggregation="session", annualize=False, strict=True)

    assert len(res.variance) == 2
    expected_first_close = EQUITIES_RTH.session.session_close_timestamp(pd.Timestamp("2024-01-02", tz="UTC").date())
    expected_second_close = EQUITIES_RTH.session.session_close_timestamp(pd.Timestamp("2024-01-03", tz="UTC").date())
    assert res.variance.index[0] == expected_first_close
    assert res.variance.index[1] == expected_second_close
    assert res.variance.index.tz is not None


def test_realized_variance_uses_intraday_returns_and_overnight():
    # Realized variance should include open->first close, intraday path, and overnight gap.
    # Session 1 closes: 100 -> 102 -> 104 (no overnight component)
    sess1 = [
        {"datetime": pd.Timestamp("2024-01-02 14:30:00Z").value // 10**6, "open": 100, "high": 104, "low": 100, "close": 102},
        {"datetime": pd.Timestamp("2024-01-02 16:30:00Z").value // 10**6, "open": 102, "high": 104, "low": 102, "close": 104},
    ]
    # Session 2 closes: overnight gap from 104 -> 103, intraday 103 -> 106 -> 107
    sess2 = [
        {"datetime": pd.Timestamp("2024-01-03 14:30:00Z").value // 10**6, "open": 103, "high": 106, "low": 103, "close": 106},
        {"datetime": pd.Timestamp("2024-01-03 16:30:00Z").value // 10**6, "open": 106, "high": 107, "low": 106, "close": 107},
    ]
    df = pd.DataFrame(sess1 + sess2)
    res = realized_vol(df, method="realized_variance", window=2, annualize=False, include_overnight=True)

    r1 = math.log(102 / 100) ** 2 + math.log(104 / 102) ** 2
    overnight = math.log(103 / 104) ** 2
    r2 = math.log(106 / 103) ** 2 + math.log(107 / 106) ** 2
    expected_var = (r1 + (overnight + r2)) / 2

    assert math.isclose(res.variance.iloc[-1], expected_var, rel_tol=1e-9)


def test_realized_variance_captures_path_difference():
    # Intraday path differences should change realized_variance but not OHLC estimators.
    # Two sessions; first session paths differ but OHLC identical
    sess1_path_a = [
        # Big swings in closes; session OHLC remains O=100, H=110, L=90, C=100.
        {"datetime": pd.Timestamp("2024-01-02 14:30:00Z").value // 10**6, "open": 100, "high": 110, "low": 99, "close": 110},
        {"datetime": pd.Timestamp("2024-01-02 15:30:00Z").value // 10**6, "open": 110, "high": 110, "low": 90, "close": 90},
        {"datetime": pd.Timestamp("2024-01-02 16:30:00Z").value // 10**6, "open": 90, "high": 110, "low": 90, "close": 110},
        {"datetime": pd.Timestamp("2024-01-02 17:30:00Z").value // 10**6, "open": 110, "high": 110, "low": 90, "close": 90},
        {"datetime": pd.Timestamp("2024-01-02 18:30:00Z").value // 10**6, "open": 90, "high": 100, "low": 90, "close": 100},
    ]
    sess1_path_b = [
        # Small close-to-close moves; use wicks to match the same session H/L as path A.
        {"datetime": pd.Timestamp("2024-01-02 14:30:00Z").value // 10**6, "open": 100, "high": 110, "low": 99, "close": 101},
        {"datetime": pd.Timestamp("2024-01-02 15:30:00Z").value // 10**6, "open": 101, "high": 102, "low": 90, "close": 100},
        {"datetime": pd.Timestamp("2024-01-02 16:30:00Z").value // 10**6, "open": 100, "high": 102, "low": 99, "close": 101},
        {"datetime": pd.Timestamp("2024-01-02 17:30:00Z").value // 10**6, "open": 101, "high": 102, "low": 99, "close": 100},
        {"datetime": pd.Timestamp("2024-01-02 18:30:00Z").value // 10**6, "open": 100, "high": 101, "low": 99, "close": 100},
    ]
    sess2 = [
        {"datetime": pd.Timestamp("2024-01-03 14:30:00Z").value // 10**6, "open": 110, "high": 112, "low": 108, "close": 112},
        {"datetime": pd.Timestamp("2024-01-03 15:30:00Z").value // 10**6, "open": 112, "high": 112, "low": 108, "close": 112},
    ]
    sess3 = [
        {"datetime": pd.Timestamp("2024-01-04 14:30:00Z").value // 10**6, "open": 112, "high": 113, "low": 111, "close": 113},
        {"datetime": pd.Timestamp("2024-01-04 15:30:00Z").value // 10**6, "open": 113, "high": 113, "low": 111, "close": 113},
    ]

    df_a = pd.DataFrame(sess1_path_a + sess2 + sess3)
    df_b = pd.DataFrame(sess1_path_b + sess2 + sess3)

    rv_a = realized_vol(df_a, method="realized_variance", window=2, annualize=False)
    rv_b = realized_vol(df_b, method="realized_variance", window=2, annualize=False)
    yz_a = realized_vol(df_a, method="yang_zhang", window=2, annualize=False, aggregation="session")
    yz_b = realized_vol(df_b, method="yang_zhang", window=2, annualize=False, aggregation="session")

    # With window=2, the first non-NaN realized_variance value includes session1 and session2.
    assert not math.isclose(rv_a.variance.dropna().iloc[0], rv_b.variance.dropna().iloc[0])
    assert math.isclose(yz_a.variance.iloc[-1], yz_b.variance.iloc[-1])


def test_realized_vol_invalid_method_raises():
    # Unknown estimator names should raise ValueError.
    df = _make_df("2024-01-01", 5, "1D", price=100.0)
    with pytest.raises(ValueError):
        realized_vol(df, method="unknown_method", window=3)


def test_realized_vol_window_must_exceed_one():
    # window <= 1 should raise before computation begins.
    df = _make_df("2024-01-01", 5, "1D", price=100.0)
    with pytest.raises(ValueError):
        realized_vol(df, window=1)


def test_realized_vol_include_moments_emits_skew_and_kurtosis_series():
    # include_moments=True should return skew/kurtosis aligned with variance.
    idx = pd.date_range("2024-01-01", periods=5, freq="1D", tz="UTC")
    closes = pd.Series([100.0, 103.0, 101.0, 104.0, 102.0], index=idx)
    df = pd.DataFrame(
        {
            "open": closes,
            "high": closes + 1.0,
            "low": closes - 1.0,
            "close": closes,
        }
    )
    res = realized_vol(
        df,
        method="close_to_close",
        window=3,
        include_moments=True,
        assume_mean_zero=False,
        annualize=False,
    )
    assert res.skew is not None and res.excess_kurtosis is not None
    assert res.skew.index.equals(res.variance.index)

    returns = np.log(df["close"] / df["close"].shift(1))

    def _skew(arr: np.ndarray) -> float:
        r = pd.Series(arr)
        mu = r.mean()
        centered = r - mu
        std = centered.std(ddof=0)
        if std == 0 or pd.isna(std):
            return math.nan
        z = centered / std
        return float((z**3).mean())

    def _kurt(arr: np.ndarray) -> float:
        r = pd.Series(arr)
        mu = r.mean()
        centered = r - mu
        std = centered.std(ddof=0)
        if std == 0 or pd.isna(std):
            return math.nan
        z = centered / std
        return float((z**4).mean() - 3.0)

    expected_skew = returns.rolling(window=3, min_periods=3).apply(_skew, raw=True)
    expected_kurt = returns.rolling(window=3, min_periods=3).apply(_kurt, raw=True)

    assert math.isclose(
        res.skew.dropna().iloc[-1],
        expected_skew.dropna().iloc[-1],
        rel_tol=1e-9,
        abs_tol=1e-12,
    )
    assert math.isclose(
        res.excess_kurtosis.dropna().iloc[-1],
        expected_kurt.dropna().iloc[-1],
        rel_tol=1e-9,
        abs_tol=1e-12,
    )


def test_realized_variance_toggle_overnight_gap_contribution():
    # include_overnight toggles whether overnight gap returns contribute to variance.
    sess1 = [
        {"datetime": pd.Timestamp("2024-01-02 14:30:00Z").value // 10**6, "open": 100, "high": 101, "low": 99, "close": 101},
        {"datetime": pd.Timestamp("2024-01-02 14:45:00Z").value // 10**6, "open": 101, "high": 102, "low": 100, "close": 102},
    ]
    sess2 = [
        {"datetime": pd.Timestamp("2024-01-03 14:30:00Z").value // 10**6, "open": 99, "high": 103, "low": 99, "close": 101},
        {"datetime": pd.Timestamp("2024-01-03 14:45:00Z").value // 10**6, "open": 101, "high": 104, "low": 101, "close": 103},
    ]
    df = pd.DataFrame(sess1 + sess2)
    rv_with = realized_vol(df, method="realized_variance", window=2, include_overnight=True, annualize=False)
    rv_without = realized_vol(df, method="realized_variance", window=2, include_overnight=False, annualize=False)
    overnight_return = math.log(99 / 102)
    expected_diff = (overnight_return**2) / 2.0
    diff = rv_with.variance.iloc[-1] - rv_without.variance.iloc[-1]
    assert diff > 0
    assert math.isclose(diff, expected_diff, rel_tol=1e-9, abs_tol=1e-12)


def test_realized_vol_bars_per_day_override_updates_metadata():
    # bars_per_day_override should control bars_per_year when not session-indexed.
    idx = pd.date_range("2024-01-02 14:30:00Z", periods=4, freq="60min")
    df = pd.DataFrame({"open": 100, "high": 101, "low": 99, "close": np.linspace(100, 101, 4)}, index=idx)
    res_default = realized_vol(
        df,
        method="close_to_close",
        window=3,
        aggregation="none",
        annualize=True,
        assume_mean_zero=True,
    )
    res_override = realized_vol(
        df,
        method="close_to_close",
        window=3,
        aggregation="none",
        annualize=True,
        assume_mean_zero=True,
        bars_per_day_override=1,
    )
    expected_bars_year = 1 * EQUITIES_RTH.days_per_year
    assert res_override.metadata["bars_per_year"] == expected_bars_year
    assert res_default.metadata["bars_per_year"] != res_override.metadata["bars_per_year"]


def test_realized_vol_invalid_aggregation_option_raises():
    # Invalid aggregation parameter should raise ValueError.
    df = _make_df("2024-01-01", 5, "1D", price=100.0)
    with pytest.raises(ValueError):
        realized_vol(df, method="close_to_close", window=3, aggregation="invalid_mode")