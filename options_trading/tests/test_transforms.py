"""Tests for options_trading.volatility.transforms.

Coverage focuses on deterministic, synthetic inputs to validate timezone handling,
payload normalization, session-aware strictness, session aggregation, and OHLC validation.
"""

import numpy as np
import pandas as pd
import pytest

from options_trading.volatility.calendar import EQUITIES_RTH
from options_trading.volatility.transforms import (
    aggregate_to_sessions,
    apply_adjustment_factor,
    candles_to_df,
    ensure_tz_aware_utc_index,
    infer_bar_minutes,
    validate_ohlc,
)


def test_ensure_tz_aware_utc_index_localizes_naive_then_converts():
    # Naive index should be localized to provided tz then converted to UTC.
    idx = pd.DatetimeIndex([pd.Timestamp("2024-01-02 09:30:00")])  # naive
    df = pd.DataFrame({"open": [1.0], "high": [1.0], "low": [1.0], "close": [1.0]}, index=idx)
    out = ensure_tz_aware_utc_index(df, tz="America/New_York")
    assert out.index.tz is not None
    assert out.index[0] == pd.Timestamp("2024-01-02 14:30:00Z")


def test_ensure_tz_aware_utc_index_converts_tz_aware_to_utc():
    # TZ-aware index should be converted to UTC.
    idx = pd.DatetimeIndex([pd.Timestamp("2024-01-02 09:30:00", tz="America/New_York")])
    df = pd.DataFrame({"open": [1.0], "high": [1.0], "low": [1.0], "close": [1.0]}, index=idx)
    out = ensure_tz_aware_utc_index(df)
    assert out.index[0] == pd.Timestamp("2024-01-02 14:30:00Z")


def test_ensure_tz_aware_utc_index_raises_on_non_datetime_index():
    # Non-DatetimeIndex should raise.
    df = pd.DataFrame({"open": [1.0], "high": [1.0], "low": [1.0], "close": [1.0]}, index=[1])
    with pytest.raises(ValueError):
        ensure_tz_aware_utc_index(df)


def test_candles_to_df_accepts_dataframe_and_sorts_and_casts():
    # DataFrame input should be sorted, UTC tz-aware, and OHLC float dtype.
    idx = pd.DatetimeIndex(
        [pd.Timestamp("2024-01-02 14:45:00Z"), pd.Timestamp("2024-01-02 14:30:00Z")]
    )
    df = pd.DataFrame(
        {"open": [100, 100], "high": [101, 101], "low": [99, 99], "close": [100, 100], "volume": ["10", "11"]},
        index=idx,
    )
    out = candles_to_df(df, strict=True)
    assert out.index.is_monotonic_increasing
    assert out.index.tz is not None
    assert str(out.index.tz) == "UTC"
    assert all(out[c].dtype == float for c in ["open", "high", "low", "close"])


def test_candles_to_df_dict_payload_parses_ms_datetime_and_drops_datetime_col_and_sorts():
    # Schwab-style dict payload should parse ms datetime to UTC and drop datetime column.
    t1 = pd.Timestamp("2024-01-02 14:30:00Z").value // 10**6
    t2 = pd.Timestamp("2024-01-02 14:45:00Z").value // 10**6
    payload = {
        "candles": [
            {"datetime": t2, "open": 100, "high": 101, "low": 99, "close": 100, "volume": 2},
            {"datetime": t1, "open": 100, "high": 101, "low": 99, "close": 100, "volume": 1},
        ]
    }
    out = candles_to_df(payload, strict=True)
    assert out.index.is_monotonic_increasing
    assert out.index[0] == pd.Timestamp("2024-01-02 14:30:00Z")
    assert "datetime" not in out.columns


def test_candles_to_df_duplicate_timestamps_strict_raises():
    # Duplicate timestamps should raise in strict=True.
    t = pd.Timestamp("2024-01-02 14:30:00Z").value // 10**6
    payload = {"candles": [{"datetime": t, "open": 1, "high": 1, "low": 1, "close": 1}, {"datetime": t, "open": 2, "high": 2, "low": 2, "close": 2}]}
    with pytest.raises(ValueError):
        candles_to_df(payload, strict=True)


def test_candles_to_df_duplicate_timestamps_strict_false_drops_keep_last():
    # Duplicate timestamps should be deduped (keep last) when strict=False.
    t = pd.Timestamp("2024-01-02 14:30:00Z").value // 10**6
    payload = {"candles": [{"datetime": t, "open": 1, "high": 1, "low": 1, "close": 1}, {"datetime": t, "open": 2, "high": 2, "low": 2, "close": 2}]}
    out = candles_to_df(payload, strict=False)
    assert len(out) == 1
    assert float(out["close"].iloc[0]) == 2.0


def test_candles_to_df_missing_candles_key_raises():
    # Dict payload without "candles" should raise.
    with pytest.raises(ValueError):
        candles_to_df({"symbol": "SPY"})


def test_candles_to_df_list_payload_works():
    # List-of-dicts payload should be accepted.
    t = pd.Timestamp("2024-01-02 14:30:00Z").value // 10**6
    candles = [{"datetime": t, "open": 1, "high": 2, "low": 1, "close": 2, "volume": 3}]
    out = candles_to_df(candles)
    assert out.index[0] == pd.Timestamp("2024-01-02 14:30:00Z")


def test_candles_to_df_missing_required_ohlc_raises():
    # Missing OHLC columns should raise.
    t = pd.Timestamp("2024-01-02 14:30:00Z").value // 10**6
    candles = [{"datetime": t, "open": 1, "close": 1}]
    with pytest.raises(ValueError):
        candles_to_df(candles)


def test_candles_to_df_no_datetime_and_non_datetime_index_raises():
    # If no datetime column and index isn't datetime, candles_to_df should raise.
    df = pd.DataFrame({"open": [1], "high": [1], "low": [1], "close": [1]})
    with pytest.raises(ValueError):
        candles_to_df(df)


def test_apply_adjustment_factor_multiplies_ohlc_and_leaves_volume():
    # Adjustment factor should multiply OHLC but not mutate volume.
    idx = pd.date_range("2024-01-01", periods=2, freq="1D", tz="UTC")
    df = pd.DataFrame({"open": [10.0, 20.0], "high": [11.0, 21.0], "low": [9.0, 19.0], "close": [10.5, 20.5], "volume": [5, 6]}, index=idx)
    factor = pd.Series([2.0, 3.0], index=idx)
    out = apply_adjustment_factor(df, factor)
    assert out["open"].tolist() == [20.0, 60.0]
    assert out["close"].tolist() == [21.0, 61.5]
    assert out["volume"].tolist() == [5, 6]


def test_apply_adjustment_factor_alignment_missing_raises():
    # If alignment introduces NA, apply_adjustment_factor should raise.
    idx = pd.date_range("2024-01-01", periods=2, freq="1D", tz="UTC")
    df = pd.DataFrame({"open": [1.0, 1.0], "high": [1.0, 1.0], "low": [1.0, 1.0], "close": [1.0, 1.0]}, index=idx)
    factor = pd.Series([2.0], index=idx[:1])
    with pytest.raises(ValueError):
        apply_adjustment_factor(df, factor)


def test_apply_adjustment_factor_none_returns_same_df():
    # None should return the original object unchanged.
    idx = pd.date_range("2024-01-01", periods=1, freq="1D", tz="UTC")
    df = pd.DataFrame({"open": [1.0], "high": [1.0], "low": [1.0], "close": [1.0]}, index=idx)
    assert apply_adjustment_factor(df, None) is df


def test_infer_bar_minutes_session_none_strict_raises_on_irregular():
    # Without session info, strict=True should raise on irregular deltas.
    idx = pd.DatetimeIndex(
        [pd.Timestamp("2024-01-01 00:00:00Z"), pd.Timestamp("2024-01-01 00:15:00Z"), pd.Timestamp("2024-01-01 00:45:00Z")]
    )
    with pytest.raises(ValueError):
        infer_bar_minutes(idx, session=None, strict=True)


def test_infer_bar_minutes_session_strict_ignores_overnight_gap():
    # With session provided, strictness should ignore cross-session gaps.
    idx = pd.DatetimeIndex(
        [
            pd.Timestamp("2024-01-02 14:30:00Z"),
            pd.Timestamp("2024-01-02 14:45:00Z"),
            pd.Timestamp("2024-01-02 15:00:00Z"),
            pd.Timestamp("2024-01-03 14:30:00Z"),  # overnight gap
            pd.Timestamp("2024-01-03 14:45:00Z"),
        ]
    )
    minutes = infer_bar_minutes(idx, session=EQUITIES_RTH.session, strict=True)
    assert minutes == 15.0


def test_infer_bar_minutes_session_strict_raises_on_within_session_irregular():
    # Within-session irregular spacing should raise in strict=True.
    idx = pd.DatetimeIndex(
        [pd.Timestamp("2024-01-02 14:30:00Z"), pd.Timestamp("2024-01-02 14:45:00Z"), pd.Timestamp("2024-01-02 15:30:00Z")]
    )
    with pytest.raises(ValueError):
        infer_bar_minutes(idx, session=EQUITIES_RTH.session, strict=True)


def test_infer_bar_minutes_session_strict_false_warns_on_within_session_irregular():
    # strict=False should warn but still infer modal bar minutes.
    idx = pd.DatetimeIndex(
        [pd.Timestamp("2024-01-02 14:30:00Z"), pd.Timestamp("2024-01-02 14:45:00Z"), pd.Timestamp("2024-01-02 15:30:00Z")]
    )
    with pytest.warns(UserWarning):
        minutes = infer_bar_minutes(idx, session=EQUITIES_RTH.session, strict=False)
    assert minutes == 15.0


def test_infer_bar_minutes_len_lt_2_raises():
    # Fewer than 2 timestamps should raise.
    idx = pd.DatetimeIndex([pd.Timestamp("2024-01-01 00:00:00Z")])
    with pytest.raises(ValueError):
        infer_bar_minutes(idx, session=EQUITIES_RTH.session)


def test_aggregate_to_sessions_aggregates_ohlc_and_sets_session_close_index():
    # Aggregation should compute correct OHLC/volume and index by session close UTC.
    idx = pd.DatetimeIndex(
        [
            pd.Timestamp("2024-01-02 14:30:00Z"),
            pd.Timestamp("2024-01-02 14:45:00Z"),
            pd.Timestamp("2024-01-02 15:00:00Z"),
            pd.Timestamp("2024-01-03 14:30:00Z"),
            pd.Timestamp("2024-01-03 14:45:00Z"),
        ]
    )
    df = pd.DataFrame(
        {
            "open": [100, 101, 102, 200, 201],
            "high": [105, 106, 104, 210, 211],
            "low": [99, 100, 98, 199, 200],
            "close": [101, 102, 103, 201, 202],
            "volume": [1, 2, 3, 4, 5],
        },
        index=idx,
    )
    out = aggregate_to_sessions(df, EQUITIES_RTH.session, strict=True)
    assert len(out) == 2
    assert out.index.is_monotonic_increasing
    assert out.index[0] == EQUITIES_RTH.session.session_close_timestamp(pd.Timestamp("2024-01-02", tz="UTC").date())
    assert out.index[1] == EQUITIES_RTH.session.session_close_timestamp(pd.Timestamp("2024-01-03", tz="UTC").date())

    first = out.iloc[0]
    assert first["open"] == 100
    assert first["high"] == 106
    assert first["low"] == 98
    assert first["close"] == 103
    assert first["volume"] == 6


def test_aggregate_to_sessions_raises_on_non_datetime_index():
    # Non-DatetimeIndex should raise.
    df = pd.DataFrame({"open": [1.0], "high": [1.0], "low": [1.0], "close": [1.0]}, index=[1])
    with pytest.raises(ValueError):
        aggregate_to_sessions(df, EQUITIES_RTH.session)


def test_validate_ohlc_raises_on_non_positive_prices():
    # Prices <= 0 should raise.
    idx = pd.date_range("2024-01-01", periods=1, freq="1D", tz="UTC")
    df = pd.DataFrame({"open": [0.0], "high": [1.0], "low": [1.0], "close": [1.0]}, index=idx)
    with pytest.raises(ValueError):
        validate_ohlc(df, strict=True)


def test_validate_ohlc_raises_on_high_below_low():
    # high < low should raise.
    idx = pd.date_range("2024-01-01", periods=1, freq="1D", tz="UTC")
    df = pd.DataFrame({"open": [1.0], "high": [0.9], "low": [1.0], "close": [1.0]}, index=idx)
    with pytest.raises(ValueError):
        validate_ohlc(df, strict=True)


def test_validate_ohlc_strict_raises_open_outside_range_with_details():
    # strict=True should raise and include count + sample row details.
    idx = pd.date_range("2024-01-01", periods=1, freq="1D", tz="UTC")
    df = pd.DataFrame({"open": [2.0], "high": [1.5], "low": [1.0], "close": [1.2]}, index=idx)
    with pytest.raises(ValueError) as exc:
        validate_ohlc(df, strict=True)
    msg = str(exc.value)
    assert "Found open outside [low, high]" in msg
    assert "for 1 rows" in msg
    assert str(idx[0]) in msg


def test_validate_ohlc_strict_raises_close_outside_range_with_details():
    # strict=True should raise and include count + sample row details.
    idx = pd.date_range("2024-01-01", periods=1, freq="1D", tz="UTC")
    df = pd.DataFrame({"open": [1.2], "high": [1.5], "low": [1.0], "close": [2.0]}, index=idx)
    with pytest.raises(ValueError) as exc:
        validate_ohlc(df, strict=True)
    msg = str(exc.value)
    assert "Found close outside [low, high]" in msg
    assert "for 1 rows" in msg
    assert str(idx[0]) in msg


def test_validate_ohlc_non_strict_warns_on_range_inconsistency():
    # strict=False should warn (not raise) for open/close outside [low, high].
    idx = pd.date_range("2024-01-01", periods=2, freq="1D", tz="UTC")
    df = pd.DataFrame(
        {"open": [2.0, 1.2], "high": [1.5, 1.5], "low": [1.0, 1.0], "close": [1.2, 2.0]},
        index=idx,
    )
    with pytest.warns(UserWarning):
        validate_ohlc(df, strict=False)


def test_validate_ohlc_missing_columns_raises():
    # Missing required OHLC columns should raise.
    idx = pd.date_range("2024-01-01", periods=1, freq="1D", tz="UTC")
    df = pd.DataFrame({"open": [1.0], "close": [1.0]}, index=idx)
    with pytest.raises(ValueError):
        validate_ohlc(df, strict=True)

