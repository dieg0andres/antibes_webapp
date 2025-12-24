"""Tests for options_trading.volatility.panel (Volatility State Panel)."""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
import pytest

from options_trading.volatility.calendar import EQUITIES_RTH
from options_trading.volatility.panel import VolatilityStatePanelResult, volatility_state_panel


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_session_payload(
    n_sessions: int,
    start_date: str = "2024-01-02",
    time_utc: str = "12:00",
    as_format: Literal["dataframe", "dict", "list"] = "dict",
    seed: int = 0,
    constant: bool = False,
):
    """Generate deterministic session/daily candles."""
    idx_dates = pd.date_range(start_date, periods=n_sessions, freq="B", tz="UTC")
    idx = pd.DatetimeIndex([pd.Timestamp(f"{d.date()} {time_utc}").tz_localize("UTC") for d in idx_dates])

    if constant:
        close = pd.Series(100.0, index=idx)
    else:
        rng = np.random.default_rng(seed)
        trend = np.linspace(0, 0.5, n_sessions)
        noise = rng.normal(scale=0.02, size=n_sessions)
        close = pd.Series(100.0 + trend + noise, index=idx)

    open_ = close.shift(1).fillna(close)
    high = np.maximum(open_, close) + 0.10
    low = np.minimum(open_, close) - 0.10
    volume = pd.Series(1000, index=idx)

    df = pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": volume})

    if as_format == "dataframe":
        return df

    candles = []
    for ts, row in df.iterrows():
        candles.append(
            {
                "datetime": ts.value // 10**6,
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                "volume": int(row["volume"]),
            }
        )

    if as_format == "list":
        return candles

    if as_format == "dict":
        return {"candles": candles, "symbol": "TEST", "empty": False}

    raise ValueError("Unsupported format")


def make_intraday_payload(
    n_sessions: int,
    bars_per_session: int = 2,
    bar_minutes: int = 15,
    start_date: str = "2024-01-02",
    as_format: Literal["dataframe", "dict", "list"] = "dict",
    seed: int = 1,
    constant: bool = False,
):
    """Generate deterministic intraday candles within RTH (winter, Jan/Feb)."""
    session_dates = pd.date_range(start_date, periods=n_sessions, freq="B", tz="UTC")
    timestamps = []
    for d in session_dates:
        base = pd.Timestamp(f"{d.date()} 14:30").tz_localize("UTC")
        for i in range(bars_per_session):
            timestamps.append(base + pd.Timedelta(minutes=i * bar_minutes))
    idx = pd.DatetimeIndex(timestamps)

    total_bars = len(idx)
    if constant:
        close = pd.Series(100.0, index=idx)
    else:
        rng = np.random.default_rng(seed)
        # small drift across bars
        drift = np.linspace(0, 0.5, total_bars)
        noise = rng.normal(scale=0.02, size=total_bars)
        close = pd.Series(100.0 + drift + noise, index=idx)

    open_ = close.shift(1).fillna(close)
    high = np.maximum(open_, close) + 0.05
    low = np.minimum(open_, close) - 0.05
    volume = pd.Series(500, index=idx)

    df = pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": volume})

    if as_format == "dataframe":
        return df

    candles = []
    for ts, row in df.iterrows():
        candles.append(
            {
                "datetime": ts.value // 10**6,
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                "volume": int(row["volume"]),
            }
        )

    if as_format == "list":
        return candles

    if as_format == "dict":
        return {"candles": candles, "symbol": "TEST", "empty": False}

    raise ValueError("Unsupported format")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("fmt", ["dataframe", "dict", "list"])
def test_panel_accepts_session_formats_dataframe_dict_list(fmt):
    session_payload = make_session_payload(30, as_format=fmt)
    res = volatility_state_panel(
        session_data=session_payload,
        intraday_data=None,
        calendar=EQUITIES_RTH,
        rv_windows=(5, 10, 20),
        heat_pairs=((10, 20), (5, 20)),
        z_lookback=30,
        z_windows=(20,),
        rank_lookback=30,
        rank_windows=(20,),
        include_rank_percentile=True,
        tail_sigma_window=10,
        tail_lookback=30,
        corr_window=20,
        corr_vol_window=20,
        trend_ma_window=10,
    )
    assert isinstance(res, VolatilityStatePanelResult)
    assert isinstance(res.series, pd.DataFrame)
    assert not res.series.empty
    assert res.series.index.tz is not None
    assert "close" in res.series.columns
    assert "ret_log" in res.series.columns


@pytest.mark.parametrize("fmt", ["dataframe", "dict", "list"])
def test_panel_accepts_intraday_formats_dataframe_dict_list(fmt):
    session_payload = make_session_payload(40, as_format="dict")
    intraday_payload = make_intraday_payload(40, as_format=fmt, bars_per_session=2)
    res = volatility_state_panel(
        session_data=session_payload,
        intraday_data=intraday_payload,
        calendar=EQUITIES_RTH,
        rv_windows=(5, 10, 20),
        primary_max_window=20,
        heat_pairs=((10, 20), (5, 20)),
        z_lookback=30,
        z_windows=(20,),
        rank_lookback=30,
        rank_windows=(20,),
        include_rank_percentile=True,
        tail_sigma_window=10,
        tail_lookback=30,
        corr_window=20,
        corr_vol_window=20,
        trend_ma_window=10,
    )
    assert "vol_rvar_20" in res.series.columns
    assert res.series["primary_src_20"].iloc[-1] == "rvar"


def test_panel_index_is_session_close_utc_for_winter_dates():
    session_payload = make_session_payload(10, start_date="2024-01-02", as_format="dict")
    res = volatility_state_panel(
        session_data=session_payload,
        intraday_data=None,
        calendar=EQUITIES_RTH,
        rv_windows=(5, 10, 20),
        primary_max_window=20,
        heat_pairs=((10, 20), (5, 20)),
        z_lookback=30,
        z_windows=(20,),
        rank_lookback=30,
        rank_windows=(20,),
        include_rank_percentile=True,
        tail_sigma_window=10,
        tail_lookback=30,
        corr_window=20,
        corr_vol_window=20,
        trend_ma_window=10,
    )
    for ts in res.series.index:
        assert ts.tz is not None
        assert ts.hour == 21 and ts.minute == 0  # 16:00 ET winter => 21:00 UTC


def test_panel_expected_columns_present_basic():
    session_payload = make_session_payload(60, as_format="dict")
    intraday_payload = make_intraday_payload(60, as_format="dict", bars_per_session=2)
    res = volatility_state_panel(
        session_data=session_payload,
        intraday_data=intraday_payload,
        calendar=EQUITIES_RTH,
        rv_windows=(5, 10, 20),
        primary_max_window=20,
        heat_pairs=((10, 20), (5, 20)),
        z_lookback=30,
        z_windows=(20,),
        rank_lookback=30,
        rank_windows=(20,),
        include_rank_percentile=True,
        tail_sigma_window=10,
        tail_lookback=30,
        corr_window=20,
        corr_vol_window=20,
        trend_ma_window=10,
    )
    cols = res.series.columns
    expected = [
        "close",
        "ret_log",
        "vol_yz_5",
        "vol_yz_10",
        "vol_yz_20",
        "vol_rvar_5",
        "vol_rvar_10",
        "vol_rvar_20",
        "vol_primary_5",
        "vol_primary_10",
        "vol_primary_20",
        "primary_src_5",
        "primary_src_10",
        "primary_src_20",
        "heat_yz_10_20",
        "heat_primary_10_20",
        "heat_yz_5_20",
        "heat_primary_5_20",
        "zlogvol_yz_20_30",
        "rv_rank_yz_20_30",
        "rv_pct_yz_20_30",
        "sigma_ref_10",
        "p_tail2_30",
        "p_tail3_30",
        "dlogvol_yz_20",
        "corr_ret_dlogvol_yz20_20",
        "sma_10",
        "is_bull_10",
    ]
    for e in expected:
        assert e in cols


def test_latest_equals_last_row():
    session_payload = make_session_payload(40, as_format="dict")
    res = volatility_state_panel(
        session_data=session_payload,
        intraday_data=None,
        calendar=EQUITIES_RTH,
        rv_windows=(5, 10, 20),
        heat_pairs=((10, 20), (5, 20)),
        z_lookback=30,
        z_windows=(20,),
        rank_lookback=30,
        rank_windows=(20,),
        include_rank_percentile=True,
        tail_sigma_window=10,
        tail_lookback=30,
        corr_window=20,
        corr_vol_window=20,
        trend_ma_window=10,
    )
    pd.testing.assert_series_equal(res.latest, res.series.iloc[-1])


def test_toggle_include_realized_variance_false():
    session_payload = make_session_payload(40, as_format="dict")
    intraday_payload = make_intraday_payload(40, as_format="dict")
    res = volatility_state_panel(
        session_data=session_payload,
        intraday_data=intraday_payload,
        include_realized_variance=False,
        calendar=EQUITIES_RTH,
        rv_windows=(5, 10, 20),
        primary_max_window=20,
        heat_pairs=((10, 20), (5, 20)),
        z_lookback=30,
        z_windows=(20,),
        rank_lookback=30,
        rank_windows=(20,),
        include_rank_percentile=True,
        tail_sigma_window=10,
        tail_lookback=30,
        corr_window=20,
        corr_vol_window=20,
        trend_ma_window=10,
    )
    assert not any(col.startswith("vol_rvar_") for col in res.series.columns)
    for w in (5, 10, 20):
        assert (res.series[f"primary_src_{w}"] == "yz").all()
        np.testing.assert_allclose(
            res.series[f"vol_primary_{w}"].dropna(), res.series[f"vol_yz_{w}"].dropna()
        )


def test_toggle_include_rank_percentile_false():
    session_payload = make_session_payload(40, as_format="dict")
    res = volatility_state_panel(
        session_data=session_payload,
        intraday_data=None,
        include_rank_percentile=False,
        calendar=EQUITIES_RTH,
        rv_windows=(5, 10, 20),
        heat_pairs=((10, 20), (5, 20)),
        z_lookback=30,
        z_windows=(20,),
        rank_lookback=30,
        rank_windows=(20,),
        tail_sigma_window=10,
        tail_lookback=30,
        corr_window=20,
        corr_vol_window=20,
        trend_ma_window=10,
    )
    assert not any(col.startswith("rv_rank_yz") for col in res.series.columns)
    assert not any(col.startswith("rv_pct_yz") for col in res.series.columns)


def test_toggle_include_regime_false():
    session_payload = make_session_payload(40, as_format="dict")
    res = volatility_state_panel(
        session_data=session_payload,
        intraday_data=None,
        include_regime=False,
        calendar=EQUITIES_RTH,
        rv_windows=(5, 10, 20),
        heat_pairs=((10, 20), (5, 20)),
        z_lookback=30,
        z_windows=(20,),
        rank_lookback=30,
        rank_windows=(20,),
        include_rank_percentile=True,
        tail_sigma_window=10,
        tail_lookback=30,
        corr_window=20,
        corr_vol_window=20,
        trend_ma_window=10,
    )
    assert not any(col.startswith("sma_") for col in res.series.columns)
    assert not any(col.startswith("is_bull_") for col in res.series.columns)


def test_returns_definition():
    session_payload = make_session_payload(20, as_format="dict")
    res = volatility_state_panel(
        session_data=session_payload,
        intraday_data=None,
        calendar=EQUITIES_RTH,
        rv_windows=(5, 10, 20),
        heat_pairs=((10, 20), (5, 20)),
        z_lookback=10,
        z_windows=(10,),
        rank_lookback=10,
        rank_windows=(10,),
        include_rank_percentile=False,
        tail_sigma_window=5,
        tail_lookback=10,
        corr_window=10,
        corr_vol_window=10,
        trend_ma_window=5,
    )
    close = res.series["close"]
    expected = np.log(close / close.shift(1))
    np.testing.assert_allclose(res.series["ret_log"], expected, equal_nan=True)


def test_heat_matches_definition():
    session_payload = make_session_payload(40, as_format="dict")
    res = volatility_state_panel(
        session_data=session_payload,
        intraday_data=None,
        calendar=EQUITIES_RTH,
        rv_windows=(10, 20),
        heat_pairs=((10, 20),),
        z_lookback=10,
        z_windows=(10,),
        rank_lookback=10,
        rank_windows=(10,),
        include_rank_percentile=False,
        tail_sigma_window=5,
        tail_lookback=10,
        corr_window=10,
        corr_vol_window=10,
        trend_ma_window=5,
    )
    v10 = res.series["vol_yz_10"]
    v20 = res.series["vol_yz_20"]
    expected = np.log(v10.where(v10 > 0)) - np.log(v20.where(v20 > 0))
    actual = res.series["heat_yz_10_20"]
    mask = expected.notna() & actual.notna()
    np.testing.assert_allclose(actual[mask], expected[mask], rtol=1e-12, atol=1e-12)


def test_zscore_matches_reference():
    session_payload = make_session_payload(60, as_format="dict")
    res = volatility_state_panel(
        session_data=session_payload,
        intraday_data=None,
        calendar=EQUITIES_RTH,
        rv_windows=(20,),
        heat_pairs=((20, 20),),
        z_lookback=30,
        z_windows=(20,),
        rank_lookback=10,
        rank_windows=(10,),
        include_rank_percentile=False,
        tail_sigma_window=5,
        tail_lookback=10,
        corr_window=10,
        corr_vol_window=20,
        trend_ma_window=5,
    )
    col = "vol_yz_20"
    x = np.log(res.series[col].where(res.series[col] > 0))
    mu = x.rolling(window=30, min_periods=30).mean()
    sd = x.rolling(window=30, min_periods=30).std(ddof=0).replace(0.0, np.nan)
    expected = (x - mu) / sd
    actual = res.series["zlogvol_yz_20_30"]
    mask = expected.notna() & actual.notna()
    np.testing.assert_allclose(actual[mask], expected[mask], rtol=1e-12, atol=1e-12)


def test_rank_percentile_match_reference():
    session_payload = make_session_payload(60, as_format="dict")
    res = volatility_state_panel(
        session_data=session_payload,
        intraday_data=None,
        calendar=EQUITIES_RTH,
        rv_windows=(20,),
        heat_pairs=((20, 20),),
        z_lookback=10,
        z_windows=(20,),
        rank_lookback=30,
        rank_windows=(20,),
        include_rank_percentile=True,
        tail_sigma_window=5,
        tail_lookback=10,
        corr_window=10,
        corr_vol_window=20,
        trend_ma_window=5,
    )
    s = res.series["vol_yz_20"]
    roll_min = s.rolling(30, min_periods=30).min()
    roll_max = s.rolling(30, min_periods=30).max()
    denom = roll_max - roll_min
    expected_rank = (s - roll_min) / denom.replace(0.0, np.nan)

    def _pct(series: pd.Series) -> float:
        cur = series.iloc[-1]
        return float((series <= cur).mean())

    expected_pct = s.rolling(30, min_periods=30).apply(_pct, raw=False)

    rank_actual = res.series["rv_rank_yz_20_30"]
    pct_actual = res.series["rv_pct_yz_20_30"]

    mask_rank = expected_rank.notna() & rank_actual.notna()
    mask_pct = expected_pct.notna() & pct_actual.notna()

    np.testing.assert_allclose(rank_actual[mask_rank], expected_rank[mask_rank], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(pct_actual[mask_pct], expected_pct[mask_pct], rtol=1e-12, atol=1e-12)


def test_rank_denominator_zero_constant_vol_yields_nan():
    session_payload = make_session_payload(30, as_format="dict", constant=True)
    res = volatility_state_panel(
        session_data=session_payload,
        intraday_data=None,
        calendar=EQUITIES_RTH,
        rv_windows=(5, 10),
        heat_pairs=((5, 10),),
        z_lookback=10,
        z_windows=(10,),
        rank_lookback=10,
        rank_windows=(5,),
        include_rank_percentile=True,
        tail_sigma_window=5,
        tail_lookback=10,
        corr_window=10,
        corr_vol_window=5,
        trend_ma_window=5,
    )
    rank_series = res.series["rv_rank_yz_5_10"]
    assert rank_series.iloc[9:].isna().all()


def test_tail_features_detect_outlier():
    n = 15
    session_payload = make_session_payload(n, as_format="dataframe")
    # Inject an outlier on the last day
    session_payload.loc[session_payload.index[-1], "close"] = session_payload["close"].iloc[-1] * 3.0
    session_payload.loc[session_payload.index[-1], "open"] = session_payload["close"].iloc[-2]
    session_payload.loc[session_payload.index[-1], "high"] = max(
        session_payload["open"].iloc[-1], session_payload["close"].iloc[-1]
    ) + 0.1
    session_payload.loc[session_payload.index[-1], "low"] = min(
        session_payload["open"].iloc[-1], session_payload["close"].iloc[-1]
    ) - 0.1

    res = volatility_state_panel(
        session_data=session_payload,
        intraday_data=None,
        calendar=EQUITIES_RTH,
        rv_windows=(5, 10),
        heat_pairs=((5, 10),),
        z_lookback=5,
        z_windows=(5,),
        rank_lookback=5,
        rank_windows=(5,),
        include_rank_percentile=False,
        tail_sigma_window=5,
        tail_lookback=10,
        corr_window=5,
        corr_vol_window=5,
        trend_ma_window=5,
    )
    assert res.series[f"p_tail3_10"].iloc[-1] > 0


def test_corr_matches_pandas_rolling_corr():
    session_payload = make_session_payload(60, as_format="dict")
    res = volatility_state_panel(
        session_data=session_payload,
        intraday_data=None,
        calendar=EQUITIES_RTH,
        rv_windows=(20,),
        heat_pairs=((20, 20),),
        z_lookback=10,
        z_windows=(20,),
        rank_lookback=10,
        rank_windows=(10,),
        include_rank_percentile=False,
        tail_sigma_window=5,
        tail_lookback=10,
        corr_window=20,
        corr_vol_window=20,
        trend_ma_window=5,
    )
    vol = res.series["vol_yz_20"]
    dlog = np.log(vol.where(vol > 0)).diff()
    expected = res.series["ret_log"].rolling(20).corr(dlog)
    actual = res.series["corr_ret_dlogvol_yz20_20"]
    mask = expected.notna() & actual.notna()
    np.testing.assert_allclose(actual[mask], expected[mask], rtol=1e-12, atol=1e-12)


def test_regime_columns_logic():
    session_payload = make_session_payload(20, as_format="dict")
    res = volatility_state_panel(
        session_data=session_payload,
        intraday_data=None,
        calendar=EQUITIES_RTH,
        rv_windows=(5, 10),
        heat_pairs=((5, 10),),
        z_lookback=5,
        z_windows=(5,),
        rank_lookback=5,
        rank_windows=(5,),
        include_rank_percentile=False,
        tail_sigma_window=5,
        tail_lookback=10,
        corr_window=5,
        corr_vol_window=5,
        trend_ma_window=5,
    )
    close = res.series["close"]
    sma = close.rolling(5, min_periods=5).mean()
    is_bull = close > sma
    mask = sma.notna()
    np.testing.assert_allclose(res.series["sma_5"][mask], sma[mask])
    assert res.series["is_bull_5"][mask].equals(is_bull[mask])


def test_primary_uses_rvar_when_available_w_le_primary_max_window():
    session_payload = make_session_payload(40, as_format="dict")
    intraday_payload = make_intraday_payload(40, as_format="dict", bars_per_session=2)
    res = volatility_state_panel(
        session_data=session_payload,
        intraday_data=intraday_payload,
        calendar=EQUITIES_RTH,
        rv_windows=(5, 10, 20),
        primary_max_window=20,
        heat_pairs=((10, 20), (5, 20)),
        z_lookback=10,
        z_windows=(10,),
        rank_lookback=10,
        rank_windows=(10,),
        include_rank_percentile=False,
        tail_sigma_window=5,
        tail_lookback=10,
        corr_window=10,
        corr_vol_window=10,
        trend_ma_window=5,
    )
    assert res.series["primary_src_20"].iloc[-1] == "rvar"
    np.testing.assert_allclose(
        res.series["vol_primary_20"].iloc[-1], res.series["vol_rvar_20"].iloc[-1], rtol=1e-12, atol=1e-12
    )


def test_primary_falls_back_to_yz_when_intraday_insufficient_for_window():
    session_payload = make_session_payload(40, as_format="dict")
    intraday_payload = make_intraday_payload(10, as_format="dict", bars_per_session=2)
    res = volatility_state_panel(
        session_data=session_payload,
        intraday_data=intraday_payload,
        calendar=EQUITIES_RTH,
        rv_windows=(20,),
        primary_max_window=20,
        heat_pairs=((20, 20),),
        z_lookback=5,
        z_windows=(20,),
        rank_lookback=5,
        rank_windows=(5,),
        include_rank_percentile=False,
        tail_sigma_window=5,
        tail_lookback=5,
        corr_window=5,
        corr_vol_window=20,
        trend_ma_window=5,
    )
    assert res.series["primary_src_20"].iloc[-1] == "yz"
    np.testing.assert_allclose(
        res.series["vol_primary_20"].iloc[-1], res.series["vol_yz_20"].iloc[-1], rtol=1e-12, atol=1e-12
    )


def test_primary_respects_primary_max_window():
    session_payload = make_session_payload(40, as_format="dict")
    intraday_payload = make_intraday_payload(40, as_format="dict", bars_per_session=2)
    res = volatility_state_panel(
        session_data=session_payload,
        intraday_data=intraday_payload,
        calendar=EQUITIES_RTH,
        rv_windows=(5, 10, 20),
        primary_max_window=10,
        heat_pairs=((10, 20), (5, 20)),
        z_lookback=10,
        z_windows=(10,),
        rank_lookback=10,
        rank_windows=(10,),
        include_rank_percentile=False,
        tail_sigma_window=5,
        tail_lookback=10,
        corr_window=10,
        corr_vol_window=10,
        trend_ma_window=5,
    )
    assert res.series["primary_src_20"].iloc[-1] == "yz"
    np.testing.assert_allclose(
        res.series["vol_primary_20"].iloc[-1], res.series["vol_yz_20"].iloc[-1], rtol=1e-12, atol=1e-12
    )


def test_validation_z_windows_must_be_subset_of_rv_windows():
    session_payload = make_session_payload(10, as_format="dict")
    with pytest.raises(ValueError):
        volatility_state_panel(
            session_data=session_payload,
            intraday_data=None,
            calendar=EQUITIES_RTH,
            rv_windows=(5, 10),
            z_windows=(20,),
        )


def test_validation_rank_windows_must_be_subset_of_rv_windows_when_enabled():
    session_payload = make_session_payload(10, as_format="dict")
    with pytest.raises(ValueError):
        volatility_state_panel(
            session_data=session_payload,
            intraday_data=None,
            calendar=EQUITIES_RTH,
            rv_windows=(5, 10),
            rank_windows=(20,),
            include_rank_percentile=True,
        )


def test_validation_corr_vol_window_must_be_in_rv_windows():
    session_payload = make_session_payload(10, as_format="dict")
    with pytest.raises(ValueError):
        volatility_state_panel(
            session_data=session_payload,
            intraday_data=None,
            calendar=EQUITIES_RTH,
            rv_windows=(5, 10),
            corr_vol_window=20,
        )


def test_validation_heat_pairs_must_be_in_rv_windows():
    session_payload = make_session_payload(10, as_format="dict")
    with pytest.raises(ValueError):
        volatility_state_panel(
            session_data=session_payload,
            intraday_data=None,
            calendar=EQUITIES_RTH,
            rv_windows=(5, 10),
            heat_pairs=((5, 20),),
        )


def test_adjustment_factor_scales_close_but_not_vol():
    session_df = make_session_payload(30, as_format="dataframe")
    factor = pd.Series(10.0, index=session_df.index, name="scale10")

    res_no = volatility_state_panel(
        session_data=session_df,
        intraday_data=None,
        calendar=EQUITIES_RTH,
        rv_windows=(20,),
        heat_pairs=((20, 20),),
        z_lookback=10,
        z_windows=(20,),
        rank_lookback=10,
        rank_windows=(10,),
        include_rank_percentile=False,
        tail_sigma_window=5,
        tail_lookback=10,
        corr_window=5,
        corr_vol_window=20,
        trend_ma_window=5,
    )

    res_with = volatility_state_panel(
        session_data=session_df,
        intraday_data=None,
        calendar=EQUITIES_RTH,
        rv_windows=(20,),
        heat_pairs=((20, 20),),
        z_lookback=10,
        z_windows=(20,),
        rank_lookback=10,
        rank_windows=(10,),
        include_rank_percentile=False,
        tail_sigma_window=5,
        tail_lookback=10,
        corr_window=5,
        corr_vol_window=20,
        trend_ma_window=5,
        adjustment_factor=factor,
    )

    # close should scale by 10
    np.testing.assert_allclose(
        res_with.series["close"], res_no.series["close"] * 10.0, rtol=1e-12, atol=1e-12
    )

    # vol_yz_20 should be scale invariant
    np.testing.assert_allclose(
        res_with.series["vol_yz_20"].dropna(), res_no.series["vol_yz_20"].dropna(), rtol=1e-12, atol=1e-12
    )

def test_panel_index_is_session_close_utc_for_summer_dates_dst():
    # July is in US DST (EDT), so 16:00 ET == 20:00 UTC.
    session_payload = make_session_payload(
        5, start_date="2024-07-01", time_utc="12:00", as_format="dict"
    )

    res = volatility_state_panel(
        session_data=session_payload,
        intraday_data=None,
        calendar=EQUITIES_RTH,
        rv_windows=(5,),
        heat_pairs=((5, 5),),
        z_lookback=2,
        z_windows=(5,),
        rank_lookback=5,
        rank_windows=(5,),
        include_rank_percentile=False,
        tail_sigma_window=2,
        tail_lookback=2,
        corr_window=2,
        corr_vol_window=5,
        include_regime=False,
        trend_ma_window=3,
        strict=True,
        annualize=False,
    )

    ts0 = res.series.index[0].tz_convert("UTC")
    assert ts0.hour == 20 and ts0.minute == 0


def _make_intraday_two_bar_reversal_path(
    n_sessions: int,
    *,
    start_date: str,
    first_leg_mult: float,
) -> pd.DataFrame:
    """
    Build intraday candles (2 bars per session) where each session:
      price goes 100 -> 100*mult -> 100  (ends flat)
    This keeps the *session* close the same, but changes the intraday path,
    hence realized variance should change.
    """
    session_dates = pd.date_range(start_date, periods=n_sessions, freq="B", tz="UTC")

    idx = []
    rows = []
    for d in session_dates:
        t0 = pd.Timestamp(f"{d.date()} 14:30").tz_localize("UTC")  # winter RTH open in UTC
        t1 = t0 + pd.Timedelta(minutes=15)

        base = 100.0
        mid = base * first_leg_mult

        # Bar 1: base -> mid
        o1, c1 = base, mid
        h1 = max(o1, c1) + 0.1
        l1 = min(o1, c1) - 0.1
        # Bar 2: mid -> base
        o2, c2 = mid, base
        h2 = max(o2, c2) + 0.1
        l2 = min(o2, c2) - 0.1

        idx.extend([t0, t1])
        rows.extend(
            [
                {"open": o1, "high": h1, "low": l1, "close": c1, "volume": 100},
                {"open": o2, "high": h2, "low": l2, "close": c2, "volume": 100},
            ]
        )

    return pd.DataFrame(rows, index=pd.DatetimeIndex(idx))


def test_realized_variance_changes_with_intraday_path_but_yz_does_not():
    """
    Two different intraday paths (same session end price each day) should produce
    different realized variance vol, while YZ (computed from session OHLC input)
    remains identical because session_data is identical in both runs.
    """
    n = 12
    w = 5

    # Session candles are identical between both runs (YZ uses these).
    session_payload = make_session_payload(n, start_date="2024-01-02", as_format="dataframe")

    intraday_hi = _make_intraday_two_bar_reversal_path(
        n, start_date="2024-01-02", first_leg_mult=1.10  # bigger intraday swings
    )
    intraday_lo = _make_intraday_two_bar_reversal_path(
        n, start_date="2024-01-02", first_leg_mult=1.02  # smaller intraday swings
    )

    res_hi = volatility_state_panel(
        session_data=session_payload,
        intraday_data=intraday_hi,
        calendar=EQUITIES_RTH,
        rv_windows=(w,),
        primary_max_window=w,
        include_realized_variance=True,
        include_overnight=False,
        annualize=False,
        strict=True,
        heat_pairs=((w, w),),
        z_lookback=2,
        z_windows=(w,),
        include_rank_percentile=False,
        tail_sigma_window=2,
        tail_lookback=2,
        corr_window=2,
        corr_vol_window=w,
        include_regime=False,
        trend_ma_window=3,
    )

    res_lo = volatility_state_panel(
        session_data=session_payload,
        intraday_data=intraday_lo,
        calendar=EQUITIES_RTH,
        rv_windows=(w,),
        primary_max_window=w,
        include_realized_variance=True,
        include_overnight=False,
        annualize=False,
        strict=True,
        heat_pairs=((w, w),),
        z_lookback=2,
        z_windows=(w,),
        include_rank_percentile=False,
        tail_sigma_window=2,
        tail_lookback=2,
        corr_window=2,
        corr_vol_window=w,
        include_regime=False,
        trend_ma_window=3,
    )

    # YZ should be identical (session_data identical)
    np.testing.assert_allclose(
        res_hi.series[f"vol_yz_{w}"].dropna().to_numpy(),
        res_lo.series[f"vol_yz_{w}"].dropna().to_numpy(),
        rtol=0,
        atol=0,
    )

    # Realized variance should be larger for the higher-swing path
    v_hi = res_hi.series[f"vol_rvar_{w}"].dropna().iloc[-1]
    v_lo = res_lo.series[f"vol_rvar_{w}"].dropna().iloc[-1]
    assert v_hi > v_lo


def _make_intraday_flat_with_overnight_gaps(n_sessions: int, *, start_date: str) -> pd.DataFrame:
    """
    Intraday is flat within each session (zero intraday variance),
    but the session-to-session level changes (overnight gaps).
    """
    session_dates = pd.date_range(start_date, periods=n_sessions, freq="B", tz="UTC")

    idx = []
    rows = []
    for i, d in enumerate(session_dates):
        # winter RTH open in UTC for this synthetic set
        t0 = pd.Timestamp(f"{d.date()} 14:30").tz_localize("UTC")
        t1 = t0 + pd.Timedelta(minutes=15)

        # flat within session, but level changes across sessions:
        level = 100.0 * (1.0 + 0.20 * i)  # big gaps
        o1 = c1 = o2 = c2 = level
        h1 = h2 = level + 0.1
        l1 = l2 = level - 0.1

        idx.extend([t0, t1])
        rows.extend(
            [
                {"open": o1, "high": h1, "low": l1, "close": c1, "volume": 100},
                {"open": o2, "high": h2, "low": l2, "close": c2, "volume": 100},
            ]
        )

    return pd.DataFrame(rows, index=pd.DatetimeIndex(idx))


def test_include_overnight_toggle_affects_realized_variance_when_gaps_exist():
    """
    If intraday is flat but there are big overnight gaps, realized variance
    should be ~0 when include_overnight=False, and >0 when include_overnight=True.
    """
    n = 12
    w = 5

    # Session data doesn't matter for rvar math; keep it simple and valid.
    session_payload = make_session_payload(n, start_date="2024-01-02", as_format="dataframe", constant=True)
    intraday_payload = _make_intraday_flat_with_overnight_gaps(n, start_date="2024-01-02")

    res_no_ov = volatility_state_panel(
        session_data=session_payload,
        intraday_data=intraday_payload,
        calendar=EQUITIES_RTH,
        rv_windows=(w,),
        primary_max_window=w,
        include_realized_variance=True,
        include_overnight=False,
        annualize=False,
        strict=True,
        heat_pairs=((w, w),),
        z_lookback=2,
        z_windows=(w,),
        include_rank_percentile=False,
        tail_sigma_window=2,
        tail_lookback=2,
        corr_window=2,
        corr_vol_window=w,
        include_regime=False,
        trend_ma_window=3,
    )

    res_yes_ov = volatility_state_panel(
        session_data=session_payload,
        intraday_data=intraday_payload,
        calendar=EQUITIES_RTH,
        rv_windows=(w,),
        primary_max_window=w,
        include_realized_variance=True,
        include_overnight=True,
        annualize=False,
        strict=True,
        heat_pairs=((w, w),),
        z_lookback=2,
        z_windows=(w,),
        include_rank_percentile=False,
        tail_sigma_window=2,
        tail_lookback=2,
        corr_window=2,
        corr_vol_window=w,
        include_regime=False,
        trend_ma_window=3,
    )

    v_no = res_no_ov.series[f"vol_rvar_{w}"].dropna().iloc[-1]
    v_yes = res_yes_ov.series[f"vol_rvar_{w}"].dropna().iloc[-1]

    # If include_overnight=False, intraday flat should yield ~0 rvar vol.
    assert v_no == pytest.approx(0.0, abs=1e-12)
    # With overnight included, the gaps should create >0 vol.
    assert v_yes > 0.0


def test_percentile_constant_window_is_one_even_when_rank_is_nan():
    """
    When the rolling window is constant:
      - rank is NaN (denom=0)
      - percentile is 1.0 once it becomes defined

    Note: percentile output has additional warmup beyond rank_lookback because
    vol_yz_w itself has an initial warmup (NaNs). Rolling uses min_periods
    based on non-NaN count.
    """
    session_payload = make_session_payload(30, as_format="dict", constant=True)
    rank_lookback = 10

    res = volatility_state_panel(
        session_data=session_payload,
        intraday_data=None,
        calendar=EQUITIES_RTH,
        rv_windows=(5, 10),
        heat_pairs=((5, 10),),
        z_lookback=10,
        z_windows=(10,),
        rank_lookback=rank_lookback,
        rank_windows=(5,),
        include_rank_percentile=True,
        tail_sigma_window=5,
        tail_lookback=10,
        corr_window=10,
        corr_vol_window=5,
        include_regime=False,
        trend_ma_window=5,
    )

    pct = res.series["rv_pct_yz_5_10"]

    # Ensure it eventually becomes defined.
    first_valid = pct.first_valid_index()
    assert first_valid is not None

    # Once defined, for a constant series it should be 1.0 everywhere.
    after = pct.loc[first_valid:]
    np.testing.assert_allclose(after.to_numpy(), np.ones(len(after)), rtol=0, atol=0)

