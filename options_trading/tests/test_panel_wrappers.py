"""Tests for options_trading.volatility.panel_wrappers."""

from __future__ import annotations

import pandas as pd
import pytest

import options_trading.volatility.panel_wrappers as wrappers


class CallRecorder:
    def __init__(self):
        self.calls = []

    def fn(self, return_value=None, raise_exc=None):
        def _f(**kwargs):
            self.calls.append(kwargs)
            if raise_exc is not None:
                raise raise_exc
            return return_value

        return _f


def test_wrapper_uses_defaults_and_skips_intraday_when_none(monkeypatch):
    sentinel_session = object()
    sentinel_result = object()

    session_rec = CallRecorder()
    session_loader = session_rec.fn(return_value=sentinel_session)

    panel_rec = CallRecorder()
    fake_panel = panel_rec.fn(return_value=sentinel_result)
    monkeypatch.setattr(wrappers, "volatility_state_panel", fake_panel)

    result = wrappers.volatility_state_panel_for_ticker(
        ticker="GLD",
        load_session_candles=session_loader,
        load_intraday_candles=None,
    )

    assert result is sentinel_result
    assert len(session_rec.calls) == 1
    assert session_rec.calls[0] == {"ticker": "GLD", "days": 500, "frequency": "daily"}

    assert len(panel_rec.calls) == 1
    panel_kwargs = panel_rec.calls[0]
    assert panel_kwargs["session_data"] is sentinel_session
    assert panel_kwargs["intraday_data"] is None
    assert panel_kwargs["calendar"] is wrappers.EQUITIES_RTH


def test_wrapper_calls_intraday_loader_when_provided_and_forwards_days_freq(monkeypatch):
    sentinel_session = object()
    sentinel_intraday = object()
    sentinel_result = object()

    session_rec = CallRecorder()
    session_loader = session_rec.fn(return_value=sentinel_session)

    intraday_rec = CallRecorder()
    intraday_loader = intraday_rec.fn(return_value=sentinel_intraday)

    panel_rec = CallRecorder()
    fake_panel = panel_rec.fn(return_value=sentinel_result)
    monkeypatch.setattr(wrappers, "volatility_state_panel", fake_panel)

    result = wrappers.volatility_state_panel_for_ticker(
        ticker="GLD",
        load_session_candles=session_loader,
        load_intraday_candles=intraday_loader,
        session_days=123,
        intraday_days=45,
        intraday_freq="30min",
    )

    assert result is sentinel_result
    assert len(session_rec.calls) == 1
    assert session_rec.calls[0] == {"ticker": "GLD", "days": 123, "frequency": "daily"}

    assert len(intraday_rec.calls) == 1
    assert intraday_rec.calls[0] == {"ticker": "GLD", "days": 45, "frequency": "30min"}

    assert len(panel_rec.calls) == 1
    panel_kwargs = panel_rec.calls[0]
    assert panel_kwargs["session_data"] is sentinel_session
    assert panel_kwargs["intraday_data"] is sentinel_intraday


def test_wrapper_forwards_panel_kwargs(monkeypatch):
    sentinel_session = object()
    sentinel_result = object()

    session_rec = CallRecorder()
    session_loader = session_rec.fn(return_value=sentinel_session)

    panel_rec = CallRecorder()
    fake_panel = panel_rec.fn(return_value=sentinel_result)
    monkeypatch.setattr(wrappers, "volatility_state_panel", fake_panel)

    rv_windows = (5, 10, 20)
    z_lookback = 30

    result = wrappers.volatility_state_panel_for_ticker(
        ticker="GLD",
        load_session_candles=session_loader,
        load_intraday_candles=None,
        calendar=wrappers.EQUITIES_RTH,
        rv_windows=rv_windows,
        z_lookback=z_lookback,
        include_rank_percentile=False,
    )

    assert result is sentinel_result
    panel_kwargs = panel_rec.calls[0]
    assert panel_kwargs["rv_windows"] == rv_windows
    assert panel_kwargs["z_lookback"] == z_lookback
    assert panel_kwargs["include_rank_percentile"] is False
    assert panel_kwargs["calendar"] is wrappers.EQUITIES_RTH


@pytest.mark.parametrize(
    "session_payload",
    [
        object(),
        {"candles": [{"open": 1, "high": 1, "low": 1, "close": 1, "volume": 1, "datetime": 0}]},
        [{"open": 1, "high": 1, "low": 1, "close": 1, "volume": 1, "datetime": 0}],
        pd.DataFrame({"open": [1.0], "high": [1.0], "low": [1.0], "close": [1.0]}, index=[pd.Timestamp("2024-01-01")]),
    ],
)
@pytest.mark.parametrize(
    "intraday_payload",
    [
        object(),
        {"candles": [{"open": 1, "high": 1, "low": 1, "close": 1, "volume": 1, "datetime": 0}]},
        [{"open": 1, "high": 1, "low": 1, "close": 1, "volume": 1, "datetime": 0}],
        pd.DataFrame({"open": [1.0], "high": [1.0], "low": [1.0], "close": [1.0]}, index=[pd.Timestamp("2024-01-01")]),
    ],
)
def test_wrapper_passes_payloads_unmodified(monkeypatch, session_payload, intraday_payload):
    session_rec = CallRecorder()
    session_loader = session_rec.fn(return_value=session_payload)

    intraday_rec = CallRecorder()
    intraday_loader = intraday_rec.fn(return_value=intraday_payload)

    sentinel_result = object()
    panel_rec = CallRecorder()
    fake_panel = panel_rec.fn(return_value=sentinel_result)
    monkeypatch.setattr(wrappers, "volatility_state_panel", fake_panel)

    result = wrappers.volatility_state_panel_for_ticker(
        ticker="GLD",
        load_session_candles=session_loader,
        load_intraday_candles=intraday_loader,
    )

    assert result is sentinel_result
    assert panel_rec.calls[0]["session_data"] is session_payload
    assert panel_rec.calls[0]["intraday_data"] is intraday_payload


def test_wrapper_propagates_session_loader_exception(monkeypatch):
    boom = RuntimeError("boom")
    session_loader = CallRecorder().fn(raise_exc=boom)
    panel_rec = CallRecorder()
    fake_panel = panel_rec.fn(return_value=object())
    monkeypatch.setattr(wrappers, "volatility_state_panel", fake_panel)

    with pytest.raises(RuntimeError, match="boom"):
        wrappers.volatility_state_panel_for_ticker(
            ticker="GLD",
            load_session_candles=session_loader,
            load_intraday_candles=None,
        )

    assert len(panel_rec.calls) == 0


def test_wrapper_propagates_intraday_loader_exception(monkeypatch):
    session_loader = CallRecorder().fn(return_value=object())
    boom = RuntimeError("boom2")
    intraday_loader = CallRecorder().fn(raise_exc=boom)

    panel_rec = CallRecorder()
    fake_panel = panel_rec.fn(return_value=object())
    monkeypatch.setattr(wrappers, "volatility_state_panel", fake_panel)

    with pytest.raises(RuntimeError, match="boom2"):
        wrappers.volatility_state_panel_for_ticker(
            ticker="GLD",
            load_session_candles=session_loader,
            load_intraday_candles=intraday_loader,
        )

    assert len(panel_rec.calls) == 0

