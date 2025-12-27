# options_trading/tests/test_iv_rv_panel.py
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import importlib

mod = importlib.import_module("options_trading.volatility.iv_rv_panel")


class FakeRVPanel:
    def __init__(self, latest: pd.Series, settings: dict | None = None):
        self.series = pd.DataFrame()  # not used by iv_rv_panel, but present
        self.latest = latest
        self.settings = settings or {}
        self.coverage = {}


class FakeIVResult:
    def __init__(
        self,
        *,
        ivx_calc,
        flags=None,
        notes=None,
        symbol=None,
        as_of=None,
    ):
        self.ivx_calc = ivx_calc
        self.flags = flags or {"EXACT_MATCH": False, "EXTRAPOLATED": False, "VENDOR_MISMATCH": False}
        self.notes = notes or []
        self.symbol = symbol
        self.as_of = as_of
        # Other fields exist in real IVXATMResult, but iv_rv_panel only needs these.


def _make_latest(
    *,
    vol_primary_20=0.20,
    primary_src_20="rvar",
    vol_yz_20=0.18,
) -> pd.Series:
    return pd.Series(
        {
            "vol_primary_20": vol_primary_20,
            "primary_src_20": primary_src_20,
            "vol_yz_20": vol_yz_20,
        }
    )


def test_calls_dependencies_and_passes_kwargs(monkeypatch):
    calls = {"panel": None, "iv": None}

    def fake_panel(*, session_data, intraday_data, calendar, **kwargs):
        calls["panel"] = {"session_data": session_data, "intraday_data": intraday_data, "calendar": calendar, "kwargs": kwargs}
        return FakeRVPanel(latest=_make_latest(), settings={"symbol": "SPY"})

    def fake_iv(chain, *, target_dte_days, **kwargs):
        calls["iv"] = {"chain": chain, "target_dte_days": target_dte_days, "kwargs": kwargs}
        return FakeIVResult(ivx_calc=0.25, symbol="SPY")

    monkeypatch.setattr(mod, "volatility_state_panel", fake_panel)
    monkeypatch.setattr(mod, "ivx_atm", fake_iv)

    session_data = {"candles": []}
    intraday_data = {"candles": []}
    option_chain = {"chain": "x"}

    res = mod.iv_rv_panel(
        session_data=session_data,
        intraday_data=intraday_data,
        option_chain=option_chain,
        target_dte_days=45.0,
        panel_kwargs={"rv_windows": (10, 20, 60, 120)},
        iv_kwargs={"price_source": "mid_then_mark"},
    )

    assert calls["panel"] is not None
    assert calls["panel"]["session_data"] is session_data
    assert calls["panel"]["intraday_data"] is intraday_data
    assert "rv_windows" in calls["panel"]["kwargs"]

    assert calls["iv"] is not None
    assert calls["iv"]["chain"] is option_chain
    assert calls["iv"]["target_dte_days"] == 45.0
    assert calls["iv"]["kwargs"]["price_source"] == "mid_then_mark"

    # Attach full objects downstream
    assert res.rv_panel is not None
    assert res.iv_result is not None


def test_rv_reference_primary20_selects_primary_and_source(monkeypatch):
    def fake_panel(*, session_data, intraday_data, calendar, **kwargs):
        latest = _make_latest(vol_primary_20=0.20, primary_src_20="rvar", vol_yz_20=0.18)
        return FakeRVPanel(latest=latest, settings={})

    def fake_iv(chain, *, target_dte_days, **kwargs):
        return FakeIVResult(ivx_calc=0.25, symbol=None)

    monkeypatch.setattr(mod, "volatility_state_panel", fake_panel)
    monkeypatch.setattr(mod, "ivx_atm", fake_iv)

    res = mod.iv_rv_panel(session_data={}, intraday_data=None, option_chain={}, rv_reference="primary20")

    assert res.rv20 == pytest.approx(0.20)
    assert res.rv20_source == "rvar"


def test_rv_reference_yz20_selects_yz_and_no_source(monkeypatch):
    def fake_panel(*, session_data, intraday_data, calendar, **kwargs):
        latest = _make_latest(vol_primary_20=0.20, primary_src_20="rvar", vol_yz_20=0.18)
        return FakeRVPanel(latest=latest, settings={})

    def fake_iv(chain, *, target_dte_days, **kwargs):
        return FakeIVResult(ivx_calc=0.25)

    monkeypatch.setattr(mod, "volatility_state_panel", fake_panel)
    monkeypatch.setattr(mod, "ivx_atm", fake_iv)

    res = mod.iv_rv_panel(session_data={}, intraday_data=None, option_chain={}, rv_reference="yz20")

    assert res.rv20 == pytest.approx(0.18)
    assert res.rv20_source is None


def test_invalid_rv_reference_raises(monkeypatch):
    def fake_panel(*, session_data, intraday_data, calendar, **kwargs):
        return FakeRVPanel(latest=_make_latest(), settings={})

    def fake_iv(chain, *, target_dte_days, **kwargs):
        return FakeIVResult(ivx_calc=0.25)

    monkeypatch.setattr(mod, "volatility_state_panel", fake_panel)
    monkeypatch.setattr(mod, "ivx_atm", fake_iv)

    with pytest.raises(ValueError):
        mod.iv_rv_panel(session_data={}, intraday_data=None, option_chain={}, rv_reference="bad")


def test_metric_math_vrp_ratio_varspread(monkeypatch):
    def fake_panel(*, session_data, intraday_data, calendar, **kwargs):
        latest = _make_latest(vol_primary_20=0.20, primary_src_20="rvar", vol_yz_20=0.18)
        return FakeRVPanel(latest=latest, settings={})

    def fake_iv(chain, *, target_dte_days, **kwargs):
        return FakeIVResult(ivx_calc=0.25)

    monkeypatch.setattr(mod, "volatility_state_panel", fake_panel)
    monkeypatch.setattr(mod, "ivx_atm", fake_iv)

    res = mod.iv_rv_panel(session_data={}, intraday_data=None, option_chain={}, rv_reference="primary20")

    assert res.ivx == pytest.approx(0.25)
    assert res.rv20 == pytest.approx(0.20)
    assert res.vrp == pytest.approx(0.05)
    assert res.iv_over_rv == pytest.approx(1.25)
    assert res.var_vrp == pytest.approx(0.25**2 - 0.20**2)


def test_iv_over_rv_guard_when_rv20_zero(monkeypatch):
    def fake_panel(*, session_data, intraday_data, calendar, **kwargs):
        latest = _make_latest(vol_primary_20=0.0, primary_src_20="yz", vol_yz_20=0.0)
        return FakeRVPanel(latest=latest, settings={})

    def fake_iv(chain, *, target_dte_days, **kwargs):
        return FakeIVResult(ivx_calc=0.25)

    monkeypatch.setattr(mod, "volatility_state_panel", fake_panel)
    monkeypatch.setattr(mod, "ivx_atm", fake_iv)

    res = mod.iv_rv_panel(session_data={}, intraday_data=None, option_chain={}, rv_reference="primary20")

    assert res.vrp == pytest.approx(0.25 - 0.0)
    assert res.iv_over_rv is None
    assert res.var_vrp == pytest.approx(0.25**2 - 0.0**2)


def test_nan_rv20_becomes_none_and_flags_notes(monkeypatch):
    def fake_panel(*, session_data, intraday_data, calendar, **kwargs):
        latest = _make_latest(vol_primary_20=np.nan, primary_src_20="rvar", vol_yz_20=0.18)
        return FakeRVPanel(latest=latest, settings={})

    def fake_iv(chain, *, target_dte_days, **kwargs):
        return FakeIVResult(ivx_calc=0.25)

    monkeypatch.setattr(mod, "volatility_state_panel", fake_panel)
    monkeypatch.setattr(mod, "ivx_atm", fake_iv)

    res = mod.iv_rv_panel(session_data={}, intraday_data=None, option_chain={}, rv_reference="primary20")

    assert res.rv20 is None
    assert res.flags["RV20_AVAILABLE"] is False
    assert "RV20_UNAVAILABLE" in res.notes
    assert res.vrp is None
    assert res.iv_over_rv is None
    assert res.var_vrp is None


def test_inf_rv20_becomes_none(monkeypatch):
    def fake_panel(*, session_data, intraday_data, calendar, **kwargs):
        latest = _make_latest(vol_primary_20=np.inf, primary_src_20="rvar", vol_yz_20=0.18)
        return FakeRVPanel(latest=latest, settings={})

    def fake_iv(chain, *, target_dte_days, **kwargs):
        return FakeIVResult(ivx_calc=0.25)

    monkeypatch.setattr(mod, "volatility_state_panel", fake_panel)
    monkeypatch.setattr(mod, "ivx_atm", fake_iv)

    res = mod.iv_rv_panel(session_data={}, intraday_data=None, option_chain={}, rv_reference="primary20")

    assert res.rv20 is None
    assert res.flags["RV20_AVAILABLE"] is False


def test_nan_ivx_becomes_none_and_flags_notes(monkeypatch):
    def fake_panel(*, session_data, intraday_data, calendar, **kwargs):
        return FakeRVPanel(latest=_make_latest(), settings={})

    def fake_iv(chain, *, target_dte_days, **kwargs):
        return FakeIVResult(ivx_calc=np.nan, notes=["SOME_IV_NOTE"], symbol="QQQ")

    monkeypatch.setattr(mod, "volatility_state_panel", fake_panel)
    monkeypatch.setattr(mod, "ivx_atm", fake_iv)

    res = mod.iv_rv_panel(session_data={}, intraday_data=None, option_chain={}, rv_reference="primary20")

    assert res.ivx is None
    assert res.flags["IVX_AVAILABLE"] is False
    assert "IVX_UNAVAILABLE" in res.notes
    assert "SOME_IV_NOTE" in res.notes
    assert res.vrp is None
    assert res.iv_over_rv is None
    assert res.var_vrp is None


def test_only_rv_available_iv_missing(monkeypatch):
    def fake_panel(*, session_data, intraday_data, calendar, **kwargs):
        return FakeRVPanel(latest=_make_latest(vol_primary_20=0.20), settings={})

    def fake_iv(chain, *, target_dte_days, **kwargs):
        return FakeIVResult(ivx_calc=None)

    monkeypatch.setattr(mod, "volatility_state_panel", fake_panel)
    monkeypatch.setattr(mod, "ivx_atm", fake_iv)

    res = mod.iv_rv_panel(session_data={}, intraday_data=None, option_chain={})

    assert res.rv20 == pytest.approx(0.20)
    assert res.ivx is None
    assert res.vrp is None
    assert "IVX_UNAVAILABLE" in res.notes


def test_only_iv_available_rv_missing(monkeypatch):
    def fake_panel(*, session_data, intraday_data, calendar, **kwargs):
        latest = _make_latest(vol_primary_20=np.nan, primary_src_20="rvar", vol_yz_20=np.nan)
        return FakeRVPanel(latest=latest, settings={})

    def fake_iv(chain, *, target_dte_days, **kwargs):
        return FakeIVResult(ivx_calc=0.25)

    monkeypatch.setattr(mod, "volatility_state_panel", fake_panel)
    monkeypatch.setattr(mod, "ivx_atm", fake_iv)

    res = mod.iv_rv_panel(session_data={}, intraday_data=None, option_chain={})

    assert res.rv20 is None
    assert res.ivx == pytest.approx(0.25)
    assert res.vrp is None
    assert "RV20_UNAVAILABLE" in res.notes


def test_flags_and_notes_include_iv_flags_and_notes(monkeypatch):
    iv_flags = {"EXACT_MATCH": True, "EXTRAPOLATED": False, "VENDOR_MISMATCH": True}
    iv_notes = ["NOTE_A", "NOTE_B"]

    def fake_panel(*, session_data, intraday_data, calendar, **kwargs):
        return FakeRVPanel(latest=_make_latest(vol_primary_20=0.20), settings={})

    def fake_iv(chain, *, target_dte_days, **kwargs):
        return FakeIVResult(ivx_calc=0.25, flags=iv_flags, notes=iv_notes, symbol="SPY")

    monkeypatch.setattr(mod, "volatility_state_panel", fake_panel)
    monkeypatch.setattr(mod, "ivx_atm", fake_iv)

    res = mod.iv_rv_panel(session_data={}, intraday_data=None, option_chain={})

    assert res.flags["IV_FLAGS"] == iv_flags
    assert "NOTE_A" in res.notes and "NOTE_B" in res.notes


def test_symbol_preference_iv_then_rv_settings(monkeypatch):
    def fake_panel(*, session_data, intraday_data, calendar, **kwargs):
        return FakeRVPanel(latest=_make_latest(), settings={"symbol": "FROM_RV"})

    def fake_iv(chain, *, target_dte_days, **kwargs):
        return FakeIVResult(ivx_calc=0.25, symbol="FROM_IV")

    monkeypatch.setattr(mod, "volatility_state_panel", fake_panel)
    monkeypatch.setattr(mod, "ivx_atm", fake_iv)

    res = mod.iv_rv_panel(session_data={}, intraday_data=None, option_chain={})
    assert res.symbol == "FROM_IV"

    def fake_iv2(chain, *, target_dte_days, **kwargs):
        return FakeIVResult(ivx_calc=0.25, symbol=None)

    monkeypatch.setattr(mod, "ivx_atm", fake_iv2)
    res2 = mod.iv_rv_panel(session_data={}, intraday_data=None, option_chain={})
    assert res2.symbol == "FROM_RV"
