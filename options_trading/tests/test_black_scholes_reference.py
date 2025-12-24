import pytest
import math

from options_trading.pricing.black_scholes import (
    BSParams,
    bs_price,
    bs_delta,
    bs_gamma,
    bs_vega,
    bs_theta,
)

# -----------------------------
# Reference Case 1 (Hull ATM)
# -----------------------------

def test_reference_case_atm_call():
    p = BSParams(S=100, K=100, T=1.0, sigma=0.20, type="call", r=0.05, q=0.0)

    assert bs_price(p) == pytest.approx(10.450583572, rel=1e-6)
    assert bs_delta(p) == pytest.approx(0.636830651, rel=1e-6)
    assert bs_gamma(p) == pytest.approx(0.018762017, rel=1e-6)
    assert bs_vega(p)  == pytest.approx(0.375240347, rel=1e-6)  # scaled vega
    assert bs_theta(p) == pytest.approx(-6.414027546, rel=1e-6)


def test_reference_case_atm_put():
    p = BSParams(S=100, K=100, T=1.0, sigma=0.20, type="put", r=0.05, q=0.0)

    assert bs_price(p) == pytest.approx(5.573526022, rel=1e-6)
    assert bs_delta(p) == pytest.approx(-0.363169349, rel=1e-6)
    assert bs_gamma(p) == pytest.approx(0.018762017, rel=1e-6)
    assert bs_vega(p)  == pytest.approx(0.375240347, rel=1e-6)
    assert bs_theta(p) == pytest.approx(-1.657880423, rel=1e-6)


# -----------------------------
# Reference Case 2 (Dividends)
# -----------------------------

def test_reference_case_with_dividends_call_put_and_parity():
    S, K, T, sigma, r, q = 100, 100, 1.0, 0.25, 0.03, 0.02

    call = BSParams(S=S, K=K, T=T, sigma=sigma, type="call", r=r, q=q)
    put  = BSParams(S=S, K=K, T=T, sigma=sigma, type="put",  r=r, q=q)

    c = bs_price(call)
    p = bs_price(put)

    assert c == pytest.approx(10.197535275462172, rel=1e-12, abs=1e-12)
    assert p == pytest.approx(9.222221299637460, rel=1e-12, abs=1e-12)

    # Put-call parity with continuous dividend yield
    rhs = S * math.exp(-q * T) - K * math.exp(-r * T)
    assert (c - p) == pytest.approx(rhs, rel=1e-12, abs=1e-12)