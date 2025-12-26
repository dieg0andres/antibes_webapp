import math
import pytest

from options_trading.pricing.black_scholes import (
    BSParams,
    bs_price,
    bs_delta,
    bs_gamma,
    bs_vega,
    bs_theta,
    bs_theta_per_day,
    implied_vol,
)

# ----------------------------
# Helpers
# ----------------------------

def _call_put_params(S=100.0, K=100.0, T=0.5, sigma=0.2, r=0.04, q=0.0):
    call = BSParams(S=S, K=K, T=T, sigma=sigma, type="call", r=r, q=q)
    put = BSParams(S=S, K=K, T=T, sigma=sigma, type="put", r=r, q=q)
    return call, put


# ----------------------------
# 1) Contract / domain tests
# ----------------------------

@pytest.mark.parametrize(
    "S,K",
    [(0.0, 100.0), (-1.0, 100.0), (100.0, 0.0), (100.0, -5.0)],
)
def test_bs_price_rejects_non_positive_S_or_K(S, K):
    p = BSParams(S=S, K=K, T=0.5, sigma=0.2, type="call")
    with pytest.raises(ValueError, match="S and K must be positive"):
        bs_price(p)


@pytest.mark.parametrize("sigma", [0.0, -0.1])
def test_bs_price_rejects_non_positive_sigma(sigma):
    p = BSParams(S=100.0, K=100.0, T=0.5, sigma=sigma, type="call")
    with pytest.raises(ValueError, match="sigma must be positive"):
        bs_price(p)


def test_bs_price_rejects_bad_type():
    p = BSParams(S=100.0, K=100.0, T=0.5, sigma=0.2, type="call")
    p_bad = BSParams(**{**p.__dict__, "type": "CALL"})  # wrong
    with pytest.raises(ValueError, match="type must be 'call' or 'put'"):
        bs_price(p_bad)


def test_bs_price_returns_intrinsic_when_T_zeroish():
    # Your contract: T <= 1e-8 returns intrinsic
    p_call = BSParams(S=120.0, K=100.0, T=0.0, sigma=0.2, type="call")
    p_put = BSParams(S=80.0, K=100.0, T=0.0, sigma=0.2, type="put")
    assert bs_price(p_call) == 20.0
    assert bs_price(p_put) == 20.0


def test_greeks_T_zero_behavior():
    p = BSParams(S=100.0, K=100.0, T=0.0, sigma=0.2, type="call")
    with pytest.raises(ValueError):
        bs_delta(p)
    with pytest.raises(ValueError):
        bs_gamma(p)
    # your design choice
    assert bs_vega(p) == 0.0
    assert bs_theta(p) == 0.0
    assert bs_theta_per_day(p) == 0.0


# ----------------------------
# 2) No-arbitrage bounds
# ----------------------------

@pytest.mark.parametrize("opt_type", ["call", "put"])
@pytest.mark.parametrize(
    "S,K,T,sigma,r,q",
    [
        (100.0, 100.0, 0.5, 0.2, 0.03, 0.00),
        (100.0, 110.0, 1.0, 0.4, 0.05, 0.02),
        (250.0, 200.0, 0.25, 0.1, 0.01, 0.00),
    ],
)
def test_bs_price_bounds(opt_type, S, K, T, sigma, r, q):
    p = BSParams(S=S, K=K, T=T, sigma=sigma, type=opt_type, r=r, q=q)
    price = bs_price(p)

    F0 = S * math.exp(-q * T)          # discounted spot / prepaid forward
    PVK = K * math.exp(-r * T)

    if opt_type == "call":
        assert price >= max(0.0, F0 - PVK) - 1e-12
        assert price <= F0 + 1e-12
    else:
        assert price >= max(0.0, PVK - F0) - 1e-12
        assert price <= PVK + 1e-12


# ----------------------------
# 3) Put-call parity
# ----------------------------

@pytest.mark.parametrize(
    "S,K,T,sigma,r,q",
    [
        (100.0, 100.0, 0.5, 0.2, 0.04, 0.01),
        (100.0, 120.0, 1.0, 0.35, 0.02, 0.00),
        (250.0, 200.0, 0.25, 0.15, 0.01, 0.03),
    ],
)
def test_put_call_parity(S, K, T, sigma, r, q):
    call = BSParams(S=S, K=K, T=T, sigma=sigma, type="call", r=r, q=q)
    put = BSParams(S=S, K=K, T=T, sigma=sigma, type="put", r=r, q=q)

    c = bs_price(call)
    p = bs_price(put)

    lhs = c - p
    rhs = S * math.exp(-q * T) - K * math.exp(-r * T)
    assert lhs == pytest.approx(rhs, rel=1e-12, abs=1e-12)


# ----------------------------
# 4) Monotonicity properties
# ----------------------------

@pytest.mark.parametrize("opt_type", ["call", "put"])
def test_price_increases_with_sigma(opt_type):
    p1 = BSParams(S=100, K=100, T=0.5, sigma=0.10, type=opt_type, r=0.03, q=0.00)
    p2 = BSParams(**{**p1.__dict__, "sigma": 0.40})
    assert bs_price(p2) > bs_price(p1)


def test_call_increases_with_S_put_decreases_with_S():
    c1, p1 = _call_put_params(S=95, K=100, T=0.5, sigma=0.25, r=0.03, q=0.01)
    c2, p2 = _call_put_params(S=105, K=100, T=0.5, sigma=0.25, r=0.03, q=0.01)
    assert bs_price(c2) > bs_price(c1)
    assert bs_price(p2) < bs_price(p1)


def test_call_decreases_with_K_put_increases_with_K():
    c1, p1 = _call_put_params(S=100, K=95, T=0.5, sigma=0.25, r=0.03, q=0.01)
    c2, p2 = _call_put_params(S=100, K=105, T=0.5, sigma=0.25, r=0.03, q=0.01)
    assert bs_price(c2) < bs_price(c1)
    assert bs_price(p2) > bs_price(p1)


# ----------------------------
# 5) Greek sanity (sign / ranges)
# ----------------------------

@pytest.mark.parametrize("S,K", [(80, 100), (100, 100), (120, 100)])
def test_delta_ranges_and_signs(S, K):
    T, sigma, r, q = 0.5, 0.25, 0.02, 0.01
    call = BSParams(S=S, K=K, T=T, sigma=sigma, type="call", r=r, q=q)
    put = BSParams(S=S, K=K, T=T, sigma=sigma, type="put", r=r, q=q)

    dc = bs_delta(call)
    dp = bs_delta(put)

    upper = math.exp(-q * T)
    assert 0.0 < dc < upper
    assert -upper < dp < 0.0


@pytest.mark.parametrize("opt_type", ["call", "put"])
def test_gamma_positive_and_same_for_call_put(opt_type):
    p = BSParams(S=100, K=100, T=0.75, sigma=0.3, type=opt_type, r=0.01, q=0.0)
    g = bs_gamma(p)
    assert g > 0.0


def test_gamma_same_for_call_and_put_same_params():
    call = BSParams(S=100, K=100, T=0.75, sigma=0.3, type="call", r=0.01, q=0.0)
    put = BSParams(S=100, K=100, T=0.75, sigma=0.3, type="put", r=0.01, q=0.0)
    assert bs_gamma(call) == pytest.approx(bs_gamma(put), rel=1e-12, abs=1e-12)


@pytest.mark.parametrize("opt_type", ["call", "put"])
def test_vega_positive_for_T_gt_zero(opt_type):
    p = BSParams(S=100, K=100, T=0.4, sigma=0.2, type=opt_type, r=0.01, q=0.0)
    assert bs_vega(p) > 0.0


def test_theta_per_day_is_theta_div_365():
    p = BSParams(S=100, K=100, T=0.5, sigma=0.2, type="call", r=0.02, q=0.0)
    assert bs_theta_per_day(p) == pytest.approx(bs_theta(p) / 365.0, rel=1e-12, abs=1e-12)


# ----------------------------
# 6) Finite-difference checks for Greeks
# ----------------------------

def test_delta_gamma_finite_difference_call():
    p = BSParams(S=100, K=100, T=0.75, sigma=0.25, type="call", r=0.02, q=0.01)
    h = 1e-3 * p.S

    c0 = bs_price(p)
    c_up = bs_price(BSParams(**{**p.__dict__, "S": p.S + h}))
    c_dn = bs_price(BSParams(**{**p.__dict__, "S": p.S - h}))

    delta_fd = (c_up - c_dn) / (2 * h)
    gamma_fd = (c_up - 2 * c0 + c_dn) / (h * h)

    assert bs_delta(p) == pytest.approx(delta_fd, rel=1e-6, abs=1e-8)
    assert bs_gamma(p) == pytest.approx(gamma_fd, rel=1e-5, abs=1e-8)


def test_delta_finite_difference_put():
    p = BSParams(S=100, K=105, T=0.6, sigma=0.3, type="put", r=0.01, q=0.0)
    h = 1e-3 * p.S

    p_up = BSParams(**{**p.__dict__, "S": p.S + h})
    p_dn = BSParams(**{**p.__dict__, "S": p.S - h})

    price_up = bs_price(p_up)
    price_dn = bs_price(p_dn)

    delta_fd = (price_up - price_dn) / (2 * h)
    assert bs_delta(p) == pytest.approx(delta_fd, rel=1e-6, abs=1e-8)


def test_vega_matches_finite_difference_scaled():
    # Your bs_vega returns "per 0.01 vol point"
    p = BSParams(S=100, K=105, T=0.4, sigma=0.2, type="put", r=0.01, q=0.0)

    h = 1e-4
    price_up = bs_price(BSParams(**{**p.__dict__, "sigma": p.sigma + h}))
    price_dn = bs_price(BSParams(**{**p.__dict__, "sigma": p.sigma - h}))

    vega_raw_fd = (price_up - price_dn) / (2 * h)
    vega_scaled_fd = 0.01 * vega_raw_fd

    assert bs_vega(p) == pytest.approx(vega_scaled_fd, rel=1e-5, abs=1e-8)


# ----------------------------
# 7) Implied vol tests
# ----------------------------

def test_implied_vol_round_trip_call_and_put():
    for opt_type in ["call", "put"]:
        true_sigma = 0.33
        p = BSParams(S=100, K=95, T=0.6, sigma=true_sigma, type=opt_type, r=0.03, q=0.01)
        target = bs_price(p)

        guess_params = BSParams(**{**p.__dict__, "sigma": 0.10})
        iv = implied_vol(target, guess_params, tol=1e-8, max_iter=300)

        assert iv == pytest.approx(true_sigma, rel=1e-6, abs=1e-6)


def test_implied_vol_rejects_non_positive_target():
    p = BSParams(S=100, K=100, T=0.5, sigma=0.2, type="call")
    with pytest.raises(ValueError, match="target_price must be positive"):
        implied_vol(0.0, p)


def test_implied_vol_rejects_below_intrinsic():
    p = BSParams(S=120, K=100, T=0.5, sigma=0.2, type="call", r=0.02, q=0.0)
    # intrinsic is 20 for a call
    with pytest.raises(ValueError, match="below intrinsic"):
        implied_vol(19.999, p)


def test_implied_vol_rejects_outside_bracket():
    # Make a target price that is impossible in the bracket [1e-6, 5.0]
    p = BSParams(S=100, K=100, T=0.5, sigma=0.2, type="call", r=0.02, q=0.0)
    # Above discounted spot is impossible for a call
    impossible_target = p.S * math.exp(-p.q * p.T) + 1.0
    with pytest.raises(ValueError, match="outside"):
        implied_vol(impossible_target, p)


def test_implied_vol_monotonic_in_target_price():
    # Higher target price => higher implied vol (all else equal)
    base = BSParams(S=100, K=100, T=0.5, sigma=0.2, type="call", r=0.02, q=0.0)
    low_sigma = 0.15
    high_sigma = 0.35

    p_low = BSParams(**{**base.__dict__, "sigma": low_sigma})
    p_high = BSParams(**{**base.__dict__, "sigma": high_sigma})
    target_low = bs_price(p_low)
    target_high = bs_price(p_high)

    iv_low = implied_vol(target_low, base, tol=1e-8, max_iter=300)
    iv_high = implied_vol(target_high, base, tol=1e-8, max_iter=300)

    assert iv_high > iv_low


# ----------------------------
# 8) Extreme / stability tests (doesn't explode, respects bounds)
# ----------------------------

@pytest.mark.parametrize(
    "S,K,T,sigma",
    [
        (100.0, 200.0, 1.0, 0.2),   # deep OTM call
        (200.0, 100.0, 1.0, 0.2),   # deep ITM call
        (100.0, 100.0, 1e-6, 0.5),  # very small T but > 1e-8? (this one is smaller; should go intrinsic path)
        (100.0, 100.0, 0.05, 5.0),  # very high sigma
    ],
)
def test_bs_price_stability_and_bounds_under_extremes(S, K, T, sigma):
    # If T is extremely tiny, your bs_price returns intrinsic (if <= 1e-8).
    # We'll pick r,q normal.
    for opt_type in ["call", "put"]:
        p = BSParams(S=S, K=K, T=T, sigma=sigma, type=opt_type, r=0.03, q=0.0)
        price = bs_price(p)
        assert math.isfinite(price)

        F0 = S * math.exp(-p.q * p.T)
        PVK = K * math.exp(-p.r * p.T)
        if opt_type == "call":
            assert price >= max(0.0, F0 - PVK) - 1e-10
            assert price <= F0 + 1e-10
        else:
            assert price >= max(0.0, PVK - F0) - 1e-10
            assert price <= PVK + 1e-10
