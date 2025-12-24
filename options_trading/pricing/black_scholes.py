"""Lightweight Black–Scholes pricing, Greek, and implied volatility helpers."""

import math
from dataclasses import dataclass
from typing import Literal

@dataclass
class BSParams:
    S: float
    K: float
    T: float
    sigma: float
    type: Literal["call", "put"]
    r: float = 0.04
    q: float = 0.0


_SQRT_2 = math.sqrt(2.0)
_SQRT_2PI = math.sqrt(2.0 * math.pi)


def _norm_cdf(x: float) -> float:
    """Standard normal cumulative distribution function."""
    return 0.5 * (1.0 + math.erf(x / _SQRT_2))


def _norm_pdf(x: float) -> float:
    """Standard normal probability density function."""
    return math.exp(-x**2 / 2.0) / _SQRT_2PI


def _d1_d2(params: BSParams) -> tuple[float, float]:
    S = params.S
    K = params.K
    T = params.T
    sigma = params.sigma
    r = params.r
    q = params.q

    if T <= 1e-8:
        raise ValueError("T must be > 0 for Greeks")

    sqrtT = math.sqrt(T)
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT
    return d1, d2



def bs_price(params: BSParams) -> float:
    """Return the European call or put price assuming constant vol/r/q."""

    S = params.S
    K = params.K
    T = params.T
    sigma = params.sigma
    type = params.type
    r = params.r
    q = params.q

    if S <= 0 or K <= 0:
        raise ValueError("S and K must be positive")

    if sigma <= 0:
        raise ValueError("sigma must be positive")

    if type not in {"call", "put"}:
        raise ValueError("type must be 'call' or 'put'")

    if T <= 1e-8:  # practically zero time to expiry
        intrinsic = max(0.0, S - K) if type == "call" else max(0.0, K - S)
        return intrinsic

    sqrtT = math.sqrt(T)
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT

    if type == "call":
        return S * math.exp(-q * T) * _norm_cdf(d1) - K * math.exp(-r * T) * _norm_cdf(d2)

    return K * math.exp(-r * T) * _norm_cdf(-d2) - S * math.exp(-q * T) * _norm_cdf(-d1)


def implied_vol(
    target_price: float,
    params: BSParams,
    tol: float = 1e-6,
    max_iter: int = 100,
    sigma_low: float = 1e-6,
    sigma_high: float = 5.0,
) -> float:
    """
    Solve for Black–Scholes implied volatility given a target option price.
    Uses bisection over [sigma_low, sigma_high].
    Returns sigma in decimal units (e.g. 0.25 for 25%).
    """

    if target_price <= 0.0:
        raise ValueError("target_price must be positive")

    # Work on a copy so we can modify sigma
    p = BSParams(**{**params.__dict__})

    # Intrinsic value is a lower bound on price
    intrinsic = bs_price(BSParams(
        S=p.S,
        K=p.K,
        T=0.0,              # force intrinsic
        sigma=1.0,          # unused since T=0
        type=p.type,
        r=p.r,
        q=p.q,
    ))

    if target_price < intrinsic:
        raise ValueError("target_price is below intrinsic value: no real implied vol")

    # Set up bracket
    p.sigma = sigma_low
    price_low = bs_price(p)

    p.sigma = sigma_high
    price_high = bs_price(p)

    if target_price < price_low or target_price > price_high:
        raise ValueError(
            f"target_price={target_price} is outside [price_low={price_low}, price_high={price_high}]; "
            "no implied vol in this bracket."
        )

    for _ in range(max_iter):
        sigma_mid = 0.5 * (sigma_low + sigma_high)
        p.sigma = sigma_mid
        price_mid = bs_price(p)

        diff = price_mid - target_price

        if abs(diff) < tol:
            return sigma_mid

        # Because price is monotonically increasing in sigma
        if diff < 0.0:
            sigma_low = sigma_mid
        else:
            sigma_high = sigma_mid

    # If not converged within max_iter, return mid as best guess
    return sigma_mid


def bs_delta(params: BSParams) -> float:
    """Delta (∂price/∂S) for European options with continuous dividends."""
    d1, _ = _d1_d2(params)
    S, T, q = params.S, params.T, params.q
    opt_type = params.type

    if T <= 1e-8:
        # At expiry: delta is 0 or 1 for call, 0 or -1 for put, but we’ll just raise for now
        raise ValueError("Delta at T=0 is path-dependent; handle intrinsic case separately.")

    if opt_type == "call":
        return math.exp(-q * T) * _norm_cdf(d1)
    else:  # put
        return math.exp(-q * T) * (_norm_cdf(d1) - 1.0)


def bs_gamma(params: BSParams) -> float:
    """Gamma (∂²price/∂S²) for European options."""
    d1, _ = _d1_d2(params)
    S, T, sigma, q = params.S, params.T, params.sigma, params.q

    if T <= 1e-8 or sigma <= 0.0:
        raise ValueError("Gamma undefined for T<=0 or sigma<=0")

    return math.exp(-q * T) * _norm_pdf(d1) / (S * sigma * math.sqrt(T))


def bs_vega(params: BSParams) -> float:
    """
    Vega (∂price/∂sigma) measuring sensitivity to volatility
    Vega per share, per 1 vol point (0.01 change in sigma).
    """
    S, T, q = params.S, params.T, params.q

    if T <= 1e-8:
        return 0.0

    d1, _ = _d1_d2(params)

    raw = S * math.exp(-q * T) * _norm_pdf(d1) * math.sqrt(T)

    return raw * 0.01


def bs_theta(params: BSParams) -> float:
    """Theta (∂price/∂t, calendar time; negative for long options), per year."""

    T = params.T
    if T <= 1e-8:
        return 0.0

    d1, d2 = _d1_d2(params)
    S, K, T, sigma, r, q = (
        params.S,
        params.K,
        params.T,
        params.sigma,
        params.r,
        params.q,
    )
    opt_type = params.type

    first_term = - (S * math.exp(-q * T) * _norm_pdf(d1) * sigma) / (2.0 * math.sqrt(T))

    if opt_type == "call":
        second_term = - r * K * math.exp(-r * T) * _norm_cdf(d2)
        third_term = + q * S * math.exp(-q * T) * _norm_cdf(d1)
    else:  # put
        second_term = + r * K * math.exp(-r * T) * _norm_cdf(-d2)
        third_term = - q * S * math.exp(-q * T) * _norm_cdf(-d1)

    return first_term + second_term + third_term


def bs_theta_per_day(params: BSParams) -> float:
    return bs_theta(params) / 365.0

