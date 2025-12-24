from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Optional, Literal

from black_scholes import (
    BSParams,
    bs_price,
    bs_delta,
    bs_gamma,
    bs_theta_per_day,
    bs_vega,
)


NEAR_EXPIRY_EPS = 5e-5


class BaseLeg(ABC):
    """
    Abstract base class for any leg in the portfolio
    (option leg, underlying leg, etc.).
    """

    underlying: Optional[str]
    label: Optional[str] = None
    trade_id: Optional[str] = None
    qty: int
    multiplier: int

    @abstractmethod
    def mark_to_model(self) -> float: ...

    
    @abstractmethod
    def greeks(self) -> Dict[str, float]: ...

    
    @abstractmethod
    def pnl(self) -> float: ...

    
    @abstractmethod
    def entry_cashflow(self) -> float: ...


    def _fmt(self, x: Optional[float], *, nd: int = 2, signed: bool = False) -> str:
        if x is None:
            return "-"
        return f"{x:+,.{nd}f}" if signed else f"{x:,.{nd}f}"


    def _extra_str(self) -> str:
        """
        Subclasses override to add type-specific info.
        Must return a short string like: "type=PUT K=550 T=0.118 sigma=0.22"
        """
        return ""


    def __str__(self) -> str:
        # Common identity
        underlying = getattr(self, "underlying", None) or "?"
        label = getattr(self, "label", None) or self.__class__.__name__
        trade_id = getattr(self, "trade_id", None) or "-"

        qty = getattr(self, "qty", 0)
        mult = getattr(self, "multiplier", 1)

        # Common computed metrics (guard pnl if entry_price missing)
        value = self.mark_to_model()
        cashflow = self.entry_cashflow()

        try:
            pnl = self.pnl()
            pnl_str = self._fmt(pnl, nd=2, signed=True)
        except Exception:
            pnl_str = "N/A"  # e.g., entry_price not set

        g = self.greeks()
        delta = self._fmt(g.get("delta"), nd=2, signed=True)
        gamma = self._fmt(g.get("gamma"), nd=4, signed=True)
        vega  = self._fmt(g.get("vega"), nd=2, signed=True)
        theta = self._fmt(g.get("theta_per_day"), nd=2, signed=True)

        extra = self._extra_str()
        extra = f" | {extra}" if extra else ""

        return (
            "-"*90 + "\n" +
            f"{underlying} | {label} | trade_id={trade_id} | "
            f"qty={qty:+d} x{mult} | "
            f"value={self._fmt(value, nd=2)} | pnl={pnl_str} | entry_cf={self._fmt(cashflow, nd=2, signed=True)} | "
            f"Δ={delta} Γ={gamma} Θ/d={theta} V={vega}"
            f"{extra}"
        )



@dataclass
class OptionLeg(BaseLeg):
    """
    One option leg priced with Black–Scholes.

    qty: number of contracts (+ long, – short)
    multiplier: 100 for standard equity options
    """

    params: BSParams
    qty: int
    multiplier: int = 100
    underlying: Optional[str] = None
    label: Optional[str] = None
    strategy_tag: Optional[str] = None
    trade_id: Optional[str] = None
    entry_price: Optional[float] = None  # per share

    def __post_init__(self) -> None:
        if self.qty == 0:
            raise ValueError("OptionLeg.qty cannot be zero.")
        if self.multiplier <= 0:
            raise ValueError("OptionLeg.multiplier must be positive.")

    # --- valuation ---

    def mark_to_model(self) -> float:
        price_per_share = bs_price(self.params)
        return price_per_share * self.qty * self.multiplier

    def value(self) -> float:
        """Alias for mark_to_model()."""
        return self.mark_to_model()

    # --- Greeks ---

    def greeks(self) -> Dict[str, float]:
        scale = self.qty * self.multiplier
        if self.params.T <= NEAR_EXPIRY_EPS:
            return self._expiry_greek_limits(scale)
        return {
            "delta": bs_delta(self.params) * scale,
            "gamma": bs_gamma(self.params) * scale,
            "vega":  bs_vega(self.params)  * scale,
            "theta_per_day": bs_theta_per_day(self.params) * scale,
        }

    def _expiry_greek_limits(self, scale: float) -> Dict[str, float]:
        S = self.params.S
        K = self.params.K
        opt_type = self.params.type
        if opt_type == "call":
            per_share_delta = 1.0 if S > K else 0.0
        else:  # put
            per_share_delta = -1.0 if S < K else 0.0
        return {
            "delta": per_share_delta * scale,
            "gamma": 0.0,
            "vega":  0.0,
            "theta_per_day": 0.0,
        }

    # --- cashflow & PnL ---

    def entry_cashflow(self) -> float:
        """
        + for premium received, − for premium paid.
        """
        if self.entry_price is None:
            return 0.0
        return -self.entry_price * self.qty * self.multiplier

    def pnl(self) -> float:
        """
        (current - entry) * qty * multiplier
        """
        if self.entry_price is None:
            raise ValueError(
                "entry_price must be set to compute PnL for OptionLeg."
            )
        current_price = bs_price(self.params)
        return (current_price - self.entry_price) * self.qty * self.multiplier

    # --- convenience constructor ---

    @classmethod
    def from_plain(
        cls,
        *,
        underlying: str,
        S: float,
        K: float,
        T_days: int,
        sigma: float,
        option_type: Literal["call", "put"],
        qty: int,
        r: float = 0.04,
        q: float = 0.0,
        label: Optional[str] = None,
        strategy_tag: Optional[str] = None,
        trade_id: Optional[str] = None,
        entry_price: Optional[float] = None,
        multiplier: int = 100,
    ) -> "OptionLeg":
        T_years = T_days / 365.0
        params = BSParams(
            S=S,
            K=K,
            T=T_years,
            sigma=sigma,
            type=option_type,
            r=r,
            q=q,
        )
        return cls(
            params=params,
            qty=qty,
            multiplier=multiplier,
            underlying=underlying,
            label=label,
            strategy_tag=strategy_tag,
            trade_id=trade_id,
            entry_price=entry_price,
        )



@dataclass
class UnderlyingLeg(BaseLeg):
    """
    Leg representing the underlying (stock, ETF, future).

    qty: number of shares/contracts (+ long, – short)
    multiplier: 1 for stocks/ETFs, >1 for futures
    """

    underlying: str
    S: float
    qty: int
    multiplier: int = 1
    label: Optional[str] = None
    trade_id: Optional[str] = None
    entry_price: Optional[float] = None  # per share/unit

    def __post_init__(self) -> None:
        if self.qty == 0:
            raise ValueError("UnderlyingLeg.qty cannot be zero.")
        if self.multiplier <= 0:
            raise ValueError("UnderlyingLeg.multiplier must be positive.")

    # --- valuation ---

    def mark_to_model(self) -> float:
        return self.S * self.qty * self.multiplier

    def value(self) -> float:
        return self.mark_to_model()

    # --- Greeks ---

    def greeks(self) -> Dict[str, float]:
        scale = self.qty * self.multiplier
        return {
            "delta": 1.0 * scale,
            "gamma": 0.0,
            "vega":  0.0,
            "theta_per_day": 0.0,
        }

    # --- cashflow & PnL ---

    def entry_cashflow(self) -> float:
        """
        + for sale proceeds, − for purchase cost.
        """
        if self.entry_price is None:
            return 0.0
        return -self.entry_price * self.qty * self.multiplier

    def pnl(self) -> float:
        if self.entry_price is None:
            raise ValueError(
                "entry_price must be set to compute PnL for UnderlyingLeg."
            )
        return (self.S - self.entry_price) * self.qty * self.multiplier

