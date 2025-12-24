from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Optional
from collections import defaultdict
from legs import BaseLeg, OptionLeg, UnderlyingLeg, NEAR_EXPIRY_EPS
from black_scholes import BSParams


@dataclass
class Portfolio:
    """
    Container for a list of legs (OptionLeg, UnderlyingLeg, etc.) with
    methods to compute value, PnL, and Greeks.

    This is meant to represent OPEN positions only. Closed trades should
    be archived elsewhere (DB/CSV/etc.).
    """

    legs: List[BaseLeg]

    # --- mutation helpers ---

    def add_leg(self, leg: BaseLeg) -> None:
        """Append a leg to the portfolio."""
        self.legs.append(leg)

    def remove_leg(self, leg: BaseLeg) -> None:
        """Remove a leg from the portfolio if present."""
        self.legs.remove(leg)

    # --- valuation / risk ---

    def value(self) -> float:
        """
        Total mark-to-model value of the portfolio.

        Returns
        -------
        float
            Portfolio value in currency units.
        """
        return sum(leg.mark_to_model() for leg in self.legs)

    def pnl(self) -> float:
        """
        Total model PnL of the portfolio, based on each leg's entry_price.

        Returns
        -------
        float
            Portfolio PnL in currency units.
        """
        total = 0.0
        for leg in self.legs:
            total += leg.pnl()
        return total

    def greeks(self) -> Dict[str, float]:
        """
        Aggregate Greeks across all legs.

        Returns
        -------
        dict
            {"delta": ..., "gamma": ..., "vega": ..., "theta_per_day": ...}
            All Greeks are scaled by contracts and multipliers.
        """
        total_delta = total_gamma = total_vega = total_theta_per_day = 0.0
        for leg in self.legs:
            g = leg.greeks()
            total_delta += g["delta"]
            total_gamma += g["gamma"]
            total_vega  += g["vega"]
            total_theta_per_day += g["theta_per_day"]
        return {
            "delta": total_delta,
            "gamma": total_gamma,
            "vega":  total_vega,
            "theta_per_day": total_theta_per_day,
        }

    def greeks_by_underlying(self) -> Dict[str, Dict[str, float]]:
        """
        Aggregate Greeks grouped by underlying symbol.

        Returns
        -------
        dict
            {
                "ORCL": {"delta": ..., "gamma": ..., "vega": ..., "theta_per_day": ...},
                "SPY":  {...},
                ...
            }
        """
        agg: Dict[str, Dict[str, float]] = defaultdict(
            lambda: {"delta": 0.0, "gamma": 0.0, "vega": 0.0, "theta_per_day": 0.0}
        )

        for leg in self.legs:
            key = getattr(leg, "underlying", None) or "UNKNOWN"
            g = leg.greeks()
            agg[key]["delta"] += g["delta"]
            agg[key]["gamma"] += g["gamma"]
            agg[key]["vega"]  += g["vega"]
            agg[key]["theta_per_day"] += g["theta_per_day"]

        return dict(agg)


    def vega_weighted_sigma(
        self,
        *,
        underlying: Optional[str] = None,
        abs_vega: bool = True,
        min_T: float = NEAR_EXPIRY_EPS,
        eps: float = 1e-12,
    ) -> float:
        """
        Compute a portfolio "anchor" volatility as a vega-weighted average sigma.

        This is NOT a true single-vol model of the portfolio; it's a useful
        summary statistic for labeling/anchoring scenario grids.

        Parameters
        ----------
        underlying : str | None
            If provided, only include OptionLegs for this underlying.
            If None, include all OptionLegs (multi-underlying portfolios allowed).
        abs_vega : bool
            If True, use |vega| as weights to avoid long/short cancellation.
            Recommended.
        min_T : float
            Ignore options with T <= min_T (near-expiry) because vega ~ 0 and
            can produce unstable weights.
        eps : float
            Numerical safeguard for divide-by-zero.

        Returns
        -------
        float
            Vega-weighted average sigma. If no eligible options exist, returns 0.0.
        """
        num = 0.0
        den = 0.0

        for leg in self.legs:
            if not isinstance(leg, OptionLeg):
                continue
            if underlying is not None and leg.underlying != underlying:
                continue

            p = leg.params
            if p.T <= min_T:
                continue

            # leg.greeks() already scales by qty * multiplier
            v = leg.greeks()["vega"]
            w = abs(v) if abs_vega else v

            # If w is negative (possible when abs_vega=False), it will cancel.
            # That's fine if you explicitly want net-vega weighting, but usually you don't.
            if abs(w) < eps:
                continue

            num += w * p.sigma
            den += w

        if abs(den) < eps:
            # Fallback: if no options contribute, return 0.0 (or consider None)
            return 0.0

        return num / den



    # --- representation ---

    def __str__(self) -> str:
        """
        Human-readable summary of the portfolio.

        Example line:
            01 | ORCL | Short 2x ORCL Dec 230C | qty=-2 | T=0.082 | sigma=0.250 | trade_id=...
        """

        port_sigma = self.vega_weighted_sigma()
        port_PnL = self.pnl()
        lines: list[str] = [
            '-'*90,
            f"Portfolio PnL: {port_PnL:.2f}",
            f"Portfolio sigma (vega-weighted): {port_sigma:.4f}",
            f"Portfolio Greeks: {self.greeks()}",
            '-'*90,
        ]

        for i, leg in enumerate(self.legs, start=1):
            if isinstance(leg, OptionLeg):
                p = leg.params
                label = leg.label or f"{p.type.upper()} {p.K}"
                underlying = leg.underlying or "?"
                trade_id = leg.trade_id or "-"
                entry = leg.entry_price
                entry_str = f"{entry:.2f}" if entry is not None else "-"
                lines.append(
                    f"{i:02d} | {underlying} | {label} | "
                    f"qty={leg.qty} | T={p.T:.3f} | sigma={p.sigma:.3f} | trade_id={trade_id} | entry={entry_str}"
                )
            elif isinstance(leg, UnderlyingLeg):
                label = leg.label or "Underlying"
                trade_id = leg.trade_id or "-"
                entry = leg.entry_price
                entry_str = f"{entry:.2f}" if entry is not None else "-"
                lines.append(
                    f"{i:02d} | {leg.underlying} | {label} | "
                    f"qty={leg.qty} | S={leg.S:.2f} | trade_id={trade_id} | entry={entry_str}"
                )
            else:
                lines.append(f"{i:02d} | UNKNOWN LEG TYPE")
        return "\n".join(lines)
