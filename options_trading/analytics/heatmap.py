from typing import Sequence, Dict
import numpy as np
import pandas as pd
from portfolio import Portfolio, OptionLeg, UnderlyingLeg, NEAR_EXPIRY_EPS, BSParams

def generate_heatmaps(
    portfolio: Portfolio,
    dS: Sequence[float],
    dT_days: Sequence[int],
    dVol: Sequence[float],
) -> Dict[int, Dict[str, pd.DataFrame]]:
    """
    Generate PnL/Greek heatmap surfaces for a single-underlying Portfolio.

    Parameters
    ----------
    portfolio : Portfolio
        Portfolio containing OptionLeg and/or UnderlyingLeg for ONE underlying.
    dS : sequence of float
        Dollar shocks to apply to the current underlying price, e.g. [-20, -10, 0, 10, 20].
    dT_days : sequence of int
        Time steps forward in calendar days, e.g. [0, 1, 2, 3].
    dVol : sequence of float
        Volatility shocks (absolute), applied as sigma_new = sigma_old + dv,
        e.g. [-0.2, -0.1, 0.0, 0.1, 0.2].

    Returns
    -------
    surfaces : dict
        Nested dict:

        {
            dt_days: {
                "pnl":   pd.DataFrame,
                "delta": pd.DataFrame,
                "gamma": pd.DataFrame,
                "theta": pd.DataFrame,
                "vega":  pd.DataFrame,
            },
            ...
        }

        Each DataFrame has:
            - index  (rows)   = volatility levels (sigma)
            - columns (cols)  = underlying prices (S)
    """

    # -------------------------------------------------------------
    # 1. Sanity check: portfolio must effectively be single-underlying
    # -------------------------------------------------------------
    underlyings = {
        getattr(leg, "underlying", None)
        for leg in portfolio.legs
        if getattr(leg, "underlying", None) is not None
    }

    if len(underlyings) == 0:
        raise ValueError("Portfolio has no identifiable underlying symbols.")
    if len(underlyings) > 1:
        # For now, we explicitly do NOT support multi-underlying portfolios
        raise ValueError(
            "make_portfolio_heatmaps expects a single-underlying portfolio. "
            f"Found multiple underlyings: {underlyings}"
        )

    # The single underlying symbol we’re shocking
    underlying_symbol = next(iter(underlyings))

    # -------------------------------------------------------------
    # 2. Infer a "base" S and sigma from the existing legs
    #    (S is needed for the x-axis, sigma for the y-axis)
    # -------------------------------------------------------------
    base_S = None

    for leg in portfolio.legs:
        # Prefer to get S and sigma from an OptionLeg if available
        if isinstance(leg, OptionLeg) and leg.underlying == underlying_symbol:
            base_S = leg.params.S
            break
        # If no options exist yet, fall back to UnderlyingLeg.S
        if isinstance(leg, UnderlyingLeg) and leg.underlying == underlying_symbol:
            base_S = leg.S
            # base_sigma may remain None if there are no options

    if base_S is None:
        raise ValueError("Could not infer base S from portfolio legs.")

    base_sigma = portfolio.vega_weighted_sigma(underlying=underlying_symbol)

    # If there are no options, we still need a sigma value for the index.
    # Use 0.0 as a placeholder; the dVol axis will still be applied.
    if base_sigma is None or base_sigma <= 0.0:
        base_sigma = 1e-6

    # -------------------------------------------------------------
    # 3. Build the actual axis levels from the shocks
    #    - S_axis: absolute underlying prices
    #    - sigma_axis: absolute volatility levels
    # -------------------------------------------------------------
    S_axis = [base_S + ds for ds in dS]

    # For sigma, apply each absolute shock to the base sigma.
    # We also clip to a small positive floor so we don't hit sigma <= 0.
    sigma_axis = [max(base_sigma + dv, 1e-6) for dv in dVol]

    # This dictionary will hold everything:
    # surfaces[dt]["pnl"], surfaces[dt]["delta"], ...
    surfaces: Dict[int, Dict[str, pd.DataFrame]] = {}

    # -------------------------------------------------------------
    # 4. Main loop over time horizons (dT_days)
    # -------------------------------------------------------------
    for dt in dT_days:
        # Grid dimensions: rows = # of sigma points, cols = # of S points
        n_sigma = len(sigma_axis)
        n_S = len(S_axis)

        # Initialize empty grids for each metric
        pnl_grid = np.zeros((n_sigma, n_S))
        delta_grid = np.zeros_like(pnl_grid)
        gamma_grid = np.zeros_like(pnl_grid)
        theta_grid = np.zeros_like(pnl_grid)
        vega_grid = np.zeros_like(pnl_grid)

        # ---------------------------------------------------------
        # 5. Loop over vol and price points on the grid
        # ---------------------------------------------------------
        for i, sigma_new_label in enumerate(sigma_axis):
            # sigma_new_label = base_sigma + dv
            # So the corresponding dv is:
            dv = sigma_new_label - base_sigma

            for j, S_new in enumerate(S_axis):
                # We will construct a "shocked" portfolio at this grid point
                shocked_legs = []

                for leg in portfolio.legs:
                    # -----------------------
                    # Option legs: S, T, sigma all move
                    # -----------------------
                    if isinstance(leg, OptionLeg) and leg.underlying == underlying_symbol:
                        p = leg.params

                        # Time: move forward dt days, but don't let T go to zero
                        new_T = max(p.T - dt / 365.0, NEAR_EXPIRY_EPS)

                        # Vol: parallel shift by dv from each leg's own sigma
                        new_sigma = max(p.sigma + dv, 1e-6)

                        # Build new BSParams with the shocked values
                        new_params = BSParams(
                            S=S_new,
                            K=p.K,
                            T=new_T,
                            sigma=new_sigma,
                            type=p.type,
                            r=p.r,
                            q=p.q,
                        )

                        # Clone the leg with updated params, but same qty, entry price, etc.
                        shocked_leg = OptionLeg(
                            params=new_params,
                            qty=leg.qty,
                            multiplier=leg.multiplier,
                            underlying=leg.underlying,
                            label=leg.label,
                            strategy_tag=leg.strategy_tag,
                            trade_id=leg.trade_id,
                            entry_price=leg.entry_price,
                        )
                        shocked_legs.append(shocked_leg)

                    # -----------------------
                    # Underlying legs: only S moves
                    # -----------------------
                    elif isinstance(leg, UnderlyingLeg) and leg.underlying == underlying_symbol:
                        shocked_leg = UnderlyingLeg(
                            underlying=leg.underlying,
                            S=S_new,  # shocked price
                            qty=leg.qty,
                            multiplier=leg.multiplier,
                            label=leg.label,
                            trade_id=leg.trade_id,
                            entry_price=leg.entry_price,
                        )
                        shocked_legs.append(shocked_leg)

                    # -----------------------
                    # Any other leg type or different underlying:
                    # leave unchanged in this simple implementation
                    # -----------------------
                    else:
                        shocked_legs.append(leg)

                # Create the portfolio at this grid point (S_new, sigma_new, t + dt)
                shocked_portfolio = Portfolio(legs=shocked_legs)

                # Compute total PnL and Greeks for the shocked portfolio
                pnl_value = shocked_portfolio.pnl()
                g = shocked_portfolio.greeks()
                # g is assumed to be:
                # {"delta": ..., "gamma": ..., "vega": ..., "theta_per_day": ...}

                # Store into the matrices at row i (sigma), col j (S)
                pnl_grid[i, j] = pnl_value
                delta_grid[i, j] = g["delta"]
                gamma_grid[i, j] = g["gamma"]
                vega_grid[i, j] = g["vega"]
                theta_grid[i, j] = g["theta_per_day"]

        # ---------------------------------------------------------
        # 6. Wrap each grid as a pandas DataFrame with labeled axes
        # ---------------------------------------------------------
        index = pd.Index(sigma_axis, name="sigma")   # y-axis: vol levels
        columns = pd.Index(S_axis, name="S")         # x-axis: price levels

        surfaces[dt] = {
            "pnl":   pd.DataFrame(pnl_grid,   index=index, columns=columns),
            "delta": pd.DataFrame(delta_grid, index=index, columns=columns),
            "gamma": pd.DataFrame(gamma_grid, index=index, columns=columns),
            "theta": pd.DataFrame(theta_grid, index=index, columns=columns),
            "vega":  pd.DataFrame(vega_grid,  index=index, columns=columns),
        }

    # At the end, you get one "block" of heatmaps per dt in dT_days
    return surfaces
