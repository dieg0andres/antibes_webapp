# AGENTS.md

## Project context
This is a Django project (antibes_webapp). We are extracting trading-related logic into a pure-Python package:
- `options_trading/` MUST be Django-agnostic.
- Django wiring lives in `main/integrations/trading_wiring.py`.

## Current refactor goal (legacy cleanup)
Remove legacy wrapper modules once all call sites are migrated:
- main/utils/schwab_client.py
- main/utils/schwab_prices.py
- main/utils/trading_log.py

All Django code should import:
- main/integrations/trading_wiring.py for Django-aware entry points (client factory, get_client, get_latest_prices, get_trading_log_worksheet)
- options_trading.* only for pure functions that require no Django settings


## Hard constraints
- Do NOT import `django.*` or `config.settings` anywhere under `options_trading/`.
- Keep diffs minimal (no formatting-only refactors).
- Do not change business logic or output formats.
- Do not add new third-party dependencies.

## Target structure
- `options_trading/schwab/client.py`: pure `get_client(api_key, app_secret, callback_url, token_path)`
- `options_trading/schwab/manual_auth.py`: pure `run_manual_auth(api_key, app_secret, callback_url, token_path)`
- `options_trading/schwab/prices.py`: `get_latest_prices(tickers, *, client_factory, cache=None, cache_key=None, ttl=None)`
- `options_trading/sheets/trading_log.py`: injected args (spreadsheet_id, sa_key_path, worksheet_gid)
- `options_trading/wiring/env.py`: create client_factory from env vars (no Django)
- `main/integrations/trading_wiring.py`: Django wrapper that reads `django.conf.settings` and optionally `django.core.cache.cache`

## Verification commands
Run after changes:
- `python manage.py check`
- `python manage.py run_schwab_manual_auth --help`
- `python -c "from options_trading.wiring.env import make_client_factory_from_env; print('ok')"`
- rg -n "main\.utils\.(schwab_client|schwab_prices|trading_log)" .   # should return no matches



## Realized Volatility (RV) module (options_trading/volatility)

### Goal
Add a pure-Python realized volatility toolkit under `options_trading/volatility/` supporting:
- close-to-close
- Parkinson
- Garman–Klass
- Rogers–Satchell
- Yang–Zhang (default when OHLC is available)
- realized_variance (intraday sum of squared log returns; path-dependent; optional overnight term)

This module will be used for systematic options/volatility research and production dashboards.

### Hard constraints (RV)
- MUST remain Django-agnostic: no `django.*` imports under `options_trading/`.
- Do NOT add new third-party dependencies beyond `numpy` and `pandas`.
- Keep diffs minimal (avoid unrelated formatting changes).
- Input prices are assumed split/dividend-adjusted by default; optionally accept an `adjustment_factor` Series.

### Time handling
- Always normalize to a tz-aware UTC `DatetimeIndex` internally.
- For intraday bars intended for IV comparison: support session aggregation ("fund RV") to include overnight effects.
- Strict timestamp regularity is enforced **within sessions** (e.g., missing/irregular 15m bars during RTH should raise).
  Cross-session gaps (overnight/weekend/holidays) are expected and MUST NOT cause strict failures.
  If `strict=False`, irregular within-session spacing should emit a warning and proceed.


### API conventions
- Window is specified in **bars** (int).
- Annualization uses trading-time: `sigma_ann = sigma_bar * sqrt(bars_per_year)`.
- close-to-close variance supports `assume_mean_zero: bool` (default True) and `ddof` overrides.
- `realized_variance` uses intraday returns and supports `include_overnight: bool` (default True).


### Target structure
- `options_trading/volatility/calendar.py`
- `options_trading/volatility/transforms.py`
- `options_trading/volatility/realized.py`
- `options_trading/volatility/__init__.py`
- Tests under `options_trading/tests/`

### Verification commands (after changes)
- `python manage.py check`
- `python -c "from options_trading.volatility.realized import realized_vol; print('ok')"`
- `python -m pytest -q`   # if pytest is available in the dev environment



## Volatility State Panel (options_trading/volatility/panel.py)

### Goal
Add a session-level “Volatility State Panel” that outputs a DataFrame of volatility features per session (index = session close timestamp in UTC), plus a latest snapshot and coverage metadata.

Inputs:
- session_data: daily/session candles (DataFrame or Schwab-style dict/list) with long history (e.g., 500 sessions)
- intraday_data: optional intraday candles for realized_variance (DataFrame or Schwab-style dict/list)

Core outputs (typical):
- vol_yz_{W} for W in (10,20,60,120)
- vol_rvar_{W} where intraday coverage exists (include_overnight=True)
- vol_primary_{W} where for W<=120: primary=rvar if available else yz; and primary_src_{W} indicating source
- heat features (log ratios) for configured window pairs
- z-scores on log(vol_yz_W) with long lookback (e.g., 252) for configured z_windows
- optional RV rank/percentile on vol_yz_W with lookback (e.g., 252) for configured rank_windows
- tail frequency (% |r| > 2σ, >3σ) and leverage-effect correlation
- regime label (SMA200 + is_bull_200) as lightweight conditioning
- conditional leverage betas (optional): rolling β⁻/β⁺ of Δlog(vol_yz_20) on returns for ret<0 vs ret>0, plus counts n_neg/n_pos

### Hard constraints (Panel)
- MUST remain Django-agnostic (no `django.*` imports).
- No new third-party dependencies beyond numpy/pandas.
- Provide thorough docstrings explaining behavior and design decisions.
- Keep diffs minimal; avoid unrelated refactors.


 ## Implied Volatility (IV) calculation (options_trading/volatility/iv_calculation.py)

### Goal
Compute constant-maturity ATM implied volatility (IVX) from an option chain snapshot (Schwab dict or normalized DataFrame).
Default uses IV computed via Black–Scholes inversion of market prices (mid then mark), with optional vendor IV comparison.

### Hard constraints (IV)
- MUST remain Django-agnostic (no django imports).
- No new third-party dependencies beyond numpy/pandas.
- Provide thorough docstrings including formulas and unit conventions (decimals vs percent).

### Notes
- Output IV values are annualized decimals (e.g., 0.18 = 18%).
- Supports user-defined target maturity via `target_dte_days` (default 30).
- Interpolate in total variance: TV = sigma^2 * T.


### Wrapper
- Provide a convenience wrapper under options_trading (still Django-agnostic) that accepts a ticker and uses injected loader callables to fetch candles, then calls the pure panel function.
- Do NOT hardcode Django settings or global Schwab clients inside the panel module.

