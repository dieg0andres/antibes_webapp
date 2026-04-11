# Volatility Trading System Overview

This document summarizes the current state of the volatility trading system built inside the **antibes_webapp** repo.  
It is intended to be a durable “single source of truth” for architecture, design decisions, and operational workflow.

---

## Objective

Build a practical, engineering‑first volatility research/trading system (inspired by Sinclair *Volatility Trading*, Ch.2–3) that:

1. Measures realized volatility (RV) robustly from OHLC and intraday data  
2. Characterizes volatility regimes (heat, z‑scores, rank/percentile, tails, leverage coupling)  
3. Adds priced volatility (IV) from option chains to compute IV/RV metrics (VRP proxies)  
4. Stores everything daily into SQLite + raw JSON files for transparency and debugging  
5. Serves dashboards from the DB (no live Schwab calls at page load)

---

## Repo Principles

### Django vs Pure Python separation
- `options_trading/` is **pure Python** and **Django‑agnostic** (no `django.*` imports).
- Django orchestration/wiring lives under `main/` (views, management commands, persistence).

### Safety + transparency
- Prefer deterministic, testable functions with clear inputs/outputs.
- Prefer “glass box” diagnostics over black box behavior.
- Store raw market inputs on disk and store pointers in DB.

### Minimal dependencies
- New third‑party dependencies are avoided (beyond numpy/pandas, plus what already exists in the repo).

---

## Module Map

### Realized vol + transforms
- `options_trading/volatility/transforms.py`
  - Normalizes candle payloads to tz‑aware UTC indices
  - Validates OHLC integrity
  - Aggregates intraday bars into session bars
  - Infers bar size and checks within‑session spacing
  - Logs data quality warnings when `strict=False`

- `options_trading/volatility/realized.py`
  - `realized_vol(...)` supports:
    - close_to_close
    - Parkinson
    - Garman–Klass
    - Rogers–Satchell
    - Yang–Zhang (default OHLC estimator)
    - realized_variance (intraday path-dependent + optional overnight)

- `options_trading/volatility/calendar.py`
  - Trading calendars and session specs (e.g., `EQUITIES_RTH`)

### Volatility State Panel (RV behavior/regime)
- `options_trading/volatility/panel.py`
  - `volatility_state_panel(...)` produces a session‑indexed DataFrame with:
    - RV levels (YZ + realized_variance where available)
    - regime indicators (heat, zscore, rank, percentile)
    - tail frequencies
    - leverage coupling (corr + conditional betas)
    - regime label (SMA200 bull/bear)
  - Returns `VolatilityStatePanelResult(series, latest, settings, coverage)`

### Implied volatility (IVX) from chain snapshot
- `options_trading/volatility/iv_calculation.py`
  - `ivx_atm(...)` computes constant‑maturity ATM implied vol:
    - ATM strike chosen by forward: `F = S * exp((r-q)T)`
    - default: compute IV via **BS inversion** on market price (mid then mark)
    - interpolates in **total variance**: `TV = σ²T`
    - extrapolation uses nearest expiry IV **as-is** (no rescaling)
    - optional vendor comparison + mismatch flag
  - Accepts Schwab dict or normalized DataFrame (explicit args + df.attrs fallback)
  - Returns `IVXATMResult` with full diagnostics (expiry lower/upper, strikes, quotes used, flags)

### IV/RV composition layer
- `options_trading/volatility/iv_rv_panel.py`
  - `iv_rv_panel(...)` composes:
    - `volatility_state_panel(...)` (RV behavior)
    - `ivx_atm(...)` (priced vol)
  - Computes snapshot metrics:
    - `vrp = ivx - rv20`
    - `iv_over_rv = ivx / rv20`
    - `var_vrp = ivx² - rv20²`
  - Attaches full `VolatilityStatePanelResult` + `IVXATMResult`

---

## Key Definitions and Formulas

### Realized volatility
- Returns: `ret_log_t = ln(C_t / C_{t-1})`
- Annualized vol: `σ_ann = σ_bar * sqrt(bars_per_year)`

### Heat (regime “temperature”)
- `heat(a,b) = ln(RV_a) - ln(RV_b) = ln(RV_a / RV_b)`
- Default: `heat_20_120`

### Z-score of log vol
- Let `x_t = ln(RV_20,t)`
- `z_t = (x_t - mean(x over 252)) / std(x over 252)`

### Rank and percentile (distribution-free)
- Rank (IV-rank style):
  - `rank_t = (RV_t - min(RV_window)) / (max(RV_window) - min(RV_window))`
- Percentile:
  - `pct_t = mean(RV_window <= RV_t)`

### Tail frequency proxies
- `sigma_ref` = rolling std of returns (default window 60 sessions)
- `p_tail2` = rolling mean of `|ret| > 2*sigma_ref` over 252
- `p_tail3` = rolling mean of `|ret| > 3*sigma_ref` over 252

### Leverage proxy
- `dlogvol_t = ln(RV_t) - ln(RV_{t-1})`
- `corr(ret_log, dlogvol)` over rolling window (default 120)

### Conditional leverage betas (with intercept)
Fit separately on ret<0 and ret>0:
- `dlogvol = α + β * ret + ε`
Store:
- `β⁻, n⁻` (neg returns subset)
- `β⁺, n⁺` (pos returns subset)

### Implied volatility (IVX)
- IV is the σ that solves: `BS(σ) ≈ market_price`
- Constant maturity interpolation uses total variance:
  - `TV = σ²T`
  - interpolate TV linearly in T, then `σ = sqrt(TV/T)`

### Priced vs realized metrics (snapshot)
- `VRP ≈ IVX - RV20` (vol points proxy)
- `IV/RV = IVX / RV20`
- `Variance VRP = IVX² - RV20²`

---

## Strictness and Data Quality Handling

### `strict=True`
- Raises on irregular timestamps within session and OHLC consistency failures.
- Intended for tests and controlled datasets.

### `strict=False`
- Warns + logs for “soft” data issues (range inconsistencies, irregular spacing).
- Still raises for OHLC ≤ 0 (log returns invalid).

### Logging
- Data quality logs: `logs/options_trading_volatility.log`
- Warnings printed to console include “see logs/...”.

---

## Data Pipeline (Production)

### Storage locations
Use `VOL_DATA_DIR` environment variable.

- **Development** (recommended):
  - `VOL_DATA_DIR=<repo_root>/data_store/vol_pipeline`
  - `data_store/` should be gitignored.

- **Production** (Raspberry Pi SSD):
  - `VOL_DATA_DIR=/mnt/data/vol_pipeline`
  - SQLite DB lives on SSD at `/mnt/data/sqlite/db.sqlite3`

### Raw file layout
`<VOL_DATA_DIR>/raw/YYYY-MM-DD/<run_label>/<SYMBOL>/`
- `daily.json`
- `intraday_15m.json`
- `option_chain.json`

### Django models (in `main/models.py`)
- `VolatilityTrackedSymbol`  
- `VolatilityPipelineRun`
- `VolatilitySnapshot` (per run × symbol × DTE × rv_reference)
- `VolatilitySessionBar` (one row per session_ts; full panel row dict)
- `ImpliedVolSnapshot` (IV time series per run × symbol × DTE)

### Management command
- `main/management/commands/update_vol_snapshots.py`
  - loops all active symbols
  - supports `--target-dtes` (default `30 45`)
  - stores both `rv_reference` variants (`primary20`, `yz20`)
  - stores raw JSON files to disk + paths in DB
  - writes session bars without duplicating history:
    - inserts bars where `session_ts > max(session_ts in DB)`

### Scheduler (cron)
- Use a .sh wrapper `run_update_vol_snapshots.sh`
- Run daily at **20:15 CST** (safe before Schwab maintenance window).

---

## Dashboards (DB-backed, no Schwab calls)

### Page 1 (snapshot)
- JSON API: `/trading/volatility_dashboard`
- HTML UI: `/trading/volatility_dashboard/ui`
- Select:
  - symbol
  - target DTE
  - rv_reference
  - run_id (scroll back)
- Display:
  - AHA metrics grouped (priced vs realized / regime / tail+leverage)
  - IV diagnostics (`iv_result`)
  - RV latest row (`rv_latest`)
  - raw file paths

### Page 2 (graphs)
- HTML graphs UI (matplotlib PNG): `/trading/volatility_dashboard/graphs/ui`
- Joins:
  - `VolatilitySessionBar` (daily/session metrics)
  - `ImpliedVolSnapshot` (IVX history)
- Supports:
  - range: 3m/6m/1y/3y/5y/max (max downsampled weekly, last)
  - rv_reference affects VRP/ratio computations
  - run_id can anchor end date

---

## Environment Variables

In `REPO_ROOT/.env` (both dev and prod):

Required for Schwab:
- `SCHWAB_API_KEY`
- `SCHWAB_APP_SECRET`
- `SCHWAB_CALLBACK_URL`

Required for pipeline storage:
- `VOL_DATA_DIR`  
  - dev example: `/Users/diegogalindo/my_stuff/01_Projects/antibes_webapp/data_store/vol_pipeline`
  - prod example: `/mnt/data/vol_pipeline`

---

## Git Hygiene

Add to `.gitignore`:
```gitignore
# Local pipeline artifacts / raw market data
data_store/
```

---

## Current Status (Milestones Achieved)

✅ RV estimators implemented + tested  
✅ Volatility State Panel implemented + tested  
✅ Conditional leverage betas implemented + tested  
✅ IVX constant‑maturity ATM IV from chain implemented + tested (synthetic + fixtures)  
✅ IV/RV composition implemented + tested  
✅ Daily pipeline stores DB rows + raw JSON files (dev + prod)  
✅ Dashboard page 1 JSON + UI working  
✅ Graphs page working with matplotlib PNG and DB‑only reads

---

## Next Planned Work (Sinclair Chapter 4 and beyond)

1) Consider an RV forecasting layer (EWMA, GARCH) and features that support forecasting (regime transitions, volatility of volatility).  
2) “Sell premium score” as a **risk governor** (not a signal) combining:
   - IV/RV richness
   - tail penalties
   - leverage penalties
   - regime/heat penalties
3) Upgrade dashboards aesthetics (Tailwind + (optional) Plotly) after correctness is locked.  
4) Add pipeline test suite for `update_vol_snapshots.py` once operational behavior is stable.

