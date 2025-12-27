# main/management/commands/update_vol_snapshots.py
"""
update_vol_snapshots management command

PURPOSE
-------
Run the daily volatility pipeline:
- Fetch raw market data from Schwab for all active tracked symbols:
    * daily candles
    * 15-minute candles
    * option chain snapshot
- Compute:
    * VolatilityStatePanelResult (RV panel)
    * IVXATMResult for multiple target maturities (e.g., 30 and 45 days)
    * Derived IV/RV metrics (VRP, IV/RV, IV^2-RV^2) for both rv_reference variants:
        - primary20
        - yz20
- Persist:
    * Raw JSON files (daily, 15m, option chain) to disk
    * Snapshot records per run per symbol per target_dte per rv_reference
    * Long-history session time series (VolatilitySessionBar): one row per session_ts, full row dict
    * IV time series (ImpliedVolSnapshot): one row per (run, symbol, target_dte)

DESIGN PRINCIPLES
-----------------
1) Keep it simple now, extensible later:
   - Run once per day (daily_close). Later add intraday runs by changing run_label + cron.
2) Transparency:
   - Store raw JSON responses (paths stored in DB).
   - Store full IVXATMResult (serialized) and RV latest row (serialized).
   - Store full session-level history rows in VolatilitySessionBar for multi-year charting.
3) Robustness:
   - Per-symbol failures do not stop the run.
   - Run is marked partial if any symbol fails.

ENVIRONMENT
-----------
This command loads environment variables from:
  REPO_ROOT/.env
in both dev and production.

Required env vars:
- SCHWAB_API_KEY
- SCHWAB_APP_SECRET
- SCHWAB_CALLBACK_URL
Optional env var:
- VOL_DATA_DIR  (recommended in prod: /mnt/data/vol_pipeline)

STORAGE
-------
VOL_DATA_DIR selection:
- If env VOL_DATA_DIR set -> use it.
- Else if /mnt/data/vol_pipeline exists -> use it (prod).
- Else -> use repo_root/data_store/vol_pipeline (dev).

Raw JSON file layout:
  <VOL_DATA_DIR>/raw/YYYY-MM-DD/<run_label>/<SYMBOL>/
      daily.json
      intraday_15m.json
      option_chain.json

NOTES ON "MISSING DAYS"
-----------------------
VolatilitySessionBar is inserted as one row per session timestamp.
If you miss a day/week, the next run inserts all sessions newer than the DB's max(session_ts),
assuming the fetched daily history includes those sessions.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, is_dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pandas as pd
from django.core.management.base import BaseCommand
from django.utils import timezone as dj_timezone
from dotenv import load_dotenv

from schwab.auth import easy_client

from main.models import (
    VolatilityTrackedSymbol,
    VolatilityPipelineRun,
    VolatilitySnapshot,
    VolatilitySessionBar,
    ImpliedVolSnapshot,
)

from options_trading.volatility.calendar import EQUITIES_RTH
from options_trading.volatility.panel import volatility_state_panel
from options_trading.volatility.iv_calculation import ivx_atm
from options_trading.volatility.transforms import candles_to_df


# -----------------------------
# Repo/env helpers
# -----------------------------

def _repo_root() -> Path:
    # .../main/management/commands/update_vol_snapshots.py
    # parents: commands -> management -> main -> repo_root
    return Path(__file__).resolve().parents[3]


def _load_repo_dotenv() -> None:
    """
    Load REPO_ROOT/.env so cron and dev behave consistently.
    Safe to call multiple times.
    """
    dotenv_path = _repo_root() / ".env"
    if dotenv_path.exists():
        load_dotenv(dotenv_path=dotenv_path, override=False)


# -----------------------------
# Path helpers
# -----------------------------

def _default_vol_data_dir() -> Path:
    env = os.getenv("VOL_DATA_DIR")
    if env:
        return Path(env)
    prod = Path("/mnt/data/vol_pipeline")
    if prod.exists():
        return prod
    return _repo_root() / "data_store" / "vol_pipeline"


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _write_json(path: Path, payload: Any) -> None:
    _ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def _today_utc_date() -> date:
    return datetime.now(timezone.utc).date()


def _last_business_day(d: date) -> date:
    # Weekend-only heuristic. Good enough for daily_close.
    if d.weekday() == 5:  # Saturday
        return d - timedelta(days=1)
    if d.weekday() == 6:  # Sunday
        return d - timedelta(days=2)
    return d


def _snap_strike(x: float, inc: float = 1.0) -> float:
    return round(x / inc) * inc


# -----------------------------
# Serialization helpers
# -----------------------------

def _jsonable(x: Any) -> Any:
    """
    Convert dataclasses / numpy scalars / pandas timestamps into JSON-safe Python types.
    """
    if x is None:
        return None

    if is_dataclass(x):
        return _jsonable(asdict(x))

    if isinstance(x, pd.Timestamp):
        return x.isoformat()

    if isinstance(x, (np.floating, np.integer)):
        v = x.item()
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            return None
        return v

    if isinstance(x, float):
        if math.isnan(x) or math.isinf(x):
            return None
        return x

    if isinstance(x, (int, str, bool)):
        return x

    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}

    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]

    return str(x)


def _series_row_to_dict(row: pd.Series) -> dict:
    return _jsonable(row.to_dict())


# -----------------------------
# Schwab fetch helpers
# -----------------------------

def _make_schwab_client() -> Any:
    """
    Create Schwab client (non-interactive) using env vars and token file.
    Assumes token file at REPO_ROOT/secret_store/schwab_token.json.
    """
    repo = _repo_root()
    token_path = repo / "secret_store" / "schwab_token.json"

    api_key = os.getenv("SCHWAB_API_KEY")
    app_secret = os.getenv("SCHWAB_APP_SECRET")
    callback_url = os.getenv("SCHWAB_CALLBACK_URL")

    if not api_key or not app_secret or not callback_url:
        raise RuntimeError("Missing SCHWAB_API_KEY/SCHWAB_APP_SECRET/SCHWAB_CALLBACK_URL in environment.")
    if not token_path.exists():
        raise FileNotFoundError(f"Schwab token not found at {token_path}")

    return easy_client(
        api_key=api_key,
        app_secret=app_secret,
        callback_url=callback_url,
        token_path=str(token_path),
        interactive=False,  # required for cron
    )


def _fetch_daily(client: Any, symbol: str) -> dict:
    resp = client.get_price_history_every_day(symbol)
    resp.raise_for_status()
    return resp.json()


def _fetch_intraday_15m(client: Any, symbol: str) -> dict:
    resp = client.get_price_history_every_fifteen_minutes(symbol)
    resp.raise_for_status()
    return resp.json()


def _fetch_quote_underlying(client: Any, symbol: str) -> float:
    q = client.get_quote(symbol)
    q.raise_for_status()
    qj = q.json()
    ticker_obj = qj.get(symbol) or next(iter(qj.values()))
    quote = ticker_obj.get("quote", ticker_obj)
    underlying = quote.get("lastPrice") or quote.get("mark") or quote.get("regularMarketLastPrice")
    if underlying is None:
        raise RuntimeError(f"Could not find {symbol} price in quote payload keys={list(quote.keys())}")
    return float(underlying)


def _fetch_option_chain(
    client: Any,
    symbol: str,
    *,
    underlying: float,
    from_date: date,
    to_date: date,
    strike_count: int,
) -> dict:
    """
    Fetch a constrained chain snapshot with defensive retries.
    """
    strike_center = _snap_strike(underlying, inc=1.0)

    resp = client.get_option_chain(
        symbol,
        contract_type=client.Options.ContractType.ALL,
        strategy=client.Options.Strategy.SINGLE,
        include_underlying_quote=True,
        strike=float(strike_center),
        strike_count=int(strike_count),
        strike_range=client.Options.StrikeRange.STRIKES_NEAR_MARKET,
        from_date=from_date,
        to_date=to_date,
    )
    if resp.status_code == httpx.codes.OK:
        return resp.json()

    if resp.status_code == 400:
        resp2 = client.get_option_chain(
            symbol,
            contract_type=client.Options.ContractType.ALL,
            strategy=client.Options.Strategy.SINGLE,
            include_underlying_quote=True,
            strike=float(strike_center),
            strike_count=int(strike_count),
            from_date=from_date,
            to_date=to_date,
        )
        if resp2.status_code == httpx.codes.OK:
            return resp2.json()

    resp.raise_for_status()
    return resp.json()


# -----------------------------
# Intraday cleaning
# -----------------------------

def _clean_intraday_payload(intraday_payload: dict) -> tuple[pd.DataFrame, dict]:
    """
    Drop intraday bars with OHLC <= 0 or NaN (holiday/bad-feed placeholders).
    Returns cleaned DataFrame (safe for log returns) + metadata for transparency.
    """
    df = candles_to_df(intraday_payload, strict=False)
    ohlc = df[["open", "high", "low", "close"]]
    bad_mask = (ohlc <= 0).any(axis=1) | ohlc.isna().any(axis=1)
    bad_n = int(bad_mask.sum())
    cleaned = df.loc[~bad_mask].copy()
    meta = {
        "intraday_total_bars": len(df),
        "intraday_bad_bars_dropped": bad_n,
        "intraday_bad_bars_example_ts": str(df.loc[bad_mask].index.min()) if bad_n else None,
    }
    return cleaned, meta


# -----------------------------
# Snapshot metric helpers
# -----------------------------

def _pick_rv20(latest: pd.Series, rv_reference: str) -> tuple[float | None, str | None]:
    """
    Pick RV20 based on rv_reference ("primary20" or "yz20").
    Returns (rv20_value, rv20_source).
    """
    if latest is None or latest.empty:
        return None, None

    if rv_reference == VolatilitySnapshot.RVREF_PRIMARY20:
        rv = latest.get("vol_primary_20")
        src = latest.get("primary_src_20")
    elif rv_reference == VolatilitySnapshot.RVREF_YZ20:
        rv = latest.get("vol_yz_20")
        src = None
    else:
        raise ValueError("rv_reference must be primary20 or yz20")

    if rv is None or pd.isna(rv) or np.isinf(rv):
        return None, src
    return float(rv), (str(src) if src is not None else None)


def _compute_pricing_metrics(ivx: float | None, rv20: float | None) -> dict:
    """
    Compute IV/RV metrics:
      vrp = ivx - rv20
      iv_over_rv = ivx/rv20 (if rv20>0)
      var_vrp = ivx^2 - rv20^2
    """
    if ivx is None or rv20 is None:
        return {"vrp": None, "iv_over_rv": None, "var_vrp": None}
    if rv20 <= 0:
        return {"vrp": ivx - rv20, "iv_over_rv": None, "var_vrp": ivx * ivx - rv20 * rv20}
    return {
        "vrp": ivx - rv20,
        "iv_over_rv": ivx / rv20,
        "var_vrp": ivx * ivx - rv20 * rv20,
    }


# -----------------------------
# Django management command
# -----------------------------

class Command(BaseCommand):
    help = "Fetch market data, compute RV/IV metrics, and persist volatility snapshots + time series."

    def add_arguments(self, parser):
        parser.add_argument("--run-label", type=str, default="daily_close")
        parser.add_argument("--target-dtes", type=float, nargs="+", default=[30.0, 45.0])
        parser.add_argument("--strike-count", type=int, default=12)
        parser.add_argument("--strict", action="store_true", help="Use strict=True (raises on quirks). Default strict=False.")
        parser.add_argument("--symbol", type=str, default="", help="If set, only run this symbol.")

    def handle(self, *args, **opts):
        _load_repo_dotenv()  # <-- key change: always load REPO_ROOT/.env

        run_label: str = opts["run_label"]
        target_dtes: list[float] = [float(x) for x in opts["target_dtes"]]
        strike_count: int = int(opts["strike_count"])
        strict: bool = bool(opts["strict"])
        symbol_filter: str = (opts["symbol"] or "").upper().strip()

        base_dir = _default_vol_data_dir()
        raw_base = base_dir / "raw"
        _ensure_dir(raw_base)
        _ensure_dir(base_dir / "logs")

        self.stdout.write(self.style.SUCCESS(f"VOL_DATA_DIR={base_dir}"))
        self.stdout.write(self.style.SUCCESS(f"run_label={run_label} target_dtes={target_dtes} strict={strict}"))

        client = _make_schwab_client()
        run = VolatilityPipelineRun.objects.create(run_label=run_label, status=VolatilityPipelineRun.STATUS_OK)

        any_errors = False

        qs = VolatilityTrackedSymbol.objects.filter(is_active=True).order_by("symbol")
        if symbol_filter:
            qs = qs.filter(symbol=symbol_filter)
        symbols = list(qs)
        if not symbols:
            run.status = VolatilityPipelineRun.STATUS_ERROR
            run.notes = "No active symbols."
            run.save(update_fields=["status", "notes"])
            self.stdout.write(self.style.WARNING("No active symbols found."))
            return

        from_dt = _last_business_day(_today_utc_date())
        to_dt = from_dt + timedelta(days=int(max(target_dtes)) + 30)

        panel_kwargs = dict(
            rv_windows=(10, 20, 60, 120),
            primary_max_window=120,
            include_realized_variance=True,
            include_overnight=True,
            annualize=True,
            strict=strict,
            heat_pairs=((20, 120), (10, 60)),
            z_lookback=252,
            z_windows=(20,),
            rank_lookback=252,
            rank_windows=(20,),
            include_rank_percentile=True,
            tail_sigma_window=60,
            tail_lookback=252,
            corr_window=120,
            corr_vol_window=20,
            include_regime=True,
            trend_ma_window=200,
            include_conditional_betas=True,
            conditional_beta_min_count=20,
        )

        for sym in symbols:
            symbol = sym.symbol
            self.stdout.write(self.style.NOTICE(f"\n=== {symbol} ==="))

            run_day = from_dt.isoformat()
            sym_dir = raw_base / run_day / run_label / symbol
            daily_path = sym_dir / "daily.json"
            intraday_path = sym_dir / "intraday_15m.json"
            chain_path = sym_dir / "option_chain.json"

            raw_paths = {
                "daily": str(daily_path),
                "intraday_15m": str(intraday_path),
                "option_chain": str(chain_path),
            }

            try:
                daily_payload = _fetch_daily(client, symbol)
                intraday_payload = _fetch_intraday_15m(client, symbol)
                underlying = _fetch_quote_underlying(client, symbol)
                chain_payload = _fetch_option_chain(
                    client,
                    symbol,
                    underlying=underlying,
                    from_date=from_dt,
                    to_date=to_dt,
                    strike_count=strike_count,
                )

                _write_json(daily_path, daily_payload)
                _write_json(intraday_path, intraday_payload)
                _write_json(chain_path, chain_payload)

            except Exception as e:
                any_errors = True
                msg = f"DATA_FETCH_ERROR for {symbol}: {e}"
                self.stdout.write(self.style.ERROR(msg))

                for dte in target_dtes:
                    for rv_ref in (VolatilitySnapshot.RVREF_PRIMARY20, VolatilitySnapshot.RVREF_YZ20):
                        VolatilitySnapshot.objects.create(
                            run=run,
                            symbol=sym,
                            target_dte_days=float(dte),
                            rv_reference=rv_ref,
                            status="error",
                            aha_metrics={},
                            iv_result={},
                            rv_latest={},
                            flags={"DATA_FETCH_ERROR": True},
                            notes=[msg],
                            raw_paths=_jsonable(raw_paths),
                            error_message=msg,
                        )
                continue

            intraday_df, intraday_meta = _clean_intraday_payload(intraday_payload)
            if intraday_meta["intraday_bad_bars_dropped"]:
                self.stdout.write(self.style.WARNING(
                    f"Dropped {intraday_meta['intraday_bad_bars_dropped']} intraday bars (<=0/NaN OHLC)."
                ))

            # Compute RV panel ONCE per symbol per run
            try:
                rv_panel = volatility_state_panel(
                    session_data=daily_payload,
                    intraday_data=intraday_df,
                    calendar=EQUITIES_RTH,
                    **panel_kwargs,
                )
            except Exception as e:
                any_errors = True
                msg = f"RV_PANEL_ERROR for {symbol}: {e}"
                self.stdout.write(self.style.ERROR(msg))

                for dte in target_dtes:
                    for rv_ref in (VolatilitySnapshot.RVREF_PRIMARY20, VolatilitySnapshot.RVREF_YZ20):
                        VolatilitySnapshot.objects.create(
                            run=run,
                            symbol=sym,
                            target_dte_days=float(dte),
                            rv_reference=rv_ref,
                            status="error",
                            aha_metrics={},
                            iv_result={},
                            rv_latest={},
                            flags={"RV_PANEL_ERROR": True},
                            notes=[msg],
                            raw_paths=_jsonable(raw_paths),
                            error_message=msg,
                        )
                continue

            # Insert session bars newer than current max
            try:
                existing_max = (
                    VolatilitySessionBar.objects.filter(symbol=sym)
                    .order_by("-session_ts")
                    .values_list("session_ts", flat=True)
                    .first()
                )

                series = rv_panel.series
                if existing_max is None:
                    to_insert = series
                else:
                    to_insert = series[series.index > existing_max]

                bars = [
                    VolatilitySessionBar(symbol=sym, session_ts=ts.to_pydatetime(), data=_series_row_to_dict(row))
                    for ts, row in to_insert.iterrows()
                ]
                if bars:
                    VolatilitySessionBar.objects.bulk_create(bars, ignore_conflicts=True)
                    self.stdout.write(self.style.SUCCESS(f"Inserted {len(bars)} new session bars."))
                else:
                    self.stdout.write(self.style.NOTICE("No new session bars to insert."))

            except Exception as e:
                any_errors = True
                self.stdout.write(self.style.WARNING(f"SESSION_BAR_STORE_WARNING for {symbol}: {e}"))

            # Latest RV row (serialized)
            rv_latest_dict = _jsonable(rv_panel.latest.to_dict() if rv_panel.latest is not None else {})

            # Compute IV and store snapshots per DTE and rv_reference
            for dte in target_dtes:
                iv_res = None
                try:
                    iv_res = ivx_atm(chain_payload, target_dte_days=float(dte), compare_vendor=True)
                except Exception as e:
                    any_errors = True
                    msg = f"IVX_ERROR dte={dte} for {symbol}: {e}"
                    self.stdout.write(self.style.ERROR(msg))

                # Store IV time series row
                try:
                    if iv_res is not None:
                        ImpliedVolSnapshot.objects.update_or_create(
                            run=run,
                            symbol=sym,
                            target_dte_days=float(dte),
                            defaults=dict(
                                ivx_calc=iv_res.ivx_calc,
                                ivx_vendor=iv_res.ivx_vendor,
                                ivx_diff=iv_res.ivx_diff,
                                flags=_jsonable(iv_res.flags),
                            ),
                        )
                except Exception as e:
                    any_errors = True
                    self.stdout.write(self.style.WARNING(f"IV_SNAPSHOT_STORE_WARNING for {symbol} dte={dte}: {e}"))

                # Derived IV values
                ivx = None
                if iv_res is not None and iv_res.ivx_calc is not None and not pd.isna(iv_res.ivx_calc) and not np.isinf(iv_res.ivx_calc):
                    ivx = float(iv_res.ivx_calc)

                for rv_ref in (VolatilitySnapshot.RVREF_PRIMARY20, VolatilitySnapshot.RVREF_YZ20):
                    try:
                        rv20, rv20_src = _pick_rv20(rv_panel.latest, rv_ref)
                        metrics = _compute_pricing_metrics(ivx, rv20)

                        aha = {
                            "ivx_calc": ivx,
                            "ivx_vendor": _jsonable(iv_res.ivx_vendor) if iv_res is not None else None,
                            "ivx_diff": _jsonable(iv_res.ivx_diff) if iv_res is not None else None,
                            "rv20": rv20,
                            "rv20_source": rv20_src,
                            "vrp": metrics["vrp"],
                            "iv_over_rv": metrics["iv_over_rv"],
                            "var_vrp": metrics["var_vrp"],

                            # Context from latest RV row:
                            "heat_primary_20_120": _jsonable(rv_panel.latest.get("heat_primary_20_120")),
                            "heat_yz_20_120": _jsonable(rv_panel.latest.get("heat_yz_20_120")),
                            "zlogvol_yz_20_252": _jsonable(rv_panel.latest.get("zlogvol_yz_20_252")),
                            "rv_rank_yz_20_252": _jsonable(rv_panel.latest.get("rv_rank_yz_20_252")),
                            "rv_pct_yz_20_252": _jsonable(rv_panel.latest.get("rv_pct_yz_20_252")),
                            "p_tail2_252": _jsonable(rv_panel.latest.get("p_tail2_252")),
                            "p_tail3_252": _jsonable(rv_panel.latest.get("p_tail3_252")),
                            "corr_ret_dlogvol_yz20_120": _jsonable(rv_panel.latest.get("corr_ret_dlogvol_yz20_120")),
                            "beta_ret_dlogvol_yz20_neg_120": _jsonable(rv_panel.latest.get("beta_ret_dlogvol_yz20_neg_120")),
                            "beta_ret_dlogvol_yz20_pos_120": _jsonable(rv_panel.latest.get("beta_ret_dlogvol_yz20_pos_120")),
                            "n_ret_neg_120": _jsonable(rv_panel.latest.get("n_ret_neg_120")),
                            "n_ret_pos_120": _jsonable(rv_panel.latest.get("n_ret_pos_120")),
                            "is_bull_200": _jsonable(rv_panel.latest.get("is_bull_200")),
                        }

                        flags = {
                            "RUN_LABEL": run_label,
                            "STRICT": strict,
                            "INTRADAY_BAD_BARS_DROPPED": intraday_meta["intraday_bad_bars_dropped"],
                            "IV_FLAGS": _jsonable(iv_res.flags) if iv_res is not None else {},
                            "IVX_AVAILABLE": ivx is not None,
                            "RV20_AVAILABLE": rv20 is not None,
                            "RV_REFERENCE": rv_ref,
                        }

                        notes = []
                        if ivx is None:
                            notes.append("IVX_UNAVAILABLE")
                        if rv20 is None:
                            notes.append("RV20_UNAVAILABLE")
                        if intraday_meta["intraday_bad_bars_dropped"]:
                            notes.append(f"Dropped {intraday_meta['intraday_bad_bars_dropped']} intraday bad bars (<=0/NaN).")

                        VolatilitySnapshot.objects.create(
                            run=run,
                            symbol=sym,
                            target_dte_days=float(dte),
                            rv_reference=rv_ref,
                            status="ok" if (iv_res is not None and rv_panel.latest is not None) else "error",
                            aha_metrics=_jsonable(aha),
                            iv_result=_jsonable(iv_res) if iv_res is not None else {},
                            rv_latest=rv_latest_dict,
                            flags=_jsonable(flags),
                            notes=_jsonable(notes),
                            raw_paths=_jsonable(raw_paths),
                            error_message="",
                        )

                    except Exception as e:
                        any_errors = True
                        msg = f"SNAPSHOT_ERROR for {symbol} dte={dte} rv_ref={rv_ref}: {e}"
                        self.stdout.write(self.style.ERROR(msg))

                        VolatilitySnapshot.objects.create(
                            run=run,
                            symbol=sym,
                            target_dte_days=float(dte),
                            rv_reference=rv_ref,
                            status="error",
                            aha_metrics={},
                            iv_result=_jsonable(iv_res) if iv_res is not None else {},
                            rv_latest=rv_latest_dict,
                            flags={"SNAPSHOT_ERROR": True},
                            notes=[msg],
                            raw_paths=_jsonable(raw_paths),
                            error_message=msg,
                        )

        run.status = VolatilityPipelineRun.STATUS_PARTIAL if any_errors else VolatilityPipelineRun.STATUS_OK
        run.notes = "partial: some errors" if any_errors else "ok"
        run.save(update_fields=["status", "notes"])

        self.stdout.write(self.style.SUCCESS(f"\nRun complete: {run}"))
        self.stdout.write(self.style.SUCCESS(f"Raw base: {raw_base / from_dt.isoformat() / run_label}"))
