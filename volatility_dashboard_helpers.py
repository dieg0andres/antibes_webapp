# --- Graph helpers (matplotlib PNG) ---

from io import BytesIO
import base64
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta

from main.models import VolatilityTrackedSymbol, VolatilityPipelineRun, VolatilitySessionBar, ImpliedVolSnapshot


_RANGE_TO_DAYS = {
    "3m": 92,
    "6m": 185,
    "1y": 365,
    "3y": 365 * 3,
    "5y": 365 * 5,
    "max": None,
}


def _dt_utc(d: datetime) -> datetime:
    return d.astimezone(timezone.utc) if d.tzinfo else d.replace(tzinfo=timezone.utc)


def _parse_range(range_str: str | None) -> str:
    r = (range_str or "1y").strip().lower()
    return r if r in _RANGE_TO_DAYS else "1y"


def _fig_to_base64(fig) -> str:
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def _build_joined_timeseries(
    *,
    symbol_obj: VolatilityTrackedSymbol,
    target_dte_days: float,
    run_id: int | None,
    range_str: str,
    rv_reference: str,
) -> tuple[pd.DataFrame, dict]:
    """
    Build a joined DataFrame:
      - index: session_ts (UTC)
      - RV/state from VolatilitySessionBar.data JSON
      - IV series from ImpliedVolSnapshot aligned by UTC date

    Returns (df, stats).
    """
    range_key = _parse_range(range_str)

    # Determine end_date (UTC) from run_id if provided, else "now"
    if run_id:
        run = VolatilityPipelineRun.objects.filter(id=run_id).first()
        end_dt = _dt_utc(run.run_at) if run else _dt_utc(datetime.now(timezone.utc))
    else:
        end_dt = _dt_utc(datetime.now(timezone.utc))

    # Start date
    days = _RANGE_TO_DAYS[range_key]
    if days is None:
        start_dt = None
    else:
        start_dt = end_dt - timedelta(days=days)

    # Pull session bars
    bars_qs = VolatilitySessionBar.objects.filter(symbol=symbol_obj)
    if start_dt:
        bars_qs = bars_qs.filter(session_ts__gte=start_dt)
    bars_qs = bars_qs.filter(session_ts__lte=end_dt).order_by("session_ts").values("session_ts", "data")

    rows = []
    for r in bars_qs:
        ts = r["session_ts"]
        data = r["data"] or {}
        # Add ts explicitly
        row = {"session_ts": ts, **data}
        rows.append(row)

    if not rows:
        return pd.DataFrame(), {
            "range": range_key,
            "end_dt": end_dt.isoformat(),
            "start_dt": start_dt.isoformat() if start_dt else None,
            "session_rows": 0,
            "iv_rows": 0,
        }

    df = pd.DataFrame(rows)
    df["session_ts"] = pd.to_datetime(df["session_ts"], utc=True)
    df = df.set_index("session_ts").sort_index()

    # Ensure numeric where possible (safe coercion)
    for c in ["close", "vol_primary_20", "vol_yz_20", "heat_primary_20_120", "heat_yz_20_120",
              "zlogvol_yz_20_252", "p_tail2_252", "p_tail3_252",
              "corr_ret_dlogvol_yz20_120",
              "beta_ret_dlogvol_yz20_neg_120", "beta_ret_dlogvol_yz20_pos_120"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Build date key for joining IV (UTC date)
    df["date_utc"] = df.index.date

    # Pull IV snapshots for symbol + target_dte_days
    iv_qs = ImpliedVolSnapshot.objects.filter(symbol=symbol_obj, target_dte_days=target_dte_days).select_related("run")
    if start_dt:
        iv_qs = iv_qs.filter(run__run_at__gte=start_dt)
    iv_qs = iv_qs.filter(run__run_at__lte=end_dt).order_by("run__run_at")

    iv_rows = []
    for iv in iv_qs:
        run_at = _dt_utc(iv.run.run_at)
        iv_rows.append({
            "run_at": run_at,
            "date_utc": run_at.date(),
            "ivx_calc": iv.ivx_calc,
            "ivx_vendor": iv.ivx_vendor,
            "ivx_diff": iv.ivx_diff,
        })

    iv_df = pd.DataFrame(iv_rows)
    if not iv_df.empty:
        # If multiple IV points per date in future, keep last per day
        iv_df = iv_df.sort_values("run_at").groupby("date_utc", as_index=False).last()
        df = df.merge(iv_df.drop(columns=["run_at"]), on="date_utc", how="left")
        df = df.set_index(df.index)  # keep session_ts index
    else:
        df["ivx_calc"] = np.nan
        df["ivx_vendor"] = np.nan
        df["ivx_diff"] = np.nan

    # Compute derived series using selected rv_reference
    if rv_reference == "yz20":
        rv_col = "vol_yz_20"
    else:
        rv_col = "vol_primary_20"  # default

    df["rv20_ref"] = df[rv_col]
    df["vrp"] = df["ivx_calc"] - df["rv20_ref"]
    df["iv_over_rv"] = df["ivx_calc"] / df["rv20_ref"]
    df.loc[df["rv20_ref"] <= 0, "iv_over_rv"] = np.nan
    df["var_vrp"] = (df["ivx_calc"] ** 2) - (df["rv20_ref"] ** 2)

    # Downsample if max range
    if range_key == "max":
        # weekly last observation (finance standard)
        df = df.resample("W-FRI").last()

    stats = {
        "range": range_key,
        "rv_reference": rv_reference,
        "target_dte_days": target_dte_days,
        "end_dt": end_dt.isoformat(),
        "start_dt": start_dt.isoformat() if start_dt else None,
        "session_rows": int(df["close"].count()) if "close" in df.columns else len(df),
        "iv_rows": int(pd.Series(df["ivx_calc"]).count()) if "ivx_calc" in df.columns else 0,
    }
    return df, stats


def _make_graph_images(df: pd.DataFrame, stats: dict) -> tuple[dict, dict]:
    """
    Create PNG charts from joined df.
    Returns (images_dict, chart_stats).
    """
    images = {}
    chart_stats = {}

    if df.empty:
        return images, {"empty": True, **stats}

    # Helper for counts
    def _count_non_nan(col: str) -> int:
        return int(pd.Series(df[col]).dropna().shape[0]) if col in df.columns else 0

    chart_stats.update({
        **stats,
        "points_total": int(len(df)),
        "non_nan_ivx": _count_non_nan("ivx_calc"),
        "non_nan_rv_primary20": _count_non_nan("vol_primary_20"),
        "non_nan_rv_yz20": _count_non_nan("vol_yz_20"),
        "non_nan_vrp": _count_non_nan("vrp"),
    })

    idx = df.index

    # 1) Price + IVX vs RV20s
    fig, ax1 = plt.subplots(figsize=(12, 5))
    if "close" in df.columns:
        ax1.plot(idx, df["close"], label="Close")
        ax1.set_ylabel("Price")

    ax2 = ax1.twinx()
    if "ivx_calc" in df.columns and df["ivx_calc"].notna().any():
        ax2.plot(idx, df["ivx_calc"], label=f"IVX ({int(stats['target_dte_days'])}d)")
    if "vol_primary_20" in df.columns and df["vol_primary_20"].notna().any():
        ax2.plot(idx, df["vol_primary_20"], label="RV20 primary")
    if "vol_yz_20" in df.columns and df["vol_yz_20"].notna().any():
        ax2.plot(idx, df["vol_yz_20"], label="RV20 YZ")
    ax2.set_ylabel("Annualized vol")

    ax1.set_title("Price vs IVX and RV20")
    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")
    images["price_iv_rv"] = _fig_to_base64(fig)

    # 2) VRP (IVX - RV20_ref)
    fig, ax = plt.subplots(figsize=(12, 4))
    if "vrp" in df.columns and df["vrp"].notna().any():
        ax.plot(idx, df["vrp"], label="VRP = IVX - RV20_ref")
        ax.axhline(0.0, linewidth=1)
    ax.set_title("VRP over time")
    ax.set_ylabel("Vol points (decimal)")
    ax.legend(loc="upper left")
    images["vrp"] = _fig_to_base64(fig)

    # 3) IV/RV
    fig, ax = plt.subplots(figsize=(12, 4))
    if "iv_over_rv" in df.columns and df["iv_over_rv"].notna().any():
        ax.plot(idx, df["iv_over_rv"], label="IV/RV")
        ax.axhline(1.0, linewidth=1)
    ax.set_title("IV/RV over time")
    ax.set_ylabel("Ratio")
    ax.legend(loc="upper left")
    images["iv_over_rv"] = _fig_to_base64(fig)

    # 4) Variance VRP
    fig, ax = plt.subplots(figsize=(12, 4))
    if "var_vrp" in df.columns and df["var_vrp"].notna().any():
        ax.plot(idx, df["var_vrp"], label="IV^2 - RV^2")
        ax.axhline(0.0, linewidth=1)
    ax.set_title("Variance VRP over time")
    ax.set_ylabel("Variance points")
    ax.legend(loc="upper left")
    images["var_vrp"] = _fig_to_base64(fig)

    # 5) IV calc-vendor diff
    fig, ax = plt.subplots(figsize=(12, 4))
    if "ivx_diff" in df.columns and df["ivx_diff"].notna().any():
        ax.plot(idx, df["ivx_diff"], label="IVX_calc - IVX_vendor")
        ax.axhline(0.0, linewidth=1)
    ax.set_title("Vendor diff (sanity)")
    ax.set_ylabel("Vol points")
    ax.legend(loc="upper left")
    images["iv_vendor_diff"] = _fig_to_base64(fig)

    # 6) Heat and Z (two charts)
    fig, ax = plt.subplots(figsize=(12, 4))
    if "heat_primary_20_120" in df.columns and df["heat_primary_20_120"].notna().any():
        ax.plot(idx, df["heat_primary_20_120"], label="Heat primary ln(vol20/vol120)")
        ax.axhline(0.0, linewidth=1)
    ax.set_title("Heat (primary 20/120)")
    ax.set_ylabel("Heat")
    ax.legend(loc="upper left")
    images["heat"] = _fig_to_base64(fig)

    fig, ax = plt.subplots(figsize=(12, 4))
    if "zlogvol_yz_20_252" in df.columns and df["zlogvol_yz_20_252"].notna().any():
        ax.plot(idx, df["zlogvol_yz_20_252"], label="Z-score log vol (YZ20 vs 252)")
        ax.axhline(0.0, linewidth=1)
        ax.axhline(2.0, linestyle="--", linewidth=1)
        ax.axhline(-2.0, linestyle="--", linewidth=1)
    ax.set_title("Z-score of log vol (YZ20)")
    ax.set_ylabel("Z")
    ax.legend(loc="upper left")
    images["zscore"] = _fig_to_base64(fig)

    # 7) Tail frequency
    fig, ax = plt.subplots(figsize=(12, 4))
    for c in ("p_tail2_252", "p_tail3_252"):
        if c in df.columns and df[c].notna().any():
            ax.plot(idx, df[c], label=c)
    ax.set_title("Tail frequencies")
    ax.set_ylabel("Probability")
    ax.legend(loc="upper right")
    images["tails"] = _fig_to_base64(fig)

    # 8) Leverage corr + betas
    fig, ax = plt.subplots(figsize=(12, 4))
    if "corr_ret_dlogvol_yz20_120" in df.columns and df["corr_ret_dlogvol_yz20_120"].notna().any():
        ax.plot(idx, df["corr_ret_dlogvol_yz20_120"], label="corr(ret, dlogvol_yz20)")
        ax.axhline(0.0, linewidth=1)
    ax.set_title("Leverage correlation")
    ax.set_ylabel("Correlation")
    ax.legend(loc="upper left")
    images["corr"] = _fig_to_base64(fig)

    fig, ax = plt.subplots(figsize=(12, 4))
    for c in ("beta_ret_dlogvol_yz20_neg_120", "beta_ret_dlogvol_yz20_pos_120"):
        if c in df.columns and df[c].notna().any():
            ax.plot(idx, df[c], label=c)
    ax.axhline(0.0, linewidth=1)
    ax.set_title("Conditional leverage betas")
    ax.set_ylabel("beta")
    ax.legend(loc="upper left")
    images["betas"] = _fig_to_base64(fig)

    return images, chart_stats


def get_volatility_dashboard_graphs_ui_payload(
    *,
    symbol: str | None = None,
    target_dte_days: float = 30.0,
    rv_reference: str = "primary20",
    run_id: int | None = None,
    range: str | None = "1y",
) -> dict:
    """
    Build a payload for the graphs UI:
    - reuse existing selection/options logic from get_volatility_dashboard_payload
    - build a joined time series df
    - generate base64 PNG charts
    """
    base_payload = get_volatility_dashboard_payload(
        symbol=symbol,
        target_dte_days=target_dte_days,
        rv_reference=rv_reference,
        run_id=run_id,
    )

    # if selection failed, return base payload + empty images
    sel = base_payload.get("selected") or {}
    sel_symbol = sel.get("symbol")
    if not sel_symbol:
        base_payload["images"] = {}
        base_payload["chart_stats"] = {"empty": True}
        base_payload["range"] = _parse_range(range)
        return base_payload

    sym_obj = VolatilityTrackedSymbol.objects.get(symbol=sel_symbol)
    range_key = _parse_range(range)

    df, stats = _build_joined_timeseries(
        symbol_obj=sym_obj,
        target_dte_days=float(sel.get("target_dte_days", target_dte_days)),
        run_id=sel.get("run_id"),
        range_str=range_key,
        rv_reference=sel.get("rv_reference", rv_reference),
    )

    images, chart_stats = _make_graph_images(df, stats)
    base_payload["images"] = images
    base_payload["chart_stats"] = chart_stats
    base_payload["range"] = range_key

    return base_payload
