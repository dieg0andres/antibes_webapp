# main/views_helpers/volatility_dashboard_helpers.py
from __future__ import annotations

from typing import Any, Dict, Optional

from django.utils import timezone

from main.models import (
    VolatilityTrackedSymbol,
    VolatilityPipelineRun,
    VolatilitySnapshot,
    VolatilitySessionBar,
)


def _iso(dt):
    return dt.isoformat() if dt else None


def get_volatility_dashboard_payload(
    *,
    symbol: Optional[str] = None,
    target_dte_days: float = 30.0,
    rv_reference: str = "primary20",
    run_id: Optional[int] = None,
    runs_limit: int = 50,
    recent_bars_limit: int = 30,
) -> Dict[str, Any]:
    """
    Return a JSON-serializable payload for the volatility dashboard.

    This is intentionally "read-only":
    - No Schwab calls
    - No caching
    - Reads only from SQLite

    Selection logic:
    - If run_id provided: pick that run, else pick latest run that has a matching snapshot.
    - If symbol not provided: pick the first symbol that has any snapshots.
    """

    errors = []
    rv_reference = (rv_reference or "primary20").strip()
    if rv_reference not in {"primary20", "yz20"}:
        errors.append(f"Invalid rv_reference '{rv_reference}'. Must be 'primary20' or 'yz20'.")
        rv_reference = "primary20"

    # Symbols that actually have snapshots
    symbol_qs = (
        VolatilityTrackedSymbol.objects.filter(is_active=True, snapshots__isnull=False)
        .distinct()
        .order_by("symbol")
    )
    symbols = list(symbol_qs.values_list("symbol", flat=True))

    if not symbols:
        return {
            "status": "error",
            "errors": ["No symbols with snapshots found."],
            "selected": {},
            "options": {"symbols": [], "rv_references": ["primary20", "yz20"], "target_dtes": [], "runs": []},
            "snapshot": None,
        }

    if symbol is None:
        symbol = symbols[0]
    symbol = symbol.upper().strip()

    # Validate symbol exists in tracked list
    try:
        sym_obj = VolatilityTrackedSymbol.objects.get(symbol=symbol)
    except VolatilityTrackedSymbol.DoesNotExist:
        errors.append(f"Symbol '{symbol}' not found in VolatilityTrackedSymbol.")
        sym_obj = VolatilityTrackedSymbol.objects.get(symbol=symbols[0])
        symbol = sym_obj.symbol

    # Available DTEs for this symbol
    dtes = (
        VolatilitySnapshot.objects.filter(symbol=sym_obj)
        .order_by("target_dte_days")
        .values_list("target_dte_days", flat=True)
        .distinct()
    )
    dtes = [float(x) for x in dtes]

    if dtes and target_dte_days not in dtes:
        # choose closest
        target_dte_days = min(dtes, key=lambda x: abs(x - target_dte_days))

    # Runs list (for future UI)
    runs_qs = VolatilityPipelineRun.objects.order_by("-run_at")[:runs_limit]
    runs = [{"id": r.id, "run_at": _iso(r.run_at), "run_label": r.run_label, "status": r.status} for r in runs_qs]

    # Choose a run
    selected_run = None
    if run_id is not None:
        selected_run = VolatilityPipelineRun.objects.filter(id=run_id).first()
        if selected_run is None:
            errors.append(f"run_id={run_id} not found. Falling back to latest matching snapshot.")
    if selected_run is None:
        # find latest run that has matching snapshot for this symbol+dte+rv_ref
        selected_run = (
            VolatilityPipelineRun.objects.filter(
                snapshots__symbol=sym_obj,
                snapshots__target_dte_days=target_dte_days,
                snapshots__rv_reference=rv_reference,
            )
            .distinct()
            .order_by("-run_at")
            .first()
        )

    if selected_run is None:
        errors.append("No run found with a matching snapshot for this selection.")
        return {
            "status": "error",
            "errors": errors,
            "selected": {
                "symbol": symbol,
                "target_dte_days": target_dte_days,
                "rv_reference": rv_reference,
                "run_id": None,
            },
            "options": {
                "symbols": symbols,
                "rv_references": ["primary20", "yz20"],
                "target_dtes": dtes,
                "runs": runs,
            },
            "snapshot": None,
        }

    # Load the snapshot
    snap = (
        VolatilitySnapshot.objects.filter(
            run=selected_run,
            symbol=sym_obj,
            target_dte_days=target_dte_days,
            rv_reference=rv_reference,
        )
        .order_by("-created_at")
        .first()
    )

    if snap is None:
        errors.append("Snapshot not found for selected run/symbol/dte/rv_reference.")
        snapshot_payload = None
    else:
        snapshot_payload = {
            "id": snap.id,
            "created_at": _iso(snap.created_at),
            "status": snap.status,
            "target_dte_days": snap.target_dte_days,
            "rv_reference": snap.rv_reference,
            "aha_metrics": snap.aha_metrics,
            "flags": snap.flags,
            "notes": snap.notes,
            "raw_paths": snap.raw_paths,
            "iv_result": snap.iv_result,
            "rv_latest": snap.rv_latest,
            "error_message": snap.error_message,
        }

    # Recent session bars for charting later (not graphs yet)
    bars_qs = (
        VolatilitySessionBar.objects.filter(symbol=sym_obj)
        .order_by("-session_ts")[:recent_bars_limit]
    )
    recent_bars = [
        {"session_ts": _iso(b.session_ts), "data": b.data}
        for b in reversed(list(bars_qs))  # chronological
    ]

    return {
        "status": "ok" if snapshot_payload else "error",
        "errors": errors,
        "selected": {
            "symbol": symbol,
            "target_dte_days": target_dte_days,
            "rv_reference": rv_reference,
            "run_id": selected_run.id,
            "run_at": _iso(selected_run.run_at),
            "run_label": selected_run.run_label,
            "run_status": selected_run.status,
        },
        "options": {
            "symbols": symbols,
            "rv_references": ["primary20", "yz20"],
            "target_dtes": dtes,
            "runs": runs,
        },
        "snapshot": snapshot_payload,
        "recent_session_bars": recent_bars,
    }


def get_volatility_dashboard_graphs_payload(**kwargs) -> Dict[str, Any]:
    """
    Placeholder for /trading/volatility_dashboard/graphs

    Later:
    - Return pre-aggregated series for charting (close/vol/heat/z/tails/etc.)
    - Possibly return downsampled data for long history
    """
    payload = get_volatility_dashboard_payload(**kwargs)
    payload["graphs_status"] = "not_implemented"
    return payload

