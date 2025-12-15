from __future__ import annotations

from main.integrations.trading_wiring import get_client as _get_client


def get_client():
    """Thin wrapper for backward compatibility."""
    return _get_client()

