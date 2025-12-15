from __future__ import annotations

from django.core.cache import cache

from main.integrations.trading_wiring import make_client_factory_from_settings
from options_trading.schwab.prices import get_latest_prices as _lib_get_latest_prices


def get_latest_prices(tickers):
    factory = make_client_factory_from_settings()
    return _lib_get_latest_prices(tickers, client_factory=factory, cache=cache, cache_key=None, ttl=None)
