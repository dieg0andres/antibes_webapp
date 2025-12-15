from __future__ import annotations

import importlib
from typing import Iterable, List, Mapping, Optional

from options_trading.cache.iface import Cache


def _normalize_tickers(tickers: Iterable[str]) -> List[str]:
    seen = set()
    symbols: List[str] = []
    for ticker in tickers:
        if ticker is None:
            continue
        symbol = str(ticker).strip()
        if not symbol or symbol in seen:
            continue
        seen.add(symbol)
        symbols.append(symbol)
    return symbols


def _get_httpx():
    return importlib.import_module("httpx")


def get_latest_prices(
    tickers: Iterable[str],
    *,
    client_factory,
    cache: Optional[Cache] = None,
    cache_key: Optional[str] = None,
    ttl: Optional[int] = None,
) -> Mapping[str, float]:
    symbols = _normalize_tickers(tickers)
    if not symbols:
        return {}

    if cache is not None and cache_key:
        cached = cache.get(cache_key)
        if cached is not None:
            return cached

    httpx = _get_httpx()
    client = client_factory()
    response = client.get_quotes(symbols)

    response.raise_for_status()
    data = response.json() or {}

    prices: dict[str, float] = {}
    for symbol in symbols:
        payload = data.get(symbol) or {}
        quote = payload.get("quote") or {}
        mark = quote.get("mark")
        if mark is None:
            continue
        prices[symbol] = float(mark)

    if cache is not None and cache_key:
        cache.set(cache_key, prices, ttl)

    return prices

