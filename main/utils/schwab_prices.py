from __future__ import annotations

import importlib
from typing import Iterable, List

from main.utils.schwab_client import get_client


def _normalize_tickers(tickers: Iterable[str]) -> List[str]:
    seen = set()
    normalized: List[str] = []
    for ticker in tickers:
        if ticker is None:
            continue
        symbol = str(ticker).strip()
        if not symbol or symbol in seen:
            continue
        seen.add(symbol)
        normalized.append(symbol)
    return normalized


def _get_httpx():
    try:
        return importlib.import_module("httpx")
    except ImportError as exc:  # pragma: no cover - dependency missing at runtime
        raise RuntimeError("httpx must be installed to fetch Schwab quotes") from exc


def get_latest_prices(tickers: Iterable[str]) -> dict[str, float]:
    """
    Fetch the latest mark price for the provided Schwab tickers.
    """
    symbols = _normalize_tickers(tickers)
    if not symbols:
        return {}

    httpx = _get_httpx()
    client = get_client()
    response = client.get_quotes(symbols)

    try:
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        raise RuntimeError("Schwab get_quotes request failed") from exc

    data = response.json() or {}
    prices: dict[str, float] = {}

    for symbol in symbols:
        payload = data.get(symbol) or {}
        quote = payload.get("quote") or {}
        mark = quote.get("mark")
        if mark is None:
            continue
        prices[symbol] = float(mark)

    return prices
