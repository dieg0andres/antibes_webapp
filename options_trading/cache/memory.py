from __future__ import annotations

import time
from typing import Any, Dict, Optional, Tuple

from options_trading.cache.iface import Cache


class InMemoryCache(Cache):
    """A minimal in-memory cache with optional TTL."""

    def __init__(self) -> None:
        self._store: Dict[str, Tuple[float, Any]] = {}

    def get(self, key: str) -> Any:
        entry = self._store.get(key)
        if not entry:
            return None
        expires_at, value = entry
        if expires_at and expires_at < time.time():
            self._store.pop(key, None)
            return None
        return value

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        expires_at = time.time() + ttl if ttl else 0
        self._store[key] = (expires_at, value)

