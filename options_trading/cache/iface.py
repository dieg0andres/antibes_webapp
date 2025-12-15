from __future__ import annotations

from typing import Any, Protocol, Optional


class Cache(Protocol):
    def get(self, key: str) -> Any:
        ...

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        ...

