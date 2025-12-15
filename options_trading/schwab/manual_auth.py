from __future__ import annotations

from pathlib import Path
from typing import Optional

try:
    from schwab.auth import client_from_manual_flow
except ImportError as exc:  # pragma: no cover - dependency optional until installed
    client_from_manual_flow = None  # type: ignore[assignment]
    _IMPORT_ERROR: Optional[ImportError] = exc
else:  # pragma: no cover
    _IMPORT_ERROR = None


def run_manual_auth(*, api_key: str, app_secret: str, callback_url: str, token_path: Path):
    if client_from_manual_flow is None:  # pragma: no cover
        raise RuntimeError(
            "schwab-py must be installed to run the Schwab manual auth flow."
        ) from _IMPORT_ERROR

    token_path = Path(token_path)
    token_path.parent.mkdir(parents=True, exist_ok=True)

    client_from_manual_flow(
        api_key=api_key,
        app_secret=app_secret,
        callback_url=callback_url,
        token_path=str(token_path),
    )

