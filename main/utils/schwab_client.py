from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Optional

from django.conf import settings

try:
    from schwab.auth import easy_client
except ImportError as exc:  # pragma: no cover - dependency optional until installed
    easy_client = None  # type: ignore[assignment]
    _EASY_CLIENT_IMPORT_ERROR: Optional[ImportError] = exc
else:  # pragma: no cover - executed when dependency present
    _EASY_CLIENT_IMPORT_ERROR = None


def _get_token_path(require_existing: bool = False) -> Path:
    token_path = Path(settings.SCHWAB_TOKEN_PATH)
    token_path.parent.mkdir(parents=True, exist_ok=True)
    if require_existing and not token_path.is_file():
        raise RuntimeError(
            f"Schwab token file not found at {token_path}. "
            "Run 'python manage.py run_schwab_manual_auth' to create one."
        )
    return token_path


@lru_cache(maxsize=1)
def get_client():
    """
    Return a cached Schwab easy_client instance configured from settings.
    """
    if easy_client is None:  # pragma: no cover
        raise RuntimeError(
            "schwab-py must be installed to use the Schwab client utilities."
        ) from _EASY_CLIENT_IMPORT_ERROR

    token_path = _get_token_path(require_existing=True)

    try:
        return easy_client(
            api_key=settings.SCHWAB_API_KEY,
            app_secret=settings.SCHWAB_APP_SECRET,
            callback_url=settings.SCHWAB_CALLBACK_URL,
            token_path=str(token_path),
            interactive=False,
        )
    except Exception as exc:  # pragma: no cover - defensive guard
        raise RuntimeError(
            "Unable to initialize Schwab client. The refresh token may be missing or expired. "
            "Re-run 'python manage.py run_schwab_manual_auth' to generate a fresh token."
        ) from exc

