from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Optional

try:
    from schwab.auth import easy_client
except ImportError as exc:  # pragma: no cover - dependency optional until installed
    easy_client = None  # type: ignore[assignment]
    _EASY_CLIENT_IMPORT_ERROR: Optional[ImportError] = exc
else:  # pragma: no cover
    _EASY_CLIENT_IMPORT_ERROR = None


def _ensure_token_path(token_path: Path, require_existing: bool = False) -> Path:
    token_path = Path(token_path)
    token_path.parent.mkdir(parents=True, exist_ok=True)
    if require_existing and not token_path.is_file():
        raise RuntimeError(f"Schwab token file not found at {token_path}. Run manual auth to create one.")
    return token_path


def get_client(
    *,
    api_key: str,
    app_secret: str,
    callback_url: str,
    token_path: Path,
    require_existing_token: bool = True,
):
    """
    Create a Schwab client using easy_client with the provided credentials.
    """
    if easy_client is None:  # pragma: no cover
        raise RuntimeError(
            "schwab-py must be installed to use Schwab client utilities."
        ) from _EASY_CLIENT_IMPORT_ERROR

    token_path = _ensure_token_path(token_path, require_existing=require_existing_token)
    return easy_client(
        api_key=api_key,
        app_secret=app_secret,
        callback_url=callback_url,
        token_path=str(token_path),
        interactive=False,
    )


def make_client_factory(*, api_key: str, app_secret: str, callback_url: str, token_path: Path):
    @lru_cache(maxsize=1)
    def _factory():
        return get_client(
            api_key=api_key,
            app_secret=app_secret,
            callback_url=callback_url,
            token_path=token_path,
            require_existing_token=True,
        )

    return _factory

