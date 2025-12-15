from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from options_trading.schwab.client import make_client_factory


def make_client_factory_from_env():
    api_key = os.environ.get("SCHWAB_API_KEY")
    app_secret = os.environ.get("SCHWAB_APP_SECRET")
    callback_url = os.environ.get("SCHWAB_CALLBACK_URL")
    token_path = os.environ.get("SCHWAB_TOKEN_PATH")

    if not all([api_key, app_secret, callback_url, token_path]):
        raise RuntimeError("SCHWAB_API_KEY, SCHWAB_APP_SECRET, SCHWAB_CALLBACK_URL, SCHWAB_TOKEN_PATH must be set")

    return make_client_factory(
        api_key=api_key,
        app_secret=app_secret,
        callback_url=callback_url,
        token_path=Path(token_path),
    )


@lru_cache(maxsize=1)
def get_client_from_env():
    factory = make_client_factory_from_env()
    return factory()

