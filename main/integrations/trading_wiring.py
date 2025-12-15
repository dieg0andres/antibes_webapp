from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from django.conf import settings

from options_trading.schwab.client import make_client_factory
from options_trading.schwab.prices import get_latest_prices as _lib_get_latest_prices
from options_trading.sheets.trading_log import get_trading_log_worksheet as _lib_get_trading_log_worksheet


def make_client_factory_from_settings():
    return make_client_factory(
        api_key=settings.SCHWAB_API_KEY,
        app_secret=settings.SCHWAB_APP_SECRET,
        callback_url=settings.SCHWAB_CALLBACK_URL,
        token_path=Path(settings.SCHWAB_TOKEN_PATH),
    )


@lru_cache(maxsize=1)
def make_client_factory_cached():
    return make_client_factory_from_settings()


def get_client():
    factory = make_client_factory_cached()
    return factory()


def get_latest_prices(tickers):
    factory = make_client_factory_cached()
    return _lib_get_latest_prices(tickers, client_factory=factory, cache=None, cache_key=None, ttl=None)


def get_trading_log_worksheet():
    return _lib_get_trading_log_worksheet(
        spreadsheet_id=settings.TRADING_LOG_SPREADSHEET_ID,
        sa_key_path=settings.TRADING_LOG_SA_KEY_PATH,
        worksheet_gid=settings.TRADING_LOG_WORKSHEET_GID,
    )

