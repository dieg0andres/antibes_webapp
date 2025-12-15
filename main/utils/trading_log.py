from __future__ import annotations

from django.conf import settings

from options_trading.sheets.trading_log import (
    append_row,
    get_trading_log_worksheet as _lib_get_trading_log_worksheet,
    read_df,
    update_pending_close_prices,
    write_df,
)


def get_trading_log_worksheet():
    return _lib_get_trading_log_worksheet(
        spreadsheet_id=settings.TRADING_LOG_SPREADSHEET_ID,
        sa_key_path=settings.TRADING_LOG_SA_KEY_PATH,
        worksheet_gid=settings.TRADING_LOG_WORKSHEET_GID,
    )