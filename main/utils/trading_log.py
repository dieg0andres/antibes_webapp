# main/utils/google_sheets.py
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import List, Mapping

import gspread
import pandas as pd
from gspread_dataframe import get_as_dataframe, set_with_dataframe

from config.settings import (
    TRADING_LOG_SPREADSHEET_ID,
    TRADING_LOG_SA_KEY_PATH,
    TRADING_LOG_WORKSHEET_GID,
)


def _ensure_key_path() -> Path:
    key_path = Path(TRADING_LOG_SA_KEY_PATH)
    if not key_path.is_file():
        raise FileNotFoundError(f"Service account key not found at: {key_path}")
    return key_path


@lru_cache(maxsize=1)
def _get_gc() -> gspread.Client:
    key_path = _ensure_key_path()
    return gspread.service_account(filename=str(key_path))


def get_trading_log_worksheet() -> gspread.Worksheet:
    """
    Return the trading worksheet defined in settings.
    """
    spreadsheet_id = TRADING_LOG_SPREADSHEET_ID
    if not spreadsheet_id:
        raise RuntimeError("TRADING_LOG_SPREADSHEET_ID must be set in config.settings")

    gc = _get_gc()
    sh = gc.open_by_key(spreadsheet_id)

    if TRADING_LOG_WORKSHEET_GID:
        return sh.get_worksheet_by_id(int(TRADING_LOG_WORKSHEET_GID))

    raise RuntimeError(
        "TRADING_LOG_WORKSHEET_GID must be set in config.settings when targeting the trading worksheet."
    )


def read_df(ws) -> pd.DataFrame:
    """Read a worksheet into a pandas DataFrame."""
    return get_as_dataframe(ws, evaluate_formulas=True)


def append_row(ws, row: list, value_input_option: str = "USER_ENTERED"):
    """Append a single row to the worksheet."""
    ws.append_row(row, value_input_option=value_input_option)


def write_df(ws, df: pd.DataFrame, start: str = "A1"):
    """
    Overwrite the sheet starting at A1 with the provided DataFrame.
    set_with_dataframe writes starting at row=1,col=1 by default.
    """
    set_with_dataframe(ws, df)  # simple and convenient


def _column_letter_from_index(index_1_based: int) -> str:
    if index_1_based < 1:
        raise ValueError("Column index must be >= 1")

    letters: List[str] = []
    idx = index_1_based
    while idx:
        idx, remainder = divmod(idx - 1, 26)
        letters.append(chr(65 + remainder))
    return "".join(reversed(letters))


def update_pending_close_prices(
    ws: gspread.Worksheet,
    df: pd.DataFrame,
    price_map: Mapping[str, float],
    mask: pd.Series,
) -> int:
    """
    Update the Price column in-place for rows matching the criteria and with provided prices.

    Only cells whose ticker appears in price_map are updated via batch_update to minimize writes.
    Returns the number of rows that were updated.
    """
    required_columns = ["TICKER", "STATUS", "OrderAction", "Price"]
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"DataFrame is missing required columns: {', '.join(missing)}")

    if not price_map:
        return 0

    if not mask.any():
        return 0

    price_col_index = df.columns.get_loc("Price") + 1  # convert to 1-based
    price_col_letter = _column_letter_from_index(price_col_index)

    updates = []
    updated_rows = 0
    for row_idx in df.index[mask]:
        ticker = str(df.at[row_idx, "TICKER"]).strip()
        if not ticker:
            continue
        price = price_map.get(ticker)
        if price is None:
            continue

        sheet_row = row_idx + 2  # account for header row in sheet
        df.at[row_idx, "Price"] = price
        updates.append(
            {
                "range": f"{price_col_letter}{sheet_row}",
                "values": [[price]],
            }
        )
        updated_rows += 1

    if updates:
        ws.batch_update(updates)

    return updated_rows