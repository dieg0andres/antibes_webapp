from __future__ import annotations

import math
from functools import lru_cache
from pathlib import Path
from typing import List, Mapping, Optional

import gspread
import pandas as pd
from gspread_dataframe import get_as_dataframe, set_with_dataframe


def _ensure_key_path(sa_key_path: Path) -> Path:
    key_path = Path(sa_key_path)
    if not key_path.is_file():
        raise FileNotFoundError(f"Service account key not found at: {key_path}")
    return key_path


@lru_cache(maxsize=1)
def _get_gc(sa_key_path: Path) -> gspread.Client:
    key_path = _ensure_key_path(sa_key_path)
    return gspread.service_account(filename=str(key_path))


def get_trading_log_worksheet(
    *,
    spreadsheet_id: str,
    sa_key_path: Path,
    worksheet_gid: int,
) -> gspread.Worksheet:
    if not spreadsheet_id:
        raise RuntimeError("TRADING_LOG_SPREADSHEET_ID must be provided.")

    gc = _get_gc(Path(sa_key_path))
    sh = gc.open_by_key(spreadsheet_id)

    if worksheet_gid is not None:
        return sh.get_worksheet_by_id(int(worksheet_gid))

    raise RuntimeError("worksheet_gid must be provided when targeting the trading worksheet.")


def read_df(ws: gspread.Worksheet) -> pd.DataFrame:
    return get_as_dataframe(ws, evaluate_formulas=True)


def append_row(ws: gspread.Worksheet, row: list, value_input_option: str = "USER_ENTERED"):
    ws.append_row(row, value_input_option=value_input_option)


def write_df(ws: gspread.Worksheet, df: pd.DataFrame, start: str = "A1"):
    set_with_dataframe(ws, df)


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
    required_columns = ["TICKER", "STATUS", "OrderAction", "Price", "Qty"]
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

        qty_value = df.at[row_idx, "Qty"]
        try:
            qty_float = float(qty_value)
        except (TypeError, ValueError):
            continue

        if qty_float == 0:
            continue

        qty_sign = math.copysign(1.0, qty_float)
        signed_price = price * (-qty_sign)

        sheet_row = row_idx + 2  # account for header row in sheet
        df.at[row_idx, "Price"] = signed_price
        updates.append(
            {
                "range": f"{price_col_letter}{sheet_row}",
                "values": [[signed_price]],
            }
        )
        updated_rows += 1

    if updates:
        ws.batch_update(updates)

    return updated_rows

