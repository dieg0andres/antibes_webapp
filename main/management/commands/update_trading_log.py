import pickle

from django.conf import settings
from django.core.cache import cache
from django.core.management.base import BaseCommand

from main.utils.trading_log import (
    get_trading_log_worksheet,
    read_df,
    update_pending_close_prices,
)
from main.utils.schwab_prices import get_latest_prices


def _get_pending_close_mask(df):
    """
    Returns a boolean mask that selects rows in the trading log DataFrame corresponding
    to 'pending close' positions.

    These are rows which satisfy ALL of the following:
    - The 'TICKER' column is present (not NaN).
    - The 'STATUS' is 'Open' (ignoring case and surrounding whitespace).
    - The 'OrderAction' value ends with 'Close' (e.g., 'SellToClose', 'BuyToClose').

    Why needed:
        When synchronizing updated market prices into the trading log (such as in
        the update_trading_log management command), we only want to update the closing
        price for option/position rows that are still open AND have a pending close
        order. This mask efficiently filters for those rows so we can batch update
        only the relevant lines in the backing spreadsheet.
    """
    ticker_series = df["TICKER"]
    ticker_present = ~ticker_series.isna()
    status_open = df["STATUS"].astype(str).str.strip().eq("Open")
    order_close = df["OrderAction"].astype(str).str.endswith("Close")
    return ticker_present & status_open & order_close


class Command(BaseCommand):
    help = "Update the trading log Google Sheet"

    def handle(self, *args, **options):

        trading_log_worksheet = get_trading_log_worksheet()
        df = read_df(trading_log_worksheet)

        mask = _get_pending_close_mask(df)
        tickers = df.loc[mask, "TICKER"].tolist()
        price_map = get_latest_prices(tickers)
        updated_rows = update_pending_close_prices(trading_log_worksheet, df, price_map, mask=mask)

        cache_key = getattr(settings, "TRADING_LOG_CACHE_KEY", None)
        cache_ttl = getattr(settings, "TRADING_LOG_CACHE_TTL", None)
        if cache_key:
            timeout = None
            if cache_ttl is not None:
                try:
                    timeout = int(cache_ttl)
                except (TypeError, ValueError):
                    timeout = None
            cache.set(cache_key, pickle.dumps(df), timeout=timeout)
            # Temporary debug output for troubleshooting; remove when no longer needed.
           # self.stdout.write("Cached trading log DataFrame:")
           # self.stdout.write(df.to_string())

        self.stdout.write(self.style.SUCCESS(f"Trading log updated ({updated_rows} rows changed)"))