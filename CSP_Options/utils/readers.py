"""
Readers for the Alphathon 2025 data set
"""

from pathlib import Path
from typing import Dict, List, Union

import csp
import pandas as pd
from csp import ts
from csp.adapters.parquet import ParquetReader
from csp.basketlib import sample_dict

from CSP_Options.structs import (
    EquityQuoteWithSize,
    EquityQuoteWoSize,
    EquityTrade,
    KalshiTrade,
    OptionQuote,
)


# =============================================================================
# MONKEY PATCH: Fix ParquetReader.subscribe_dict_basket
# =============================================================================
def _fixed_subscribe_dict_basket(self, typ, name, shape, push_mode=None, field_map=None):
    """
    Fixed version of subscribe_dict_basket that doesn't incorrectly prefix column names.

    The original implementation creates field mappings like "{name}.{field}" even when
    name="" (empty string), which results in ".{field}" - causing the "Missing column .ask_price" error.

    This version bypasses the problematic field mapping logic by calling _subscribe_impl
    with field_map=None and basket_name=None, which triggers the correct auto-mapping behavior.

    Args:
        field_map: Optional dict mapping struct field names to parquet column names
                   e.g., {"symbol": "ticker"} maps struct's symbol field to parquet's ticker column
    """
    from csp.impl.types.common_definitions import PushMode

    if push_mode is None:
        push_mode = PushMode.NON_COLLAPSING

    # Call _subscribe_impl with basket_name=None to avoid the problematic prefix logic
    return {v: self._subscribe_impl(v, typ, field_map, push_mode, basket_name=None) for v in shape}


# Apply the monkey patch
ParquetReader.subscribe_dict_basket = _fixed_subscribe_dict_basket

# =============================================================================
# Paths and constants
# =============================================================================

BASE_DATA_PATH = Path("data/alphathon_2025")
DATE_OF_INTEREST = "2024-11-05"  # Nov 5, 2024 - Election Day
EXPIRATION = "20241115"  # Nov 15, 202
ALL_1S_QUOTES_PATH = BASE_DATA_PATH / "all_1s_quotes" / f"{DATE_OF_INTEREST}.parquet"
TAQ_QUOTES_PATH = BASE_DATA_PATH / "quotes_selected" / f"{DATE_OF_INTEREST}.parquet"
TAQ_TRADES_PATH = BASE_DATA_PATH / "trades_selected" / f"{DATE_OF_INTEREST}.parquet"
KALSHI_TRADES_PATH = BASE_DATA_PATH / "kalshi_election" / f"{DATE_OF_INTEREST}.parquet"

# =============================================================================
# Load 20 EQUITIES AND 20 ETFs
# =============================================================================
# Load instruments from JSON file

import json
import os

# Get the absolute path to the project root (2 levels up from this file)
_PROJECT_ROOT = Path(__file__).parent.parent.parent
_INSTRUMENTS_FILE = _PROJECT_ROOT / "instrument_selected_instruments.json"

with open(_INSTRUMENTS_FILE, "r") as f:
    instruments_data = json.load(f)

ETFS = instruments_data["selected_ETFs"]
EQUITIES = instruments_data["selected_stocks"]
ALL_TICKERS = ETFS + EQUITIES

# Kalshi election contracts
KALSHI_CONTRACTS = ["PRES-2024-KH", "PRES-2024-DJT"]

# =============================================================================
# Helpers
# =============================================================================


# Build file paths for each ticker's options
def get_option_file(ticker: str) -> str:
    """Get the path to the option file for a ticker"""
    # Note: BRK.B is stored as BRKB in the files
    file_ticker = "BRKB" if ticker == "BRK.B" else ticker
    date_numeric = DATE_OF_INTEREST.replace("-", "")  # Convert to YYYYMMDD format
    return str(
        BASE_DATA_PATH
        / "options_selected_formatted"
        / f"{file_ticker}_{EXPIRATION}_{date_numeric}_{date_numeric}_1000.parquet"
    )


# =============================================================================
# READERS: Same structure as FitHappensHere, but for ALL 40 tickers
# =============================================================================


# Data Ingress Graphs
@csp.graph
def get_quotes_1s_wo_size(symbols: Union[List[str], None]) -> Dict[str, ts[EquityQuoteWoSize]]:
    """
    Get underlying quotes without size from 1-second downsampled data.

    The all_1s_quotes file contains thousands of tickers downsampled to 1-second intervals,
    with timestamps representing the right endpoint of each interval. This function
    efficiently filters to only the requested tickers.

    Args:
        symbols: List of ticker symbols to subscribe to, or None to use ALL_TICKERS

    Returns:
        Dict basket: {ticker: ts[EquityQuoteWoSize]} - quotes without size information
    """
    parquet_reader = ParquetReader(
        filename_or_list=str(ALL_1S_QUOTES_PATH),
        time_column="timestamp",  # all_1s_quotes uses 'timestamp' column
        symbol_column="symbol",  # all_1s_quotes uses 'symbol' column
    )

    # Subscribe to specified tickers - efficiently filters from the large file
    # Returns dict of time series: {ticker: ts[EquityQuoteWoSize]}
    quotes_basket = parquet_reader.subscribe_dict_basket(
        typ=EquityQuoteWoSize,
        name="",  # No prefix needed since columns match struct fields
        shape=symbols if symbols is not None else ALL_TICKERS,
    )

    return quotes_basket


from datetime import timedelta


@csp.graph
def get_underlying_quotes() -> Dict[str, ts[EquityQuoteWoSize]]:
    """
    Get quotes for all tickers using the 1-second downsampled data.

    Returns:
        Dict basket: {ticker: ts[EquityQuoteWoSize]} - quotes without size information for all tickers
    """
    timer = csp.timer(timedelta(minutes=1))
    one_second_quotes = get_quotes_1s_wo_size(ALL_TICKERS)
    sampled_quotes = sample_dict(timer, one_second_quotes)
    return sampled_quotes


@csp.graph
def get_taq_quotes() -> Dict[str, ts[EquityQuoteWithSize]]:
    """
    Get quotes for all 40 tickers using subscribe_dict_basket.
    """
    parquet_reader = ParquetReader(
        filename_or_list=str(TAQ_QUOTES_PATH),
        time_column="timestamp",
        symbol_column="symbol",  # This is for filtering by symbol, not field mapping
    )

    # Map the parquet's 'ticker' column to the struct's 'symbol' field
    # Format: {parquet_column: struct_field}
    quotes_basket = parquet_reader.subscribe_dict_basket(
        typ=EquityQuoteWithSize, name="", shape=ALL_TICKERS
    )
    return quotes_basket


@csp.graph
def get_taq_trades() -> Dict[str, ts[EquityTrade]]:
    """
    Get trades for all 40 tickers using subscribe_dict_basket.
    """
    parquet_reader = ParquetReader(
        filename_or_list=str(TAQ_TRADES_PATH),
        time_column="timestamp",
        symbol_column="symbol",  # This is for filtering by symbol, not field mapping
    )

    # Map the parquet's 'ticker' column to the struct's 'symbol' field
    # Format: {parquet_column: struct_field}
    trades_basket = parquet_reader.subscribe_dict_basket(
        typ=EquityTrade, name="", shape=ALL_TICKERS
    )
    return trades_basket


@csp.graph
def get_kalshi_trades() -> Dict[str, ts[KalshiTrade]]:
    """
    Get Kalshi election trades for presidential contracts.

    Returns a dict basket: {contract_symbol: ts[KalshiTrade]}
    e.g., {"PRES-2024-KH": ts[KalshiTrade], "PRES-2024-DJT": ts[KalshiTrade]}
    """
    parquet_reader = ParquetReader(
        filename_or_list=str(KALSHI_TRADES_PATH),
        time_column="timestamp",
        symbol_column="symbol",
    )

    # Subscribe to Kalshi contracts
    kalshi_basket = parquet_reader.subscribe_dict_basket(
        typ=KalshiTrade, name="", shape=KALSHI_CONTRACTS
    )
    return kalshi_basket


@csp.graph
def get_option_quotes(file_name: str) -> ts[OptionQuote]:
    """
    Load all option quotes from a parquet file.

    Args:
        file_name: Path to the parquet file containing option quotes

    Returns:
        ts[OptionQuote]: Stream of all option quotes from the file
    """

    parquet_reader = ParquetReader(filename_or_list=file_name, time_column="timestamp")
    return parquet_reader.subscribe_all(OptionQuote)
