"""
Helper functions for converting CSP microstructure bars to David's VECM model format.

This module provides utilities to convert the CSP output to the exact
DataFrame format expected by David's VECM leadership model.
"""

from datetime import datetime
from typing import Dict, List

import pandas as pd


def bars_dict_to_long_df(bars_dict: Dict[str, object], timestamp: datetime) -> pd.DataFrame:
    """
    Convert a dictionary of EquityBar1m to a long-format DataFrame.

    This is typically called from a CSP node using csp.validitems().

    Args:
        bars_dict: Dictionary of {symbol: EquityBar1m} from the basket
        timestamp: Current timestamp for the bar

    Returns:
        DataFrame in long format with one row per symbol
    """
    rows = []

    for symbol, bar in bars_dict.items():
        rows.append(
            {
                "bucket_ts": timestamp,
                "ticker": bar.symbol,
                "log_mid": bar.log_mid,
                "iso_flow_intensity": bar.iso_flow_intensity,
                "total_flow": bar.total_flow,
                "total_flow_non_iso": bar.total_flow_non_iso,
                "num_trades": bar.num_trades,
                "quote_updates": bar.quote_updates,
                "avg_rsprd": bar.avg_rsprd,
                "pct_trades_iso": bar.pct_trades_iso,
            }
        )

    return pd.DataFrame(rows)


def pivot_to_wide_format(df_long: pd.DataFrame) -> pd.DataFrame:
    """
    Pivot from long to wide format for VECM model.

    This creates one column per (feature, ticker) combination.
    For example: log_mid_AAPL, log_mid_MSFT, iso_flow_intensity_AAPL, etc.

    Args:
        df_long: Long-format DataFrame with columns:
                 bucket_ts, ticker, log_mid, iso_flow_intensity, ...

    Returns:
        Wide-format DataFrame with columns:
        bucket_ts, log_mid_AAPL, log_mid_MSFT, ..., iso_flow_intensity_AAPL, ...
    """

    # Define the features to pivot
    feature_columns = [
        "log_mid",
        "iso_flow_intensity",
        "total_flow",
        "total_flow_non_iso",
        "num_trades",
        "quote_updates",
        "avg_rsprd",
        "pct_trades_iso",
    ]

    # Pivot each feature separately and concatenate
    pivoted_dfs = []

    for feature in feature_columns:
        if feature in df_long.columns:
            pivoted = df_long.pivot(index="bucket_ts", columns="ticker", values=feature)
            # Rename columns: AAPL -> feature_AAPL
            pivoted.columns = [f"{feature}_{col}" for col in pivoted.columns]
            pivoted_dfs.append(pivoted)

    # Concatenate all features
    df_wide = pd.concat(pivoted_dfs, axis=1)
    df_wide = df_wide.reset_index()

    return df_wide


def add_delta_returns(df_wide: pd.DataFrame, tickers: List[str]) -> pd.DataFrame:
    """
    Add delta log returns (David's optional feature).

    Computes the first difference of log_mid for each ticker.

    Args:
        df_wide: Wide-format DataFrame
        tickers: List of ticker symbols

    Returns:
        DataFrame with added delta_log_mid_{TICKER} columns
    """
    df = df_wide.copy()

    for ticker in tickers:
        log_mid_col = f"log_mid_{ticker}"
        delta_col = f"delta_log_mid_{ticker}"

        if log_mid_col in df.columns:
            df[delta_col] = df[log_mid_col].diff()

    return df


def add_time_of_day_bins(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add time-of-day bins (David's optional feature).

    Bins:
    - 'open60': First 60 minutes (9:30-10:30 ET)
    - 'mid': Middle of day (10:30-15:00 ET)
    - 'close60': Last 60 minutes (15:00-16:00 ET)

    Args:
        df: DataFrame with bucket_ts column

    Returns:
        DataFrame with added 'tod_bin' column
    """
    df = df.copy()

    # Convert to ET if not already
    df["bucket_ts"] = pd.to_datetime(df["bucket_ts"])

    # Extract time
    times = df["bucket_ts"].dt.time

    # Define bins
    open_end = pd.Timestamp("10:30:00").time()
    close_start = pd.Timestamp("15:00:00").time()

    def classify_time(t):
        if t < open_end:
            return "open60"
        elif t < close_start:
            return "mid"
        else:
            return "close60"

    df["tod_bin"] = times.apply(classify_time)

    return df


def prepare_for_vecm(
    df_long: pd.DataFrame, tickers: List[str], add_returns: bool = False, add_tod_bins: bool = False
) -> pd.DataFrame:
    """
    Complete pipeline: Convert long-format CSP output to VECM-ready wide format.

    This is the main function to use after collecting bars from CSP.

    Args:
        df_long: Long-format DataFrame from CSP
        tickers: List of ticker symbols
        add_returns: If True, add delta log returns
        add_tod_bins: If True, add time-of-day bins

    Returns:
        Wide-format DataFrame ready for VECM model
    """

    # Pivot to wide
    df_wide = pivot_to_wide_format(df_long)

    # Add optional features
    if add_returns:
        df_wide = add_delta_returns(df_wide, tickers)

    if add_tod_bins:
        df_wide = add_time_of_day_bins(df_wide)

    return df_wide


# =============================================================================
# CSP Integration: Collecting Bars During Runtime
# =============================================================================


class BarCollector:
    """
    Utility class to collect bars during CSP runtime for later analysis.

    Usage in CSP node:
        with csp.state():
            s_collector = BarCollector()

        if csp.ticked(timer):
            bars_dict = {s: b for s, b in bars_basket.validitems()}
            s_collector.add_bars(bars_dict, csp.now())

        with csp.stop():
            return s_collector.to_dataframe()
    """

    def __init__(self):
        self.bars_history: List[pd.DataFrame] = []

    def add_bars(self, bars_dict: Dict[str, object], timestamp: datetime):
        """Add a set of bars for a given timestamp."""
        df = bars_dict_to_long_df(bars_dict, timestamp)
        self.bars_history.append(df)

    def to_dataframe(self) -> pd.DataFrame:
        """Concatenate all collected bars into a single DataFrame."""
        if not self.bars_history:
            return pd.DataFrame()
        return pd.concat(self.bars_history, ignore_index=True)

    def to_wide_format(self, tickers: List[str], **kwargs) -> pd.DataFrame:
        """Get wide-format DataFrame ready for VECM."""
        df_long = self.to_dataframe()
        return prepare_for_vecm(df_long, tickers, **kwargs)
