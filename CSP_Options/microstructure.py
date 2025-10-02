"""
Microstructure feature computation nodes for high-frequency equity data.

This module provides CSP nodes to compute features for David's VECM model:
- Log mid-price
- Lee-Ready trade classification (buy/sell sign)
- Order flow metrics (ISO flow, signed dollar flow)
- Quote quality metrics (spread, quote updates)
- Aggregated 1-minute bars with all features
"""

from datetime import timedelta
from typing import Dict

import numpy as np

import csp
from csp import ts
from CSP_Options.structs import EquityBar1m, EquityQuoteWithSize, EquityTrade

# =============================================================================
# Basic Feature Nodes
# =============================================================================


@csp.node
def compute_log_mid(quote: ts[EquityQuoteWithSize]) -> ts[float]:
    """
    Compute log of mid-price from bid/ask quotes.

    Args:
        quote: Time series of quotes with bid_price and ask_price

    Returns:
        Log mid-price: log((bid + ask) / 2)
    """
    if csp.ticked(quote):
        mid_price = (quote.bid_price + quote.ask_price) / 2.0
        if mid_price > 0:
            return np.log(mid_price)
        else:
            # Return NaN for invalid prices
            return np.nan


@csp.node
def compute_rel_spread(quote: ts[EquityQuoteWithSize]) -> ts[float]:
    """
    Compute relative spread: (ask - bid) / mid.

    Args:
        quote: Time series of quotes with bid_price and ask_price

    Returns:
        Relative spread
    """
    if csp.ticked(quote):
        mid_price = (quote.bid_price + quote.ask_price) / 2.0
        if mid_price > 0:
            spread = quote.ask_price - quote.bid_price
            return spread / mid_price
        else:
            return np.nan


@csp.node
def lee_ready_classifier(trade: ts[EquityTrade], quote: ts[EquityQuoteWithSize]) -> ts[int]:
    """
    Lee-Ready trade classification algorithm.

    Classifies each trade as buy (+1) or sell (-1) by comparing trade price
    to the prevailing quote midpoint (tick rule).

    Args:
        trade: Time series of trades
        quote: Time series of quotes (must have valid quote before trade)

    Returns:
        +1 for buy-initiated trade, -1 for sell-initiated trade, 0 if uncertain
    """
    with csp.state():
        s_last_mid = None
        s_last_bid = None
        s_last_ask = None

    # Update quote midpoint whenever quote ticks
    if csp.ticked(quote):
        s_last_mid = (quote.bid_price + quote.ask_price) / 2.0
        s_last_bid = quote.bid_price
        s_last_ask = quote.ask_price

    # Classify trade when it ticks
    if csp.ticked(trade):
        if s_last_mid is None:
            # No quote available yet
            return 0

        # Tick rule: compare trade price to mid
        if trade.price > s_last_mid:
            return 1  # Buy (aggressive buyer hit the ask)
        elif trade.price < s_last_mid:
            return -1  # Sell (aggressive seller hit the bid)
        else:
            # At midpoint - use quote rule (distance to bid/ask)
            if s_last_bid is not None and s_last_ask is not None:
                distance_to_bid = abs(trade.price - s_last_bid)
                distance_to_ask = abs(trade.price - s_last_ask)

                if distance_to_ask < distance_to_bid:
                    return 1  # Closer to ask -> buy
                elif distance_to_bid < distance_to_ask:
                    return -1  # Closer to bid -> sell

            # Default: return 0 if we can't determine
            return 0


@csp.node
def is_iso_trade(trade: ts[EquityTrade]) -> ts[bool]:
    """
    Detect if a trade is an Intermarket Sweep Order (ISO).

    ISO trades are typically indicated by specific condition codes.
    Common ISO condition codes in TAQ data: 'F' or 'I'

    Args:
        trade: Time series of trades

    Returns:
        True if trade is ISO, False otherwise
    """
    if csp.ticked(trade):
        try:
            if trade.conditions:
                conditions = str(trade.conditions)
                return "14" in conditions
        except (AttributeError, TypeError):
            pass
        return False


@csp.node
def compute_signed_dollar_flow(trade: ts[EquityTrade], trade_sign: ts[int]) -> ts[float]:
    """
    Compute signed dollar flow: sign * price * size.

    Positive for buy-initiated trades, negative for sell-initiated.

    Args:
        trade: Time series of trades
        trade_sign: Time series of trade signs from Lee-Ready (+1/-1/0)

    Returns:
        Signed dollar flow
    """
    if csp.ticked(trade) and csp.valid(trade_sign):
        return float(trade_sign) * trade.price * trade.size


# =============================================================================
# 1-Minute Aggregation Node
# =============================================================================


@csp.node
def aggregate_1m_bar(
    symbol: str,
    trade: ts[EquityTrade],
    quote: ts[EquityQuoteWithSize],
    trade_sign: ts[int],
    is_iso: ts[bool],
    signed_flow: ts[float],
    log_mid: ts[float],
    rel_spread: ts[float],
    timer: ts[object],  # Trigger for bar output (every 1 minute)
) -> ts[EquityBar1m]:
    """
    Aggregate trades and quotes into 1-minute bars.

    This node accumulates all ticks within each minute and outputs
    aggregated features when the timer ticks.

    Args:
        symbol: Ticker symbol
        trade: Trade time series
        quote: Quote time series
        trade_sign: Trade sign time series (+1/-1/0)
        is_iso: ISO flag time series
        signed_flow: Signed dollar flow time series
        log_mid: Log mid-price time series
        rel_spread: Relative spread time series
        timer: Timer that triggers bar output (1 minute interval)

    Returns:
        Aggregated 1-minute bar with all features
    """
    with csp.state():
        # Trade accumulators
        s_num_trades = 0
        s_num_iso_trades = 0
        s_total_volume = 0.0
        s_iso_volume = 0.0
        s_total_flow = 0.0
        s_non_iso_flow = 0.0

        # Quote accumulators
        s_num_quote_updates = 0
        s_rel_spread_sum = 0.0
        s_rel_spread_count = 0

        # Last values
        s_last_log_mid = np.nan

    # Accumulate trade data
    if csp.ticked(trade) and csp.valid(trade_sign) and csp.valid(is_iso):
        s_num_trades += 1
        volume = float(trade.size)
        s_total_volume += volume

        if csp.valid(signed_flow):
            s_total_flow += signed_flow

        if is_iso:
            s_num_iso_trades += 1
            s_iso_volume += volume
        else:
            if csp.valid(signed_flow):
                s_non_iso_flow += signed_flow

    # Accumulate quote data
    if csp.ticked(quote):
        s_num_quote_updates += 1

        if csp.valid(rel_spread) and not np.isnan(rel_spread):
            s_rel_spread_sum += rel_spread
            s_rel_spread_count += 1

    # Update last log mid-price
    if csp.ticked(log_mid) and not np.isnan(log_mid):
        s_last_log_mid = log_mid

    # Output bar when timer ticks
    if csp.ticked(timer):
        # Calculate derived metrics
        iso_flow_intensity = s_iso_volume / s_total_volume if s_total_volume > 0 else 0.0

        avg_rsprd = s_rel_spread_sum / s_rel_spread_count if s_rel_spread_count > 0 else np.nan

        pct_trades_iso = float(s_num_iso_trades) / s_num_trades if s_num_trades > 0 else 0.0

        # Create bar
        bar = EquityBar1m(
            symbol=symbol,
            log_mid=s_last_log_mid,
            iso_flow_intensity=iso_flow_intensity,
            total_flow=s_total_flow,
            total_flow_non_iso=s_non_iso_flow,
            num_trades=s_num_trades,
            quote_updates=s_num_quote_updates,
            avg_rsprd=avg_rsprd,
            pct_trades_iso=pct_trades_iso,
        )

        # Reset accumulators for next bar
        s_num_trades = 0
        s_num_iso_trades = 0
        s_total_volume = 0.0
        s_iso_volume = 0.0
        s_total_flow = 0.0
        s_non_iso_flow = 0.0
        s_num_quote_updates = 0
        s_rel_spread_sum = 0.0
        s_rel_spread_count = 0

        return bar


# =============================================================================
# Graph-Level Wiring
# =============================================================================


@csp.graph
def compute_microstructure_features_single(
    symbol: str,
    trades: ts[EquityTrade],
    quotes: ts[EquityQuoteWithSize],
    bar_interval: timedelta = timedelta(minutes=1),
) -> ts[EquityBar1m]:
    """
    Compute all microstructure features for a single symbol.

    This graph wires together all the feature computation nodes and
    produces aggregated bars at the specified interval.

    Args:
        symbol: Ticker symbol
        trades: Trade time series for this symbol
        quotes: Quote time series for this symbol
        bar_interval: Aggregation interval (default 1 minute)

    Returns:
        Time series of aggregated bars
    """
    # Create timer for bar aggregation
    timer = csp.timer(bar_interval)

    # Compute basic features
    log_mid = compute_log_mid(quotes)
    rel_spread = compute_rel_spread(quotes)

    # Classify trades
    trade_sign = lee_ready_classifier(trades, quotes)
    is_iso = is_iso_trade(trades)

    # Compute flow
    signed_flow = compute_signed_dollar_flow(trades, trade_sign)

    # Aggregate into bars
    bars = aggregate_1m_bar(
        symbol=symbol,
        trade=trades,
        quote=quotes,
        trade_sign=trade_sign,
        is_iso=is_iso,
        signed_flow=signed_flow,
        log_mid=log_mid,
        rel_spread=rel_spread,
        timer=timer,
    )

    return bars


@csp.graph
def compute_microstructure_features_basket(
    trades_basket: Dict[str, ts[EquityTrade]],
    quotes_basket: Dict[str, ts[EquityQuoteWithSize]],
    bar_interval: timedelta = timedelta(minutes=1),
) -> Dict[str, ts[EquityBar1m]]:
    """
    Compute microstructure features for a basket of symbols.

    Args:
        trades_basket: Dictionary basket of trades {symbol: ts[EquityTrade]}
        quotes_basket: Dictionary basket of quotes {symbol: ts[EquityQuoteWithSize]}
        bar_interval: Aggregation interval (default 1 minute)

    Returns:
        Dictionary basket of bars {symbol: ts[EquityBar1m]}
    """
    bars_basket = {}

    # Process each symbol
    for symbol in trades_basket.keys():
        if symbol in quotes_basket:
            bars_basket[symbol] = compute_microstructure_features_single(
                symbol=symbol,
                trades=trades_basket[symbol],
                quotes=quotes_basket[symbol],
                bar_interval=bar_interval,
            )

    return bars_basket


# =============================================================================
# Utility: Convert to DataFrame (for David's model)
# =============================================================================


@csp.node
def bars_basket_to_dataframe(
    bars_basket: {str: ts[EquityBar1m]}, timer: ts[object]
) -> ts[object]:  # Returns pd.DataFrame
    """
    Convert a basket of bars to a pandas DataFrame in long format.

    This outputs a DataFrame whenever the timer ticks, containing all
    valid bars across all symbols. The DataFrame is in "long" format
    ready for pivoting to David's wide format.

    Args:
        bars_basket: Dictionary basket of bar time series
        timer: Timer to trigger DataFrame output

    Returns:
        DataFrame with columns: bucket_ts, ticker, log_mid, iso_flow_intensity, etc.
    """
    import pandas as pd

    if csp.ticked(timer):
        rows = []
        current_time = csp.now()

        # Collect all valid bars
        for symbol, bar_ts in bars_basket.validitems():
            bar = bar_ts
            rows.append(
                {
                    "bucket_ts": current_time,
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

        if rows:
            return pd.DataFrame(rows)
