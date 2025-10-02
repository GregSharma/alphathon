"""
Test microstructure feature computation nodes.

This test file demonstrates and validates the microstructure computation
pipeline for David's VECM model.
"""

import logging
from datetime import timedelta

import csp
import pandas as pd
import pytz
from csp.basketlib import sample_dict

from CSP_Options.microstructure import (
    compute_log_mid,
    compute_microstructure_features_basket,
    compute_microstructure_features_single,
    compute_rel_spread,
    compute_signed_dollar_flow,
    is_iso_trade,
    lee_ready_classifier,
)
from CSP_Options.utils.readers import get_taq_quotes, get_taq_trades

logging.basicConfig(level=logging.INFO)

DATE_OF_INTEREST = "2024-11-05"

# Test parameters
start_ts = pd.Timestamp(f"{DATE_OF_INTEREST} 09:30:00", tz="America/New_York")
end_ts = start_ts + pd.Timedelta(minutes=5)  # 5 minute test


# =============================================================================
# Test 1: Basic Feature Computation (Single Symbol)
# =============================================================================


@csp.graph
def test_basic_features():
    """Test basic feature nodes: log_mid, rel_spread for a single symbol."""

    # Get data baskets
    quotes_basket = get_taq_quotes()

    # Pick one symbol for detailed testing
    test_symbol = "AAPL"
    aapl_quotes = quotes_basket[test_symbol]

    # Compute features
    log_mid = compute_log_mid(aapl_quotes)
    rel_spread = compute_rel_spread(aapl_quotes)

    # Log outputs
    csp.log(
        logging.INFO,
        "AAPL_log_mid",
        log_mid,
        logger_tz=pytz.timezone("America/New_York"),
        use_thread=True,
    )

    csp.log(
        logging.INFO,
        "AAPL_rel_spread",
        rel_spread,
        logger_tz=pytz.timezone("America/New_York"),
        use_thread=True,
    )


# =============================================================================
# Test 2: Lee-Ready Classification
# =============================================================================


@csp.graph
def test_lee_ready():
    """Test Lee-Ready trade classification."""

    # Get data
    quotes_basket = get_taq_quotes()
    trades_basket = get_taq_trades()

    # Test on AAPL
    test_symbol = "AAPL"
    aapl_quotes = quotes_basket[test_symbol]
    aapl_trades = trades_basket[test_symbol]

    # Classify trades
    trade_sign = lee_ready_classifier(aapl_trades, aapl_quotes)
    is_iso = is_iso_trade(aapl_trades)
    signed_flow = compute_signed_dollar_flow(aapl_trades, trade_sign)

    # Log results
    csp.log(
        logging.INFO,
        "AAPL_trade",
        aapl_trades,
        logger_tz=pytz.timezone("America/New_York"),
        use_thread=True,
    )

    csp.log(
        logging.INFO,
        "AAPL_trade_sign",
        trade_sign,
        logger_tz=pytz.timezone("America/New_York"),
        use_thread=True,
    )

    csp.log(
        logging.INFO,
        "AAPL_is_iso",
        is_iso,
        logger_tz=pytz.timezone("America/New_York"),
        use_thread=True,
    )

    csp.log(
        logging.INFO,
        "AAPL_signed_flow",
        signed_flow,
        logger_tz=pytz.timezone("America/New_York"),
        use_thread=True,
    )


# =============================================================================
# Test 3: 1-Minute Bar Aggregation (Single Symbol)
# =============================================================================


@csp.graph
def test_1m_bars_single():
    """Test 1-minute bar aggregation for a single symbol."""

    # Get data
    quotes_basket = get_taq_quotes()
    trades_basket = get_taq_trades()

    # Test on AAPL
    test_symbol = "AAPL"
    aapl_quotes = quotes_basket[test_symbol]
    aapl_trades = trades_basket[test_symbol]

    # Compute 1-minute bars
    bars = compute_microstructure_features_single(
        symbol=test_symbol,
        trades=aapl_trades,
        quotes=aapl_quotes,
        bar_interval=timedelta(minutes=1),
    )

    # Log bars
    csp.log(
        logging.INFO,
        "AAPL_1m_bar",
        bars,
        logger_tz=pytz.timezone("America/New_York"),
        use_thread=True,
    )


# =============================================================================
# Test 4: Basket Aggregation (All 40 Tickers)
# =============================================================================


@csp.graph
def test_1m_bars_basket():
    """Test 1-minute bar aggregation for all tickers."""

    # Get data baskets
    quotes_basket = get_taq_quotes()
    trades_basket = get_taq_trades()

    # Compute bars for all symbols
    bars_basket = compute_microstructure_features_basket(
        trades_basket=trades_basket, quotes_basket=quotes_basket, bar_interval=timedelta(minutes=1)
    )

    # Sample and log every minute
    timer = csp.timer(timedelta(minutes=1))
    sampled_bars = sample_dict(timer, bars_basket)

    csp.log(
        logging.INFO,
        "all_bars_sampled",
        sampled_bars,
        logger_tz=pytz.timezone("America/New_York"),
        use_thread=True,
    )


# =============================================================================
# Test 5: DataFrame Output (David's Format)
# =============================================================================


@csp.graph
def test_dataframe_output():
    """
    Test converting bars to DataFrame format for David's model.

    This produces a DataFrame in long format ready for pivoting.
    """

    from CSP_Options.microstructure import bars_basket_to_dataframe

    # Get data baskets
    quotes_basket = get_taq_quotes()
    trades_basket = get_taq_trades()

    # Compute bars at 1-minute intervals
    bars_basket = compute_microstructure_features_basket(
        trades_basket=trades_basket, quotes_basket=quotes_basket, bar_interval=timedelta(minutes=1)
    )

    # Convert to DataFrame every minute
    timer = csp.timer(timedelta(minutes=1))
    df = bars_basket_to_dataframe(bars_basket, timer)

    # Log the DataFrame
    csp.log(
        logging.INFO,
        "dataframe_output",
        df,
        logger_tz=pytz.timezone("America/New_York"),
        use_thread=True,
    )


# =============================================================================
# Run Tests
# =============================================================================


def run_test(test_func, test_name):
    """Helper to run a test graph."""
    print(f"\n{'=' * 80}")
    print(f"Running: {test_name}")
    print(f"From {start_ts} to {end_ts}")
    print(f"{'=' * 80}\n")

    csp.run(
        test_func,
        starttime=start_ts,
        endtime=end_ts,
        realtime=False,
    )

    print(f"\n{test_name} completed!\n")


if __name__ == "__main__":
    # Run tests individually - comment out tests you don't want to run

    # Test 1: Basic features (log mid, spread)
    run_test(test_basic_features, "Test 1: Basic Features (Log Mid, Rel Spread)")

    # Test 2: Lee-Ready classification
    run_test(test_lee_ready, "Test 2: Lee-Ready Trade Classification")

    # Test 3: 1-minute bars (single symbol)
    run_test(test_1m_bars_single, "Test 3: 1-Minute Bars (Single Symbol)")

    # Test 4: 1-minute bars (all symbols)
    run_test(test_1m_bars_basket, "Test 4: 1-Minute Bars (All Symbols)")

    # Test 5: DataFrame output
    run_test(test_dataframe_output, "Test 5: DataFrame Output (David's Format)")
