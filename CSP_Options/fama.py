"""
CSP nodes for computing Fama-French 5-factor model in real-time.

Efficiently calculates Fama factor log prices and returns from market quote data
using precomputed factor weights. Handles missing data via weight renormalization.
"""

import json
from typing import Dict

import numpy as np

import csp
from csp import ts
from CSP_Options.structs import EquityQuoteWoSize, FamaFactors, FamaReturns


def load_factor_weights(json_path: str) -> Dict[str, Dict[str, float]]:
    """Load factor weights from JSON file."""
    with open(json_path, "r") as f:
        return json.load(f)


@csp.node
def compute_fama_log_prices(
    quotes_basket: {str: ts[EquityQuoteWoSize]},
    factor_weights: Dict[str, Dict[str, float]],
) -> ts[FamaFactors]:
    """
    Compute Fama-French factor log prices as weighted sum of individual log mid-prices.

    Factor_log_price = sum(w_i * log(mid_price_i))

    Handles missing data by renormalizing weights to maintain equivalent total weighting.
    """
    with csp.state():
        s_factor_names = ["HML", "SMB", "RMW", "CMA", "MKT-RF"]
        s_weights = {factor: factor_weights[factor] for factor in s_factor_names}

    if csp.ticked(quotes_basket):
        log_mid_prices = {}

        for ticker, quote in quotes_basket.validitems():
            mid_price = (quote.bid_price + quote.ask_price) / 2.0
            if mid_price > 0:
                log_mid_prices[ticker] = np.log(mid_price)

        factor_values = {}

        for factor_name in s_factor_names:
            weights = s_weights[factor_name]
            weighted_sum = 0.0
            valid_weight_sum = 0.0

            for ticker, weight in weights.items():
                if ticker in log_mid_prices:
                    weighted_sum += weight * log_mid_prices[ticker]
                    valid_weight_sum += weight

            if valid_weight_sum != 0:
                factor_log_price = weighted_sum / valid_weight_sum
            else:
                factor_log_price = np.nan

            factor_values[factor_name] = factor_log_price

        return FamaFactors(
            HML=factor_values["HML"],
            SMB=factor_values["SMB"],
            RMW=factor_values["RMW"],
            CMA=factor_values["CMA"],
            MKT_RF=factor_values["MKT-RF"],
        )


@csp.node
def compute_fama_returns(
    fama_log_prices: ts[FamaFactors],
) -> ts[FamaReturns]:
    """
    Compute Fama-French factor returns (innovations) from log prices.

    Factor_return = Factor_log_price(t) - Factor_log_price(t-1)

    Due to linearity: Factor_return = sum(w_i * log_return_i)
    """
    with csp.state():
        s_prev_log_prices = None

    if csp.ticked(fama_log_prices):
        current = fama_log_prices

        if s_prev_log_prices is not None:
            returns = FamaReturns(
                HML=current.HML - s_prev_log_prices.HML,
                SMB=current.SMB - s_prev_log_prices.SMB,
                RMW=current.RMW - s_prev_log_prices.RMW,
                CMA=current.CMA - s_prev_log_prices.CMA,
                MKT_RF=current.MKT_RF - s_prev_log_prices.MKT_RF,
            )
            s_prev_log_prices = current
            return returns
        else:
            s_prev_log_prices = current


@csp.node
def compute_fama_log_prices_efficient(
    quotes_basket: {str: ts[EquityQuoteWoSize]},
    factor_weights: Dict[str, Dict[str, float]],
) -> ts[FamaFactors]:
    """
    Optimized version that pre-filters to only relevant tickers.
    Use when quotes_basket contains many more tickers than needed.
    """
    with csp.state():
        s_factor_names = ["HML", "SMB", "RMW", "CMA", "MKT-RF"]
        s_relevant_tickers = set()
        for factor in s_factor_names:
            s_relevant_tickers.update(factor_weights[factor].keys())
        s_weights = {factor: factor_weights[factor] for factor in s_factor_names}

    if csp.ticked(quotes_basket):
        log_mid_prices = {}

        for ticker, quote in quotes_basket.validitems():
            if ticker in s_relevant_tickers:
                mid_price = (quote.bid_price + quote.ask_price) / 2.0
                if mid_price > 0:
                    log_mid_prices[ticker] = np.log(mid_price)

        factor_values = {}

        for factor_name in s_factor_names:
            weights = s_weights[factor_name]
            weighted_sum = 0.0
            valid_weight_sum = 0.0

            for ticker, weight in weights.items():
                if ticker in log_mid_prices:
                    weighted_sum += weight * log_mid_prices[ticker]
                    valid_weight_sum += weight

            if valid_weight_sum != 0:
                factor_log_price = weighted_sum / valid_weight_sum
            else:
                factor_log_price = np.nan

            factor_values[factor_name] = factor_log_price

        return FamaFactors(
            HML=factor_values["HML"],
            SMB=factor_values["SMB"],
            RMW=factor_values["RMW"],
            CMA=factor_values["CMA"],
            MKT_RF=factor_values["MKT-RF"],
        )


@csp.graph
def compute_fama_factors_graph(
    quotes_basket: {str: ts[EquityQuoteWoSize]},
    factor_weights_path: str,
    use_efficient: bool = True,
) -> csp.Outputs(
    log_prices=ts[FamaFactors],
    returns=ts[FamaReturns],
):
    """
    Complete graph for computing Fama-French factors from quotes.

    Returns named outputs: log_prices (FamaFactors) and returns (FamaReturns).
    """
    weights = load_factor_weights(factor_weights_path)

    if use_efficient:
        log_prices = compute_fama_log_prices_efficient(quotes_basket, weights)
    else:
        log_prices = compute_fama_log_prices(quotes_basket, weights)

    returns = compute_fama_returns(log_prices)

    return csp.output(log_prices=log_prices, returns=returns)




