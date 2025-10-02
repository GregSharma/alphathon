"""
Reusable option processing graph components.

This module contains graph-level components for processing option chains,
including the full pipeline from raw quotes to fitted volatility models.
"""

from datetime import timedelta

import csp
import pandas as pd
from csp import ts

from CSP_Options.nodes import (
    create_model_params_dict,
    filter_vector_quote,
    quote_list_to_vector,
    sample_dynamic,
)
from CSP_Options.option_models import node_cw_fit, node_kde_fit
from CSP_Options.rnd_nodes import RNDResult, extract_rnd_features, extract_rnd_from_vq
from CSP_Options.structs import EquityQuoteWoSize, OptionQuote, VectorizedOptionQuote, VolFit
from CSP_Options.utils.readers import get_option_quotes


@csp.graph
def process_single_ticker(
    ticker: str,
    underlying_quote: ts[EquityQuoteWoSize],
    option_filename: str,
    expiration_ts: pd.Timestamp,
    max_dist: float = 0.1,
    min_bid: float = 0.05,
    grid_points: int = 300,
) -> csp.Outputs(
    ticker_out=ts[str],
    vec_quotes=ts[VectorizedOptionQuote],
    kde_fits=ts[VolFit],
    cw_fits=ts[VolFit],
    rnd_result=ts[RNDResult],
    rnd_mean=ts[float],
    rnd_std=ts[float],
    rnd_skew=ts[float],
    rnd_kurt=ts[float],
):
    """
    Process a single ticker through the complete volatility fitting + RND pipeline.

    Returns:
        ticker_out: Echo of ticker name for tracking
        vec_quotes: Vectorized option quotes with IVs
        kde_fits: KDE model fits
        cw_fits: Carr-Wu model fits
        rnd_result: Full RND extraction result
        rnd_mean, rnd_std, rnd_skew, rnd_kurt: RND moments
    """
    # 1. Ingest option data
    option_quotes: ts[OptionQuote] = get_option_quotes(option_filename)

    # 2. Dynamic demultiplex options stream into a basket keyed by symbol
    demuxed_options = csp.dynamic_demultiplex(x=option_quotes, key=option_quotes.symbol)

    # 3. Create a 1-minute timer to sample the chain
    sampling_timer = csp.timer(timedelta(minutes=1))

    # 4. Sample the latest quote for each option on the timer
    sampled_chain = sample_dynamic(sampling_timer, demuxed_options)

    # 5. Calculate time to expiration (TTE) in years
    times_ns = csp.times_ns(sampling_timer)
    expiration_ns = int(expiration_ts.timestamp() * 1e9)
    seconds_in_year = 365 * 24 * 60 * 60
    tte_yrs = (expiration_ns - times_ns) / (1e9 * seconds_in_year)

    # 6. Vectorize quotes
    list_of_quotes = csp.apply(sampled_chain, lambda d: list(d.values()), list)
    quote_output = quote_list_to_vector(list_of_quotes, tte_yrs)
    vec_quotes = quote_output.vq
    impl_spot_price = quote_output.impl_spot_price

    # 7. Filter for calibration
    # Keep only options struck <=10% from spot, with a minimum bid of $0.05
    vq_filt = filter_vector_quote(
        max_dist=max_dist, min_bid=min_bid, vq=vec_quotes, impl_spot_price=impl_spot_price
    )

    # 8. Fit models
    kde_fits = node_kde_fit(vq_filt, impl_spot_price)
    cw_fits = node_cw_fit(vq_filt)

    # 9. RND Extraction
    # Compute spot price from underlying quote
    spot_price = 0.5 * underlying_quote.ask_price + 0.5 * underlying_quote.bid_price

    # Extract RND from vectorized quotes
    rnd_result = extract_rnd_from_vq(
        vq=vec_quotes,
        spot_price=spot_price,
        grid_points=grid_points,
    )

    # Compute RND moments (mean, std, skew, kurtosis)
    rnd_moments = extract_rnd_features(rnd_result)

    # Echo ticker name whenever vec_quotes ticks
    ticker_out = csp.sample(vec_quotes, csp.const(ticker))

    return csp.output(
        ticker_out=ticker_out,
        vec_quotes=vec_quotes,
        kde_fits=kde_fits,
        cw_fits=cw_fits,
        rnd_result=rnd_result,
        rnd_mean=rnd_moments.mean,
        rnd_std=rnd_moments.std,
        rnd_skew=rnd_moments.skew,
        rnd_kurt=rnd_moments.kurt,
    )
