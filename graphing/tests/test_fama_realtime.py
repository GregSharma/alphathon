"""
Test real-time Fama-French factor computation using 1s quotes.
Demonstrates calculation of all 5 factors with 3240 tickers.
"""

import json

import pandas as pd

import csp
from CSP_Options.fama import compute_fama_factors_graph
from CSP_Options.utils.readers import get_quotes_1s_wo_size

DATE_OF_INTEREST = "2024-11-05"
FACTOR_WEIGHTS_PATH = "/home/grego/Alphathon/fama/factor_weights_tickers.json"

with open(FACTOR_WEIGHTS_PATH, "r") as f:
    factor_weights = json.load(f)

ALL_FACTOR_TICKERS = set()
for factor_dict in factor_weights.values():
    ALL_FACTOR_TICKERS.update(factor_dict.keys())

print(f"Total unique tickers in Fama factors: {len(ALL_FACTOR_TICKERS)}")
print(f"Sample tickers: {sorted(list(ALL_FACTOR_TICKERS))[:10]}")

start_ts = pd.Timestamp(f"{DATE_OF_INTEREST} 09:31:00", tz="America/New_York")
end_ts = pd.Timestamp(f"{DATE_OF_INTEREST} 10:00:00", tz="America/New_York")


@csp.graph
def main_graph():
    """Main graph that computes Fama factors from 1-second quotes."""
    quotes_basket = get_quotes_1s_wo_size(list(ALL_FACTOR_TICKERS))

    fama = compute_fama_factors_graph(
        quotes_basket=quotes_basket,
        factor_weights_path=FACTOR_WEIGHTS_PATH,
        use_efficient=True,
    )

    csp.print("FAMA_LOG_PRICES", fama.log_prices)
    csp.print("FAMA_RETURNS", fama.returns)


print(f"\nRunning from {start_ts} to {end_ts}...")
print("=" * 80)

csp.run(
    main_graph,
    starttime=start_ts,
    endtime=end_ts,
    realtime=False,
)

print("=" * 80)
print("\nFama factor computation complete!")
