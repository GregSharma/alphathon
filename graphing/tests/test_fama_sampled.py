"""
Cleaner demonstration of Fama-French factors with periodic sampling.
Prints factor values every 30 seconds for easier analysis.
"""

import json

import pandas as pd

import csp
from csp import ts
from CSP_Options.fama import compute_fama_factors_graph
from CSP_Options.structs import FamaFactors, FamaReturns
from CSP_Options.utils.readers import get_quotes_1s_wo_size

DATE_OF_INTEREST = "2024-11-05"
FACTOR_WEIGHTS_PATH = "/home/grego/Alphathon/fama/factor_weights_tickers.json"

with open(FACTOR_WEIGHTS_PATH, "r") as f:
    factor_weights = json.load(f)

ALL_FACTOR_TICKERS = set()
for factor_dict in factor_weights.values():
    ALL_FACTOR_TICKERS.update(factor_dict.keys())

print(f"Computing Fama-French 5-Factor Model in Real-Time")
print(f"=" * 80)
print(f"Total unique tickers: {len(ALL_FACTOR_TICKERS)}")
print(f"Factors: HML, SMB, RMW, CMA, MKT-RF")
print(f"=" * 80)

start_ts = pd.Timestamp(f"{DATE_OF_INTEREST} 09:31:00", tz="America/New_York")
end_ts = pd.Timestamp(f"{DATE_OF_INTEREST} 09:41:00", tz="America/New_York")


@csp.node
def sample_and_print(
    log_prices: ts[FamaFactors],
    returns: ts[FamaReturns],
    timer: ts[object],
):
    """Print factors at regular intervals."""
    if csp.ticked(timer):
        if csp.valid(log_prices) and csp.valid(returns):
            print(f"\n{csp.now()}")
            print(
                f"  Log Prices - HML: {log_prices.HML:.6f}, SMB: {log_prices.SMB:.6f}, "
                f"RMW: {log_prices.RMW:.6f}, CMA: {log_prices.CMA:.6f}, MKT-RF: {log_prices.MKT_RF:.6f}"
            )
            print(
                f"  Returns    - HML: {returns.HML:.8f}, SMB: {returns.SMB:.8f}, "
                f"RMW: {returns.RMW:.8f}, CMA: {returns.CMA:.8f}, MKT-RF: {returns.MKT_RF:.8f}"
            )


@csp.graph
def main_graph():
    """Main graph that computes Fama factors from 1-second quotes."""
    quotes_basket = get_quotes_1s_wo_size(list(ALL_FACTOR_TICKERS))

    fama = compute_fama_factors_graph(
        quotes_basket=quotes_basket,
        factor_weights_path=FACTOR_WEIGHTS_PATH,
        use_efficient=True,
    )

    timer = csp.timer(pd.Timedelta(seconds=30))
    sample_and_print(fama.log_prices, fama.returns, timer)


print(f"\nRunning from {start_ts} to {end_ts}...\n")

csp.run(
    main_graph,
    starttime=start_ts,
    endtime=end_ts,
    realtime=False,
)

print("\n" + "=" * 80)
print("Fama-French factor computation complete!")
print("=" * 80)
