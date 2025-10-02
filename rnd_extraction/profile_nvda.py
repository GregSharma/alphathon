"""Profile NVDA RND extraction to find bottlenecks."""

import time
import numpy as np
import pandas as pd
import warnings
from pathlib import Path
from rnd_extraction import extract_rnd_ultra_simple, MarketData

warnings.filterwarnings('ignore')

# Load NVDA
print("Loading NVDA data...")
df = pd.read_parquet('/home/grego/Alphathon/AlphathonDataSets/options_selected_formatted/NVDA_20241115_20241105_20241105_1000.parquet')
df['ts'] = pd.to_datetime(df['timestamp']).dt.tz_convert("America/New_York")
slice_time = pd.Timestamp("2024-11-05 10:00:00", tz="America/New_York")
data = df[df['ts'] == slice_time]

vol = data[["strike", "right", "bid", "ask", "bid_size", "ask_size"]].reset_index(drop=True)
vol = vol[vol.bid > 0.01].copy()

# Infer spot
temp_mid = (data['bid'] + data['ask']) / 2
temp_strike = data['strike'].values
temp_right = data['right'].str.lower().values
unique_strikes = np.unique(temp_strike)
atm_strike = unique_strikes[np.argmin(np.abs(unique_strikes - np.median(unique_strikes)))]
call_mask = (temp_right == 'c') & (temp_strike == atm_strike)
put_mask = (temp_right == 'p') & (temp_strike == atm_strike)
call_mid = temp_mid[call_mask].iloc[0]
put_mid = temp_mid[put_mask].iloc[0]
spot = call_mid - put_mid + atm_strike * np.exp(-0.05341 * 0.027)

options_df = vol[["strike", "right", "bid", "ask", "bid_size", "ask_size"]]
market_data = MarketData(spot, 0.05341, 0.027, options_df)

print(f"NVDA: {len(vol)} options, Spot: ${spot:.2f}")

# Profile multiple runs
print("\n=== Profiling ===")
print("Run | Exact GP | LowRank GP")
print("----|----------|------------")

for i in range(5):
    # Exact
    t0 = time.perf_counter()
    result_exact = extract_rnd_ultra_simple(market_data, grid_points=50, min_bid=0.01, use_lowrank=False)
    exact_time = (time.perf_counter() - t0) * 1000
    
    # LowRank
    t0 = time.perf_counter()
    result_lowrank = extract_rnd_ultra_simple(market_data, grid_points=50, min_bid=0.01, use_lowrank=True)
    lowrank_time = (time.perf_counter() - t0) * 1000
    
    print(f" {i+1}  | {exact_time:7.1f}ms | {lowrank_time:9.1f}ms")

print("\n=== Comparison ===")
print(f"Exact RND integral: {np.trapz(result_exact.rnd_density, result_exact.log_moneyness):.6f}")
print(f"LowRank RND integral: {np.trapz(result_lowrank.rnd_density, result_lowrank.log_moneyness):.6f}")
print(f"L2 difference: {np.sqrt(np.mean((result_exact.rnd_density - result_lowrank.rnd_density)**2)):.6f}")

