"""
Incremental RND Extraction Test - NVDA Single Ticker
=====================================================

Tests incremental update performance on NVDA over 5 consecutive minutes:
- Time 0 (10:00): Full extraction (baseline)
- Times 1-4 (10:01-10:04): Compare incremental vs full recompute
"""

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
import time
import warnings
import pyarrow.parquet as pq
from rnd_extraction import extract_rnd_ultra, MarketData

warnings.filterwarnings("ignore")

# ==================== CONFIGURATION ====================
DATA_DIR = Path("/home/grego/Alphathon/AlphathonDataSets/options_selected_formatted")
OUTPUT_DIR = Path("/home/grego/Alphathon/example_RND_incremental_nvda")

TICKER = "SPY"
BASE_TIME = pd.Timestamp("2024-11-05 10:00:00", tz="America/New_York")
EXPIRATION = pd.Timestamp("2024-11-15 16:00:00", tz="America/New_York")
RISK_FREE_RATE = 0.05341
MIN_BID = 0.01
# ========================================================

OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
print(f"Output directory: {OUTPUT_DIR}\n")
print(f"Testing {TICKER} incremental updates")
print(f"Base time: {BASE_TIME.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Time slices: 10:00, 10:01, 10:02, 10:03, 10:04\n")

# Generate 5 time slices
time_slices = [BASE_TIME + pd.Timedelta(minutes=i) for i in range(30)]

# Load data efficiently
data_path = DATA_DIR / f"{TICKER}_20241115_20241105_20241105_1000.parquet"
print(f"Loading {data_path.name}...", end=" ", flush=True)
table = pq.read_table(data_path)
df = table.to_pandas()
print(f"loaded ({len(df)} rows)\n")

results = []
prev_state = None

print("=" * 80)
print("EXTRACTION COMPARISON: Incremental vs Full Recompute")
print("=" * 80)

# Process each time slice
for time_idx, slice_time in enumerate(time_slices):
    print(f"\n[Time {time_idx}] {slice_time.strftime('%H:%M:%S')}")

    # Get data for this time slice
    data = df[df["timestamp"] == slice_time].copy()
    if len(data) == 0:
        unique_times = df["timestamp"].unique()
        nearest_idx = np.argmin(np.abs(unique_times - slice_time))
        actual_time = unique_times[nearest_idx]
        data = df[df["timestamp"] == actual_time].copy()
        print(f"  Using nearest time: {actual_time.strftime('%H:%M:%S')}")
    else:
        actual_time = slice_time

    # Calculate TTE
    tte = (EXPIRATION - actual_time).total_seconds() / (24 * 3600) / 365.25

    # Infer spot price
    temp_mid = (data["bid"] + data["ask"]) / 2
    temp_strike = data["strike"].values
    temp_right = data["right"].str.lower().values
    unique_strikes = np.unique(temp_strike)
    atm_strike = unique_strikes[
        np.argmin(np.abs(unique_strikes - np.median(unique_strikes)))
    ]

    call_mask = (temp_right == "c") & (temp_strike == atm_strike)
    put_mask = (temp_right == "p") & (temp_strike == atm_strike)

    if call_mask.sum() == 0 or put_mask.sum() == 0:
        print(f"  SKIP: No ATM options")
        continue

    call_mid = temp_mid[call_mask].iloc[0]
    put_mid = temp_mid[put_mask].iloc[0]
    spot = call_mid - put_mid + atm_strike * np.exp(-RISK_FREE_RATE * tte)

    # Prepare options
    vol = data[["strike", "right", "bid", "ask", "bid_size", "ask_size"]].reset_index(
        drop=True
    )
    vol = vol[vol.bid > MIN_BID].copy()
    vol["right"] = vol["right"].str.lower()

    if len(vol) < 10:
        print(f"  SKIP: Only {len(vol)} options")
        continue

    print(f"  Options: {len(vol)}, Spot: ${spot:.2f}")

    # Prepare market data
    market_data = MarketData(
        spot_price=spot,
        risk_free_rate=RISK_FREE_RATE,
        time_to_expiry=tte,
        options_df=vol,
    )

    if time_idx == 0:
        # TIME 0: Full extraction (baseline)
        print(f"  Baseline (full): ", end="", flush=True)
        t0 = time.perf_counter()
        result, prev_state = extract_rnd_ultra(market_data, 300, None, MIN_BID, True)
        full_time = (time.perf_counter() - t0) * 1000
        print(f"{full_time:.2f} ms")

        results.append(
            {
                "time_idx": time_idx,
                "time": actual_time,
                "n_options": len(vol),
                "full_time_ms": full_time,
                "incremental_time_ms": None,
                "speedup": None,
            }
        )
    else:
        # TIMES 1-4: Compare incremental vs full

        # Incremental (with cached state)
        print(f"  Incremental: ", end="", flush=True)
        t0 = time.perf_counter()
        result_inc, state_inc = extract_rnd_ultra(
            market_data, 300, prev_state, MIN_BID, True
        )
        inc_time = (time.perf_counter() - t0) * 1000
        print(f"{inc_time:.2f} ms")

        # Full recompute (no cached state)
        print(f"  Full recompute: ", end="", flush=True)
        t0 = time.perf_counter()
        result_full, state_full = extract_rnd_ultra(
            market_data, 300, None, MIN_BID, True
        )
        full_time = (time.perf_counter() - t0) * 1000
        print(f"{full_time:.2f} ms")

        speedup = full_time / inc_time
        print(f"  Speedup: {speedup:.2f}x {'✓ FASTER' if speedup > 1 else '✗ SLOWER'}")

        # Update prev_state for next iteration
        prev_state = state_inc

        results.append(
            {
                "time_idx": time_idx,
                "time": actual_time,
                "n_options": len(vol),
                "full_time_ms": full_time,
                "incremental_time_ms": inc_time,
                "speedup": speedup,
            }
        )

# Save results
results_df = pd.DataFrame(results)
results_df.to_csv(OUTPUT_DIR / "incremental_comparison.csv", index=False)

# Print summary
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"\n{TICKER} - {len(results)} time slices processed\n")
print(results_df.to_string(index=False))

if len(results) > 1:
    incremental_only = results_df[results_df["time_idx"] > 0]
    print(f"\n--- Incremental Performance (Times 1-4) ---")
    print(f"Mean speedup: {incremental_only['speedup'].mean():.3f}x")
    print(f"Median speedup: {incremental_only['speedup'].median():.3f}x")
    print(f"Min speedup: {incremental_only['speedup'].min():.3f}x")
    print(f"Max speedup: {incremental_only['speedup'].max():.3f}x")

    print(f"\nMean full recompute: {incremental_only['full_time_ms'].mean():.2f} ms")
    print(f"Mean incremental: {incremental_only['incremental_time_ms'].mean():.2f} ms")
    print(
        f"Savings: {incremental_only['full_time_ms'].mean() - incremental_only['incremental_time_ms'].mean():.2f} ms"
    )

print(f"\n\nResults saved to: {OUTPUT_DIR / 'incremental_comparison.csv'}")
print("=" * 80)
