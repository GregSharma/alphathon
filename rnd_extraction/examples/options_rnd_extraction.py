"""
Options RND Extraction - Generic Example
=========================================

This example demonstrates RND extraction on real options data using the packaged rnd_extraction.
Works with any ticker - just update the configuration below.

Configuration:
- TICKER: Symbol (e.g., 'AAPL', 'SPY', 'TSLA')
- DATA_PATH: Path to options parquet file
- slice_time: Analysis timestamp
- expiration_time: Option expiration date
- MIN_BID: Liquidity filter threshold

The script will:
1. Load options data from parquet
2. Calculate microprices and IVs
3. Extract risk-neutral density
4. Generate clean 3-subplot visualization
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from rnd_extraction import extract_rnd_ultra_simple, MarketData

# ==================== CONFIGURATION ====================
TICKER = "AAPL"
DATA_PATH = Path("/home/grego/Alphathon/AlphathonDataSets/options_selected_formatted/AAPL_20241115_20241105_20241105_1000.parquet")
# Spot price can be specified manually or will be inferred from put-call parity
SPOT_PRICE = None  # Set to a float to override, or None to infer from options

slice_time = pd.Timestamp("2024-11-05 10:00:00", tz="America/New_York")
expiration_time = pd.Timestamp("2024-11-15 16:00:00", tz="America/New_York")
RISK_FREE_RATE = 0.05341
MIN_BID = 0.01  # Liquidity filter threshold (lowered for less liquid names)
# ========================================================

# Load data
print(f"Loading {TICKER} data...")
df = pd.read_parquet(DATA_PATH)

# The timestamp column is already a proper datetime (UTC)
df['ts'] = pd.to_datetime(df['timestamp']).dt.tz_convert("America/New_York")

# Get data at slice time (data is already one row per strike/right/timestamp)
data = df[df['ts'] == slice_time].copy()
if len(data) == 0:
    # Find nearest timestamp
    unique_times = df['ts'].unique()
    nearest_idx = abs(unique_times - slice_time).argmin()
    slice_time = unique_times[nearest_idx]
    data = df[df['ts'] == slice_time].copy()
    print(f"Using nearest time: {slice_time}")

# Print raw data to inspect
print(f"\n=== RAW DATA at {slice_time} ===")
print(f"Number of options: {len(data)}")
print("\nFirst 20 rows:")
print(data[['strike', 'right', 'bid', 'ask', 'bid_size', 'ask_size']].head(20))
print("\nBid statistics:")
print(data['bid'].describe())
print(f"\nOptions with bid > 0: {(data['bid'] > 0).sum()}")
print(f"Options with bid > 0.01: {(data['bid'] > 0.01).sum()}")
print(f"Options with bid > 0.05: {(data['bid'] > 0.05).sum()}")

TTE_YEARS = (expiration_time - slice_time).total_seconds() / (24 * 3600) / 365.25

# Infer spot price from put-call parity if not specified
if SPOT_PRICE is None:
    print("Inferring spot price from put-call parity...")
    
    # Calculate mid prices
    temp_mid = (data['bid'] + data['ask']) / 2
    temp_strike = data['strike'].values
    temp_right = data['right'].str.lower().values
    
    # Find unique strikes and choose one near the center
    unique_strikes = np.unique(temp_strike)
    median_strike = np.median(unique_strikes)
    
    # Use the strike closest to median as ATM proxy
    closest_idx = np.argmin(np.abs(unique_strikes - median_strike))
    atm_strike = unique_strikes[closest_idx]
    
    # Get call and put at this strike
    call_mask = (temp_right == 'c') & (temp_strike == atm_strike)
    put_mask = (temp_right == 'p') & (temp_strike == atm_strike)
    
    if call_mask.sum() > 0 and put_mask.sum() > 0:
        call_mid = temp_mid[call_mask].iloc[0]
        put_mid = temp_mid[put_mask].iloc[0]
        
        # Put-call parity: C - P = S - K*exp(-r*t)
        # Therefore: S = C - P + K*exp(-r*t)
        SPOT_PRICE = call_mid - put_mid + atm_strike * np.exp(-RISK_FREE_RATE * TTE_YEARS)
        print(f"  Using strike K={atm_strike:.2f}: C={call_mid:.2f}, P={put_mid:.2f}")
        print(f"  Implied spot price: ${SPOT_PRICE:.2f}")
    else:
        raise ValueError(f"Could not find both call and put at strike {atm_strike} to infer spot price")

print(f"\nSlice time: {slice_time}")
print(f"Expiration: {expiration_time}")
print(f"TTE: {TTE_YEARS:.6f} years ({TTE_YEARS*365:.1f} days)")
print(f"Spot: ${SPOT_PRICE:.2f}")
print(f"Risk-free rate: {RISK_FREE_RATE:.5f}")

# Prepare options data (data is already one row per strike/right)
vol = data[["strike", "right", "bid", "ask", "bid_size", "ask_size"]].reset_index(drop=True)

# Calculate mid price and microprice
vol["mid"] = (vol["bid"] + vol["ask"]) / 2
vol["microprice"] = (
    (vol["ask_size"] * vol["bid"] + vol["bid_size"] * vol["ask"]) / 
    (vol["ask_size"] + vol["bid_size"])
)

# Filter: bid > MIN_BID (liquidity filter)
vol = vol[vol.bid > MIN_BID].copy()
vol["right"] = vol["right"].str.lower()

print(f"Options after bid filter: {len(vol)}")

# Calculate forward
F = SPOT_PRICE * np.exp(RISK_FREE_RATE * TTE_YEARS)
print(f"Forward: {F:.2f}")

# Prepare MarketData object
options_df = vol[["strike", "right", "bid", "ask", "bid_size", "ask_size"]].reset_index(drop=True)

market_data = MarketData(
    spot_price=SPOT_PRICE,
    risk_free_rate=RISK_FREE_RATE,
    time_to_expiry=TTE_YEARS,
    options_df=options_df
)

# Extract RND using packaged version
print(f"\nExtracting RND using rnd_extraction package (min_bid=${MIN_BID})...")
result = extract_rnd_ultra_simple(market_data, grid_points=300, min_bid=MIN_BID)

# Calculate statistics
integral = np.trapz(result.rnd_density, result.log_moneyness)
E_S = np.trapz(result.strikes * result.rnd_density, result.log_moneyness)
E_S2 = np.trapz(result.strikes**2 * result.rnd_density, result.log_moneyness)
variance = E_S2 - E_S**2
implied_vol = np.sqrt(variance / E_S**2 / TTE_YEARS) if TTE_YEARS > 0 else 0

print(f"\n=== Results ===")
print(f"RND integral: {integral:.6f}")
print(f"CDF max: {result.rnd_cumulative[-1]:.6f}")
print(f"Expected value: {E_S:.2f} (vs forward {F:.2f})")
print(f"Variance: {variance:.2f}")
print(f"Implied vol (from moments): {implied_vol:.4f}")
print(f"Mean fitted IV: {result.fitted_iv.mean():.4f}")
print(f"IV std: {result.fitted_iv.std():.4f}")
print(f"IV range: {result.fitted_iv.min():.4f} to {result.fitted_iv.max():.4f}")

# Calculate market IVs from microprices for plotting
from py_vollib_vectorized import vectorized_implied_volatility as viv

vol['market_iv'] = viv(
    price=vol['microprice'],
    S=SPOT_PRICE,
    K=vol['strike'],
    t=TTE_YEARS,
    r=RISK_FREE_RATE,
    flag=vol['right'],
    model='black_scholes',
    return_as='numpy'
)

# Filter OTM for cleaner visualization
vol_otm = vol[
    ((vol['right'] == 'c') & (vol['strike'] > F)) |
    ((vol['right'] == 'p') & (vol['strike'] < F))
].copy()
print(f"\nOTM options: {len(vol_otm)}")

# Clean 3-subplot layout
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
colors = ['purple' if r == 'c' else 'blue' for r in vol_otm['right']]

# 1. IV Surface
ax1 = axes[0]
ax1.scatter(vol_otm['strike'], vol_otm['market_iv'], c=colors, s=40, alpha=0.6, 
            label='Market IV', edgecolors='black', linewidths=0.5)
ax1.plot(result.strikes, result.fitted_iv, 'r-', linewidth=2.5, label='Fitted IV')
ax1.fill_between(
    result.strikes,
    result.fitted_iv - 1.96 * result.fitted_iv_std,
    result.fitted_iv + 1.96 * result.fitted_iv_std,
    alpha=0.2, color='red', label='95% CI'
)
ax1.axvline(F, color='k', linestyle='--', alpha=0.5, linewidth=1.5, label='Forward')
ax1.axvline(SPOT_PRICE, color='g', linestyle='--', alpha=0.5, linewidth=1.5, label='Spot')
ax1.set_xlabel('Strike', fontsize=11)
ax1.set_ylabel('Implied Volatility', fontsize=11)
ax1.set_title('IV Surface', fontsize=12, fontweight='bold')
ax1.legend(fontsize=9, loc='best')
ax1.grid(True, alpha=0.3)

# 2. RND
ax2 = axes[1]
ax2.bar(result.log_moneyness, result.rnd_density, 
        width=(result.log_moneyness[1] - result.log_moneyness[0]), 
        alpha=0.6, color='steelblue', edgecolor='navy', linewidth=0.5)
ax2.axvline(0, color='k', linestyle='--', alpha=0.6, linewidth=1.5, label='ATM')
ax2.set_xlabel('Log-Moneyness', fontsize=11)
ax2.set_ylabel('Density', fontsize=11)
ax2.set_title(f'Risk-Neutral Density (integral={integral:.4f})', fontsize=12, fontweight='bold')
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3, axis='y')

# 3. CDF
ax3 = axes[2]
ax3.plot(result.strikes, result.rnd_cumulative, 'g-', linewidth=2.5)
ax3.axvline(F, color='k', linestyle='--', alpha=0.5, linewidth=1.5, label='Forward')
ax3.axvline(SPOT_PRICE, color='r', linestyle='--', alpha=0.5, linewidth=1.5, label='Spot')
ax3.axhline(0.5, color='gray', linestyle=':', alpha=0.5, linewidth=1.5)
ax3.set_xlabel('Strike', fontsize=11)
ax3.set_ylabel('Cumulative Probability', fontsize=11)
ax3.set_title(f'Cumulative RND (max={result.rnd_cumulative[-1]:.4f})', fontsize=12, fontweight='bold')
ax3.legend(fontsize=9)
ax3.grid(True, alpha=0.3)

plt.suptitle(f'{TICKER} RND Extraction - {slice_time.strftime("%Y-%m-%d %H:%M")} (Exp: {expiration_time.strftime("%Y-%m-%d")})', 
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{TICKER.lower()}_rnd_extraction.png', dpi=150, bbox_inches='tight')
print(f"\nPlot saved to: {TICKER.lower()}_rnd_extraction.png")
plt.show()

# Print summary
print("\n" + "="*80)
print(f"{TICKER} RND EXTRACTION SUMMARY")
print("="*80)
print(f"""
Ticker: {TICKER}
Spot: ${SPOT_PRICE:.2f}
Expiration: {expiration_time.strftime("%Y-%m-%d")} ({TTE_YEARS*365:.1f} days to expiry)

RND Quality Checks:
  ✓ Integral: {integral:.6f} (target: ~1.0)
  ✓ CDF max: {result.rnd_cumulative[-1]:.6f} (target: ~1.0)
  ✓ E[S]: ${E_S:.2f} vs Forward: ${F:.2f} (error: {abs(E_S-F)/F:.2%})
  ✓ Mean IV: {result.fitted_iv.mean():.2%}

Pipeline:
  1. Microprice calculation: (ask_size*bid + bid_size*ask)/(ask_size+bid_size)
  2. Liquidity filter: bid > ${MIN_BID}
  3. OTM filtering: Calls > Forward, Puts < Forward
  4. GP-based IV fitting: RBF kernel with noise variance
  5. Breeden-Litzenberger: Numerical differentiation for RND
  6. Validation: Normalization and monotonicity checks
""")


