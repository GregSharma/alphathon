"""
Basic Usage Example: RND Extraction from Option Chain
======================================================

This example demonstrates how to extract risk-neutral density from a simple
option chain using the rnd_extraction package.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from rnd_extraction import extract_rnd_ultra_simple, MarketData
from py_vollib_vectorized import vectorized_black_scholes

# Example: Create sample option chain data
# In practice, this would come from market data (S3 parquet, API, etc.)

spot_price = 5500.0
risk_free_rate = 0.05
time_to_expiry = 30 / 365  # 30 days in years
constant_iv = 0.20  # 20% constant implied volatility

# Generate option chain using Black-Scholes pricing with constant volatility
# Use reasonable strike range (±15% around spot) to avoid deep OTM with tiny prices
strikes = np.arange(int(spot_price * 0.85), int(spot_price * 1.15), 25)  
n_options = len(strikes)

# Determine call or put (OTM options typically have better liquidity)
flags = ['c' if strike > spot_price else 'p' for strike in strikes]

# Price options using Black-Scholes with constant volatility
mid_prices = vectorized_black_scholes(
    flag=flags,
    S=spot_price,
    K=strikes,
    t=time_to_expiry,
    r=risk_free_rate,
    sigma=constant_iv,
    return_as='numpy'
)

# Add realistic bid-ask spread (1% of mid price, minimum $0.10)
spreads = np.maximum(0.10, mid_prices * 0.01)

# Create option chain dataframe with non-negative bids
options_data = []
for i, strike in enumerate(strikes):
    bid_price = max(0.10, mid_prices[i] - spreads[i] / 2)  # Ensure bid >= $0.10
    ask_price = bid_price + spreads[i]
    
    options_data.append({
        'strike': strike,
        'right': flags[i],
        'bid': bid_price,
        'ask': ask_price,
        'bid_size': np.random.randint(10, 100),
        'ask_size': np.random.randint(10, 100)
    })

options_df = pd.DataFrame(options_data)

# Create MarketData object
market_data = MarketData(
    spot_price=spot_price,
    risk_free_rate=risk_free_rate,
    time_to_expiry=time_to_expiry,
    options_df=options_df
)

# Extract RND
print("Extracting RND from Black-Scholes priced options...")
print(f"  Input constant IV: {constant_iv:.4f}")
result = extract_rnd_ultra_simple(market_data, grid_points=300)

# Display results
print("\nResults:")
print(f"  Forward price: ${result.forward_price:.2f}")
print(f"  Grid points: {len(result.strikes)}")
print(f"  RND integral: {np.trapz(result.rnd_density, result.log_moneyness):.6f}")
print(f"  Fitted IV range: {result.fitted_iv.min():.4f} to {result.fitted_iv.max():.4f}")
print(f"  Input constant IV: {constant_iv:.4f}")

# Plot results
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 1. Risk-neutral density
axes[0, 0].plot(result.strikes, result.rnd_density, 'b-', linewidth=2)
axes[0, 0].axvline(spot_price, color='r', linestyle='--', label='Spot')
axes[0, 0].axvline(result.forward_price, color='g', linestyle='--', label='Forward')
axes[0, 0].set_xlabel('Strike')
axes[0, 0].set_ylabel('Probability Density')
axes[0, 0].set_title('Risk-Neutral Density')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# 2. Cumulative RND
axes[0, 1].plot(result.strikes, result.rnd_cumulative, 'g-', linewidth=2)
axes[0, 1].set_xlabel('Strike')
axes[0, 1].set_ylabel('Cumulative Probability')
axes[0, 1].set_title('Cumulative RND (CDF)')
axes[0, 1].grid(True, alpha=0.3)

# 3. Implied volatility surface
axes[1, 0].plot(result.strikes, result.fitted_iv, 'b-', linewidth=2, label='Fitted IV')
axes[1, 0].fill_between(
    result.strikes,
    result.fitted_iv - 1.96 * result.fitted_iv_std,
    result.fitted_iv + 1.96 * result.fitted_iv_std,
    alpha=0.3,
    label='95% CI'
)
axes[1, 0].set_xlabel('Strike')
axes[1, 0].set_ylabel('Implied Volatility')
axes[1, 0].set_title('GP-Fitted IV Surface')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# 4. Characteristic function
axes[1, 1].plot(result.characteristic_function_u, 
                np.real(result.characteristic_function_values), 
                'b-', label='Real part')
axes[1, 1].plot(result.characteristic_function_u, 
                np.imag(result.characteristic_function_values), 
                'r--', label='Imaginary part')
axes[1, 1].set_xlabel('u')
axes[1, 1].set_ylabel('φ(u)')
axes[1, 1].set_title('Characteristic Function')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('rnd_extraction_example.png', dpi=150, bbox_inches='tight')
print(f"\nPlot saved to: rnd_extraction_example.png")

# Example: Compute statistics from RND
print("\n--- RND Statistics ---")

# Expected value (mean)
E_S = np.trapz(result.strikes * result.rnd_density, result.log_moneyness)
print(f"Expected value: ${E_S:.2f}")

# Variance
E_S2 = np.trapz(result.strikes**2 * result.rnd_density, result.log_moneyness)
variance = E_S2 - E_S**2
volatility = np.sqrt(variance / time_to_expiry)
print(f"Implied volatility (from moments): {volatility:.4f}")

# Probabilities
prob_otm_call = 1 - result.rnd_cumulative[result.strikes >= spot_price][0]
prob_otm_put = result.rnd_cumulative[result.strikes <= spot_price][-1]
print(f"P(S > {spot_price}): {prob_otm_call:.4f}")
print(f"P(S < {spot_price}): {prob_otm_put:.4f}")

