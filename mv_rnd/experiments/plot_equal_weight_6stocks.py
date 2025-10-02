"""
Equal-Weighted Portfolio of 6 Equities using RND and CBRA
===========================================================

Uses extracted risk-neutral densities from options to create an equal-weighted
portfolio distribution using CBRA (Constraint-Based Rearrangement Algorithm).

Stocks: AAPL, AMD, AMZN, AVGO, CEG, DJT
"""

import os
import sys
import pickle
import numpy as np
from pathlib import Path
from scipy import stats, interpolate
import matplotlib.pyplot as plt

# Add cbrapipe to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from cbrapipe import (
    discretize_instruments,
    discretize_constraints,
    build_initial_matrix,
    expand_coefficients,
    identify_admissible_blocks,
    cbra_optimize,
    extract_joint_distribution,
)

# Configuration
PICKLE_DIR = Path("/home/grego/Alphathon/example_RND_10am_1105/rnd_pickles")
TICKERS = ['AAPL', 'AMD', 'AMZN', 'AVGO', 'CEG', 'DJT']
OUTPUT_DIR = Path("/home/grego/Alphathon/MVRND/experiments/output")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

print("="*80)
print("EQUAL-WEIGHTED PORTFOLIO: 6 EQUITY RND → CBRA")
print("="*80)
print(f"\nStocks: {', '.join(TICKERS)}")
print(f"Weights: Equal (1/6 each)")
print(f"Data source: Options-implied RND from {PICKLE_DIR}")


def load_rnd_pickle(ticker: str) -> dict:
    """Load RND data from pickle file."""
    pickle_path = PICKLE_DIR / f"{ticker.lower()}_rnd.pkl"
    with open(pickle_path, 'rb') as f:
        return pickle.load(f)


def rnd_to_quantile_function(strikes: np.ndarray, cdf: np.ndarray):
    """
    Convert RND (strikes, CDF) to a quantile function F_inv(p).
    
    For CBRA, we need F_inv: [0,1] → R that gives the value at probability p.
    This is the inverse of the CDF.
    """
    # Ensure CDF is strictly increasing for interpolation
    # Add small epsilon to ensure monotonicity
    cdf_monotonic = np.maximum.accumulate(cdf)
    
    # Normalize to [0, 1]
    cdf_normalized = (cdf_monotonic - cdf_monotonic[0]) / (cdf_monotonic[-1] - cdf_monotonic[0])
    
    # Create inverse CDF (quantile function)
    # ppf(p) returns the strike value at probability p
    def ppf(p):
        # Handle array or scalar
        p = np.atleast_1d(p)
        result = np.interp(p, cdf_normalized, strikes)
        return result if len(p) > 1 else result[0]
    
    return ppf


# Load RND data for all 6 stocks
print("\nLoading RND data...")
rnd_data = {}
for ticker in TICKERS:
    try:
        data = load_rnd_pickle(ticker)
        rnd_data[ticker] = data
        print(f"  ✓ {ticker}: Spot=${data['spot_price']:.2f}, {len(data['strikes'])} RND points")
    except FileNotFoundError:
        print(f"  ✗ {ticker}: Pickle not found at {PICKLE_DIR}/{ticker.lower()}_rnd.pkl")
        sys.exit(1)

# Convert RNDs to quantile functions
print("\nCreating quantile functions from RND cumulative distributions...")
F_inv_stocks = []
for ticker in TICKERS:
    data = rnd_data[ticker]
    ppf = rnd_to_quantile_function(data['strikes'], data['rnd_cumulative'])
    F_inv_stocks.append(ppf)
    
    # Test the quantile function
    test_probs = [0.01, 0.25, 0.50, 0.75, 0.99]
    test_vals = [ppf(p) for p in test_probs]
    print(f"  {ticker}: P(0.50)=${test_vals[2]:.2f}, P(0.99)=${test_vals[4]:.2f}")

# Build CBRA setup
print("\n" + "="*80)
print("CBRA SETUP")
print("="*80)

d = len(TICKERS)  # 6 stocks
K = 1  # One constraint: equal-weighted portfolio sum

# Constraint: S1 = (1/6) * (X1 + X2 + X3 + X4 + X5 + X6)
# This is the equal-weighted portfolio return
A = np.ones((d, K)) / d  # Each stock contributes 1/6

print(f"\nDimensions:")
print(f"  d = {d} stocks")
print(f"  K = {K} constraint (equal-weighted portfolio)")
print(f"\nWeight matrix A:")
print(A)

# For the constraint distribution, we need the distribution of the portfolio sum
# We'll use Monte Carlo to approximate it from the marginals
print("\nComputing portfolio constraint distribution via Monte Carlo...")
mc_n = 10000  # Smaller for faster computation
mc_X = np.zeros((mc_n, d))
for j, ticker in enumerate(TICKERS):
    print(f"  Sampling {ticker}...", end=" ", flush=True)
    # Sample from each marginal using quantile function
    u = np.random.uniform(0.001, 0.999, mc_n)
    mc_X[:, j] = np.array([F_inv_stocks[j](p) for p in u])
    print("done")

# Compute equal-weighted portfolio (mean of returns)
S1_mc = mc_X.mean(axis=1)
print(f"  Portfolio MC stats: mean={S1_mc.mean():.4f}, std={S1_mc.std():.4f}")

# Create empirical quantile function for constraint
S1_sorted = np.sort(S1_mc)
def portfolio_ppf(p):
    p = np.atleast_1d(p)
    idx = np.clip((p * len(S1_sorted)).astype(int), 0, len(S1_sorted) - 1)
    result = S1_sorted[idx]
    return result if len(p) > 1 else result[0]

F_inv_constraints = [portfolio_ppf]

# Discretize for CBRA
n_cbra = 1000  # Small for fast testing
print(f"\nDiscretizing with n={n_cbra} samples...")
print("  Discretizing instruments...", end=" ", flush=True)
X = discretize_instruments(n_cbra, F_inv_stocks)
print("done")
print("  Discretizing constraints...", end=" ", flush=True)
S = discretize_constraints(n_cbra, F_inv_constraints)
print("done")

print(f"  X shape: {X.shape}, S shape: {S.shape}")

# Expand coefficients
print("  Expanding coefficients...", end=" ", flush=True)
tilde_alpha = expand_coefficients(A)
print(f"done (shape: {tilde_alpha.shape})")

# Build initial matrix
print("  Building initial matrix...", end=" ", flush=True)
Y = build_initial_matrix(X, S)
print("done")

# Randomize to avoid comonotonic start
for j in range(Y.shape[1]):
    np.random.shuffle(Y[:, j])

# Identify admissible blocks
blocks = identify_admissible_blocks(tilde_alpha)
print(f"  Admissible blocks: {len(blocks)}")

# Run CBRA with lower iterations for speed
max_iter = 100  # Very low for quick testing
print(f"\nRunning CBRA optimization (max_iter={max_iter})...")
print("-" * 80)
Y_final = cbra_optimize(Y, tilde_alpha, blocks, max_iter=max_iter, verbose=True)
X_final = extract_joint_distribution(Y_final, d=d)
print("-" * 80)
print(f"✓ CBRA completed! Generated {len(X_final)} joint samples.")

# Extract portfolio distribution from CBRA result
P_cbra = X_final.mean(axis=1)

# Plot portfolio distribution
print("\nGenerating portfolio distribution plot...")
from scipy.stats import gaussian_kde

# KDE for smooth PDF
kde_cbra = gaussian_kde(P_cbra)
x_grid = np.linspace(P_cbra.min(), P_cbra.max(), 1000)
pdf_cbra = kde_cbra(x_grid)

# Empirical CDF
P_cbra_sorted = np.sort(P_cbra)
cdf_cbra_y = np.arange(1, len(P_cbra_sorted) + 1) / len(P_cbra_sorted)

# Create figure
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# PDF
ax_pdf = axes[0]
ax_pdf.fill_between(x_grid, pdf_cbra, alpha=0.4, color='steelblue')
ax_pdf.plot(x_grid, pdf_cbra, 'b-', lw=2.5)
ax_pdf.axvline(P_cbra.mean(), color='r', linestyle='--', lw=2, label=f'Mean={P_cbra.mean():.2f}')
ax_pdf.set_title(f"PDF: Equal-Weighted Portfolio\n{', '.join(TICKERS)}", 
                 fontsize=12, fontweight='bold')
ax_pdf.set_xlabel("Portfolio Value (Strike Units)", fontsize=11)
ax_pdf.set_ylabel("Density", fontsize=11)
ax_pdf.legend(fontsize=10)
ax_pdf.grid(True, alpha=0.3)

# CDF
ax_cdf = axes[1]
ax_cdf.plot(P_cbra_sorted, cdf_cbra_y, 'b-', lw=2.5)
ax_cdf.axvline(P_cbra.mean(), color='r', linestyle='--', lw=2, alpha=0.7, label=f'Mean={P_cbra.mean():.2f}')
ax_cdf.axhline(0.5, color='gray', linestyle=':', alpha=0.5)
ax_cdf.set_title(f"CDF: Equal-Weighted Portfolio\n{', '.join(TICKERS)}", 
                 fontsize=12, fontweight='bold')
ax_cdf.set_xlabel("Portfolio Value (Strike Units)", fontsize=11)
ax_cdf.set_ylabel("Cumulative Probability", fontsize=11)
ax_cdf.legend(fontsize=10)
ax_cdf.grid(True, alpha=0.3)

fig.tight_layout()
out_path = OUTPUT_DIR / "portfolio_6stocks_equal_weight.png"
fig.savefig(out_path, dpi=200, bbox_inches='tight')
plt.close(fig)

# Save portfolio distribution as pickle
portfolio_data = {
    'tickers': TICKERS,
    'weights': [1/6] * 6,
    'portfolio_samples': P_cbra,
    'portfolio_sorted': P_cbra_sorted,
    'portfolio_cdf': cdf_cbra_y,
    'pdf_grid': x_grid,
    'pdf_values': pdf_cbra,
}
pickle_out = OUTPUT_DIR / "portfolio_6stocks_distribution.pkl"
with open(pickle_out, 'wb') as f:
    pickle.dump(portfolio_data, f)

# Print statistics
print("\n" + "="*80)
print("PORTFOLIO DISTRIBUTION STATISTICS")
print("="*80)
print(f"\nStocks: {', '.join(TICKERS)}")
print(f"Weights: Equal (1/{d} each)")
print(f"\nPortfolio Statistics:")
print(f"  Mean: {P_cbra.mean():.4f}")
print(f"  Std:  {P_cbra.std():.4f}")
print(f"  Skew: {stats.skew(P_cbra):.4f}")
print(f"  Kurt: {stats.kurtosis(P_cbra):.4f}")
print(f"  Min:  {P_cbra.min():.4f}")
print(f"  Max:  {P_cbra.max():.4f}")
print(f"\nPlot saved to: {out_path}")
print(f"Distribution saved to: {pickle_out}")
print("="*80 + "\n")

