"""
Benchmark CBRA for REAL-TIME trading scenarios.

Target: d=40 assets (20 stocks + 20 ETFs), K=10 indices
Goal: <500ms per run for intraday updates
"""
import time
import numpy as np
from scipy import stats

from cbrapipe import (
    discretize_instruments,
    discretize_constraints,
    build_initial_matrix,
    expand_coefficients,
    identify_admissible_blocks,
    cbra_optimize,
)


def setup_realtime_problem(d=40, K=10, n=10000, seed=42):
    """
    Set up a realistic real-time trading problem.
    
    d=40: 20 individual stocks + 20 ETFs
    K=10: Various sector/market indices
    n=10000: Good balance of accuracy and speed
    """
    if seed is not None:
        np.random.seed(seed)
    
    # All assets have standard normal marginals (simplified)
    F_inv_stocks = [lambda p: stats.norm.ppf(p, loc=0, scale=1) for _ in range(d)]
    
    # Create K random index constraints
    # Each index is a weighted sum of a subset of assets
    F_inv_constraints = []
    A_list = []
    
    for k in range(K):
        # Each index contains 5-15 random assets
        num_assets = np.random.randint(5, 16)
        asset_indices = np.random.choice(d, num_assets, replace=False)
        
        # Random weights (normalized)
        weights = np.random.rand(num_assets)
        weights /= weights.sum()
        
        # Constraint distribution (sum of normals with correlation)
        # Variance depends on weights and assumed correlation structure
        var_sum = num_assets * (1 + 0.3 * (num_assets - 1))  # Assume avg corr = 0.3
        
        F_inv_constraints.append(
            lambda p, v=var_sum: stats.norm.ppf(p, loc=0, scale=np.sqrt(v))
        )
        
        # Build weight matrix column
        A_col = np.zeros(d)
        A_col[asset_indices] = weights
        A_list.append(A_col)
    
    A = np.column_stack(A_list)  # Shape (d, K)
    
    # Discretize
    X = discretize_instruments(n, F_inv_stocks)
    S = discretize_constraints(n, F_inv_constraints)
    Y = build_initial_matrix(X, S)
    
    # Randomize
    for j in range(Y.shape[1]):
        np.random.shuffle(Y[:, j])
    
    tilde_alpha = expand_coefficients(A)
    blocks = identify_admissible_blocks(tilde_alpha)
    
    return Y, tilde_alpha, blocks


def benchmark_realtime(runs=5):
    """Benchmark real-time trading scenario."""
    print("\n" + "="*70)
    print("REAL-TIME TRADING BENCHMARK")
    print("="*70)
    print(f"Target: d=40 assets, K=10 indices, n=10,000")
    print(f"Goal: <500ms per run")
    print("="*70 + "\n")
    
    times = []
    
    for run in range(runs):
        print(f"Run {run+1}/{runs}...")
        Y, tilde_alpha, blocks = setup_realtime_problem(d=40, K=10, n=10000, seed=42+run)
        
        start = time.perf_counter()
        Y_final = cbra_optimize(Y, tilde_alpha, blocks, max_iter=5000, verbose=False)
        elapsed = time.perf_counter() - start
        
        times.append(elapsed)
        print(f"  Time: {elapsed:.3f}s")
    
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    print(f"Mean:   {np.mean(times):.3f}s")
    print(f"Median: {np.median(times):.3f}s")
    print(f"Std:    {np.std(times):.3f}s")
    print(f"Min:    {np.min(times):.3f}s")
    print(f"Max:    {np.max(times):.3f}s")
    
    if np.median(times) < 0.5:
        print(f"\n✅ SUCCESS! Median {np.median(times):.3f}s < 0.5s target")
    else:
        print(f"\n⚠️  Median {np.median(times):.3f}s > 0.5s target - needs more optimization")
    
    print("="*70)
    
    return times


def benchmark_scaling():
    """Test how performance scales with d and n."""
    print("\n" + "="*70)
    print("SCALING ANALYSIS")
    print("="*70)
    
    configs = [
        (20, 5, 5000),   # Small
        (40, 10, 5000),  # Medium-fast
        (40, 10, 10000), # Target
        (60, 15, 10000), # Large
    ]
    
    for d, K, n in configs:
        Y, tilde_alpha, blocks = setup_realtime_problem(d=d, K=K, n=n, seed=42)
        
        start = time.perf_counter()
        Y_final = cbra_optimize(Y, tilde_alpha, blocks, max_iter=2000, verbose=False)
        elapsed = time.perf_counter() - start
        
        print(f"d={d:2d}, K={K:2d}, n={n:5d}:  {elapsed:.3f}s")
    
    print("="*70)


if __name__ == "__main__":
    benchmark_realtime(runs=5)
    benchmark_scaling()
