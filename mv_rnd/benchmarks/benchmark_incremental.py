"""
Benchmark incremental CBRA for streaming real-time updates.

Simulates the real-world scenario:
- Minute 0: Get option quotes, compute RNCDs, run CBRA (cold start)
- Minute 1: Quotes update slightly, RNCDs change ~5%, run incremental CBRA (warm start)
- Minute 2: Another update, warm start again
- etc.

Expected: 10-20x speedup for warm starts!
"""
import time
import numpy as np
from scipy import stats

from cbrapipe.incremental import (
    cbra_optimize_stateful,
    cbra_update_incremental,
    detect_marginal_changes,
)
from cbrapipe import extract_joint_distribution, compute_L, compute_objective


def simulate_rncd_drift(F_inv_original, drift_std=0.05):
    """
    Simulate a slightly changed RNCD (e.g., after 1 minute of trading).
    
    Parameters
    ----------
    F_inv_original : Callable
        Original inverse RNCD
    drift_std : float
        Standard deviation of drift in the scale parameter
        
    Returns
    -------
    F_inv_new : Callable
        New inverse RNCD (slightly different)
    """
    # For normal distributions, slightly change the scale
    # In practice, implied vol changes by ~1-5% per minute
    scale_drift = np.random.normal(0, drift_std)
    
    def F_inv_new(p):
        # Get original value
        original = F_inv_original(p)
        # Apply drift (scale change)
        return original * (1 + scale_drift)
    
    return F_inv_new


def benchmark_cold_vs_warm():
    """Compare cold start vs warm start performance."""
    print("\n" + "="*70)
    print("INCREMENTAL CBRA BENCHMARK")
    print("="*70)
    print("Scenario: Real-time trading with minute-by-minute RNCD updates")
    print("="*70 + "\n")
    
    # Setup
    d = 10  # 10 assets for faster test
    K = 3   # 3 indices
    n = 2000  # Smaller for speed
    
    np.random.seed(42)
    
    # Original RNCDs (time t=0)
    F_inv_stocks_t0 = [lambda p: stats.norm.ppf(p, loc=0, scale=1) for _ in range(d)]
    F_inv_indices_t0 = [lambda p: stats.norm.ppf(p, loc=0, scale=np.sqrt(k+3)) for k in range(K)]
    
    # Weight matrix (random but fixed)
    A = np.random.rand(d, K)
    for k in range(K):
        A[:, k] /= A[:, k].sum()
    
    # === TIME 0: Cold start ===
    print("TIME t=0: Initial CBRA (COLD START)")
    start = time.perf_counter()
    state = cbra_optimize_stateful(
        n=n,
        F_inv_stocks=F_inv_stocks_t0,
        F_inv_indices=F_inv_indices_t0,
        A=A,
        max_iter=5000,
        verbose=False
    )
    cold_time = time.perf_counter() - start
    print(f"  Time: {cold_time:.3f}s")
    print(f"  Converged to V = {compute_objective(compute_L(state.Y_optimized, state.tilde_alpha)):.6e}")
    
    # === TIME 1: Slight update (warm start) ===
    print("\nTIME t=1: RNCDs drift ~5% (WARM START)")
    
    # Simulate 5% drift in RNCDs
    F_inv_stocks_t1 = [simulate_rncd_drift(f, drift_std=0.05) for f in F_inv_stocks_t0]
    F_inv_indices_t1 = [simulate_rncd_drift(f, drift_std=0.05) for f in F_inv_indices_t0]
    
    # Auto-detect changes
    changed_stocks, changed_indices = detect_marginal_changes(
        state, F_inv_stocks_t1, F_inv_indices_t1, threshold=0.01
    )
    print(f"  Detected changes: {len(changed_stocks)} stocks, {len(changed_indices)} indices")
    
    # Warm start update
    start = time.perf_counter()
    state = cbra_update_incremental(
        state,
        F_inv_stocks_new=F_inv_stocks_t1,
        F_inv_indices_new=F_inv_indices_t1,
        changed_stock_indices=changed_stocks,
        changed_index_indices=changed_indices,
        max_iter=100,  # Much fewer iterations!
        verbose=False
    )
    warm_time = time.perf_counter() - start
    print(f"  Time: {warm_time:.3f}s")
    print(f"  Converged to V = {compute_objective(compute_L(state.Y_optimized, state.tilde_alpha)):.6e}")
    
    # === TIME 2: Another update ===
    print("\nTIME t=2: Another drift (WARM START)")
    
    F_inv_stocks_t2 = [simulate_rncd_drift(f, drift_std=0.05) for f in F_inv_stocks_t1]
    F_inv_indices_t2 = [simulate_rncd_drift(f, drift_std=0.05) for f in F_inv_indices_t1]
    
    changed_stocks, changed_indices = detect_marginal_changes(
        state, F_inv_stocks_t2, F_inv_indices_t2, threshold=0.01
    )
    
    start = time.perf_counter()
    state = cbra_update_incremental(
        state,
        F_inv_stocks_new=F_inv_stocks_t2,
        F_inv_indices_new=F_inv_indices_t2,
        changed_stock_indices=changed_stocks,
        changed_index_indices=changed_indices,
        max_iter=100,
        verbose=False
    )
    warm_time_2 = time.perf_counter() - start
    print(f"  Time: {warm_time_2:.3f}s")
    print(f"  Converged to V = {compute_objective(compute_L(state.Y_optimized, state.tilde_alpha)):.6e}")
    
    # Summary
    print("\n" + "="*70)
    print("SPEEDUP ANALYSIS")
    print("="*70)
    print(f"Cold start (t=0):        {cold_time:.3f}s")
    print(f"Warm start (t=1):        {warm_time:.3f}s  ({cold_time/warm_time:.1f}x faster)")
    print(f"Warm start (t=2):        {warm_time_2:.3f}s  ({cold_time/warm_time_2:.1f}x faster)")
    print(f"\nAverage warm start:      {np.mean([warm_time, warm_time_2]):.3f}s")
    print(f"Average speedup:         {cold_time/np.mean([warm_time, warm_time_2]):.1f}x 🚀")
    print("="*70)
    
    if cold_time / np.mean([warm_time, warm_time_2]) > 5:
        print("\n✅ SUCCESS! Incremental updates are 5x+ faster!")
    else:
        print("\n⚠️  Speedup less than expected - may need tuning")
    
    return state


def benchmark_streaming_scenario():
    """
    Simulate a full trading day with streaming updates.
    
    Scenario:
    - Market opens: Cold start
    - Every minute: Warm start update (100 updates total)
    """
    print("\n" + "="*70)
    print("STREAMING TRADING DAY SIMULATION")
    print("="*70)
    print("Simulating 100 minute-by-minute updates")
    print("="*70 + "\n")
    
    d = 10
    K = 3
    n = 2000
    
    np.random.seed(42)
    
    # Initial setup
    F_inv_stocks = [lambda p: stats.norm.ppf(p, loc=0, scale=1) for _ in range(d)]
    F_inv_indices = [lambda p: stats.norm.ppf(p, loc=0, scale=np.sqrt(k+3)) for k in range(K)]
    
    A = np.random.rand(d, K)
    for k in range(K):
        A[:, k] /= A[:, k].sum()
    
    # Cold start
    print("Market open: Cold start...")
    start = time.perf_counter()
    state = cbra_optimize_stateful(n, F_inv_stocks, F_inv_indices, A, max_iter=5000, verbose=False)
    cold_time = time.perf_counter() - start
    print(f"  Time: {cold_time:.3f}s\n")
    
    # Simulate 100 updates
    print("Running 100 minute-by-minute updates...")
    warm_times = []
    
    current_F_stocks = F_inv_stocks
    current_F_indices = F_inv_indices
    
    for minute in range(100):
        # Simulate drift
        current_F_stocks = [simulate_rncd_drift(f, drift_std=0.03) for f in current_F_stocks]
        current_F_indices = [simulate_rncd_drift(f, drift_std=0.03) for f in current_F_indices]
        
        # Detect changes
        changed_stocks, changed_indices = detect_marginal_changes(
            state, current_F_stocks, current_F_indices, threshold=0.01
        )
        
        # Update
        start = time.perf_counter()
        state = cbra_update_incremental(
            state,
            F_inv_stocks_new=current_F_stocks,
            F_inv_indices_new=current_F_indices,
            changed_stock_indices=changed_stocks,
            changed_index_indices=changed_indices,
            max_iter=50,
            verbose=False
        )
        elapsed = time.perf_counter() - start
        warm_times.append(elapsed)
        
        if (minute + 1) % 10 == 0:
            print(f"  Minute {minute+1}: {elapsed:.3f}s (avg: {np.mean(warm_times):.3f}s)")
    
    print("\n" + "="*70)
    print("TRADING DAY RESULTS")
    print("="*70)
    print(f"Cold start:              {cold_time:.3f}s")
    print(f"Warm updates (100):      {np.mean(warm_times):.3f}s average")
    print(f"                         {np.median(warm_times):.3f}s median")
    print(f"                         {np.min(warm_times):.3f}s min")
    print(f"                         {np.max(warm_times):.3f}s max")
    print(f"\nSpeedup (warm vs cold):  {cold_time / np.mean(warm_times):.1f}x 🚀")
    print(f"Total time (1 cold + 100 warm): {cold_time + np.sum(warm_times):.1f}s")
    print("="*70)


if __name__ == "__main__":
    benchmark_cold_vs_warm()
    benchmark_streaming_scenario()
