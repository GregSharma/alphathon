"""
Benchmark CBRA performance to identify optimization targets.
"""
import time
import numpy as np
from scipy import stats
import cProfile
import pstats
from io import StringIO

from cbrapipe import (
    discretize_instruments,
    discretize_constraints,
    build_initial_matrix,
    expand_coefficients,
    identify_admissible_blocks,
    cbra_optimize,
)


def setup_paper_example(n=10000, seed=42):
    """Set up the paper Section 4.1 example."""
    if seed is not None:
        np.random.seed(seed)
    
    d = 6
    K = 3
    
    F_inv_stocks = [lambda p: stats.norm.ppf(p, loc=0, scale=1) for _ in range(d)]
    F_inv_constraints = [
        lambda p: stats.norm.ppf(p, loc=0, scale=np.sqrt(10)),
        lambda p: stats.norm.ppf(p, loc=0, scale=np.sqrt(10)),
        lambda p: stats.norm.ppf(p, loc=0, scale=np.sqrt(24)),
    ]
    
    A = np.array([
        [1.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0],
        [0.0, 1.0, 1.0],
        [0.0, 1.0, 1.0],
    ])
    
    X = discretize_instruments(n, F_inv_stocks)
    S = discretize_constraints(n, F_inv_constraints)
    Y = build_initial_matrix(X, S)
    
    # Randomize
    for j in range(Y.shape[1]):
        np.random.shuffle(Y[:, j])
    
    tilde_alpha = expand_coefficients(A)
    blocks = identify_admissible_blocks(tilde_alpha)
    
    return Y, tilde_alpha, blocks


def benchmark_cbra(n_values=[1000, 5000, 10000], num_runs=3):
    """Benchmark CBRA for different problem sizes."""
    print("\n" + "="*70)
    print("CBRA PERFORMANCE BENCHMARK")
    print("="*70)
    
    results = []
    
    for n in n_values:
        times = []
        for run in range(num_runs):
            Y, tilde_alpha, blocks = setup_paper_example(n=n, seed=42+run)
            
            start = time.perf_counter()
            Y_final = cbra_optimize(Y, tilde_alpha, blocks, max_iter=20000, verbose=False)
            elapsed = time.perf_counter() - start
            
            times.append(elapsed)
        
        avg_time = np.mean(times)
        std_time = np.std(times)
        results.append((n, avg_time, std_time))
        
        print(f"\nn = {n:5d}:  {avg_time:.3f} ± {std_time:.3f} seconds")
    
    print("\n" + "="*70)
    return results


def profile_cbra(n=10000):
    """Profile CBRA to identify hotspots."""
    print("\n" + "="*70)
    print("PROFILING CBRA (n=10000)")
    print("="*70)
    
    Y, tilde_alpha, blocks = setup_paper_example(n=n, seed=42)
    
    profiler = cProfile.Profile()
    profiler.enable()
    
    Y_final = cbra_optimize(Y, tilde_alpha, blocks, max_iter=20000, verbose=False)
    
    profiler.disable()
    
    # Print stats
    s = StringIO()
    ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
    ps.print_stats(20)
    
    print(s.getvalue())
    print("="*70)


if __name__ == "__main__":
    # Run benchmarks
    benchmark_cbra(n_values=[1000, 5000, 10000], num_runs=3)
    
    # Profile
    profile_cbra(n=10000)
