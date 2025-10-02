"""
Detailed profiling to find every last optimization opportunity.
"""
import numpy as np
from scipy import stats
import time

from cbrapipe import (
    discretize_instruments,
    discretize_constraints,
    build_initial_matrix,
    expand_coefficients,
    identify_admissible_blocks,
    cbra_optimize,
)


def setup_problem(n=10000, seed=42):
    """Set up the paper example."""
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
    
    for j in range(Y.shape[1]):
        np.random.shuffle(Y[:, j])
    
    tilde_alpha = expand_coefficients(A)
    blocks = identify_admissible_blocks(tilde_alpha)
    
    return Y, tilde_alpha, blocks


def time_components(n=10000):
    """Time individual components."""
    print(f"\n{'='*70}")
    print(f"COMPONENT TIMING (n={n})")
    print('='*70)
    
    Y, tilde_alpha, blocks = setup_problem(n=n, seed=42)
    
    from cbrapipe.optimize import compute_L, compute_objective, block_rearrangement
    
    # Time compute_L
    start = time.perf_counter()
    for _ in range(1000):
        L = compute_L(Y, tilde_alpha)
    elapsed = time.perf_counter() - start
    print(f"compute_L (1000 calls):        {elapsed:.3f}s  ({elapsed/1000*1000:.3f} ms/call)")
    
    # Time compute_objective
    start = time.perf_counter()
    for _ in range(1000):
        V = compute_objective(L)
    elapsed = time.perf_counter() - start
    print(f"compute_objective (1000 calls): {elapsed:.3f}s  ({elapsed/1000*1000:.3f} ms/call)")
    
    # Time block_rearrangement for a single block
    block_mask = blocks[0]  # First singleton block
    start = time.perf_counter()
    for _ in range(100):
        Y_copy = Y.copy()
        block_rearrangement(Y_copy, block_mask, tilde_alpha)
    elapsed = time.perf_counter() - start
    print(f"block_rearrangement (100 calls): {elapsed:.3f}s  ({elapsed/100*1000:.3f} ms/call)")
    
    # Time full CBRA
    Y, tilde_alpha, blocks = setup_problem(n=n, seed=42)
    start = time.perf_counter()
    Y_final = cbra_optimize(Y, tilde_alpha, blocks, max_iter=1000, verbose=False)
    elapsed = time.perf_counter() - start
    print(f"Full CBRA:                      {elapsed:.3f}s")
    print('='*70)


def analyze_block_rearrangement_internals():
    """Analyze what's slow inside block_rearrangement."""
    print(f"\n{'='*70}")
    print("BLOCK REARRANGEMENT INTERNALS")
    print('='*70)
    
    n = 10000
    Y, tilde_alpha, blocks = setup_problem(n=n, seed=42)
    
    # Pick a multi-column block for realistic timing
    multi_block = [b for b in blocks if np.sum(b) > 1][0]
    
    in_block = np.where(multi_block)[0]
    out_block = np.where(~multi_block)[0]
    alpha_block = tilde_alpha[:, in_block[0]]
    K = tilde_alpha.shape[0]
    
    # Time individual operations
    iterations = 1000
    
    # 1. Computing S
    start = time.perf_counter()
    for _ in range(iterations):
        S = np.sum(Y[:, in_block], axis=1)
    elapsed = time.perf_counter() - start
    print(f"1. Computing S:           {elapsed:.3f}s  ({elapsed/iterations*1000:.3f} ms/call)")
    
    # 2. Computing Z_agg (numpy version)
    start = time.perf_counter()
    for _ in range(iterations):
        Z_agg = np.zeros(n)
        for k in range(K):
            Z_agg += alpha_block[k] * (Y[:, out_block] @ tilde_alpha[k, out_block])
    elapsed = time.perf_counter() - start
    print(f"2. Computing Z_agg:       {elapsed:.3f}s  ({elapsed/iterations*1000:.3f} ms/call)")
    
    # Compute once for next steps
    S = np.sum(Y[:, in_block], axis=1)
    Z_agg = np.zeros(n)
    for k in range(K):
        Z_agg += alpha_block[k] * (Y[:, out_block] @ tilde_alpha[k, out_block])
    
    # 3. Argsort
    start = time.perf_counter()
    for _ in range(iterations):
        sort_Z = np.argsort(Z_agg)
        sort_S = np.argsort(S)[::-1]
    elapsed = time.perf_counter() - start
    print(f"3. Argsort (both):        {elapsed:.3f}s  ({elapsed/iterations*1000:.3f} ms/call)")
    
    # 4. Create permutation
    start = time.perf_counter()
    for _ in range(iterations):
        permutation = np.empty(n, dtype=np.int64)
        permutation[sort_Z] = sort_S
    elapsed = time.perf_counter() - start
    print(f"4. Create permutation:    {elapsed:.3f}s  ({elapsed/iterations*1000:.3f} ms/call)")
    
    # 5. Apply permutation
    sort_Z = np.argsort(Z_agg)
    sort_S = np.argsort(S)[::-1]
    permutation = np.empty(n, dtype=np.int64)
    permutation[sort_Z] = sort_S
    
    start = time.perf_counter()
    for _ in range(iterations):
        Y_copy = Y.copy()
        Y_block_copy = Y_copy[:, in_block].copy()
        Y_copy[:, in_block] = Y_block_copy[permutation, :]
    elapsed = time.perf_counter() - start
    print(f"5. Apply permutation:     {elapsed:.3f}s  ({elapsed/iterations*1000:.3f} ms/call)")
    
    print('='*70)


if __name__ == "__main__":
    time_components(n=10000)
    analyze_block_rearrangement_internals()
