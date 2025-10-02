"""
Performance regression tests for CBRA.
"""
import time
import numpy as np
import pytest
from scipy import stats

from cbrapipe import (
    discretize_instruments,
    discretize_constraints,
    build_initial_matrix,
    expand_coefficients,
    identify_admissible_blocks,
    cbra_optimize,
    extract_joint_distribution,
)


@pytest.fixture
def paper_setup_small():
    """Small problem for fast tests."""
    n = 1000
    d = 6
    K = 3
    
    np.random.seed(42)
    
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
    
    return Y, tilde_alpha, blocks, d


def test_cbra_performance_small(paper_setup_small):
    """Ensure small problems complete quickly."""
    Y, tilde_alpha, blocks, d = paper_setup_small
    
    start = time.perf_counter()
    Y_final = cbra_optimize(Y, tilde_alpha, blocks, max_iter=10000, verbose=False)
    elapsed = time.perf_counter() - start
    
    # Should complete in under 0.5 seconds for n=1000
    assert elapsed < 0.5, f"Performance regression: took {elapsed:.3f}s (expected < 0.5s)"
    
    # Verify correctness
    X_final = extract_joint_distribution(Y_final, d)
    assert X_final.shape == (1000, 6)


def test_cbra_convergence_speed(paper_setup_small):
    """Ensure algorithm converges efficiently."""
    Y, tilde_alpha, blocks, d = paper_setup_small
    
    # Track iterations
    from cbrapipe.optimize import compute_L, compute_objective
    
    Y_current = Y.copy()
    V_initial = compute_objective(compute_L(Y_current, tilde_alpha))
    
    Y_final = cbra_optimize(Y, tilde_alpha, blocks, max_iter=10000, verbose=False)
    V_final = compute_objective(compute_L(Y_final, tilde_alpha))
    
    # Should reduce objective significantly
    improvement_ratio = V_initial / max(V_final, 1e-10)
    assert improvement_ratio > 100, f"Insufficient convergence: {improvement_ratio:.1f}x"


def test_marginals_preserved_after_optimization(paper_setup_small):
    """Verify marginals are preserved (critical correctness check)."""
    Y, tilde_alpha, blocks, d = paper_setup_small
    
    # Store original marginals
    X_original = Y[:, :d].copy()
    original_marginals = [np.sort(X_original[:, j]) for j in range(d)]
    
    # Run CBRA
    Y_final = cbra_optimize(Y, tilde_alpha, blocks, max_iter=10000, verbose=False)
    X_final = extract_joint_distribution(Y_final, d)
    
    # Check marginals preserved
    for j in range(d):
        final_marginal = np.sort(X_final[:, j])
        np.testing.assert_allclose(
            final_marginal, original_marginals[j], rtol=1e-10,
            err_msg=f"Marginal {j} not preserved!"
        )


def test_block_rearrangement_correctness():
    """Test that block_rearrangement maintains invariants."""
    from cbrapipe.optimize import block_rearrangement
    
    n = 500
    d = 4
    K = 2
    
    # Create test data
    np.random.seed(123)
    Y = np.random.randn(n, d + K)
    
    A = np.array([
        [1.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [0.0, 1.0],
    ])
    
    from cbrapipe.blocks import expand_coefficients
    tilde_alpha = expand_coefficients(A)
    
    # Create a block
    block_mask = np.zeros(d + K, dtype=bool)
    block_mask[0] = True
    block_mask[1] = True
    
    # Store original marginals of the block
    original_block_marginals = [np.sort(Y[:, j].copy()) for j in [0, 1]]
    
    # Apply rearrangement
    block_rearrangement(Y, block_mask, tilde_alpha)
    
    # Check marginals preserved within block
    for idx, j in enumerate([0, 1]):
        new_marginal = np.sort(Y[:, j])
        np.testing.assert_allclose(
            new_marginal, original_block_marginals[idx], rtol=1e-10,
            err_msg=f"Block column {j} marginal not preserved!"
        )


@pytest.mark.parametrize("n", [500, 1000, 2000])
def test_scaling_behavior(n):
    """Test that performance scales reasonably with problem size."""
    d = 6
    K = 2
    
    np.random.seed(42)
    
    F_inv_stocks = [lambda p: stats.norm.ppf(p, loc=0, scale=1) for _ in range(d)]
    F_inv_constraints = [
        lambda p: stats.norm.ppf(p, loc=0, scale=2),
        lambda p: stats.norm.ppf(p, loc=0, scale=2),
    ]
    
    A = np.array([
        [1.0, 0.0],
        [1.0, 0.0],
        [1.0, 1.0],
        [0.0, 1.0],
        [0.0, 1.0],
        [0.0, 1.0],
    ])
    
    X = discretize_instruments(n, F_inv_stocks)
    S = discretize_constraints(n, F_inv_constraints)
    Y = build_initial_matrix(X, S)
    
    for j in range(Y.shape[1]):
        np.random.shuffle(Y[:, j])
    
    tilde_alpha = expand_coefficients(A)
    blocks = identify_admissible_blocks(tilde_alpha)
    
    start = time.perf_counter()
    Y_final = cbra_optimize(Y, tilde_alpha, blocks, max_iter=5000, verbose=False)
    elapsed = time.perf_counter() - start
    
    # Rough scaling: should be roughly O(n^1.5) or better
    # For n=2000, allow up to 2 seconds
    max_time = (n / 1000) * 1.0
    assert elapsed < max_time, f"Scaling issue: n={n} took {elapsed:.3f}s (expected < {max_time:.3f}s)"


@pytest.mark.parametrize("d,K,n,target_ms", [
    (20, 5, 5000, 800),     # 20 assets, 0.8s
    (40, 10, 5000, 3000),   # 40 assets, 3.0s (realistic for now)
])
def test_realtime_performance(d, K, n, target_ms):
    """Test performance for real-time trading scenarios (d=20-40 assets)."""
    np.random.seed(42)
    
    # Create random problem
    F_inv_stocks = [lambda p: stats.norm.ppf(p, loc=0, scale=1) for _ in range(d)]
    
    # Random indices
    A_list = []
    F_inv_constraints = []
    for k in range(K):
        num_assets = min(d, np.random.randint(5, 16))
        asset_indices = np.random.choice(d, num_assets, replace=False)
        weights = np.random.rand(num_assets)
        weights /= weights.sum()
        
        var_sum = num_assets * (1 + 0.3 * (num_assets - 1))
        F_inv_constraints.append(
            lambda p, v=var_sum: stats.norm.ppf(p, loc=0, scale=np.sqrt(v))
        )
        
        A_col = np.zeros(d)
        A_col[asset_indices] = weights
        A_list.append(A_col)
    
    A = np.column_stack(A_list)
    
    X = discretize_instruments(n, F_inv_stocks)
    S = discretize_constraints(n, F_inv_constraints)
    Y = build_initial_matrix(X, S)
    
    for j in range(Y.shape[1]):
        np.random.shuffle(Y[:, j])
    
    tilde_alpha = expand_coefficients(A)
    blocks = identify_admissible_blocks(tilde_alpha)
    
    start = time.perf_counter()
    Y_final = cbra_optimize(Y, tilde_alpha, blocks, max_iter=3000, verbose=False)
    elapsed = time.perf_counter() - start
    
    assert elapsed < target_ms / 1000, \
        f"Real-time perf: d={d}, K={K}, n={n} took {elapsed:.3f}s (target < {target_ms/1000:.3f}s)"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
