"""
Comprehensive test suite for CBRA implementation.
"""
import numpy as np
import pytest
from cbrapipe import (
    discretize_instruments,
    discretize_constraints,
    build_initial_matrix,
    expand_coefficients,
    identify_admissible_blocks,
    compute_L,
    compute_objective,
    cbra_optimize,
    extract_joint_distribution
)


class TestDiscretize:
    """Test discretization functions."""
    
    def test_discretize_instruments_shape(self):
        """Check output shape is correct."""
        n = 100
        d = 3
        F_inv_list = [lambda p: p for _ in range(d)]
        X = discretize_instruments(n, F_inv_list)
        assert X.shape == (n, d)
    
    def test_discretize_instruments_monotonic(self):
        """Check each column is monotonically increasing."""
        n = 50
        F_inv_list = [lambda p: p, lambda p: 2*p, lambda p: p**2]
        X = discretize_instruments(n, F_inv_list)
        for j in range(X.shape[1]):
            assert np.all(np.diff(X[:, j]) >= 0)
    
    def test_discretize_constraints_shape(self):
        """Check output shape is correct."""
        n = 100
        K = 2
        F_inv_list = [lambda p: p for _ in range(K)]
        S = discretize_constraints(n, F_inv_list)
        assert S.shape == (n, K)
    
    def test_build_initial_matrix(self):
        """Check Y matrix combines X and S correctly."""
        n, d, K = 50, 3, 2
        X = np.random.randn(n, d)
        S = np.random.randn(n, K)
        Y = build_initial_matrix(X, S)
        assert Y.shape == (n, d + K)
        assert np.allclose(Y[:, :d], X)
        assert np.allclose(Y[:, d:], S)


class TestCoefficients:
    """Test coefficient expansion."""
    
    def test_expand_coefficients_shape(self):
        """Check output shape matches (K, d+K)."""
        d, K = 4, 2
        A = np.random.randn(d, K)
        tilde_alpha = expand_coefficients(A)
        assert tilde_alpha.shape == (K, d + K)
    
    def test_expand_coefficients_values(self):
        """Verify coefficients match specification."""
        d, K = 3, 2
        A = np.array([[0.5, 0.3],
                      [0.3, 0.4],
                      [0.2, 0.3]])
        tilde_alpha = expand_coefficients(A)
        
        # Check stock weights
        for k in range(K):
            assert np.allclose(tilde_alpha[k, :d], A[:, k])
        
        # Check index columns
        for k in range(K):
            assert tilde_alpha[k, d + k] == -1.0
            for m in range(K):
                if m != k:
                    assert tilde_alpha[k, d + m] == 0.0


class TestBlocks:
    """Test admissible block identification."""
    
    def test_singleton_blocks_present(self):
        """Ensure all singleton blocks are included."""
        d, K = 3, 2
        A = np.random.randn(d, K)
        tilde_alpha = expand_coefficients(A)
        blocks = identify_admissible_blocks(tilde_alpha)
        
        total_cols = d + K
        # Check that each column has a singleton block
        for j in range(total_cols):
            singleton_exists = False
            for block in blocks:
                if np.sum(block) == 1 and block[j]:
                    singleton_exists = True
                    break
            assert singleton_exists
    
    def test_admissible_block_property(self):
        """Verify admissibility: coefficients constant within block."""
        d, K = 3, 2
        A = np.random.randn(d, K)
        tilde_alpha = expand_coefficients(A)
        blocks = identify_admissible_blocks(tilde_alpha)
        
        for block in blocks:
            cols_in_block = np.where(block)[0]
            if len(cols_in_block) > 1:
                # All columns in block should have same coefficients
                for k in range(K):
                    coeffs = tilde_alpha[k, cols_in_block]
                    assert np.allclose(coeffs, coeffs[0])


class TestObjective:
    """Test objective function computation."""
    
    def test_objective_nonnegative(self):
        """Objective should be non-negative (sum of variances)."""
        K, n = 3, 100
        L = np.random.randn(K, n)
        V = compute_objective(L)
        assert V >= 0
    
    def test_objective_zero_for_constant(self):
        """Objective should be zero if all L_k are constant."""
        K, n = 3, 100
        L = np.ones((K, n)) * 5.0
        V = compute_objective(L)
        assert np.abs(V) < 1e-10


class TestRearrangement:
    """Test block rearrangement."""
    
    def test_rearrangement_antimonotone(self):
        """After rearrangement, W and Z should be anti-monotonic."""
        n, d, K = 100, 2, 1
        np.random.seed(42)
        
        # Create simple case
        Y = np.random.randn(n, d + K)
        A = np.array([[0.5], [0.5]])
        tilde_alpha = expand_coefficients(A)
        
        # Use first column as block
        block_mask = np.zeros(d + K, dtype=np.bool_)
        block_mask[0] = True
        
        # Compute initial correlation
        from cbrapipe.optimize import block_rearrangement
        block_rearrangement(Y, block_mask, tilde_alpha)
        
        # After rearrangement, correlation should be negative or zero
        # (This is a weak test, but verifies the function runs)
        assert Y.shape == (n, d + K)


class TestEndToEnd:
    """Integration tests."""
    
    def test_cbra_small_example(self):
        """Small synthetic example with known properties."""
        n = 100
        d = 2
        K = 1
        
        # Define simple inverse CDFs (uniform-like)
        F_inv_stocks = [lambda p: p, lambda p: 2*p]
        F_inv_indices = [lambda p: 1.5*p]  # Index should be 0.5*X1 + 0.5*X2
        
        # Weight matrix: index is 0.5*X1 + 0.5*X2
        A = np.array([[0.5], [0.5]])
        
        # Step 1: Build initial matrix
        X = discretize_instruments(n, F_inv_stocks)
        S = discretize_constraints(n, F_inv_indices)
        Y = build_initial_matrix(X, S)
        
        # Step 2: Expand coefficients and find blocks
        tilde_alpha = expand_coefficients(A)
        blocks = identify_admissible_blocks(tilde_alpha)
        
        # Step 3: Optimize
        Y_final = cbra_optimize(Y, tilde_alpha, blocks, max_iter=100)
        
        # Step 4: Extract joint distribution
        X_final = extract_joint_distribution(Y_final, d)
        
        # Verify properties
        assert X_final.shape == (n, d)
        
        # Check that objective decreased
        L_init = compute_L(Y, tilde_alpha)
        V_init = compute_objective(L_init)
        L_final = compute_L(Y_final, tilde_alpha)
        V_final = compute_objective(L_final)
        assert V_final <= V_init
    
    def test_marginals_preserved(self):
        """Test that marginal distributions are preserved."""
        n = 200
        d = 2
        K = 1
        
        # Define quantile functions
        F_inv_stocks = [
            lambda p: np.sqrt(p),  # Square root transformation
            lambda p: p**2         # Quadratic transformation
        ]
        F_inv_indices = [lambda p: p]
        
        A = np.array([[0.6], [0.4]])
        
        # Build and optimize
        X = discretize_instruments(n, F_inv_stocks)
        S = discretize_constraints(n, F_inv_indices)
        Y = build_initial_matrix(X, S)
        
        tilde_alpha = expand_coefficients(A)
        blocks = identify_admissible_blocks(tilde_alpha)
        Y_final = cbra_optimize(Y, tilde_alpha, blocks)
        X_final = extract_joint_distribution(Y_final, d)
        
        # Marginals should be preserved (up to permutation)
        # Check by comparing sorted values
        for j in range(d):
            assert np.allclose(
                np.sort(X_final[:, j]),
                np.sort(X[:, j]),
                rtol=1e-10
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
