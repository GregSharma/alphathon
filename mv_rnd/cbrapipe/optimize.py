"""
Step 3 & 4: CBRA optimization loop with block rearrangement.
"""
import numpy as np
from numpy.typing import NDArray
from typing import List

# Try to import Numba for JIT compilation
try:
    from numba import njit, prange
    _USE_NUMBA = True
except ImportError:
    _USE_NUMBA = False
    # Dummy decorators if numba not available
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        if len(args) == 1 and callable(args[0]):
            return args[0]
        return decorator
    prange = range  # Fallback to regular range


def compute_L(
    Y: NDArray[np.float64],
    tilde_alpha: NDArray[np.float64]
) -> NDArray[np.float64]:
    """
    Compute constraint vectors L_k for all k.
    
    L[k, i] = sum_j tilde_alpha[k, j] * Y[i, j]
    
    Parameters
    ----------
    Y : NDArray[np.float64]
        Current state matrix of shape (n, d+K)
    tilde_alpha : NDArray[np.float64]
        Coefficient matrix of shape (K, d+K)
        
    Returns
    -------
    NDArray[np.float64]
        Constraint matrix L of shape (K, n)
    """
    # Use numpy's optimized matmul (already very fast)
    # Cython version doesn't help much here
    return tilde_alpha @ Y.T  # Shape (K, n)


@njit(cache=True, fastmath=True, nogil=True)
def _compute_weighted_sums_fast(Y, in_block, out_block, tilde_alpha, alpha_block):
    """
    Numba-JIT optimized computation of S and Z_agg with GIL released.
    
    This is the compute-intensive part - tight loops over large arrays.
    Everything else (argsort, indexing) stays in NumPy.
    
    Note: nogil=True allows Python to use threads elsewhere while this runs.
    """
    n = Y.shape[0]
    K = tilde_alpha.shape[0]
    num_in = len(in_block)
    num_out = len(out_block)
    
    # Compute S = sum of block columns
    S = np.zeros(n, dtype=np.float64)
    for i in range(n):
        s_val = 0.0
        for j_idx in range(num_in):
            s_val += Y[i, in_block[j_idx]]
        S[i] = s_val
    
    # Compute Z_agg = weighted sum outside block
    Z_agg = np.zeros(n, dtype=np.float64)
    if num_out > 0:
        for k in range(K):
            alpha_k = alpha_block[k]
            for i in range(n):
                z_val = 0.0
                for j_idx in range(num_out):
                    j = out_block[j_idx]
                    z_val += tilde_alpha[k, j] * Y[i, j]
                Z_agg[i] += alpha_k * z_val
    
    return S, Z_agg


def compute_objective(L: NDArray[np.float64]) -> float:
    """
    Compute objective function V as sum of variances of L_k.
    
    Parameters
    ----------
    L : NDArray[np.float64]
        Constraint matrix of shape (K, n)
        
    Returns
    -------
    float
        Sum of variances across all K constraints
    """
    return np.sum(np.var(L, axis=1, ddof=0))


def block_rearrangement(
    Y: NDArray[np.float64],
    block_mask: NDArray[np.bool_],
    tilde_alpha: NDArray[np.float64]
) -> None:
    """
    Apply block rearrangement to achieve anti-monotonicity (in-place).
    
    For the given block, rearrange rows to make the weighted sum within
    the block anti-monotonic with the weighted sum outside the block.
    
    Parameters
    ----------
    Y : NDArray[np.float64]
        Current state matrix of shape (n, d+K), modified in-place
    block_mask : NDArray[np.bool_]
        Boolean mask of shape (d+K,) indicating block membership
    tilde_alpha : NDArray[np.float64]
        Coefficient matrix of shape (K, d+K)
    """
    # Partition columns (NumPy's where is very fast)
    n = Y.shape[0]
    in_block = np.where(block_mask)[0]
    out_block = np.where(~block_mask)[0]
    
    if len(in_block) == 0:
        return
    
    alpha_block = tilde_alpha[:, in_block[0]]
    
    # Compute S and Z_agg - use Numba for tight loops if available
    if _USE_NUMBA:
        S, Z_agg = _compute_weighted_sums_fast(Y, in_block, out_block, tilde_alpha, alpha_block)
    else:
        # Pure NumPy fallback
        S = np.sum(Y[:, in_block], axis=1)
        K = tilde_alpha.shape[0]
        Z_agg = np.zeros(n, dtype=np.float64)
        if len(out_block) > 0:
            for k in range(K):
                Z_agg += alpha_block[k] * (Y[:, out_block] @ tilde_alpha[k, out_block])
    
    # Use NumPy's highly optimized argsort (faster than Numba's)
    sort_indices_Z = np.argsort(Z_agg)
    sort_indices_S = np.argsort(S)[::-1]
    
    # Create permutation (fast)
    permutation = np.empty(n, dtype=np.int64)
    permutation[sort_indices_Z] = sort_indices_S
    
    # Apply permutation using NumPy's optimized fancy indexing
    Y_block_copy = Y[:, in_block].copy()
    Y[:, in_block] = Y_block_copy[permutation, :]


def cbra_optimize(
    Y: NDArray[np.float64],
    tilde_alpha: NDArray[np.float64],
    blocks: List[NDArray[np.bool_]],
    max_iter: int = 1000,
    tol: float = 1e-10,
    rel_tol: float = 1e-6,
    verbose: bool = False
) -> NDArray[np.float64]:
    """
    Execute CBRA optimization loop with adaptive convergence.
    
    Parameters
    ----------
    Y : NDArray[np.float64]
        Initial matrix of shape (n, d+K)
    tilde_alpha : NDArray[np.float64]
        Coefficient matrix of shape (K, d+K)
    blocks : List[NDArray[np.bool_]]
        List of admissible block masks
    max_iter : int, optional
        Maximum iterations
    tol : float, optional
        Absolute convergence tolerance
    rel_tol : float, optional
        Relative convergence tolerance (fraction of initial V)
    verbose : bool, optional
        Print progress information
        
    Returns
    -------
    NDArray[np.float64]
        Optimized matrix of shape (n, d+K)
    """
    Y_current = Y.copy()
    
    # Smart block scheduling: process larger blocks first (bigger impact)
    # Skip singleton index columns (they have coef=-1, don't help)
    d_plus_K = Y.shape[1]
    d = d_plus_K - tilde_alpha.shape[0]
    
    sorted_blocks = sorted(
        blocks,
        key=lambda b: (-np.sum(b), -int(np.any(b[:d]))),  # Larger blocks first, prefer stock columns
        reverse=False
    )
    
    L = compute_L(Y_current, tilde_alpha)
    V_current = compute_objective(L)
    V_initial = V_current
    
    # Adaptive tolerance: stop when improvement < rel_tol * V_initial
    adaptive_tol = max(tol, rel_tol * V_initial)
    
    if verbose:
        print(f"Initial objective V = {V_current:.6f}")
        print(f"Adaptive tolerance = {adaptive_tol:.6e}")
    
    for iteration in range(max_iter):
        V_prev = V_current
        
        # Iterate through blocks (smart order)
        for block_mask in sorted_blocks:
            # Apply rearrangement
            block_rearrangement(Y_current, block_mask, tilde_alpha)
            
            # Compute new objective
            L = compute_L(Y_current, tilde_alpha)
            V_current = compute_objective(L)
        
        # Check convergence (adaptive)
        improvement = V_prev - V_current
        
        if verbose and (iteration < 10 or iteration % 100 == 0):
            print(f"Iteration {iteration+1}: V = {V_current:.6f}, improvement = {improvement:.6e}")
        
        # Stop if improvement below adaptive threshold
        if improvement < adaptive_tol:
            if verbose:
                print(f"Converged after {iteration+1} iterations (improvement < {adaptive_tol:.6e})")
                print(f"Final objective V = {V_current:.6f}")
                print(f"Total improvement: {V_initial - V_current:.6e}")
            break
    else:
        if verbose:
            print(f"Reached max iterations ({max_iter})")
            print(f"Final objective V = {V_current:.6f}")
            print(f"Total improvement: {V_initial - V_current:.6e}")
    
    return Y_current


def extract_joint_distribution(
    Y_final: NDArray[np.float64],
    d: int
) -> NDArray[np.float64]:
    """
    Extract joint distribution from final matrix.
    
    Parameters
    ----------
    Y_final : NDArray[np.float64]
        Final optimized matrix of shape (n, d+K)
    d : int
        Number of instruments
        
    Returns
    -------
    NDArray[np.float64]
        Joint distribution matrix of shape (n, d)
    """
    return Y_final[:, :d]
