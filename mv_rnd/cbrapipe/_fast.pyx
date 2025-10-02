# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: initializedcheck=False

"""
Cython-optimized core functions for CBRA.
"""

import numpy as np
cimport numpy as cnp
cimport cython

cnp.import_array()


@cython.boundscheck(False)
@cython.wraparound(False)
def block_rearrangement_fast(
    cnp.ndarray[cnp.float64_t, ndim=2] Y,
    cnp.ndarray[cnp.uint8_t, ndim=1] block_mask,
    cnp.ndarray[cnp.float64_t, ndim=2] tilde_alpha
):
    """
    Cython-optimized block rearrangement - minimal overhead version.
    
    Parameters
    ----------
    Y : ndarray (n, d+K)
        State matrix, modified in-place
    block_mask : ndarray (d+K,)
        Boolean mask (as uint8)
    tilde_alpha : ndarray (K, d+K)
        Coefficient matrix
    """
    cdef int n = Y.shape[0]
    cdef int total_cols = Y.shape[1]
    cdef int K = tilde_alpha.shape[0]
    cdef int j, k, i
    
    # Quick exit for empty block
    cdef int num_in_block = 0
    for j in range(total_cols):
        if block_mask[j]:
            num_in_block += 1
    
    if num_in_block == 0:
        return
    
    # Use numpy arrays directly to avoid copying overhead
    cdef cnp.ndarray[cnp.int64_t, ndim=1] in_block = np.where(block_mask)[0]
    cdef cnp.ndarray[cnp.int64_t, ndim=1] out_block = np.where(block_mask == 0)[0]
    
    cdef int num_out = len(out_block)
    cdef int num_in = len(in_block)
    
    # Get alpha_block (first column of block, all same for admissible blocks)
    cdef cnp.ndarray[cnp.float64_t, ndim=1] alpha_block = tilde_alpha[:, in_block[0]].copy()
    
    # Compute S = sum of block columns (vectorized)
    cdef cnp.ndarray[cnp.float64_t, ndim=1] S = np.sum(Y[:, in_block], axis=1)
    
    # Compute Z_agg = weighted sum outside block
    cdef cnp.ndarray[cnp.float64_t, ndim=1] Z_agg = np.zeros(n, dtype=np.float64)
    
    if num_out > 0:
        for k in range(K):
            Z_agg += alpha_block[k] * (Y[:, out_block] @ tilde_alpha[k, out_block])
    
    # Sort and create permutation (use numpy's optimized routines)
    cdef cnp.ndarray[cnp.int64_t, ndim=1] sort_Z = np.argsort(Z_agg)
    cdef cnp.ndarray[cnp.int64_t, ndim=1] sort_S = np.argsort(S)[::-1]
    
    cdef cnp.ndarray[cnp.int64_t, ndim=1] permutation = np.empty(n, dtype=np.int64)
    permutation[sort_Z] = sort_S
    
    # Apply permutation to block columns
    cdef cnp.ndarray[cnp.float64_t, ndim=2] Y_block_copy = Y[:, in_block].copy()
    Y[:, in_block] = Y_block_copy[permutation, :]


@cython.boundscheck(False)
@cython.wraparound(False)  
def compute_L_fast(double[:, :] Y, double[:, :] tilde_alpha):
    """
    Cython-optimized compute_L.
    
    L[k, i] = sum_j tilde_alpha[k, j] * Y[i, j]
    
    Returns
    -------
    L : ndarray (K, n)
    """
    cdef int K = tilde_alpha.shape[0]
    cdef int n = Y.shape[0]
    cdef int d_plus_K = Y.shape[1]
    
    cdef cnp.ndarray[double, ndim=2] L = np.zeros((K, n), dtype=np.float64)
    cdef int k, i, j
    
    for k in range(K):
        for i in range(n):
            for j in range(d_plus_K):
                L[k, i] += tilde_alpha[k, j] * Y[i, j]
    
    return L
