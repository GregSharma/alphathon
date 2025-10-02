"""
Step 2: Coefficient expansion and admissible block identification.
"""
import numpy as np
from numpy.typing import NDArray
from typing import List


def expand_coefficients(A: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Expand weight matrix A to coefficient matrix tilde_alpha.
    
    For constraint k and column j:
    - j in [0, d): tilde_alpha[k, j] = A[j, k] (stock weight)
    - j == d+k: tilde_alpha[k, j] = -1 (index column for constraint k)
    - j in [d, d+K) and j != d+k: tilde_alpha[k, j] = 0 (other indices)
    
    Parameters
    ----------
    A : NDArray[np.float64]
        Weight matrix of shape (d, K) where A[j, k] is weight of stock j in index k
        
    Returns
    -------
    NDArray[np.float64]
        Coefficient matrix of shape (K, d+K)
    """
    d, K = A.shape
    tilde_alpha = np.zeros((K, d + K), dtype=np.float64)
    
    for k in range(K):
        # Stock columns: use weights from A
        tilde_alpha[k, :d] = A[:, k]
        # Index column for this constraint: -1
        tilde_alpha[k, d + k] = -1.0
        # Other index columns: remain 0
    
    return tilde_alpha


def identify_admissible_blocks(
    tilde_alpha: NDArray[np.float64]
) -> List[NDArray[np.bool_]]:
    """
    Identify all admissible blocks from coefficient matrix.
    
    A block (binary mask) is admissible if for every constraint k,
    the coefficients tilde_alpha[k, j] are constant for all j in the block.
    
    Parameters
    ----------
    tilde_alpha : NDArray[np.float64]
        Coefficient matrix of shape (K, d+K)
        
    Returns
    -------
    List[NDArray[np.bool_]]
        List of boolean masks, each of shape (d+K,) indicating block membership
    """
    K, total_cols = tilde_alpha.shape
    blocks = []
    
    # Always include singleton blocks (single column)
    for j in range(total_cols):
        mask = np.zeros(total_cols, dtype=np.bool_)
        mask[j] = True
        blocks.append(mask)
    
    # Find multi-column admissible blocks
    # Two columns can be in the same block if they have the same coefficient
    # across all constraints
    for j1 in range(total_cols):
        for j2 in range(j1 + 1, total_cols):
            # Check if columns j1 and j2 have same coefficients for all constraints
            if np.allclose(tilde_alpha[:, j1], tilde_alpha[:, j2]):
                mask = np.zeros(total_cols, dtype=np.bool_)
                mask[j1] = True
                mask[j2] = True
                blocks.append(mask)
    
    # Also check for larger blocks (all columns with same coefficient pattern)
    # Group columns by their coefficient pattern
    from collections import defaultdict
    pattern_groups = defaultdict(list)
    
    for j in range(total_cols):
        # Use tuple as hashable key
        pattern = tuple(tilde_alpha[:, j])
        pattern_groups[pattern].append(j)
    
    # Create blocks for groups with 3+ columns
    for pattern, cols in pattern_groups.items():
        if len(cols) >= 3:
            mask = np.zeros(total_cols, dtype=np.bool_)
            mask[cols] = True
            blocks.append(mask)
    
    return blocks
