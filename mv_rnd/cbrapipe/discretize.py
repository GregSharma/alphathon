"""
Step 1: Discretization utilities for instruments and constraints.
"""
from typing import Callable, Sequence
import numpy as np
from numpy.typing import NDArray


def discretize_instruments(
    n: int, 
    F_inv_list: Sequence[Callable[[float], float]]
) -> NDArray[np.float64]:
    """
    Discretize individual instruments using their inverse RNCDs.
    
    Parameters
    ----------
    n : int
        Number of equiprobable states
    F_inv_list : Sequence[Callable[[float], float]]
        List of inverse RNCD functions (quantile functions) for each instrument
        
    Returns
    -------
    NDArray[np.float64]
        Matrix X of shape (n, d) where d = len(F_inv_list)
    """
    d = len(F_inv_list)
    X = np.zeros((n, d), dtype=np.float64)
    
    for i in range(n):
        prob = (i + 0.5) / n
        for j, F_inv in enumerate(F_inv_list):
            X[i, j] = F_inv(prob)
    
    return X


def discretize_constraints(
    n: int, 
    F_inv_list: Sequence[Callable[[float], float]]
) -> NDArray[np.float64]:
    """
    Discretize constraint indices using their inverse RNCDs.
    
    Parameters
    ----------
    n : int
        Number of equiprobable states
    F_inv_list : Sequence[Callable[[float], float]]
        List of inverse RNCD functions for each index
        
    Returns
    -------
    NDArray[np.float64]
        Matrix S of shape (n, K) where K = len(F_inv_list)
    """
    K = len(F_inv_list)
    S = np.zeros((n, K), dtype=np.float64)
    
    for i in range(n):
        prob = (i + 0.5) / n
        for k, F_inv in enumerate(F_inv_list):
            S[i, k] = F_inv(prob)
    
    return S


def build_initial_matrix(
    X: NDArray[np.float64], 
    S: NDArray[np.float64]
) -> NDArray[np.float64]:
    """
    Combine instrument and constraint matrices into initial Y matrix.
    
    Parameters
    ----------
    X : NDArray[np.float64]
        Instrument matrix of shape (n, d)
    S : NDArray[np.float64]
        Constraint matrix of shape (n, K)
        
    Returns
    -------
    NDArray[np.float64]
        Combined matrix Y of shape (n, d+K)
    """
    return np.hstack([X, S])
