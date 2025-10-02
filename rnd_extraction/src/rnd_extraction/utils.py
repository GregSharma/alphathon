"""Utility functions for RND extraction with Numba optimization."""
import numpy as np
from numba import jit
from scipy.stats import norm


@jit(nopython=True, cache=True)
def rbf_kernel_fast(X1: np.ndarray, X2: np.ndarray, ls: float, sf2: float) -> np.ndarray:
    """RBF kernel matrix (Numba-optimized)."""
    n1, n2 = X1.shape[0], X2.shape[0]
    K = np.zeros((n1, n2))
    for i in range(n1):
        for j in range(n2):
            dist2 = (X1[i, 0] - X2[j, 0]) ** 2
            K[i, j] = sf2 * np.exp(-0.5 * dist2 / (ls * ls))
    return K


def bs_price_vectorized(S: float, K: np.ndarray, t: float, r: float, 
                        sigma: np.ndarray, flag: np.ndarray) -> np.ndarray:
    """Black-Scholes price (vectorized)."""
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * t) / (sigma * np.sqrt(t) + 1e-8)
    d2 = d1 - sigma * np.sqrt(t)
    call_price = S * norm.cdf(d1) - K * np.exp(-r * t) * norm.cdf(d2)
    put_price = K * np.exp(-r * t) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return np.where(flag == 'c', call_price, put_price)


def vega_vectorized(S: float, K: np.ndarray, t: float, r: float, 
                   sigma: np.ndarray) -> np.ndarray:
    """Option vega (vectorized)."""
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * t) / (sigma * np.sqrt(t) + 1e-8)
    return S * np.sqrt(t) * norm.pdf(d1)


@jit(nopython=True, cache=True)
def trapz_fast(y: np.ndarray, x: np.ndarray) -> float:
    """Trapezoidal integration (Numba-optimized)."""
    n = len(y)
    result = 0.0
    for i in range(n - 1):
        result += 0.5 * (y[i] + y[i + 1]) * (x[i + 1] - x[i])
    return result


@jit(nopython=True, cache=True)
def cumulative_trapz_fast(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Cumulative trapezoidal integration (Numba-optimized)."""
    n = len(y)
    result = np.zeros(n)
    for i in range(1, n):
        result[i] = result[i-1] + 0.5 * (y[i-1] + y[i]) * (x[i] - x[i-1])
    return result


@jit(nopython=True, cache=True)
def compute_char_func(u: np.ndarray, rnd_k: np.ndarray, grid_k: np.ndarray) -> np.ndarray:
    """Characteristic function computation (Numba-optimized)."""
    dk = grid_k[1] - grid_k[0]
    n_u, n_k = len(u), len(grid_k)
    phi = np.zeros(n_u)
    for i in range(n_u):
        for j in range(n_k):
            phi[i] += np.cos(u[i] * grid_k[j]) * rnd_k[j]
    return phi * dk

