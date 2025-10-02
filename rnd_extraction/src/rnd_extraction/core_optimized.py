"""Heavily optimized RND extraction with Numba JIT compilation."""
import numpy as np
import pandas as pd
from numba import jit
from py_vollib_vectorized.implied_volatility import vectorized_implied_volatility as viv
from scipy.linalg import cho_factor, cho_solve
from scipy.stats import norm

from .types import GPHyperparameters, MarketData, RNDResult
from .utils import (compute_char_func, cumulative_trapz_fast,
                                   rbf_kernel_fast, trapz_fast)


def bs_price_numba(S: float, K: np.ndarray, t: float, r: float,
                   sigma: np.ndarray) -> np.ndarray:
    """Black-Scholes call price (using scipy for accuracy)."""
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * t) / (sigma * np.sqrt(t) + 1e-8)
    d2 = d1 - sigma * np.sqrt(t)
    return S * norm.cdf(d1) - K * np.exp(-r * t) * norm.cdf(d2)


@jit(nopython=True, cache=True)
def vega_numba(S: float, K: np.ndarray, t: float, r: float,
              sigma: np.ndarray) -> np.ndarray:
    """Option vega (Numba-optimized)."""
    n = len(K)
    vegas = np.zeros(n)
    sqrt_t = np.sqrt(t)
    inv_sqrt_2pi = 0.3989422804014327
    for i in range(n):
        d1 = (np.log(S / K[i]) + (r + 0.5 * sigma[i]**2) * t) / (sigma[i] * sqrt_t + 1e-8)
        vegas[i] = S * sqrt_t * inv_sqrt_2pi * np.exp(-0.5 * d1 * d1)
    return vegas


@jit(nopython=True, cache=True)
def gradient_numba(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Numerical gradient (Numba-optimized)."""
    n = len(y)
    grad = np.zeros(n)
    # Forward difference at start
    grad[0] = (y[1] - y[0]) / (x[1] - x[0])
    # Central differences in middle
    for i in range(1, n-1):
        grad[i] = (y[i+1] - y[i-1]) / (x[i+1] - x[i-1])
    # Backward difference at end
    grad[n-1] = (y[n-1] - y[n-2]) / (x[n-1] - x[n-2])
    return grad


def prepare_market_data_fast(market_data: MarketData, min_bid: float = 0.05) -> pd.DataFrame:
    """Optimized market data preparation.
    
    Args:
        market_data: Market data containing options
        min_bid: Minimum bid price filter (default: 0.05)
    """
    df = market_data.options_df.copy()
    df['microprice'] = (df['ask_size'] * df['bid'] + df['bid_size'] * df['ask']) / (
        df['ask_size'] + df['bid_size']
    )
    df = df[df['bid'] > min_bid].copy()
    df['right'] = df['right'].str.lower()
    
    # Compute IVs - only microprice (skip bid/ask)
    df['iv'] = viv(
        price=df['microprice'], S=market_data.spot_price, K=df['strike'],
        t=market_data.time_to_expiry, r=market_data.risk_free_rate,
        flag=df['right'], model='black_scholes', return_as='numpy'
    )
    return df


def fit_gp_fast(X: np.ndarray, y: np.ndarray, noise_var: np.ndarray,
                grid_k: np.ndarray, ls: float, sf2: float) -> tuple[np.ndarray, np.ndarray]:
    """Optimized GP fitting using Numba kernel."""
    X2d = X[:, None]
    K = rbf_kernel_fast(X2d, X2d, ls, sf2) + np.diag(noise_var)
    c, lower = cho_factor(K)
    alpha = cho_solve((c, lower), y)
    grid_X = grid_k[:, None]
    k_star = rbf_kernel_fast(grid_X, X2d, ls, sf2)
    mean = k_star @ alpha
    cov = rbf_kernel_fast(grid_X, grid_X, ls, sf2) - k_star @ cho_solve((c, lower), k_star.T)
    var = np.maximum(np.diag(cov), 0)
    return mean, var


def extract_rnd_fast(iv_mean: np.ndarray, grid_k: np.ndarray,
                     spot: float, forward: float, r: float, t: float) -> np.ndarray:
    """Optimized RND extraction using Numba."""
    strikes = forward * np.exp(grid_k)
    prices = bs_price_numba(spot, strikes, t, r, iv_mean)
    price_diff1 = gradient_numba(prices, strikes)
    price_diff2 = gradient_numba(price_diff1, strikes)
    q_K = np.exp(r * t) * price_diff2
    q_k = np.maximum(q_K * strikes, 0)
    total = trapz_fast(q_k, grid_k)
    rnd_k = q_k / total if abs(total) > 1e-10 else q_k
    return rnd_k / trapz_fast(rnd_k, grid_k)


def extract_rnd_optimized(market_data: MarketData, grid_points: int = 300, 
                          min_bid: float = 0.05) -> RNDResult:
    """Fully optimized RND extraction pipeline.
    
    Args:
        market_data: Market data
        grid_points: Number of grid points
        min_bid: Minimum bid price filter (default: 0.05)
        
    Returns:
        RND extraction result
    """
    # Prepare data
    df = prepare_market_data_fast(market_data, min_bid=min_bid)
    forward = market_data.spot_price * np.exp(market_data.risk_free_rate * market_data.time_to_expiry)
    
    # Filter OTM
    otm_mask = ((df['right'] == 'c') & (df['strike'] > forward)) | \
               ((df['right'] == 'p') & (df['strike'] < forward))
    df_otm = df[otm_mask].dropna()
    
    # Uncertainties
    spread = df_otm['ask'].values - df_otm['bid'].values
    total_size = df_otm['bid_size'].values + df_otm['ask_size'].values
    var_price = (spread ** 2) / (12 * total_size)
    vega_vals = vega_numba(market_data.spot_price, df_otm['strike'].values,
                          market_data.time_to_expiry, market_data.risk_free_rate,
                          df_otm['iv'].values)
    noise_var = np.clip(var_price / (vega_vals ** 2 + 1e-8), 1e-6, None)
    
    X = np.log(df_otm['strike'].values / forward)
    y = df_otm['iv'].values
    
    # Hyperparameters
    pairwise_dist = np.abs(X[:, None] - X[None, :])
    ls_init = np.median(pairwise_dist[pairwise_dist > 0])
    sf2_init = np.var(y)
    
    # Grid and GP
    grid_k = np.linspace(X.min(), X.max(), grid_points)
    iv_mean, iv_var = fit_gp_fast(X, y, noise_var, grid_k, ls_init, sf2_init)
    
    # Extract RND
    rnd = extract_rnd_fast(iv_mean, grid_k, market_data.spot_price, forward,
                          market_data.risk_free_rate, market_data.time_to_expiry)
    
    # Cumulative
    rnd_cumulative = cumulative_trapz_fast(rnd, grid_k)
    
    # CF
    u_samples = np.linspace(-10, 10, 101)
    cf_values = compute_char_func(u_samples, rnd, grid_k)
    
    return RNDResult(
        log_moneyness=grid_k, strikes=forward * np.exp(grid_k),
        rnd_density=rnd, rnd_cumulative=rnd_cumulative,
        fitted_iv=iv_mean, fitted_iv_std=np.sqrt(iv_var),
        characteristic_function_u=u_samples, characteristic_function_values=cf_values,
        forward_price=forward
    )
