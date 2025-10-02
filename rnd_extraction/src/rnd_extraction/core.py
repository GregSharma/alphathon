"""Core RND extraction pipeline with GP fitting and optimization."""
import numpy as np
import pandas as pd
from py_vollib_vectorized.implied_volatility import vectorized_implied_volatility as viv
from scipy.linalg import cho_factor, cho_solve

from .types import GPHyperparameters, MarketData, RNDResult
from .utils import (bs_price_vectorized, compute_char_func, cumulative_trapz_fast,
                    rbf_kernel_fast, trapz_fast, vega_vectorized)


def prepare_market_data(market_data: MarketData, min_bid: float = 0.05) -> pd.DataFrame:
    """Prepare options data with IV and microprice calculations.
    
    Args:
        market_data: Market data containing options
        min_bid: Minimum bid price filter (default: 0.05)
    """
    df = market_data.options_df.copy()
    df['spread'] = df['ask'] - df['bid']
    df['total_size'] = df['ask_size'] + df['bid_size']
    df['microprice'] = (df['ask_size'] * df['bid'] + df['bid_size'] * df['ask']) / df['total_size']
    df = df[df['bid'] > min_bid].copy()
    df['right'] = df['right'].str.lower()
    
    # Compute IVs
    for price_measure in ['bid', 'microprice', 'ask']:
        df[f'{price_measure}_iv'] = viv(
            price=df[price_measure], S=market_data.spot_price, K=df['strike'],
            t=market_data.time_to_expiry, r=market_data.risk_free_rate,
            flag=df['right'], model='black_scholes', return_as='numpy'
        )
    
    df['iv_spread'] = np.where(~np.isnan(df['bid_iv']), df['ask_iv'] - df['bid_iv'], 1000)
    return df


def filter_otm_options(df: pd.DataFrame, forward_price: float) -> pd.DataFrame:
    """Filter for OTM options only."""
    otm_mask = ((df['right'] == 'c') & (df['strike'] > forward_price)) | \
               ((df['right'] == 'p') & (df['strike'] < forward_price))
    return df[otm_mask].dropna().copy()


def fit_gp_to_iv(X: np.ndarray, y: np.ndarray, noise_var: np.ndarray,
                 grid_k: np.ndarray, hypers: GPHyperparameters) -> tuple[np.ndarray, np.ndarray]:
    """Fit GP to IV surface."""
    if hypers.method == "exact":
        K = rbf_kernel_fast(X[:, None], X[:, None], hypers.length_scale, 
                           hypers.signal_variance) + np.diag(noise_var)
        c, lower = cho_factor(K)
        alpha = cho_solve((c, lower), y)
        grid_X = grid_k[:, None]
        k_star = rbf_kernel_fast(grid_X, X[:, None], hypers.length_scale, hypers.signal_variance)
        mean = k_star @ alpha
        cov = rbf_kernel_fast(grid_X, grid_X, hypers.length_scale, 
                             hypers.signal_variance) - k_star @ cho_solve((c, lower), k_star.T)
        var = np.maximum(np.diag(cov), 0)
    else:
        raise NotImplementedError(f"Method {hypers.method} not implemented")
    
    return mean, var


def extract_rnd_from_iv(iv_mean: np.ndarray, grid_k: np.ndarray, 
                        spot: float, forward: float, r: float, t: float) -> np.ndarray:
    """Extract RND from fitted IV surface using Breeden-Litzenberger."""
    strikes = forward * np.exp(grid_k)
    prices = bs_price_vectorized(spot, strikes, t, r, iv_mean, np.array(['c'] * len(strikes)))
    
    # Numerical differentiation
    price_diff1 = np.gradient(prices, strikes)
    price_diff2 = np.gradient(price_diff1, strikes)
    
    # RND in strike space
    q_K = np.exp(r * t) * price_diff2
    q_k = np.maximum(q_K * strikes, 0)
    
    # Normalize
    total = trapz_fast(q_k, grid_k)
    rnd_k = q_k / total if abs(total) > 1e-10 else q_k
    return rnd_k / trapz_fast(rnd_k, grid_k)


def extract_rnd(market_data: MarketData, grid_points: int = 300, 
                min_bid: float = 0.05) -> RNDResult:
    """Main pipeline: extract continuous arbitrage-free RND from option chain.
    
    Args:
        market_data: Market data
        grid_points: Number of grid points
        min_bid: Minimum bid price filter (default: 0.05)
        
    Returns:
        RND extraction result
    """
    # Prepare data
    df = prepare_market_data(market_data, min_bid=min_bid)
    forward = market_data.spot_price * np.exp(market_data.risk_free_rate * market_data.time_to_expiry)
    df_otm = filter_otm_options(df, forward)
    
    # Compute uncertainties and log-moneyness
    spread = df_otm['ask'].values - df_otm['bid'].values
    total_size = df_otm['bid_size'].values + df_otm['ask_size'].values
    var_price = (spread ** 2) / (12 * total_size)
    vega_vals = vega_vectorized(market_data.spot_price, df_otm['strike'].values,
                               market_data.time_to_expiry, market_data.risk_free_rate,
                               df_otm['microprice_iv'].values)
    noise_var = np.clip(var_price / (vega_vals ** 2 + 1e-8), 1e-6, None)
    
    X = np.log(df_otm['strike'].values / forward)
    y = df_otm['microprice_iv'].values
    
    # Heuristic hyperparameters
    pairwise_dist = np.abs(X[:, None] - X[None, :])
    ls_init = np.median(pairwise_dist[pairwise_dist > 0])
    sf2_init = np.var(y)
    hypers = GPHyperparameters(length_scale=ls_init, signal_variance=sf2_init, method="exact")
    
    # Grid and GP fit
    grid_k = np.linspace(X.min(), X.max(), grid_points)
    iv_mean, iv_var = fit_gp_to_iv(X, y, noise_var, grid_k, hypers)
    
    # Extract RND
    rnd = extract_rnd_from_iv(iv_mean, grid_k, market_data.spot_price, forward,
                              market_data.risk_free_rate, market_data.time_to_expiry)
    
    # Cumulative RND
    rnd_cumulative = cumulative_trapz_fast(rnd, grid_k)
    
    # Characteristic function
    u_samples = np.linspace(-10, 10, 101)
    cf_values = compute_char_func(u_samples, rnd, grid_k)
    
    return RNDResult(
        log_moneyness=grid_k, strikes=forward * np.exp(grid_k),
        rnd_density=rnd, rnd_cumulative=rnd_cumulative,
        fitted_iv=iv_mean, fitted_iv_std=np.sqrt(iv_var),
        characteristic_function_u=u_samples, characteristic_function_values=cf_values,
        forward_price=forward
    )
