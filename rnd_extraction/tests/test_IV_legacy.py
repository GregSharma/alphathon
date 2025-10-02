#!/usr/bin/env python
# coding: utf-8

# In[1]:


import math  # For factorial

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from py_vollib_vectorized.implied_volatility import vectorized_implied_volatility as viv
from scipy.fft import fft, fftfreq
from scipy.integrate import trapezoid as trapz

# Imports (add these if not already present; e.g., at top of Cell or in Cell 1)
from scipy.interpolate import interp1d
from scipy.linalg import cho_factor, cho_solve
from scipy.special import hermite  # For Hermite polynomials
from scipy.stats import lognorm, norm  # For baseline densities


# Gram-Charlier CF (analytical approximation)
def gram_charlier_cf(
    u: np.ndarray, mu: float, sigma: float, coeffs: np.ndarray, baseline="normal"
) -> np.ndarray:
    """
    Analytical CF from GC expansion.
    - mu, sigma: Mean/std of baseline density (fit to RND moments).
    - coeffs: [c0, c1, ..., cM] GC coefficients (c0=1 enforced).
    - baseline: 'normal' (default) or 'lognormal' (better for RND tails).
    Returns: Complex array of φ(u) (use np.real for plots).
    """
    iu = 1j * u
    if baseline == "lognormal":
        # Lognormal CF for density on k=log(K/F): exp(i u mu - 0.5 sigma^2 u^2 (1 + i u))
        # (Assumes centered at mu; full for lognormal under RN measure)
        phi_base = np.exp(iu * mu - 0.5 * sigma**2 * u**2 * (1 + iu))
    else:  # Normal baseline
        phi_base = np.exp(iu * mu - 0.5 * sigma**2 * u**2)

    # GC correction: polynomial sum c_m (i u)^m (for probabilists' Hermite projection)
    poly_term = np.zeros_like(u, dtype=complex)
    power = np.ones_like(u, dtype=complex)
    for m, c_m in enumerate(coeffs):
        poly_term += c_m * power
        power *= iu  # Cumulative (i u)^m
    return phi_base * poly_term


# Gram-Charlier Fit (fixed dimension/normalization)
def fit_gram_charlier(
    rnd_k: np.ndarray, grid_k: np.ndarray, M: int = 10, baseline="normal"
) -> tuple:
    """
    Project numerical RND onto GC basis (fixed dimension issue).
    - M: # terms (e.g., 8; higher = better fit but more oscillation).
    - baseline: 'normal' (simpler, default) or 'lognormal' (better for RND tails).
    - Returns: (mu, sigma, coeffs), with coeffs[0]=1 for exact φ(0)=1.
    """
    dk = grid_k[1] - grid_k[0]  # Scalar for uniform grid; used in moments
    total_mass = trapz(rnd_k, grid_k)  # Should be ~1

    # Baseline moments via trapezoidal integration
    weights = rnd_k * dk
    mu0 = trapz(grid_k * weights, grid_k) / total_mass
    var0 = trapz((grid_k - mu0) ** 2 * weights, grid_k) / total_mass
    sigma0 = np.sqrt(np.maximum(var0, 1e-8))  # Clamp to avoid div0

    # Standardized z for Hermite
    z = (grid_k - mu0) / sigma0
    phi0 = norm.pdf(z)  # Standard normal baseline

    if baseline == "lognormal":
        # Lognormal baseline on strikes s=exp(grid_k), projected to k-scale
        s = np.exp(grid_k)  # K = F * exp(k); F implicit=1 for density
        # Lognorm params: shape=sigma0, scale=exp(mu0 + sigma0^2/2) for mean, but approx for pdf on k
        phi0_log = lognorm.pdf(s, s=sigma0, scale=np.exp(mu0), loc=0)
        phi0 = phi0_log / s  # Jacobian: pdf_k(k) = pdf_s(s) * |ds/dk| = pdf_s * s
        phi0 /= trapz(phi0, grid_k)  # Renormalize ∫ pdf_k dk =1

    # Basis matrix: N rows (grid pts), M+1 cols (terms); psi_m(j) = phi0(j) * He_m(z_j) / norm_m
    N = len(grid_k)
    basis_matrix = np.zeros((N, M + 1))
    for m in range(M + 1):
        H_m = hermite(m)(z)  # Probabilists' Hermite polynomial
        norm_factor = np.sqrt(math.factorial(m)) if m > 0 else 1.0
        basis_matrix[:, m] = (phi0 * H_m) / norm_factor

    # Least-squares: A c ≈ rnd_k (A: N x (M+1), b: N x 1) → c: (M+1,)
    # Discrete LS approximates integral projection (uniform dk makes it exact up to scaling)
    coeffs, residuals, rank, s = np.linalg.lstsq(basis_matrix, rnd_k, rcond=None)

    # Normalize so ∫ recon dk =1
    recon_c0 = basis_matrix[:, 0] * coeffs[0]
    integral_c0 = trapz(recon_c0, grid_k)
    if abs(integral_c0) > 1e-10:
        coeffs /= integral_c0 / total_mass
    coeffs[0] = 1.0  # Enforce c0=1 for φ(0)=1 (GC convention)

    # Reconstruction and errors
    recon = basis_matrix @ coeffs
    mse_discrete = np.mean((rnd_k - recon) ** 2)
    mse_integral = trapz((rnd_k - recon) ** 2, grid_k)
    print(
        f"GC Fit: M={M}, Discrete MSE={mse_discrete:.2e}, Integral MSE={mse_integral:.2e}, mu={mu0:.4f}, sigma={sigma0:.4f}"
    )

    return mu0, sigma0, coeffs


# Placeholder for Legendre Projection (TODO: Implement when revisiting)
def fit_legendre(rnd_k: np.ndarray, grid_k: np.ndarray, M: int = 10) -> np.ndarray:
    """
    TODO: Project RND onto Legendre polynomials on [grid_k.min(), grid_k.max()].
    - Map grid_k to [-1,1], build basis_matrix (N x M+1) with legendre(m)(x).
    - LS: coeffs = lstsq(basis_matrix, rnd_k).
    - Normalize: Ensure trapz(recon, grid_k) =1.
    - Return: coeffs (M+1,).
    Placeholder: Returns zeros for now.
    """
    print("TODO: Implement Legendre projection (scipy.special.legendre).")
    # Stub:
    a, b = grid_k.min(), grid_k.max()
    x = 2 * (grid_k - a) / (b - a) - 1  # To [-1,1]
    # from scipy.special import legendre
    # basis = np.array([[legendre(m)(xi) for m in range(M+1)] for xi in x]).T  # (M+1, N) -> transpose for LS
    # coeffs = np.linalg.lstsq(basis.T, rnd_k)[0]  # A=(N,M+1)
    # ... normalize ...
    return np.zeros(M + 1)  # Placeholder


def legendre_cf(u: np.ndarray, coeffs: np.ndarray, a: float, b: float) -> np.ndarray:
    """
    TODO: Analytical CF for Legendre basis on [a,b].
    - For each u, integrate exp(i u k) * sum c_m P_m( map(k) ) dk (via quad or Bessel approx).
    - Placeholder: Returns zeros.
    """
    print(
        "TODO: Implement Legendre CF (e.g., via scipy.integrate.quad per u, or known Fourier-Legendre transform)."
    )
    return np.zeros_like(u, dtype=complex)  # Placehold


import time
from typing import Any, Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from py_vollib_vectorized.implied_volatility import vectorized_implied_volatility as viv
from scipy.integrate import trapezoid as trapz
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize
from scipy.stats import beta as beta_dist
from scipy.stats import norm

RISK_FREE_RATE = 0.05


# Abstracted GP Fitting Function (Pluggable: 'exact' or 'lowrank')
def fit_gp(
    method: str,
    X: np.ndarray,
    y: np.ndarray,
    noise_var: np.ndarray,
    ls: float,
    sf2: float,
    grid_k: np.ndarray,
    r: float = RISK_FREE_RATE,
) -> Dict[str, Any]:
    """
    Pluggable GP fit: 'exact' (full Cholesky) or 'lowrank' (SoR approximation).
    Returns: {'mean': np.ndarray, 'var': np.ndarray, 'F_adj': float, 'runtime': float}
    """
    start_time = time.perf_counter()
    if method == "exact":
        # Exact GP (heuristic hypers, no solver)
        K = rbf_kernel(X[:, None], X[:, None], ls, sf2) + np.diag(noise_var)
        c, lower = cho_factor(K)
        alpha = cho_solve((c, lower), y)
        grid_X = grid_k[:, None]
        k_star = rbf_kernel(grid_X, X[:, None], ls, sf2)
        mean = k_star @ alpha
        cov = rbf_kernel(grid_X, grid_X, ls, sf2) - k_star @ cho_solve((c, lower), k_star.T)
        var = np.maximum(np.diag(cov), 0)
    elif method == "lowrank":
        # Low-rank SoR: Fixed inducing points (deterministic, p = n//4)
        n = len(X)
        p = max(10, n // 4)
        bar_k_idx = np.linspace(0, n - 1, p, dtype=int)
        bar_k = X[bar_k_idx]
        bar_K = rbf_kernel(bar_k[:, None], bar_k[:, None], ls, sf2)
        D = np.diag(noise_var[bar_k_idx])
        bar_c, bar_lower = cho_factor(bar_K + D)
        bar_alpha = cho_solve((bar_c, bar_lower), y[bar_k_idx])
        # Mean: project to inducing
        a_star = rbf_kernel(grid_k[:, None], bar_k[:, None], ls, sf2)
        mean = a_star @ bar_alpha
        # Var approx (diag only, simplified) - Fixed dimension mismatch
        k_ss = rbf_kernel(grid_k[:, None], grid_k[:, None], ls, sf2)
        k_ss_diag = k_ss.diagonal()
        # Compute Q = a_star @ inv(bar_K + D) @ a_star.T, then diagonal
        Q = a_star @ cho_solve((bar_c, bar_lower), a_star.T)
        var = np.maximum(k_ss_diag - np.diag(Q), 0)
    else:
        raise ValueError("Method must be 'exact' or 'lowrank'")

    F_adj = SPOT_PRICE * np.exp(r * TTE_YEARS)
    runtime = time.perf_counter() - start_time
    return {"mean": mean, "var": var, "F_adj": F_adj, "runtime": runtime}


# RBF Kernel (shared)
def rbf_kernel(X1: np.ndarray, X2: np.ndarray, ls: float, sf2: float = 1.0) -> np.ndarray:
    """RBF kernel matrix."""
    dist2 = np.sum(X1**2, axis=1)[:, None] + np.sum(X2**2, axis=1)[None, :] - 2 * np.dot(X1, X2.T)
    return sf2 * np.exp(-0.5 * dist2 / ls**2)


# Vectorized BS Price
def bs_price(
    S: float, K: np.ndarray, t: float, r: float, sigma: np.ndarray, flag: np.ndarray
) -> np.ndarray:
    """Black-Scholes price vectorized over strikes with flag array."""
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * t) / (sigma * np.sqrt(t) + 1e-8)
    d2 = d1 - sigma * np.sqrt(t)
    call_part = S * norm.cdf(d1) - K * np.exp(-r * t) * norm.cdf(d2)
    put_part = K * np.exp(-r * t) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return np.where(flag == "c", call_part, put_part)


# Smooth Wing Blend (shared)
def smooth_wing_blend(
    mean: np.ndarray,
    grid_k: np.ndarray,
    X: np.ndarray,
    mu: float,
    beta: float,
    zone_width: float = 0.15,
) -> np.ndarray:
    """Blend GP mean to linear wings for extrapolation."""
    mean_blend = mean.copy()
    # Left wing
    left_boundary = X.min()
    left_trans_start = left_boundary - zone_width
    left_mask = grid_k < left_boundary
    left_trans_mask = (grid_k >= left_trans_start) & (grid_k < left_boundary)
    if np.any(left_mask):
        linear_left = mu + beta * np.abs(grid_k[left_mask])
        mean_blend[left_mask] = linear_left
        if np.any(left_trans_mask):
            linear_trans_left = mu + beta * np.abs(grid_k[left_trans_mask])
            dist_to_boundary = (left_boundary - grid_k[left_trans_mask]) / zone_width
            mean_blend[left_trans_mask] = (
                dist_to_boundary * mean[left_trans_mask]
                + (1 - dist_to_boundary) * linear_trans_left
            )
    # Right wing
    right_boundary = X.max()
    right_trans_end = right_boundary + zone_width
    right_mask = grid_k > right_boundary
    right_trans_mask = (grid_k > right_boundary) & (grid_k <= right_trans_end)
    if np.any(right_mask):
        linear_right = mu + beta * grid_k[right_mask]
        mean_blend[right_mask] = linear_right
        if np.any(right_trans_mask):
            linear_trans_right = mu + beta * grid_k[right_trans_mask]
            dist_to_boundary = (grid_k[right_trans_mask] - right_boundary) / zone_width
            mean_blend[right_trans_mask] = (1 - dist_to_boundary) * mean[
                right_trans_mask
            ] + dist_to_boundary * linear_trans_right
    return mean_blend


# CF (shared)
def char_func(u: np.ndarray, rnd_k: np.ndarray, grid_k: np.ndarray) -> np.ndarray:
    dk = grid_k[1] - grid_k[0]
    exp_term = np.exp(1j * u[:, None] * grid_k)
    phi = dk * np.sum(exp_term * rnd_k[None, :], axis=1)
    return np.real(phi)


# IV setup
def vega(S: float, K: np.ndarray, t: float, r: float, sigma: np.ndarray) -> np.ndarray:
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * t) / (sigma * np.sqrt(t) + 1e-8)
    return S * np.sqrt(t) * norm.pdf(d1)


# Derive Prices, RND, CF (shared for both; use exact for now, but could loop)
def derive_quantities(mean: np.ndarray, F_adj: float, r: float) -> Tuple[np.ndarray, np.ndarray]:
    fine_K = F_adj * np.exp(grid_k)
    fine_prices = bs_price(SPOT_PRICE, fine_K, TTE_YEARS, r, mean, ["c"] * len(fine_K))
    price_diff1 = np.gradient(fine_prices, fine_K)
    price_diff2 = np.gradient(price_diff1, fine_K)
    q_K = np.exp(r * TTE_YEARS) * price_diff2
    q_k = np.maximum(q_K * fine_K, 0)
    total = trapz(q_k, grid_k)
    rnd_k = q_k / total if abs(total) > 1e-10 else q_k
    integral_rnd_raw = trapz(rnd_k, grid_k)
    rnd_k = rnd_k / integral_rnd_raw if abs(integral_rnd_raw) > 1e-10 else rnd_k
    return fine_prices, rnd_k


# RND Reconstruction (validate projection quality)
def gc_rnd(
    k: np.ndarray, mu: float, sigma: float, coeffs: np.ndarray, baseline="normal"
) -> np.ndarray:
    """Reconstruct RND from GC basis (for validation)."""
    z = (k - mu) / sigma
    if baseline == "normal":
        phi0 = norm.pdf(z)
    else:  # lognormal (approx)
        s = np.exp(k)
        phi0 = lognorm.pdf(s, s=sigma, scale=np.exp(mu), loc=0) / s  # Jacobian
        phi0 /= trapz(phi0, k)  # Renorm

    recon = np.zeros_like(k)
    for m, c_m in enumerate(coeffs):
        H_m = hermite(m)(z)
        norm_factor = np.sqrt(math.factorial(m)) if m > 0 else 1.0
        recon += c_m * (phi0 * H_m / norm_factor)
    return np.maximum(recon, 0)  # Clip negatives


# Validation (for both)
def validate(
    mean: np.ndarray,
    var: np.ndarray,
    y: np.ndarray,
    X: np.ndarray,
    grid_k: np.ndarray,
    rnd_k: np.ndarray,
) -> Dict[str, float]:
    interp_iv = np.interp(X, grid_k, mean)
    rmse_iv = np.sqrt(np.mean((y - interp_iv) ** 2))
    integral_rnd = trapz(rnd_k, grid_k)
    return {"rmse_iv": rmse_iv, "integral_rnd": integral_rnd}


# Negative Log-Likelihood (for validation, optional)
def nll(
    theta: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    noise_var: np.ndarray,
    alpha_params: np.ndarray,
    beta_params: np.ndarray,
    spread: np.ndarray,
    strikes: np.ndarray,
    rights: np.ndarray,
    S: float,
    t: float,
    bid_prices: np.ndarray,
) -> float:
    """Hybrid NLL: Gaussian on IV + Beta on prices (weighted 0.5). Includes r adjustment."""
    log_ls, log_sf2, r_adj = theta
    ls, sf2, r = np.exp(log_ls), np.exp(log_sf2), RISK_FREE_RATE + r_adj
    K_gp = rbf_kernel(X[:, None], X[:, None], ls, sf2) + np.diag(noise_var)
    try:
        c, lower = cho_factor(K_gp)
        alpha = cho_solve((c, lower), y)
        gp_nll = 0.5 * (np.dot(y, alpha) + np.log(np.diag(K_gp)).sum())
    except:
        return np.inf
    k_pred = rbf_kernel(X[:, None], X[:, None], ls, sf2)
    mean_iv = np.dot(k_pred, alpha)
    hat_p = bs_price(S, strikes, t, r, mean_iv, rights)
    x_beta = np.clip((hat_p - bid_prices) / spread, 0.001, 0.999)
    beta_logpdf = beta_dist.logpdf(x_beta, alpha_params, beta_params).sum()
    beta_nll = -beta_logpdf
    total_nll = gp_nll  # + 0.5 * beta_nll
    return total_nll

df = pd.read_parquet("../data/options_2024/SPXW.parquet").set_index("ts")
df.index = df.index.tz_localize("America/New_York")
slice_time = pd.Timestamp("2024-09-04 10:00:00", tz="America/New_York")
expiration_time = pd.Timestamp("2024-09-04 16:00:00", tz="America/New_York")
tte_delta = expiration_time - slice_time
TTE_YEARS: float = tte_delta.total_seconds() / (365.25 * 24 * 3600)
RISK_FREE_RATE: float = 0.05341

# Cell 1: Data Loading and IV Computation

df = pd.read_parquet("../data/options_2024/SPXW.parquet").set_index("ts")
df.index = df.index.tz_localize("America/New_York")
slice_time = pd.Timestamp("2024-09-04 10:05:00", tz="America/New_York")
expiration_time = pd.Timestamp("2024-09-04 16:00:00", tz="America/New_York")
tte_delta = expiration_time - slice_time
TTE_YEARS: float = tte_delta.total_seconds() / (365.25 * 24 * 3600)
RISK_FREE_RATE: float = 0.05341
data = df[df.index == slice_time]
SPOT_PRICE: float = 5524.19#data["spx"].values[0]
vol: pd.DataFrame = data[
    ["strike", "right", "bid", "ask", "mid", "bid_size", "ask_size"]
].reset_index(drop=True)
vol['spread'] = vol['ask'] - vol['bid']
vol["total_size"] = vol["ask_size"] + vol["bid_size"]
vol["microprice"] = (vol["ask_size"] * vol["bid"] + vol["bid_size"] * vol["ask"]) / (
    vol["total_size"]
)
vol = vol[vol.bid > 0.05]
vol["right"] = vol["right"].str.lower()

for price_measure in ["bid", "microprice", "ask"]:
    vol[f"{price_measure}_iv"] = viv(
        price=vol[price_measure],
        S=SPOT_PRICE,
        K=vol["strike"],
        t=TTE_YEARS,
        r=RISK_FREE_RATE,
        flag=vol["right"],
        model="black_scholes",
        return_as="numpy",
    )

vol['iv_spread'] = np.where(
    ~np.isnan(vol['bid_iv']), 
    vol['ask_iv'] - vol['bid_iv'], 
    1000
)
# vol = vol.loc[vol.groupby('strike')['iv_spread'].idxmin()].reset_index(drop=True)


# In[3]:


import matplotlib.pyplot as plt

DF = np.exp(-1*TTE_YEARS*RISK_FREE_RATE)
# Pivot the dataframe by strike to have call and put columns
call_cols = ['right', 'bid', 'ask', 'mid', 'bid_size', 'ask_size', 'spread', 'total_size', 'microprice', 'bid_iv', 'microprice_iv', 'ask_iv']
put_cols = [col + '_put' for col in call_cols]
call_cols = [col + '_call' for col in call_cols]

# Create separate dataframes for calls and puts
calls = vol[vol['right'] == 'c'].copy()
puts = vol[vol['right'] == 'p'].copy()

# Select relevant columns and set strike as index for pivoting
calls_pivot = calls[['strike'] + ['right', 'bid', 'ask', 'mid', 'bid_size', 'ask_size', 'spread', 'total_size', 'microprice', 'bid_iv', 'microprice_iv', 'ask_iv']].set_index('strike')
puts_pivot = puts[['strike'] + ['right', 'bid', 'ask', 'mid', 'bid_size', 'ask_size', 'spread', 'total_size', 'microprice', 'bid_iv', 'microprice_iv', 'ask_iv']].set_index('strike')

# Rename columns with suffixes
calls_pivot.columns = [col + '_call' for col in calls_pivot.columns]
puts_pivot.columns = [col + '_put' for col in puts_pivot.columns]

# Join on strike
vol_pivoted = calls_pivot.join(puts_pivot, how='outer').reset_index()

vol_pivoted['implied_forward'] = vol_pivoted['microprice_call'] - vol_pivoted['microprice_put'] + vol_pivoted['strike'] * DF

# Filter for valid implied forwards and plot
valid_forwards = vol_pivoted.dropna(subset=['implied_forward'])

# Find the unique strike closest to spot price
closest_idx = (valid_forwards['strike'] - SPOT_PRICE).abs().idxmin()
closest_strike = valid_forwards.loc[closest_idx, 'strike']
closest_forward = valid_forwards.loc[closest_idx, 'implied_forward']

plt.figure(figsize=(10, 6))
plt.plot(valid_forwards['strike'], valid_forwards['implied_forward'], 'b-', alpha=0.7, label='Implied Forward')
plt.axhline(y=SPOT_PRICE, color='r', linestyle='--', label='Spot Price')
plt.axhline(y=valid_forwards['implied_forward'].mean(), color='g', linestyle='-', label='Mean')
plt.axhline(y=valid_forwards['implied_forward'].median(), color='orange', linestyle='-', label='Median')

# Calculate size-weighted average implied forward
total_size = (valid_forwards['total_size_call'].fillna(0) + valid_forwards['total_size_put'].fillna(0))
weighted_avg_forward = np.average(valid_forwards['implied_forward'], weights=total_size)
plt.axhline(y=weighted_avg_forward, color='purple', linestyle='-', label=f'Size-Weighted Average ({weighted_avg_forward:.2f})')

# Add vertical line for closest strike and horizontal line for its implied forward
plt.axvline(x=closest_strike, color='cyan', linestyle=':', alpha=0.8, label=f'Closest Strike: {closest_strike}')
plt.axhline(y=closest_forward, color='cyan', linestyle=':', alpha=0.8, label=f'Forward at {closest_strike}: {closest_forward:.2f}')

plt.xlabel('Strike Price')
plt.ylabel('Implied Forward')
plt.title('Implied Forward vs Strike Price')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# Display the pivoted dataframe
vol_pivoted


# In[4]:


# Cell 2: OTM Filter and Uncertainty Computation
F: float = SPOT_PRICE * np.exp(RISK_FREE_RATE * TTE_YEARS)
vol_otm: pd.DataFrame = vol[
    ((vol["right"] == "c") & (vol["strike"] > F)) | ((vol["right"] == "p") & (vol["strike"] < F))
].copy().dropna()
print("OTM points after bid filter:", len(vol_otm))

# Price uncertainties
spread: np.ndarray = vol_otm["ask"].values - vol_otm["bid"].values
total_size: np.ndarray = vol_otm["bid_size"].values + vol_otm["ask_size"].values
microprice: np.ndarray = vol_otm["microprice"].values
bid_prices: np.ndarray = vol_otm["bid"].values
ask_prices: np.ndarray = vol_otm["ask"].values
rights: np.ndarray = vol_otm["right"].values
strikes: np.ndarray = vol_otm["strike"].values

# Beta parameters
frac_mean: np.ndarray = (microprice - bid_prices) / spread
var_target: np.ndarray = (spread**2) / (12 * total_size)
alpha_beta_sums: np.ndarray = frac_mean * (1 - frac_mean) / np.maximum(var_target, 1e-12) - 1
alpha_params: np.ndarray = np.maximum(frac_mean * alpha_beta_sums, 1e-6)
beta_params: np.ndarray = np.maximum((1 - frac_mean) * alpha_beta_sums, 1e-6)
nu_scale: float = 10.0
alpha_params *= nu_scale
beta_params *= nu_scale

vol_otm["price_tau2"] = var_target
vol_otm["vega"] = vega(
    SPOT_PRICE, strikes, TTE_YEARS, RISK_FREE_RATE, vol_otm["microprice_iv"].values
)
vol_otm["iv_tau2"] = vol_otm["price_tau2"] / (vol_otm["vega"] ** 2)
vol_otm["log_moneyness"] = np.log(strikes / F)

X: np.ndarray = vol_otm["log_moneyness"].values
y: np.ndarray = vol_otm["microprice_iv"].values
noise_var: np.ndarray = vol_otm["iv_tau2"].values.clip(min=1e-6)

atm_idx: int = np.argmin(np.abs(X))
mu: float = y[atm_idx]

# Heuristic hypers (deterministic)
pairwise_dist = np.abs(X[:, None] - X[None, :])
ls_init: float = np.median(pairwise_dist[pairwise_dist > 0])
sf2_init: float = np.var(y)

# Grid (no buffer for stability)
buffer: float = 0
grid_k: np.ndarray = np.linspace(X.min() - buffer, X.max() + buffer, 300)

# Fit Both Methods
print("Fitting Exact GP...")
exact_fit = fit_gp("exact", X, y, noise_var, ls_init, sf2_init, grid_k, RISK_FREE_RATE)
print(f"Exact runtime: {exact_fit['runtime']:.4f}s")

print("Fitting Low-Rank GP...")
lowrank_fit = fit_gp("lowrank", X, y, noise_var, ls_init, sf2_init, grid_k, RISK_FREE_RATE)
print(f"Low-Rank runtime: {lowrank_fit['runtime']:.4f}s")

# Blend Wings for Both
mu_exact = exact_fit["mean"][np.argmin(np.abs(grid_k))]
beta_skew_exact = (
    (np.max(exact_fit["mean"]) - mu_exact) / (2 * np.max(np.abs(grid_k)))
    if np.max(np.abs(grid_k)) > 0
    else 0.2
)
exact_mean = smooth_wing_blend(exact_fit["mean"], grid_k, X, mu_exact, beta_skew_exact)

mu_lowrank = lowrank_fit["mean"][np.argmin(np.abs(grid_k))]
beta_skew_lowrank = (
    (np.max(lowrank_fit["mean"]) - mu_lowrank) / (2 * np.max(np.abs(grid_k)))
    if np.max(np.abs(grid_k)) > 0
    else 0.2
)
lowrank_mean = smooth_wing_blend(lowrank_fit["mean"], grid_k, X, mu_lowrank, beta_skew_lowrank)


exact_prices, exact_rnd = derive_quantities(exact_mean, exact_fit["F_adj"], RISK_FREE_RATE)
lowrank_prices, lowrank_rnd = derive_quantities(lowrank_mean, lowrank_fit["F_adj"], RISK_FREE_RATE)


sample_u = np.linspace(-2, 2, 21)
exact_cf = char_func(sample_u, exact_rnd, grid_k)
lowrank_cf = char_func(sample_u, lowrank_rnd, grid_k)

M_gc = 8  # Tune: 5-12; higher = tighter fit but potential oscillation
mu_gc, sigma_gc, coeffs_gc = fit_gram_charlier(
    exact_rnd, grid_k, M=M_gc, baseline="normal"
)  # Try 'lognormal' for tails

# Analytical GC CF (at your sample_u)
gc_cf = gram_charlier_cf(sample_u, mu_gc, sigma_gc, coeffs_gc)

# Quick comparison at sample_u
print("GC vs Numerical CF (first 3 points):", np.real(gc_cf[:3]), "vs", np.real(exact_cf[:3]))

# Interpolate numerical CF for smooth plotting
exact_cf_interp = interp1d(
    sample_u, np.real(exact_cf), kind="linear", bounds_error=False, fill_value=0.0
)

# Plot CF approximation
plt.figure(figsize=(8, 5))
u_plot = np.linspace(-10, 10, 200)  # Finer u for smooth curves
plt.plot(u_plot, exact_cf_interp(u_plot), "k--", label="Numerical CF", linewidth=2)
plt.plot(
    u_plot,
    np.real(gram_charlier_cf(u_plot, mu_gc, sigma_gc, coeffs_gc)),
    "r-",
    label="GC Approx",
    linewidth=2,
)
plt.xlabel("u")
plt.ylabel("Re[φ(u)]")
plt.legend()
plt.title("CF Approximation")
plt.grid(True, alpha=0.3)
plt.show()


recon_rnd = gc_rnd(grid_k, mu_gc, sigma_gc, coeffs_gc, baseline="normal")
recon_rnd /= trapz(recon_rnd, grid_k)  # Renormalize after clipping
rmse_rnd = np.sqrt(np.mean((exact_rnd - recon_rnd) ** 2))
print(f"RND Reconstruction RMSE: {rmse_rnd:.4f} (aim <0.001; increase M if higher)")

# Optional: Legendre Placeholder Usage (uncomment when implementing)
# M_leg = 8
# coeffs_leg = fit_legendre(exact_rnd, grid_k, M=M_leg)
# leg_cf = legendre_cf(sample_u, coeffs_leg, grid_k.min(), grid_k.max())
# print("Legendre CF (placeholder):", leg_cf[:3])


exact_val = validate(exact_mean, exact_fit["var"], y, X, grid_k, exact_rnd)
lowrank_val = validate(lowrank_mean, lowrank_fit["var"], y, X, grid_k, lowrank_rnd)

print("\nExact Validation:")
print(f"IV RMSE: {exact_val['rmse_iv']:.4f}, RND integral: {exact_val['integral_rnd']:.4f}")
print(
    "Exact Fit successful"
    if exact_val["rmse_iv"] < 0.02 and 0.95 <= exact_val["integral_rnd"] <= 1.05
    else "Exact Fit warning"
)

print("\nLow-Rank Validation:")
print(f"IV RMSE: {lowrank_val['rmse_iv']:.4f}, RND integral: {lowrank_val['integral_rnd']:.4f}")
print(
    "Low-Rank Fit successful"
    if lowrank_val["rmse_iv"] < 0.02 and 0.95 <= lowrank_val["integral_rnd"] <= 1.05
    else "Low-Rank Fit warning"
)

# Output DFs
results_exact = pd.DataFrame(
    {
        "log_moneyness": grid_k,
        "strike": exact_fit["F_adj"] * np.exp(grid_k),
        "fitted_iv_mean": exact_mean,
        "fitted_iv_std": np.sqrt(exact_fit["var"]),
        "fitted_price": exact_prices,
        "rnd_density": exact_rnd,
    }
)

results_lowrank = pd.DataFrame(
    {
        "log_moneyness": grid_k,
        "strike": lowrank_fit["F_adj"] * np.exp(grid_k),
        "fitted_iv_mean": lowrank_mean,
        "fitted_iv_std": np.sqrt(lowrank_fit["var"]),
        "fitted_price": lowrank_prices,
        "rnd_density": lowrank_rnd,
    }
)

print("\nExact Results Summary:")
print(results_exact.describe())
print("\nLow-Rank Results Summary:")
print(results_lowrank.describe())

# Original Data
print("\nOriginal Data for Comparison:")
print(vol_otm[["strike", "right", "mid", "microprice_iv", "iv_tau2"]].head(10))

# Enhanced Plotting: Both Fits
fig = plt.figure(figsize=(14, 16))
colors = ["purple" if r == "c" else "blue" for r in vol_otm["right"]]

# IV Comparison (top left)
ax1 = plt.subplot2grid((4, 2), (0, 0), colspan=1, rowspan=1)
ax1.scatter(vol_otm["strike"], y, c=colors, label="Data (OTM)", s=30)
ax1.plot(results_exact["strike"], exact_mean, "r-", linewidth=2, label="Exact GP")
ax1.plot(results_lowrank["strike"], lowrank_mean, "g--", linewidth=2, label="Low-Rank GP")
ax1.axvline(exact_fit["F_adj"], color="k", linestyle="--", alpha=0.5, label="Forward")
ax1.set_xlabel("Strike")
ax1.set_ylabel("IV")
ax1.legend()
ax1.set_title("IV Fit Comparison")

# Error Plots (top right: subplots)
ax_err_exact = plt.subplot2grid((4, 2), (0, 1), colspan=1, rowspan=2)
interp_exact = np.interp(vol_otm["strike"], results_exact["strike"], exact_mean)
errors_exact = interp_exact - y
ax_err_exact.scatter(vol_otm["strike"], errors_exact, c=colors, s=30, alpha=0.7)
ax_err_exact.axhline(0, color="k", linestyle="--", alpha=0.5)
ax_err_exact.set_xlabel("Strike")
ax_err_exact.set_ylabel("Error")
ax_err_exact.set_title("Exact Errors")

ax_err_lowrank = plt.subplot2grid((4, 2), (2, 1), colspan=1, rowspan=1)
interp_lowrank = np.interp(vol_otm["strike"], results_lowrank["strike"], lowrank_mean)
errors_lowrank = interp_lowrank - y
ax_err_lowrank.scatter(vol_otm["strike"], errors_lowrank, c=colors, s=30, alpha=0.7)
ax_err_lowrank.axhline(0, color="k", linestyle="--", alpha=0.5)
ax_err_lowrank.set_xlabel("Strike")
ax_err_lowrank.set_ylabel("Error")
ax_err_lowrank.set_title("Low-Rank Errors")

# Price Comparison (middle)
ax2 = plt.subplot2grid((4, 2), (2, 0), colspan=2, rowspan=1)
ax2.scatter(vol_otm["strike"], vol_otm["microprice"], c=colors, label="Data Microprices", s=30)
ax2.plot(results_exact["strike"], exact_prices, "r-", linewidth=2, label="Exact Prices")
ax2.plot(results_lowrank["strike"], lowrank_prices, "g--", linewidth=2, label="Low-Rank Prices")
ax2.axvline(exact_fit["F_adj"], color="k", linestyle="--", alpha=0.5)
ax2.set_ylim(
    0, max(vol_otm["microprice"].max(), max(exact_prices.max(), lowrank_prices.max())) * 1.1
)
ax2.set_xlabel("Strike")
ax2.set_ylabel("Price")
ax2.legend()

# RND Comparison (bottom)
ax3 = plt.subplot2grid((4, 2), (3, 0), colspan=1, rowspan=1)
ax3.bar(grid_k, exact_rnd, width=(grid_k[1] - grid_k[0]), alpha=0.5, color="red", label="Exact RND")
ax3.bar(
    grid_k,
    lowrank_rnd,
    width=(grid_k[1] - grid_k[0]),
    alpha=0.5,
    color="green",
    label="Low-Rank RND",
)


ax3.axvline(0, color="k", linestyle="--", alpha=0.5, label="ATM")
ax3.set_xlabel("Log-Moneyness")
ax3.set_ylabel("RND")
ax3.legend()

# Runtime Bar (extra subplot)
ax_time = plt.subplot2grid((4, 2), (3, 1), colspan=1, rowspan=1)
methods = ["Exact", "Low-Rank"]
runtimes = [exact_fit["runtime"], lowrank_fit["runtime"]]
ax_time.bar(methods, runtimes, color=["red", "green"])
ax_time.set_ylabel("Runtime (s)")
ax_time.set_title("Fit Runtimes")

plt.tight_layout()
plt.show()

# CF Samples
print("\nExact CF at u=0:", exact_cf[len(sample_u) // 2])
print("Low-Rank CF at u=0:", lowrank_cf[len(sample_u) // 2])


# In[102]:


plt.scatter(vol_otm['strike'], vol_otm['microprice_iv'])


# In[5]:


vol


# In[8]:


# Join calls and puts horizontally, then sort by strike
calls = vol[vol["right"] == "c"].copy().set_index("strike").drop(columns=["right"]).add_suffix("_c")
puts = vol[vol["right"] == "p"].copy().set_index("strike").drop(columns=["right"]).add_suffix("_p")
chain = calls.join(puts, how="outer", lsuffix="_c", rsuffix="_p")
chain


# In[ ]:


print(chain)


# In[ ]:


fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))

# Combined Calls and Puts envelopes
ax1.fill_between(
    chain.index,
    chain["bid_iv_c"],
    chain["ask_iv_c"],
    alpha=0.3,
    color="blue",
    label="Calls Bid-Ask",
)
ax1.fill_between(
    chain.index, chain["bid_iv_p"], chain["ask_iv_p"], alpha=0.3, color="red", label="Puts Bid-Ask"
)
ax1.plot(chain.index, chain["microprice_iv_c"], color="blue", linewidth=2, label="Calls Microprice")
ax1.plot(chain.index, chain["microprice_iv_p"], color="red", linewidth=2, label="Puts Microprice")
ax1.set_title("IV Bid-Ask Envelopes and Microprices: Calls vs Puts")
ax1.set_ylabel("IV")
ax1.legend()

# Combined microprice comparison
ax2.plot(chain.index, chain["microprice_iv_c"], color="blue", linewidth=2, label="Calls Microprice")
ax2.plot(chain.index, chain["microprice_iv_p"], color="red", linewidth=2, label="Puts Microprice")
ax2.set_title("Microprice IV: Calls vs Puts")
ax2.set_ylabel("IV")
ax2.legend()

# Calls-only view (for reference)
ax3.fill_between(
    chain.index,
    chain["bid_iv_c"],
    chain["ask_iv_c"],
    alpha=0.3,
    color="blue",
    label="Calls Bid-Ask",
)
ax3.plot(chain.index, chain["microprice_iv_c"], color="blue", linewidth=2, label="Calls Microprice")
ax3.set_title("Calls IV: Bid-Ask Envelope and Microprice")
ax3.set_ylabel("IV")
ax3.set_xlabel("Strike")
ax3.legend()

plt.tight_layout()
plt.show()


# In[ ]:


s


# In[ ]:




