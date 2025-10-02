"""Ultra-optimized RND extraction with Cython and incremental updates."""

import numpy as np
import pandas as pd  # type: ignore[import-not-found]
from py_vollib_vectorized.implied_volatility import vectorized_implied_volatility as viv  # type: ignore[import-not-found]
from scipy.linalg import cho_factor, cho_solve  # type: ignore[import-not-found,unused-ignore]
from scipy.stats import norm  # type: ignore[import-not-found,unused-ignore]
from typing import Optional, Tuple

from .types import MarketData, RNDResult
from .incremental import (
    GPState,
    fit_gp_initial,
    update_gp_fast,
    update_gp_uniform_shift,
)

# Try to import Cython kernels, fall back to Numba if not available
try:
    from cy_kernels import (  # type: ignore[import-not-found]
        rbf_kernel_cy,
        trapz_cy,
        cumtrapz_cy,
        compute_char_func_cy,
        gradient_cy,
    )

    CYTHON_AVAILABLE = True
except ImportError:
    CYTHON_AVAILABLE = False
    from .utils import (
        rbf_kernel_fast,
        trapz_fast,
        cumulative_trapz_fast,
        compute_char_func,
    )


def prepare_data_minimal(
    market_data: MarketData, min_bid: float = 0.05
) -> pd.DataFrame:
    """Minimal data prep - compute only what's needed.

    OPTIMIZED: Uses numpy arrays directly to avoid pandas overhead.

    Args:
        market_data: Market data containing options
        min_bid: Minimum bid price filter (default: 0.05)
    """
    df = market_data.options_df

    # Extract to numpy arrays first (faster than repeated .loc[] calls)
    strikes_all = df["strike"].values
    bids_all = df["bid"].values
    asks_all = df["ask"].values
    bid_sizes_all = df["bid_size"].values
    ask_sizes_all = df["ask_size"].values
    rights_all = df["right"].values

    # Filter by min_bid using numpy boolean indexing (faster than pandas)
    mask = bids_all > min_bid
    strikes = strikes_all[mask]
    bids = bids_all[mask]
    asks = asks_all[mask]
    bid_sizes = bid_sizes_all[mask]
    ask_sizes = ask_sizes_all[mask]
    rights = rights_all[mask]

    # Compute microprice in numpy (faster)
    microprices = (ask_sizes * bids + bid_sizes * asks) / (ask_sizes + bid_sizes)

    # Convert rights to lowercase - vectorized numpy operation
    # Assumes rights are single chars 'C'/'P' or 'c'/'p'
    rights_lower = (
        np.char.lower(rights.astype(str))
        if rights.dtype.kind in ("U", "S", "O")
        else rights
    )

    # Vectorized IV computation (already fast after warmup)
    ivs = viv(
        price=microprices,
        S=market_data.spot_price,
        K=strikes,
        t=market_data.time_to_expiry,
        r=market_data.risk_free_rate,
        flag=rights_lower,
        model="black_scholes",
        return_as="numpy",
    )

    # Return DataFrame (needed downstream, but minimize overhead)
    return pd.DataFrame(
        {
            "strike": strikes,
            "right": rights_lower,
            "iv": ivs,
            "bid": bids,
            "ask": asks,
            "bid_size": bid_sizes,
            "ask_size": ask_sizes,
        },
        copy=False,
    )  # copy=False to avoid unnecessary data copying


def vega_fast(
    S: float, K: np.ndarray, t: float, r: float, sigma: np.ndarray
) -> np.ndarray:
    """Fast vega computation."""
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * t) / (sigma * np.sqrt(t) + 1e-8)
    return S * np.sqrt(t) * norm.pdf(d1)


def extract_rnd_ultra(
    market_data: MarketData,
    grid_points: int = 300,
    prev_state: Optional[GPState] = None,
    min_bid: float = 0.05,
    use_inducing_points: bool = False,
    n_inducing: Optional[int] = None,
) -> Tuple[RNDResult, GPState]:
    """Ultra-fast RND extraction with optional incremental update.

    Args:
        market_data: Market data
        grid_points: Number of grid points
        prev_state: Previous GP state for incremental update
        min_bid: Minimum bid price filter (default: 0.05)
        use_inducing_points: Use low-rank GP with inducing points for speed (default: False)
        n_inducing: Number of inducing points (default: n_options//4, minimum 10)

    Returns:
        (result, gp_state) - Result and cached state for next update

    Note:
        use_inducing_points=True gives ~64x speedup on GP for large option chains (>200 options)
        at the cost of slight accuracy reduction. Recommended for high-frequency streaming.
    """
    # Data preparation
    df = prepare_data_minimal(market_data, min_bid=min_bid)
    forward = market_data.spot_price * np.exp(
        market_data.risk_free_rate * market_data.time_to_expiry
    )

    # Filter OTM - OPTIMIZED: Use numpy arrays directly (faster than pandas)
    strikes = df["strike"].values
    rights = df["right"].values
    ivs = df["iv"].values
    bids = df["bid"].values
    asks = df["ask"].values
    bid_sizes = df["bid_size"].values
    ask_sizes = df["ask_size"].values

    # OTM mask using numpy (faster than pandas boolean indexing)
    otm_mask = ((rights == "c") & (strikes > forward)) | (
        (rights == "p") & (strikes < forward)
    )

    # Apply OTM filter to all arrays
    strikes_otm = strikes[otm_mask]
    ivs_otm = ivs[otm_mask]
    bids_otm = bids[otm_mask]
    asks_otm = asks[otm_mask]
    bid_sizes_otm = bid_sizes[otm_mask]
    ask_sizes_otm = ask_sizes[otm_mask]

    # Compute uncertainties (all numpy operations - fast)
    spread = asks_otm - bids_otm
    total_size = bid_sizes_otm + ask_sizes_otm
    var_price = (spread**2) / (12 * total_size)
    vega_vals = vega_fast(
        market_data.spot_price,
        strikes_otm,
        market_data.time_to_expiry,
        market_data.risk_free_rate,
        ivs_otm,
    )
    noise_var = np.clip(var_price / (vega_vals**2 + 1e-8), 1e-6, None)

    X = np.log(strikes_otm / forward)
    y = ivs_otm

    # Check if we can do incremental update
    if prev_state is not None:
        if len(X) == len(prev_state.X):
            # Case A: identical X
            if np.array_equal(X, prev_state.X):
                return update_gp_fast(y, prev_state), prev_state

            # Case B: exact uniform shift in log-moneyness due to forward change
            # For stationary kernels, if X_new = X_old - delta elementwise (same delta),
            # we can update exactly by shifting state and reusing K_chol/k_star.
            diffs = X - prev_state.X
            if np.allclose(diffs, diffs[0], rtol=0.0, atol=1e-12):
                # Determine implied forward change; X = log(K/F), so delta = log(F_old/F_new)
                # Use numeric relation: X_new = X_old - delta => delta = X_old[0]-X_new[0]
                # We already computed the current forward above; use it directly
                forward_new = forward
                result_shift, shifted_state = update_gp_uniform_shift(
                    y, prev_state, forward_new
                )
                return result_shift, shifted_state

    # Full path: initial fit or structure changed
    pairwise_dist = np.abs(X[:, None] - X[None, :])
    ls_init = np.median(pairwise_dist[pairwise_dist > 0])
    sf2_init = max(np.var(y), 1e-4)  # Ensure non-zero variance for constant IV case

    grid_k = np.linspace(X.min(), X.max(), grid_points)

    result, state = fit_gp_initial(
        X,
        y,
        noise_var,
        grid_k,
        ls_init,
        sf2_init,
        market_data.spot_price,
        forward,
        market_data.risk_free_rate,
        market_data.time_to_expiry,
        use_lowrank=use_inducing_points,
        n_inducing=n_inducing,
    )

    return result, state


def extract_rnd_ultra_simple(
    strikes: np.ndarray,
    rights: np.ndarray,
    bids: np.ndarray,
    asks: np.ndarray,
    bid_sizes: np.ndarray,
    ask_sizes: np.ndarray,
    spot_price: float,
    risk_free_rate: float,
    time_to_expiry: float,
    grid_points: int = 300,
    min_bid: float = 0.05,
    use_lowrank: bool = True,
) -> RNDResult:
    """Ultra-fast RND extraction with zero-copy numpy interface.

    Optimized for real-time streaming and CSP nodes. Pass numpy arrays directly
    for maximum performance (no pandas overhead).

    Args:
        strikes: Strike prices (numpy array)
        rights: Option types - 'c'/'C' for calls, 'p'/'P' for puts (numpy array)
        bids: Bid prices (numpy array)
        asks: Ask prices (numpy array)
        bid_sizes: Bid sizes (numpy array)
        ask_sizes: Ask sizes (numpy array)
        spot_price: Current underlying price
        risk_free_rate: Annualized risk-free rate
        time_to_expiry: Time to expiry in years
        grid_points: Number of grid points for RND (default: 300)
        min_bid: Minimum bid filter (default: 0.05)
        use_lowrank: Use low-rank GP for speed (default: True, recommended)

    Returns:
        RNDResult with extracted risk-neutral density

    Example (Real-Time CSP):
        >>> result = extract_rnd_ultra_simple(
        ...     strikes_arr, rights_arr, bids_arr, asks_arr,
        ...     bid_sizes_arr, ask_sizes_arr,
        ...     spot=223.46, risk_free_rate=0.05341, time_to_expiry=0.0274,
        ...     use_lowrank=True
        ... )
        >>> rnd = result.rnd_density
        >>> strikes_grid = result.strikes

    Performance:
        ~5-7ms per extraction (after viv JIT warmup) with use_lowrank=True
        First call in session: ~90ms (one-time JIT compilation)
        Subsequent calls: ~5-7ms consistently
    """
    # Create minimal pandas DataFrame (zero-copy wrapper)
    df = pd.DataFrame(
        {
            "strike": strikes,
            "right": rights,
            "bid": bids,
            "ask": asks,
            "bid_size": bid_sizes,
            "ask_size": ask_sizes,
        },
        copy=False,
    )

    market_data = MarketData(
        spot_price=spot_price,
        risk_free_rate=risk_free_rate,
        time_to_expiry=time_to_expiry,
        options_df=df,
    )

    result, _ = extract_rnd_ultra(
        market_data,
        grid_points,
        prev_state=None,
        min_bid=min_bid,
        use_inducing_points=use_lowrank,
        n_inducing=None,
    )
    return result
