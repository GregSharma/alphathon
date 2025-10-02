"""
CSP Node wrappers for Risk-Neutral Density (RND) extraction.

Integrates the rnd_extraction package with CSP for real-time streaming.
Supports incremental CBRA optimization with csp.state() caching for 10-20x speedup.
"""

import sys
from pathlib import Path

# Add rnd_extraction to path
_RND_PATH = Path(__file__).parent.parent / "rnd_extraction" / "src"
if str(_RND_PATH) not in sys.path:
    sys.path.insert(0, str(_RND_PATH))

# Add mv_rnd to path for CBRA
_CBRA_PATH = Path(__file__).parent.parent / "mv_rnd"
if str(_CBRA_PATH) not in sys.path:
    sys.path.insert(0, str(_CBRA_PATH))

import numpy as np

# Import CBRA for multivariate distribution
from cbrapipe import (
    CBRAState,
    cbra_optimize_stateful,
    cbra_update_incremental,
    detect_marginal_changes,
    extract_joint_distribution,
)

import csp
from csp import ts
from CSP_Options.structs import VectorizedOptionQuote

# Import RND extraction
from rnd_extraction import RNDResult, extract_rnd_ultra_simple


@csp.node
def extract_rnd_from_vq(
    vq: ts[VectorizedOptionQuote],
    spot_price: ts[float],
    risk_free_rate: float = 0.05431,
    grid_points: int = 300,
    min_bid: float = 0.05,
) -> ts[RNDResult]:
    """
    Extract risk-neutral density from VectorizedOptionQuote.

    This node takes a vectorized option quote (from quote_list_to_vector)
    and extracts the continuous risk-neutral probability density.

    Args:
        vq: VectorizedOptionQuote with strikes, IVs, etc.
        spot_price: Current underlying price (from impl_spot_price output)
        risk_free_rate: Annualized risk-free rate (default: 0.05431)
        grid_points: Number of grid points for RND (default: 300)
        min_bid: Minimum bid filter (default: 0.05)

    Returns:
        ts[RNDResult]: Stream of RND results with:
            - rnd_density: Continuous probability density
            - rnd_cumulative: Cumulative distribution function
            - fitted_iv: GP-smoothed IV curve
            - strikes: Strike grid
            - forward_price: Computed forward price

    Example Usage:
        >>> # In a graph after filtering quotes
        >>> rnd_result = extract_rnd_from_vq(
        ...     vq=vec_quotes,
        ...     spot_price=impl_spot_price,
        ...     grid_points=300
        ... )
        >>> csp.print("RND", rnd_result.rnd_density)
    """
    if csp.ticked(vq) and csp.valid(spot_price):
        # Check if we have valid data (non-empty arrays)
        if len(vq.strike) == 0 or len(vq.bid) == 0:
            # No valid quotes - skip this tick (don't return anything)
            return

        try:
            # Convert VectorizedOptionQuote to numpy arrays for RND extraction
            result = extract_rnd_ultra_simple(
                strikes=vq.strike,
                rights=vq.right,
                bids=vq.bid,
                asks=vq.ask,
                bid_sizes=vq.bid_size,
                ask_sizes=vq.ask_size,
                spot_price=spot_price,
                risk_free_rate=risk_free_rate,
                time_to_expiry=vq.tte,
                grid_points=grid_points,
                min_bid=min_bid,
                use_lowrank=True,  # Fast incremental updates
            )
            return result
        except (ValueError, RuntimeError) as e:
            # RND extraction failed (e.g., insufficient data, bad fit)
            # Skip this tick - downstream nodes will continue with last valid value
            return


@csp.node
def extract_rnd_features(rnd_result: ts[RNDResult]) -> csp.Outputs(
    mean=ts[float],
    std=ts[float],
    skew=ts[float],
    kurt=ts[float],
):
    """
    Compute statistical moments from the RND.

    Extracts mean, standard deviation, skewness, and kurtosis from the
    risk-neutral density for use in downstream models.

    Args:
        rnd_result: RND extraction result

    Returns:
        mean: Expected value under RND
        std: Standard deviation under RND
        skew: Skewness under RND
        kurt: Kurtosis under RND
    """
    if csp.ticked(rnd_result):
        strikes = rnd_result.strikes
        density = rnd_result.rnd_density

        # Ensure density is normalized
        density = density / np.trapz(density, strikes)

        # Compute moments
        mean_val = np.trapz(strikes * density, strikes)

        # Centered moments
        variance = np.trapz((strikes - mean_val) ** 2 * density, strikes)
        std_val = np.sqrt(variance)

        # Standardized moments
        m3 = np.trapz((strikes - mean_val) ** 3 * density, strikes)
        skew_val = m3 / (std_val**3) if std_val > 0 else 0.0

        m4 = np.trapz((strikes - mean_val) ** 4 * density, strikes)
        kurt_val = m4 / (std_val**4) - 3.0 if std_val > 0 else 0.0  # Excess kurtosis

        return csp.output(mean=mean_val, std=std_val, skew=skew_val, kurt=kurt_val)


def _rnd_to_inverse_cdf(rnd_result):
    """Convert RND result to inverse CDF function."""
    strikes = rnd_result.strikes
    cdf = rnd_result.rnd_cumulative
    return lambda p: np.interp(p, cdf, strikes)


@csp.node
def compute_joint_rnd_incremental(
    rnd_stocks: ts[list[RNDResult]],  # List of RND results for stocks
    rnd_indices: ts[list[RNDResult]],  # List of RND results for indices
    weight_matrix: np.ndarray,  # A: (d, K) weight matrix
    n_states: int = 2000,
    max_iter_cold: int = 5000,
    max_iter_warm: int = 50,
    change_threshold: float = 0.01,
) -> ts[CBRAState]:
    """
    Compute joint multivariate distribution using incremental CBRA.

    This node caches the CBRA optimization state using csp.state() and performs
    incremental updates when RNDs change slightly. This gives 10-20x speedup
    compared to cold-start optimization every tick.

    Key insight from benchmark_incremental.py:
    - First tick: Cold start with full optimization (~5s for d=10, K=3)
    - Subsequent ticks: Warm start from cached state (~0.2s for same setup)
    - Average speedup: 10-20x for typical minute-by-minute updates

    Args:
        rnd_stocks: List of RND results for d stocks
        rnd_indices: List of RND results for K indices
        weight_matrix: (d, K) matrix defining index = A @ stocks constraints
        n_states: Number of equiprobable states (default: 2000)
        max_iter_cold: Max iterations for cold start (default: 5000)
        max_iter_warm: Max iterations for warm start (default: 50)
        change_threshold: Threshold for auto-detecting marginal changes (default: 0.01)

    Returns:
        ts[CBRAState]: Stream of CBRA states containing:
            - Y_optimized: (n, d+K) joint realization matrix
            - tilde_alpha: Coefficient matrix
            - blocks: Admissible blocks for optimization
            - X_marginals, S_marginals: Cached marginals for change detection

    Example Usage:
        >>> # After extracting RNDs for multiple stocks and indices
        >>> A = np.array([[0.5, 0.3], [0.3, 0.4], [0.2, 0.3]])  # 3 stocks, 2 indices
        >>> cbra_state = compute_joint_rnd_incremental(
        ...     rnd_stocks=[rnd_aapl, rnd_msft, rnd_googl],
        ...     rnd_indices=[rnd_spy, rnd_qqq],
        ...     weight_matrix=A,
        ...     n_states=2000
        ... )
        >>> # Extract joint samples
        >>> joint_dist = extract_joint_distribution(cbra_state.Y_optimized)
    """
    with csp.state():
        s_cbra_state = None  # Cached CBRA state for warm starts

    if csp.ticked(rnd_stocks) and csp.ticked(rnd_indices):
        # Validate inputs
        if len(rnd_stocks) == 0 or len(rnd_indices) == 0:
            # No valid RNDs - skip this tick
            return

        try:
            # Convert RND results to inverse CDFs (F_inv functions)
            F_inv_stocks = [_rnd_to_inverse_cdf(rnd) for rnd in rnd_stocks]
            F_inv_indices = [_rnd_to_inverse_cdf(rnd) for rnd in rnd_indices]

            # Check if this is first tick (cold start) or warm start
            if s_cbra_state is None:
                # COLD START: Full optimization from scratch
                s_cbra_state = cbra_optimize_stateful(
                    n=n_states,
                    F_inv_stocks=F_inv_stocks,
                    F_inv_indices=F_inv_indices,
                    A=weight_matrix,
                    max_iter=max_iter_cold,
                    verbose=False,
                )
            else:
                # WARM START: Incremental update from cached state
                # Auto-detect which marginals changed
                changed_stocks, changed_indices = detect_marginal_changes(
                    s_cbra_state, F_inv_stocks, F_inv_indices, threshold=change_threshold
                )

                # Only run incremental update if something changed
                if len(changed_stocks) > 0 or len(changed_indices) > 0:
                    s_cbra_state = cbra_update_incremental(
                        s_cbra_state,
                        F_inv_stocks_new=F_inv_stocks,
                        F_inv_indices_new=F_inv_indices,
                        changed_stock_indices=changed_stocks,
                        changed_index_indices=changed_indices,
                        max_iter=max_iter_warm,
                        verbose=False,
                    )

            return s_cbra_state
        except (ValueError, RuntimeError) as e:
            # CBRA optimization failed - return cached state if available
            # Otherwise skip this tick
            if s_cbra_state is not None:
                return s_cbra_state
            else:
                return


@csp.node
def extract_joint_samples(
    cbra_state: ts[CBRAState],
) -> ts[dict]:
    """
    Extract the joint distribution matrix from CBRA state.

    The Y_optimized matrix represents n equiprobable states of the joint distribution.
    Each row is a joint realization occurring with probability 1/n.

    Args:
        cbra_state: CBRA optimization state

    Returns:
        ts[dict]: Dictionary with:
            - 'stocks': (n, d) array of stock realizations (equiprobable states)
            - 'indices': (n, K) array of index realizations
            - 'correlations': (d+K, d+K) correlation matrix
            - 'n_states': Number of equiprobable states
    """
    if csp.ticked(cbra_state):
        # Extract joint distribution (first d columns = stocks)
        stocks = extract_joint_distribution(cbra_state.Y_optimized, d=cbra_state.d)

        # Indices are remaining columns
        indices = cbra_state.Y_optimized[:, cbra_state.d :]

        # Compute correlation matrix
        full_matrix = cbra_state.Y_optimized
        correlations = np.corrcoef(full_matrix, rowvar=False)

        return {
            "stocks": stocks,
            "indices": indices,
            "correlations": correlations,
            "n_states": cbra_state.n,
        }


# =============================================================================
# USAGE EXAMPLE: Incremental Joint RND with CSP State Caching
# =============================================================================
"""
Example: Real-time joint distribution for 3 stocks + 2 indices

import csp
import numpy as np
from CSP_Options.rnd_nodes import (
    extract_rnd_from_vq, 
    compute_joint_rnd_incremental,
    extract_joint_samples
)

@csp.graph
def joint_rnd_graph():
    # Assume we have vectorized quotes for 3 stocks and 2 indices
    vq_aapl = ...  # ts[VectorizedOptionQuote]
    vq_msft = ...
    vq_googl = ...
    vq_spy = ...
    vq_qqq = ...
    
    spot_aapl = ...  # ts[float]
    spot_msft = ...
    spot_googl = ...
    spot_spy = ...
    spot_qqq = ...
    
    # Extract individual RNDs
    rnd_aapl = extract_rnd_from_vq(vq_aapl, spot_aapl)
    rnd_msft = extract_rnd_from_vq(vq_msft, spot_msft)
    rnd_googl = extract_rnd_from_vq(vq_googl, spot_googl)
    rnd_spy = extract_rnd_from_vq(vq_spy, spot_spy)
    rnd_qqq = extract_rnd_from_vq(vq_qqq, spot_qqq)
    
    # Collect into lists (using csp.collect or manual basket)
    rnd_stocks = csp.flatten([rnd_aapl, rnd_msft, rnd_googl])
    rnd_indices = csp.flatten([rnd_spy, rnd_qqq])
    
    # Define weight matrix: SPY = 0.4*AAPL + 0.3*MSFT + 0.3*GOOGL
    #                        QQQ = 0.3*AAPL + 0.5*MSFT + 0.2*GOOGL
    A = np.array([
        [0.4, 0.3],  # AAPL weights in [SPY, QQQ]
        [0.3, 0.5],  # MSFT weights
        [0.3, 0.2],  # GOOGL weights
    ])
    
    # Compute joint distribution with incremental updates (10-20x faster!)
    cbra_state = compute_joint_rnd_incremental(
        rnd_stocks=rnd_stocks,
        rnd_indices=rnd_indices,
        weight_matrix=A,
        n_states=2000,
        max_iter_cold=5000,  # First tick: ~5s
        max_iter_warm=50,    # Subsequent ticks: ~0.2s (25x speedup!)
    )
    
    # Extract joint samples for pricing/analysis
    joint_samples = extract_joint_samples(cbra_state)
    
    # Use the samples to price multivariate derivatives
    # joint_samples['stocks'] = (n, 3) array of [AAPL, MSFT, GOOGL] realizations
    # joint_samples['indices'] = (n, 2) array of [SPY, QQQ] realizations
    # joint_samples['correlations'] = (5, 5) correlation matrix
    
    csp.print("joint_correlation", joint_samples['correlations'])
    
    return cbra_state

# Run the graph
if __name__ == "__main__":
    csp.run(joint_rnd_graph, starttime=..., endtime=...)
"""
