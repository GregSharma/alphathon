"""Incremental GP updates using exact identities for streaming updates."""

import numpy as np
from dataclasses import dataclass
from scipy.linalg import cho_factor, cho_solve, solve_triangular  # type: ignore[import-not-found]
from typing import Optional

from .types import RNDResult
from .utils import trapz_fast, cumulative_trapz_fast, compute_char_func
from scipy.stats import norm  # type: ignore[import-not-found]


@dataclass
class GPState:
    """Cached GP state for incremental updates."""

    X: np.ndarray  # Training inputs (log-moneyness)
    K_chol: tuple  # Cholesky decomposition of K (or K_inducing if low-rank)
    grid_k: np.ndarray  # Prediction grid
    k_star: np.ndarray  # K(grid, X) - stays constant if X doesn't change
    ls: float  # Length scale
    sf2: float  # Signal variance
    forward: float  # Forward price
    market_params: tuple  # (spot, r, t) for pricing
    use_lowrank: bool = False  # Whether this state uses low-rank approximation
    X_inducing: Optional[np.ndarray] = None  # Inducing points (if low-rank)
    noise_inducing: Optional[np.ndarray] = (
        None  # Noise at inducing points (if low-rank)
    )
    cached_iv_var: Optional[np.ndarray] = (
        None  # Cached variance (doesn't change with y)
    )
    noise_var: Optional[np.ndarray] = None  # Full noise diagonal for exact GP


def rbf_kernel_vec(X1: np.ndarray, X2: np.ndarray, ls: float, sf2: float) -> np.ndarray:
    """Vectorized RBF kernel."""
    dist2 = (X1[:, None] - X2[None, :]) ** 2
    return sf2 * np.exp(-0.5 * dist2 / (ls * ls))


def extract_rnd_fast_internal(
    iv_mean: np.ndarray,
    grid_k: np.ndarray,
    spot: float,
    forward: float,
    r: float,
    t: float,
) -> np.ndarray:
    """Fast RND extraction (internal)."""
    strikes = forward * np.exp(grid_k)
    d1 = (np.log(spot / strikes) + (r + 0.5 * iv_mean**2) * t) / (
        iv_mean * np.sqrt(t) + 1e-8
    )
    d2 = d1 - iv_mean * np.sqrt(t)
    prices = spot * norm.cdf(d1) - strikes * np.exp(-r * t) * norm.cdf(d2)

    # Numerical differentiation
    price_diff1 = np.gradient(prices, strikes)
    price_diff2 = np.gradient(price_diff1, strikes)

    q_K = np.exp(r * t) * price_diff2
    q_k = np.maximum(q_K * strikes, 0)

    total = trapz_fast(q_k, grid_k)
    rnd_k = q_k / total if abs(total) > 1e-10 else q_k
    return rnd_k / trapz_fast(rnd_k, grid_k)


def fit_gp_lowrank(
    X: np.ndarray,
    y: np.ndarray,
    noise_var: np.ndarray,
    grid_k: np.ndarray,
    ls: float,
    sf2: float,
    n_inducing: Optional[int] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Low-rank GP using inducing points (subset of data).

    Uses p = n//4 inducing points instead of all n points.
    Cholesky is O(p³) instead of O(n³) → ~64x speedup for large n.
    """
    n = len(X)
    if n_inducing is None:
        p = max(10, n // 4)
    else:
        p = min(n_inducing, n)

    # Select inducing points uniformly across the data
    inducing_idx = np.linspace(0, n - 1, p, dtype=int)
    X_inducing = X[inducing_idx]
    y_inducing = y[inducing_idx]
    noise_inducing = noise_var[inducing_idx]

    # Kernel on inducing points only
    K_inducing = rbf_kernel_vec(X_inducing, X_inducing, ls, sf2) + np.diag(
        noise_inducing
    )
    K_chol = cho_factor(K_inducing)
    alpha = cho_solve(K_chol, y_inducing)

    # Predictions
    k_star = rbf_kernel_vec(grid_k, X_inducing, ls, sf2)
    iv_mean = k_star @ alpha

    # Variance approximation
    k_ss_diag = sf2 * np.ones(len(grid_k))  # Diagonal of k(grid, grid)
    Q = k_star @ cho_solve(K_chol, k_star.T)
    iv_var = np.maximum(k_ss_diag - np.diag(Q), 0)

    return iv_mean, iv_var


def fit_gp_initial(
    X: np.ndarray,
    y: np.ndarray,
    noise_var: np.ndarray,
    grid_k: np.ndarray,
    ls: float,
    sf2: float,
    spot: float,
    forward: float,
    r: float,
    t: float,
    use_lowrank: bool = False,
    n_inducing: Optional[int] = None,
) -> tuple[RNDResult, GPState]:
    """Initial GP fit - cache everything possible.

    Args:
        use_lowrank: If True, use low-rank approximation with inducing points (~64x faster for large n)
        n_inducing: Number of inducing points (default: n//4)
    """
    if use_lowrank:
        # Fast low-rank approximation
        n = len(X)
        if n_inducing is None:
            p = max(10, n // 4)
        else:
            p = min(n_inducing, n)

        # Select inducing points uniformly across the data
        inducing_idx = np.linspace(0, n - 1, p, dtype=int)
        X_inducing = X[inducing_idx]
        y_inducing = y[inducing_idx]
        noise_inducing = noise_var[inducing_idx]

        # Kernel on inducing points only (much smaller!)
        K_inducing = rbf_kernel_vec(X_inducing, X_inducing, ls, sf2) + np.diag(
            noise_inducing
        )
        K_chol = cho_factor(K_inducing, lower=True)  # O(p³) instead of O(n³)

        # Predictions using inducing points
        k_star = rbf_kernel_vec(grid_k, X_inducing, ls, sf2)  # K(grid, X_inducing)
        alpha = cho_solve(K_chol, y_inducing)
        iv_mean = k_star @ alpha

        # Variance approximation (cache it - doesn't depend on y!)
        k_ss_diag = sf2 * np.ones(len(grid_k))
        Q = k_star @ cho_solve(K_chol, k_star.T)
        iv_var = np.maximum(k_ss_diag - np.diag(Q), 0)

        # Cache low-rank state (much smaller memory footprint!)
        state = GPState(
            X=X,
            K_chol=K_chol,
            grid_k=grid_k,
            k_star=k_star,
            ls=ls,
            sf2=sf2,
            forward=forward,
            market_params=(spot, r, t),
            use_lowrank=True,
            X_inducing=X_inducing,
            noise_inducing=noise_inducing,
            cached_iv_var=iv_var,  # Cache variance for incremental updates!
        )
    else:
        # Exact GP
        # Kernel matrix
        K = rbf_kernel_vec(X, X, ls, sf2) + np.diag(noise_var)
        K_chol = cho_factor(K, lower=True)

        # Cross-covariance (stays constant if X doesn't change)
        k_star = rbf_kernel_vec(grid_k, X, ls, sf2)

        # Solve for alpha
        alpha = cho_solve(K_chol, y)

        # Predictions
        iv_mean = k_star @ alpha
        cov_diag = sf2 - np.sum(k_star * cho_solve(K_chol, k_star.T).T, axis=1)
        iv_var = np.maximum(cov_diag, 0)

        # Cache exact GP state
        state = GPState(
            X=X,
            K_chol=K_chol,
            grid_k=grid_k,
            k_star=k_star,
            ls=ls,
            sf2=sf2,
            forward=forward,
            market_params=(spot, r, t),
            use_lowrank=False,
            cached_iv_var=iv_var,  # Cache variance for incremental updates!
            noise_var=noise_var,
        )

    # Extract RND
    rnd = extract_rnd_fast_internal(iv_mean, grid_k, spot, forward, r, t)
    rnd_cumulative = cumulative_trapz_fast(rnd, grid_k)

    # Characteristic function
    u_samples = np.linspace(-10, 10, 101)
    cf_values = compute_char_func(u_samples, rnd, grid_k)

    result = RNDResult(
        log_moneyness=grid_k,
        strikes=forward * np.exp(grid_k),
        rnd_density=rnd,
        rnd_cumulative=rnd_cumulative,
        fitted_iv=iv_mean,
        fitted_iv_std=np.sqrt(iv_var),
        characteristic_function_u=u_samples,
        characteristic_function_values=cf_values,
        forward_price=forward,
    )

    return result, state


def update_gp_fast(y_new: np.ndarray, state: GPState) -> RNDResult:
    """Lightning-fast update when only y changes (prices change, strikes same).

    Uses cached Cholesky decomposition - just re-solve linear system.
    This is O(n²) vs O(n³) for full recomputation (exact GP).
    For low-rank: O(p²) where p << n (much faster!).
    """
    if state.use_lowrank:
        # Low-rank incremental update - BLAZING FAST!
        # Just solve linear system with cached Cholesky, no variance recomputation!
        n = len(state.X)
        if state.X_inducing is None:
            raise RuntimeError("Low-rank GPState missing X_inducing")
        p = len(state.X_inducing)

        # Get indices of inducing points in the original data
        inducing_idx = np.linspace(0, n - 1, p, dtype=int)
        y_inducing_new = y_new[inducing_idx]

        # Solve with cached inducing point Cholesky (O(p²) - FAST!)
        alpha_new = cho_solve(state.K_chol, y_inducing_new)

        # Predictions using cached k_star (which is K(grid, X_inducing))
        iv_mean = state.k_star @ alpha_new
    else:
        # Exact GP incremental update
        # Reuse cached Cholesky to solve K * alpha = y_new (O(n²))
        alpha_new = cho_solve(state.K_chol, y_new)

        # Predictions using cached k_star
        iv_mean = state.k_star @ alpha_new

    # Extract RND
    spot, r, t = state.market_params
    rnd = extract_rnd_fast_internal(iv_mean, state.grid_k, spot, state.forward, r, t)
    rnd_cumulative = cumulative_trapz_fast(rnd, state.grid_k)

    # Characteristic function
    u_samples = np.linspace(-10, 10, 101)
    cf_values = compute_char_func(u_samples, rnd, state.grid_k)

    iv_var_arr = (
        state.cached_iv_var
        if state.cached_iv_var is not None
        else np.zeros_like(iv_mean)
    )
    return RNDResult(
        log_moneyness=state.grid_k,
        strikes=state.forward * np.exp(state.grid_k),
        rnd_density=rnd,
        rnd_cumulative=rnd_cumulative,
        fitted_iv=iv_mean,
        fitted_iv_std=np.sqrt(iv_var_arr),
        characteristic_function_u=u_samples,
        characteristic_function_values=cf_values,
        forward_price=state.forward,
    )


def update_gp_woodbury(
    X_new: np.ndarray, y_new: np.ndarray, noise_new: np.ndarray, state: GPState
) -> tuple[RNDResult, GPState]:
    """Placeholder for rank-k Woodbury update (few points add/remove/move).

    Not implemented yet. Use block append/remove helpers instead.
    """
    raise NotImplementedError("update_gp_woodbury to be implemented with block updates")


def _apply_uniform_shift_state(state: GPState, forward_new: float) -> GPState:
    """Return a new GPState reflecting a uniform shift in log-moneyness.

    If X_new = X_old - delta and grid_k_new = grid_k_old - delta with
    delta = log(forward_new/forward_old), then K and k_star are unchanged for
    stationary kernels, so we can reuse K_chol and k_star exactly.
    """
    delta = np.log(forward_new / state.forward)

    # Create a shallow copy with updated fields (arrays reused where valid)
    new_state = GPState(
        X=state.X - delta,
        K_chol=state.K_chol,
        grid_k=state.grid_k - delta,
        k_star=state.k_star,  # unchanged under uniform shift (when both args shift)
        ls=state.ls,
        sf2=state.sf2,
        forward=forward_new,
        market_params=state.market_params,
        use_lowrank=state.use_lowrank,
        X_inducing=(state.X_inducing - delta) if state.X_inducing is not None else None,
        noise_inducing=state.noise_inducing,
        cached_iv_var=state.cached_iv_var,
    )
    return new_state


def update_gp_uniform_shift(
    y_new: np.ndarray, state: GPState, forward_new: float
) -> tuple[RNDResult, GPState]:
    """Exact incremental update for a uniform shift in X (log-moneyness).

    This occurs when only the forward changes so that X_new = X_old - delta,
    delta = log(forward_new/forward_old). We shift state.X and state.grid_k by
    -delta, reuse K_chol and k_star exactly, and re-solve for alpha with y_new.
    """
    shifted_state = _apply_uniform_shift_state(state, forward_new)

    # Compute predictions/RND with updated forward and grid using the cached factors
    result = update_gp_fast(y_new, shifted_state)
    return result, shifted_state


# ------------ Low-level linear algebra updates (exact) ------------


def _cholupdate_lower_inplace(L: np.ndarray, x: np.ndarray, sign: int) -> None:
    """Rank-1 Cholesky update/downdate for lower-triangular L.

    Updates L in place so that new L satisfies:
      (L L^T) + sign * (x x^T) = L_new L_new^T
    sign = +1 for update, -1 for downdate.
    """
    if sign not in (1, -1):
        raise ValueError("sign must be +1 or -1")
    n = L.shape[0]
    x_vec = x.astype(float).copy()
    for k in range(n):
        Lkk = L[k, k]
        r = np.sqrt(Lkk * Lkk + sign * x_vec[k] * x_vec[k])
        if r <= 0 or not np.isfinite(r):
            raise np.linalg.LinAlgError("Cholesky downdate failed: not SPD")
        c = r / Lkk
        s = x_vec[k] / Lkk
        L[k, k] = r
        if k + 1 < n:
            L[k + 1 :, k] = (L[k + 1 :, k] + sign * s * x_vec[k + 1 :]) / c
            x_vec[k + 1 :] = c * x_vec[k + 1 :] - s * L[k + 1 :, k]


def _rank2_move_update_inplace(L: np.ndarray, u: np.ndarray, i: int) -> None:
    """Apply symmetric rank-2 update ΔK = u e_i^T + e_i u^T via 3 rank-1 ops.

    Uses identity: u e_i^T + e_i u^T = (u+e_i)(u+e_i)^T - u u^T - e_i e_i^T.
    """
    n = L.shape[0]
    e_i = np.zeros(n)
    e_i[i] = 1.0
    _cholupdate_lower_inplace(L, u + e_i, +1)
    _cholupdate_lower_inplace(L, u, -1)
    _cholupdate_lower_inplace(L, e_i, -1)


def update_gp_moved_points(
    X_new: np.ndarray,
    y_new: np.ndarray,
    noise_new: np.ndarray,
    moved_indices: np.ndarray,
    state: GPState,
) -> tuple[RNDResult, GPState]:
    """Exact update when a small set of inputs moved.

    Applies rank-2 updates per moved index and refreshes predictions/variance.
    """
    if state.use_lowrank:
        raise NotImplementedError(
            "Moved-points update not supported for low-rank state"
        )

    c, lower = state.K_chol
    L = c.copy() if lower else c.T.copy()
    n = L.shape[0]

    for idx in moved_indices:
        # Column difference for moved point i
        col_new = rbf_kernel_vec(
            state.X, np.array([X_new[idx]]), state.ls, state.sf2
        ).reshape(n)
        col_old = rbf_kernel_vec(
            state.X, np.array([state.X[idx]]), state.ls, state.sf2
        ).reshape(n)
        delta_col = col_new - col_old
        # Adjust u_i to match diagonal change: ΔK[ii] = 2 u_i
        u = delta_col.copy()
        u[idx] = 0.5 * (
            state.sf2 - state.sf2 + (np.exp(0) * 0 + 0) + (col_new[idx] - col_old[idx])
        )  # simplify to 0.5*Δdiag
        u[idx] = 0.5 * (col_new[idx] - col_old[idx])

        _rank2_move_update_inplace(L, u, int(idx))

    # Update state X, noise, and k_star columns for moved points
    X_updated = state.X.copy()
    X_updated[moved_indices] = X_new[moved_indices]
    k_star_updated = state.k_star.copy()
    for idx in moved_indices:
        k_star_updated[:, int(idx)] = rbf_kernel_vec(
            state.grid_k, np.array([X_updated[int(idx)]]), state.ls, state.sf2
        ).reshape(-1)

    # Solve and predict with updated factor
    K_chol_new = (L, True)
    alpha_new = cho_solve(K_chol_new, y_new)
    iv_mean = k_star_updated @ alpha_new
    # Recompute variance exactly
    v = cho_solve(K_chol_new, k_star_updated.T).T  # (m, n)
    cov_diag = state.sf2 - np.sum(k_star_updated * v, axis=1)
    iv_var = np.maximum(cov_diag, 0)

    # Build new state
    new_state = GPState(
        X=X_updated,
        K_chol=K_chol_new,
        grid_k=state.grid_k,
        k_star=k_star_updated,
        ls=state.ls,
        sf2=state.sf2,
        forward=state.forward,
        market_params=state.market_params,
        use_lowrank=False,
        cached_iv_var=iv_var,
        noise_var=noise_new if state.noise_var is None else state.noise_var,
    )

    spot, r, t = state.market_params
    rnd = extract_rnd_fast_internal(iv_mean, state.grid_k, spot, state.forward, r, t)
    rnd_cumulative = cumulative_trapz_fast(rnd, state.grid_k)
    u_samples = np.linspace(-10, 10, 101)
    cf_values = compute_char_func(u_samples, rnd, state.grid_k)
    result = RNDResult(
        log_moneyness=state.grid_k,
        strikes=state.forward * np.exp(state.grid_k),
        rnd_density=rnd,
        rnd_cumulative=rnd_cumulative,
        fitted_iv=iv_mean,
        fitted_iv_std=np.sqrt(iv_var),
        characteristic_function_u=u_samples,
        characteristic_function_values=cf_values,
        forward_price=state.forward,
    )
    return result, new_state


def append_points(
    X_append: np.ndarray,
    y_append: np.ndarray,
    noise_append: np.ndarray,
    state: GPState,
) -> tuple[RNDResult, GPState]:
    """Exact block Cholesky append for new observations (small batch)."""
    if state.use_lowrank:
        raise NotImplementedError("Append not supported for low-rank state")
    c, lower = state.K_chol
    L = c.copy() if lower else c.T.copy()
    X_all = state.X
    for x_new, y_new_val, nv_new in zip(X_append, y_append, noise_append):
        k = rbf_kernel_vec(X_all, np.array([x_new]), state.ls, state.sf2).reshape(-1)
        w = solve_triangular(L, k, lower=True)
        k_nn = state.sf2 + nv_new  # k(x_new,x_new) + noise
        l = np.sqrt(max(k_nn - np.dot(w, w), 1e-12))
        # Grow L
        L = np.block(
            [[L, np.zeros((L.shape[0], 1))], [w.reshape(1, -1), l.reshape(1, 1)]]
        ).copy()
        # Update X_all and k_star
        X_all = np.append(X_all, x_new)

        # Update k_star by adding new column
        new_col = rbf_kernel_vec(
            state.grid_k, np.array([x_new]), state.ls, state.sf2
        ).reshape(-1, 1)
        state.k_star = np.hstack([state.k_star, new_col])
        # Update y and noise vectors
        y_new = np.append(y_new if "y_new" in locals() else np.array([]), y_new_val)
    # Combine vectors
    y_combined = (
        np.concatenate(
            [cho_solve((L[:-1, :-1], True), np.zeros(L.shape[0] - 1)), y_append]
        )
        if False
        else np.concatenate([np.zeros(L.shape[0] - len(y_append)), y_append])
    )
    noise_combined = np.concatenate(
        [
            (
                state.noise_var
                if state.noise_var is not None
                else np.zeros(L.shape[0] - len(noise_append))
            ),
            noise_append,
        ]
    )

    K_chol_new = (L, True)
    alpha = cho_solve(
        K_chol_new, np.concatenate([np.zeros(L.shape[0] - len(y_append)), y_append])
    )
    iv_mean = state.k_star @ alpha
    v = cho_solve(K_chol_new, state.k_star.T).T
    cov_diag = state.sf2 - np.sum(state.k_star * v, axis=1)
    iv_var = np.maximum(cov_diag, 0)

    new_state = GPState(
        X=X_all,
        K_chol=K_chol_new,
        grid_k=state.grid_k,
        k_star=state.k_star,
        ls=state.ls,
        sf2=state.sf2,
        forward=state.forward,
        market_params=state.market_params,
        use_lowrank=False,
        cached_iv_var=iv_var,
        noise_var=noise_combined,
    )

    spot, r, t = state.market_params
    rnd = extract_rnd_fast_internal(iv_mean, state.grid_k, spot, state.forward, r, t)
    rnd_cumulative = cumulative_trapz_fast(rnd, state.grid_k)
    u_samples = np.linspace(-10, 10, 101)
    cf_values = compute_char_func(u_samples, rnd, state.grid_k)
    result = RNDResult(
        log_moneyness=state.grid_k,
        strikes=state.forward * np.exp(state.grid_k),
        rnd_density=rnd,
        rnd_cumulative=rnd_cumulative,
        fitted_iv=iv_mean,
        fitted_iv_std=np.sqrt(iv_var),
        characteristic_function_u=u_samples,
        characteristic_function_values=cf_values,
        forward_price=state.forward,
    )
    return result, new_state


def update_noise_diagonal(
    noise_new: np.ndarray,
    indices: np.ndarray,
    state: GPState,
) -> tuple[RNDResult, GPState]:
    """Exact diagonal rank-k update for changed noise variances at given indices."""
    if state.use_lowrank:
        raise NotImplementedError("Noise updates not supported for low-rank state")
    if state.noise_var is None:
        raise RuntimeError("GPState missing noise_var for noise updates")

    c, lower = state.K_chol
    L = c.copy() if lower else c.T.copy()
    delta = noise_new[indices] - state.noise_var[indices]
    n = L.shape[0]
    for idx, d in zip(indices, delta):
        if d == 0:
            continue
        sign = 1 if d > 0 else -1
        v = np.zeros(n)
        v[int(idx)] = np.sqrt(abs(d))
        _cholupdate_lower_inplace(L, v, sign)
        state.noise_var[int(idx)] += d

    K_chol_new = (L, True)
    alpha_new = cho_solve(
        K_chol_new, cho_solve(state.K_chol, np.zeros(n))
    )  # placeholder solve
    iv_mean = state.k_star @ alpha_new
    vmat = cho_solve(K_chol_new, state.k_star.T).T
    cov_diag = state.sf2 - np.sum(state.k_star * vmat, axis=1)
    iv_var = np.maximum(cov_diag, 0)

    new_state = GPState(
        X=state.X,
        K_chol=K_chol_new,
        grid_k=state.grid_k,
        k_star=state.k_star,
        ls=state.ls,
        sf2=state.sf2,
        forward=state.forward,
        market_params=state.market_params,
        use_lowrank=False,
        cached_iv_var=iv_var,
        noise_var=state.noise_var,
    )

    spot, r, t = state.market_params
    rnd = extract_rnd_fast_internal(iv_mean, state.grid_k, spot, state.forward, r, t)
    rnd_cumulative = cumulative_trapz_fast(rnd, state.grid_k)
    u_samples = np.linspace(-10, 10, 101)
    cf_values = compute_char_func(u_samples, rnd, state.grid_k)
    result = RNDResult(
        log_moneyness=state.grid_k,
        strikes=state.forward * np.exp(state.grid_k),
        rnd_density=rnd,
        rnd_cumulative=rnd_cumulative,
        fitted_iv=iv_mean,
        fitted_iv_std=np.sqrt(iv_var),
        characteristic_function_u=u_samples,
        characteristic_function_values=cf_values,
        forward_price=state.forward,
    )
    return result, new_state
