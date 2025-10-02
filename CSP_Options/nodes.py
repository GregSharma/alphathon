# refactored/CSP_Options/nodes.py

import logging
from typing import Dict, List

import numpy as np
from py_vollib_vectorized.implied_volatility import vectorized_implied_volatility as viv

import csp
from csp import ts

# Explicit module-level constant for the risk-free rate used in IV calculation
from CSP_Options.constants import RISK_FREE_RATE
from CSP_Options.structs import KalshiTrade, OptionQuote, VectorizedOptionQuote, VolFit


@csp.node
def quote_list_to_vector(list_of_quotes: ts[list], tte_yrs: ts[float]) -> csp.Outputs(
    vq=ts[VectorizedOptionQuote], impl_spot_price=ts[float]
):
    """
    Converts a list of option quotes into a single vectorized time series,
    calculating implied volatility in the process.
    Derives spot price from options using put-call parity.
    """
    # The node triggers when the list ticks and requires valid tte
    if csp.ticked(list_of_quotes) and csp.valid(tte_yrs):
        valid_quotes = list_of_quotes
        if not valid_quotes:
            return
        if tte_yrs < 0:
            return

        # Extract data into numpy arrays
        fields = ["strike", "right", "bid", "ask", "bid_size", "ask_size"]
        arrays = {field: np.array([getattr(s, field) for s in valid_quotes]) for field in fields}

        strike = arrays["strike"]
        right = np.char.lower(arrays["right"])  # enforce "c" for call, "p" for put
        bid = arrays["bid"]
        ask = arrays["ask"]
        mid = (bid + ask) / 2
        bid_size = arrays["bid_size"]
        ask_size = arrays["ask_size"]
        total_size = bid_size + ask_size

        # Initial spot estimate: use median strike as a rough proxy
        unique_strikes = np.unique(strike)
        spot_estimate = np.median(unique_strikes)

        # Find the closest strike to spot estimate
        closest_idx = np.argmin(np.abs(unique_strikes - spot_estimate))
        closest_strike = unique_strikes[closest_idx]

        # Masks for call and put at closest strike
        call_mask = (right == "c") & (strike == closest_strike)
        put_mask = (right == "p") & (strike == closest_strike)

        # Get mids, default to nan if not found
        call_mid = mid[call_mask][0]
        put_mid = mid[put_mask][0]

        # Calculate implied spot price
        impl_spot_price = call_mid - put_mid + closest_strike * np.exp(-RISK_FREE_RATE * tte_yrs)
        # logging.info(f"Implied spot price for closest strike {closest_strike}: {impl_spot_price}")

        # ------------------------------------------------------------------
        # Vectorised filtering by strike following rules:
        # 1. Forward price F = S * exp(t).
        # 2. For each strike, keep the call if strike > F, else keep the put.
        # 3. If only one of call/put exists for that strike, keep whatever exists.
        # ------------------------------------------------------------------

        # Forward price under the (constant) risk-free-rate assumption
        fwd_price = impl_spot_price * np.exp(tte_yrs)

        # Preferred-leg mask according to rule 2
        preferred_mask = ((right == "c") & (strike > fwd_price)) | (
            (right == "p") & (strike <= fwd_price)
        )

        # Sorting keys for efficient vectorised selection
        # Goal: For each strike, choose preferred leg if available; otherwise the one with larger size.
        # `np.lexsort` sorts ascending; negative values achieve descending behaviour.
        sort_keys = (
            -total_size,  # third priority: larger size first
            -preferred_mask.astype(int),  # second: preferred legs (True -> -1) precede others
            strike,  # primary: group by strike
        )
        sorted_indices = np.lexsort(sort_keys)

        # Pick the first occurrence per strike (after the above ordering)
        _, unique_idx = np.unique(strike[sorted_indices], return_index=True)
        filtered_indices = sorted_indices[unique_idx]

        # Create filtered arrays
        filtered_strike = strike[filtered_indices]
        filtered_right_str = right[filtered_indices]
        filtered_right = np.where(filtered_right_str == "c", "c", "p")
        filtered_bid = bid[filtered_indices]
        filtered_ask = ask[filtered_indices]
        filtered_mid = mid[filtered_indices]
        filtered_bid_size = bid_size[filtered_indices]
        filtered_ask_size = ask_size[filtered_indices]

        # Calculate implied volatility
        try:
            # logging.info('Calculating implied volatility')
            iv = viv(
                price=filtered_mid,
                S=impl_spot_price,
                K=filtered_strike,
                t=tte_yrs,
                r=RISK_FREE_RATE,  # constant risk-free rate
                flag=filtered_right,
                return_as="numpy",
            )

        except Exception as e:
            logging.info(e)
            iv = np.full_like(filtered_strike, np.nan)

        return csp.output(
            vq=VectorizedOptionQuote(
                strike=filtered_strike,
                right=filtered_right_str,
                bid=filtered_bid,
                ask=filtered_ask,
                mid=filtered_mid,
                bid_size=filtered_bid_size,
                ask_size=filtered_ask_size,
                iv=iv,
                tte=tte_yrs,
            ),
            impl_spot_price=impl_spot_price,
        )


@csp.node
def filter_vector_quote(
    max_dist: float, min_bid: float, vq: ts[VectorizedOptionQuote], impl_spot_price: ts[float]
) -> ts[VectorizedOptionQuote]:
    """Filters a vectorized quote based on log-moneyness and minimum bid price."""
    if csp.ticked(vq) and csp.valid(impl_spot_price):
        # Create filters
        final_filter = (np.abs(np.log(impl_spot_price / vq.strike)) <= max_dist) & (
            vq.bid >= min_bid
        )

        # Apply filters to all fields at once
        return VectorizedOptionQuote(
            strike=vq.strike[final_filter],
            right=vq.right[final_filter],
            bid=vq.bid[final_filter],
            ask=vq.ask[final_filter],
            mid=vq.mid[final_filter],
            bid_size=vq.bid_size[final_filter],
            ask_size=vq.ask_size[final_filter],
            iv=vq.iv[final_filter],
            tte=vq.tte,
        )


@csp.node
def sample_dynamic(trigger: ts["Y"], x: {ts["K"]: ts["T"]}) -> ts[Dict["K", "T"]]:
    """Sample valid items from a dynamic basket on trigger"""
    with csp.start():
        csp.make_passive(x)

    if csp.ticked(trigger):
        result = {k: v for k, v in x.validitems()}
        if result:
            return result


@csp.node
def create_model_params_dict(kde_fits: ts[VolFit], cw_fits: ts[VolFit]) -> ts[dict]:
    """Create a dictionary of model parameters when both fits are available"""
    if csp.valid(kde_fits) and csp.valid(cw_fits):
        return {
            kde_fits.model_name: kde_fits.params,
            cw_fits.model_name: cw_fits.params,
        }


# =============================================================================
# BIVARIATE KALMAN FILTER FOR KALSHI BINARIES (UCBM Extension)
# =============================================================================


def probit_transform(p: float) -> float:
    """Transform probability [0,1] to probit gauge U ∈ ℝ via Φ⁻¹(p)"""
    from scipy.stats import norm

    # Clip to avoid inf at boundaries
    p_clipped = np.clip(p, 1e-6, 1 - 1e-6)
    return norm.ppf(p_clipped)


def probit_inverse(u: float) -> float:
    """Transform probit U back to probability via Φ(u)"""
    from scipy.stats import norm

    return norm.cdf(u)


def phi_pdf(u: float) -> float:
    """Standard normal PDF φ(u) = (1/√2π) exp(-u²/2)"""
    from scipy.stats import norm

    return norm.pdf(u)


@csp.node
def bivariate_kalshi_filter(
    djt_trades: ts[KalshiTrade],
    kh_trades: ts[KalshiTrade],
    obs_noise_std: float = 0.005,  # ~0.5¢ trade noise (volume-weighted)
    process_var_base: float = 1e-5,  # Q₀ for clock when idle
    correlation_prior: float = -0.97,  # ρ₀ for anti-correlation (from math doc)
    window_size: int = 60,  # seconds for local vol estimation
) -> csp.Outputs(
    u_djt=ts[float],
    u_kh=ts[float],
    b_djt=ts[float],
    b_kh=ts[float],
    nu_squared=ts[float],
    rho=ts[float],
    innovation_djt=ts[float],
    innovation_kh=ts[float],
    vig=ts[float],  # Vig = (B^D_raw + B^H_raw) - 1 before normalization
):
    """
    Bivariate Kalman Filter for coupled Kalshi binary contracts (DJT & KH).

    Implements Joint-UCBM from the mathematical framework:
    - State: U_t = (U_t^D, U_t^H) in probit gauge
    - Observations: B_t = (Φ(U_t^D), Φ(U_t^H)) with trade noise
    - Coupling: Brownian correlation ρ_t ≈ -1 (complementarity)
    - Clock: Joint information flow rate ν_t²

    Following pairs trading Kalman pattern from your working code.
    """
    with csp.state():
        # Kalman state: [U^D, U^H] in probit gauge
        s_u_mean = np.array([0.0, 0.0])  # Initialize at 50-50
        s_P = np.eye(2) * 0.1  # Initial covariance

        # Time-varying parameters
        s_rho = correlation_prior  # Current correlation estimate
        s_nu_sq = process_var_base  # Current clock rate ν²

        # Buffer for local volatility estimation
        s_price_history_djt = []
        s_price_history_kh = []
        s_time_history = []

        # Last observations for complementarity check
        s_last_b_djt = 0.5
        s_last_b_kh = 0.5

    # Process DJT trade
    if csp.ticked(djt_trades):
        # Observation: noisy probability
        y_djt = djt_trades.price / 100.0  # Convert cents to [0,1]
        s_last_b_djt = y_djt

        # Current state estimate in probability space
        b_djt_pred = probit_inverse(s_u_mean[0])
        b_kh_pred = probit_inverse(s_u_mean[1])

        # Time update (predict) - using current clock
        # dU = ½ν²U dt (curvature drift for martingality in B-space)
        dt = 1.0  # Assume 1 second between trades (will refine with timestamps)
        F = np.eye(2) + 0.5 * s_nu_sq * dt * np.eye(2)  # Drift matrix
        s_u_mean = F @ s_u_mean

        # Process covariance with correlation
        Q = s_nu_sq * dt * np.array([[1.0, s_rho], [s_rho, 1.0]])
        s_P = F @ s_P @ F.T + Q

        # Measurement update for DJT
        # Linearized observation: B ≈ Φ(U) ≈ Φ(μ) + φ(μ)(U - μ)
        phi_djt = phi_pdf(s_u_mean[0])
        H = np.array([[phi_djt, 0.0]])  # Observation matrix (only DJT observed)

        # Innovation
        y_pred = probit_inverse(s_u_mean[0])
        innovation_djt = y_djt - y_pred

        # Measurement noise (trade noise)
        R = np.array([[obs_noise_std**2]])

        # Kalman gain
        S = H @ s_P @ H.T + R  # Innovation covariance
        K = s_P @ H.T / S[0, 0]  # Kalman gain

        # Update state
        s_u_mean = s_u_mean + K.flatten() * innovation_djt
        s_P = (np.eye(2) - K @ H) @ s_P

        # Update price history for local vol estimation
        s_price_history_djt.append(y_djt)
        s_time_history.append(csp.now())

        # Trim history to window
        if len(s_time_history) > window_size:
            s_price_history_djt.pop(0)
            if s_price_history_kh:
                s_price_history_kh.pop(0)
            s_time_history.pop(0)

        # Estimate clock ν² and correlation ρ from recent history
        if len(s_price_history_djt) >= 15 and len(s_price_history_kh) >= 15:
            try:
                # Convert to probit for vol estimation
                u_djt_hist = np.array([probit_transform(p) for p in s_price_history_djt[-15:]])
                u_kh_hist = np.array([probit_transform(p) for p in s_price_history_kh[-15:]])

                # Compute returns in probit space
                du_djt = np.diff(u_djt_hist)
                du_kh = np.diff(u_kh_hist)

                if len(du_djt) > 5 and len(du_kh) > 5:
                    # Bivariate quadratic variation (jump-robust via bipower)
                    # Use robust estimators to handle jumps
                    var_djt = np.var(du_djt)
                    var_kh = np.var(du_kh)
                    cov_djt_kh = np.cov(du_djt, du_kh)[0, 1]

                    # Estimate ν² from marginal variances (averaged)
                    phi_djt_curr = phi_pdf(s_u_mean[0])
                    phi_kh_curr = phi_pdf(s_u_mean[1])
                    if phi_djt_curr > 1e-6 and phi_kh_curr > 1e-6:
                        nu_sq_djt = var_djt / (phi_djt_curr**2 * dt)
                        nu_sq_kh = var_kh / (phi_kh_curr**2 * dt)
                        # EMA update for clock (smoother adaptation)
                        alpha_nu = 0.1
                        s_nu_sq = (1 - alpha_nu) * s_nu_sq + alpha_nu * (
                            (nu_sq_djt + nu_sq_kh) / 2.0
                        )
                        s_nu_sq = np.clip(s_nu_sq, 1e-6, 1.0)  # Reasonable bounds

                    # Estimate correlation (EMA for stability)
                    if var_djt > 1e-8 and var_kh > 1e-8:
                        rho_new = cov_djt_kh / np.sqrt(var_djt * var_kh)
                        if np.isfinite(rho_new):
                            alpha_rho = 0.2
                            s_rho = (1 - alpha_rho) * s_rho + alpha_rho * rho_new
                            s_rho = np.clip(s_rho, -0.999, -0.5)  # Keep anti-correlated
            except Exception as e:
                # Fallback: keep prior values
                pass

        # Apply complementarity constraint: normalize filtered probabilities
        # to sum to 1 (remove vig), weighted by current estimates
        b_djt_raw = probit_inverse(s_u_mean[0])
        b_kh_raw = probit_inverse(s_u_mean[1])
        prob_sum = b_djt_raw + b_kh_raw

        # Normalize to enforce complementarity (B^D + B^H = 1)
        if prob_sum > 0:
            b_djt_normalized = b_djt_raw / prob_sum
            b_kh_normalized = b_kh_raw / prob_sum
        else:
            b_djt_normalized = 0.5
            b_kh_normalized = 0.5

        # Return current estimates (with normalized probabilities)
        return csp.output(
            u_djt=s_u_mean[0],
            u_kh=s_u_mean[1],
            b_djt=b_djt_normalized,
            b_kh=b_kh_normalized,
            nu_squared=s_nu_sq,
            rho=s_rho,
            innovation_djt=innovation_djt,
            innovation_kh=0.0,  # No KH observation this tick
            vig=prob_sum - 1.0,  # Vig before normalization
        )

    # Process KH trade (symmetric logic)
    if csp.ticked(kh_trades):
        y_kh = kh_trades.price / 100.0
        s_last_b_kh = y_kh

        b_djt_pred = probit_inverse(s_u_mean[0])
        b_kh_pred = probit_inverse(s_u_mean[1])

        # Time update
        dt = 1.0
        F = np.eye(2) + 0.5 * s_nu_sq * dt * np.eye(2)
        s_u_mean = F @ s_u_mean
        Q = s_nu_sq * dt * np.array([[1.0, s_rho], [s_rho, 1.0]])
        s_P = F @ s_P @ F.T + Q

        # Measurement update for KH
        phi_kh = phi_pdf(s_u_mean[1])
        H = np.array([[0.0, phi_kh]])

        y_pred = probit_inverse(s_u_mean[1])
        innovation_kh = y_kh - y_pred

        R = np.array([[obs_noise_std**2]])
        S = H @ s_P @ H.T + R
        K = s_P @ H.T / S[0, 0]

        s_u_mean = s_u_mean + K.flatten() * innovation_kh
        s_P = (np.eye(2) - K @ H) @ s_P

        # Update history
        s_price_history_kh.append(y_kh)
        s_time_history.append(csp.now())

        if len(s_time_history) > window_size:
            if s_price_history_djt:
                s_price_history_djt.pop(0)
            s_price_history_kh.pop(0)
            s_time_history.pop(0)

        # Estimate parameters (same as above)
        if len(s_price_history_djt) >= 15 and len(s_price_history_kh) >= 15:
            try:
                u_djt_hist = np.array([probit_transform(p) for p in s_price_history_djt[-15:]])
                u_kh_hist = np.array([probit_transform(p) for p in s_price_history_kh[-15:]])

                du_djt = np.diff(u_djt_hist)
                du_kh = np.diff(u_kh_hist)

                if len(du_djt) > 5 and len(du_kh) > 5:
                    var_djt = np.var(du_djt)
                    var_kh = np.var(du_kh)
                    cov_djt_kh = np.cov(du_djt, du_kh)[0, 1]

                    phi_djt_curr = phi_pdf(s_u_mean[0])
                    phi_kh_curr = phi_pdf(s_u_mean[1])
                    if phi_djt_curr > 1e-6 and phi_kh_curr > 1e-6:
                        nu_sq_djt = var_djt / (phi_djt_curr**2 * dt)
                        nu_sq_kh = var_kh / (phi_kh_curr**2 * dt)
                        alpha_nu = 0.1
                        s_nu_sq = (1 - alpha_nu) * s_nu_sq + alpha_nu * (
                            (nu_sq_djt + nu_sq_kh) / 2.0
                        )
                        s_nu_sq = np.clip(s_nu_sq, 1e-6, 1.0)

                    if var_djt > 1e-8 and var_kh > 1e-8:
                        rho_new = cov_djt_kh / np.sqrt(var_djt * var_kh)
                        if np.isfinite(rho_new):
                            alpha_rho = 0.2
                            s_rho = (1 - alpha_rho) * s_rho + alpha_rho * rho_new
                            s_rho = np.clip(s_rho, -0.999, -0.5)
            except Exception as e:
                pass

        # Apply complementarity constraint: normalize filtered probabilities
        b_djt_raw = probit_inverse(s_u_mean[0])
        b_kh_raw = probit_inverse(s_u_mean[1])
        prob_sum = b_djt_raw + b_kh_raw

        # Normalize to enforce complementarity (B^D + B^H = 1)
        if prob_sum > 0:
            b_djt_normalized = b_djt_raw / prob_sum
            b_kh_normalized = b_kh_raw / prob_sum
        else:
            b_djt_normalized = 0.5
            b_kh_normalized = 0.5

        return csp.output(
            u_djt=s_u_mean[0],
            u_kh=s_u_mean[1],
            b_djt=b_djt_normalized,
            b_kh=b_kh_normalized,
            nu_squared=s_nu_sq,
            rho=s_rho,
            innovation_djt=0.0,
            innovation_kh=innovation_kh,
            vig=prob_sum - 1.0,  # Vig before normalization
        )
