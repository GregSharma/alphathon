"""
Regime Decomposition: Moment-matching decomposition of RNDs into regime-conditional densities.

Provides both online and offline decomposition functions for splitting observed RNDs
into conditional distributions based on regime probabilities (e.g., election outcomes).
"""

import numpy as np
from scipy.optimize import minimize


def lognorm_moment(mu, sig, k=1):
    """
    Compute k-th raw moment of lognormal distribution.

    Args:
        mu: Location parameter (log-space mean)
        sig: Scale parameter (log-space std)
        k: Moment order (1=mean, 2=second moment, etc.)

    Returns:
        k-th raw moment E[X^k] for X ~ Lognormal(mu, sig^2)
    """
    return np.exp(k * mu + 0.5 * k**2 * sig**2)


def decompose_rnd_online(p_regime_1, obs_mean, obs_std):
    """
    Decompose observed RND into two regime-conditional lognormal densities.

    Uses moment-matching to find regime parameters (mu, sigma) for each regime
    such that the mixture matches observed moments.

    Args:
        p_regime_1: Probability of regime 1 (e.g., p(Trump))
        obs_mean: Observed RND mean (forward price)
        obs_std: Observed RND standard deviation

    Returns:
        Tuple of (mu_1, sig_1, mu_2, sig_2) - lognormal parameters for each regime

    Notes:
        - Fast enough for real-time CSP streaming (~1ms per call)
        - Uses L-BFGS-B optimization with Powell fallback
        - Includes regularization to ensure reasonable parameter ranges
    """
    obs_var = obs_std**2

    def objective(x):
        mu_1, sig_1, mu_2, sig_2 = x

        # Compute moments under each regime
        m1_1 = lognorm_moment(mu_1, sig_1, 1)
        m1_2 = lognorm_moment(mu_2, sig_2, 1)
        m2_1 = lognorm_moment(mu_1, sig_1, 2)
        m2_2 = lognorm_moment(mu_2, sig_2, 2)

        # Mixture moments
        m1_mix = p_regime_1 * m1_1 + (1 - p_regime_1) * m1_2
        m2_mix = p_regime_1 * m2_1 + (1 - p_regime_1) * m2_2
        var_mix = m2_mix - m1_mix**2

        # Match observed moments (normalized)
        err_mean = (m1_mix - 1.0) ** 2
        err_var = (var_mix - (obs_var / obs_mean**2)) ** 2

        # Regularization: regime 1 more optimistic, regime 2 more volatile
        reg_drift = 0.5 * ((mu_1 - 0.015) ** 2 + (mu_2 - 0.005) ** 2)
        reg_vol = 0.5 * ((sig_1 - 0.08) ** 2 + (sig_2 - 0.12) ** 2)

        return err_mean + err_var + reg_drift + reg_vol

    # Optimize
    x0 = [0.015, 0.08, 0.005, 0.12]  # [mu_1, sig_1, mu_2, sig_2]
    bounds = [(-0.05, 0.10), (0.03, 0.30), (-0.05, 0.10), (0.03, 0.30)]

    try:
        res = minimize(objective, x0, bounds=bounds, method="L-BFGS-B")
        if not res.success:
            res = minimize(objective, x0, bounds=bounds, method="Powell")
        return res.x
    except Exception:
        # Fallback to regularization targets if optimization fails
        return np.array([0.015, 0.08, 0.005, 0.12])
