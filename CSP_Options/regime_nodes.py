"""
CSP nodes for regime decomposition and conditional RND analysis.

Provides streaming nodes for decomposing RNDs into regime-conditional densities
based on external regime probabilities (e.g., from prediction markets).
"""

import csp
from csp import ts

from .regime_decomposition import decompose_rnd_online


@csp.node
def regime_decomposer(
    trigger: ts[object],  # Triggers decomposition (e.g., vec_quotes)
    p_regime_1: ts[float],  # Probability of regime 1
    rnd_mean: ts[float],  # RND mean (forward price)
    rnd_std: ts[float],  # RND standard deviation
) -> csp.Outputs(
    mu_1=ts[float],  # Regime 1 log-drift
    sig_1=ts[float],  # Regime 1 log-volatility
    mu_2=ts[float],  # Regime 2 log-drift
    sig_2=ts[float],  # Regime 2 log-volatility
    expected_return_1=ts[float],  # Regime 1 expected return (%)
    expected_return_2=ts[float],  # Regime 2 expected return (%)
    vol_1=ts[float],  # Regime 1 volatility (%)
    vol_2=ts[float],  # Regime 2 volatility (%)
    return_premium=ts[float],  # Return differential (regime 1 - regime 2, %)
):
    """
    CSP node that decomposes observed RND into regime-conditional densities.

    Triggers on RND updates, sampling latest regime probability.
    Uses moment-matching with lognormal assumption for fast real-time decomposition.

    Args:
        trigger: Stream to trigger decomposition (typically vec_quotes or RND updates)
        p_regime_1: Probability of regime 1 (e.g., p(Trump) from prediction market)
        rnd_mean: Current RND mean (forward price)
        rnd_std: Current RND standard deviation

    Outputs:
        mu_1, sig_1: Lognormal parameters for regime 1
        mu_2, sig_2: Lognormal parameters for regime 2
        expected_return_1, expected_return_2: Expected returns (%)
        vol_1, vol_2: Implied volatilities (%)
        return_premium: Return differential (regime 1 - regime 2, %)

    Example:
        >>> regime_params = regime_decomposer(
        ...     trigger=vec_quotes,
        ...     p_regime_1=p_trump,
        ...     rnd_mean=rnd_moments.mean,
        ...     rnd_std=rnd_moments.std,
        ... )
        >>> csp.add_graph_output("return_premium", regime_params.return_premium)
    """
    # Trigger on RND moments, sample regime probability
    if csp.ticked(trigger) and csp.valid(p_regime_1, rnd_mean, rnd_std):
        # Decompose using current values
        mu_1, sig_1, mu_2, sig_2 = decompose_rnd_online(p_regime_1, rnd_mean, rnd_std)

        # Output raw parameters
        csp.output(mu_1=mu_1)
        csp.output(sig_1=sig_1)
        csp.output(mu_2=mu_2)
        csp.output(sig_2=sig_2)

        # Output interpretable metrics (percentages)
        csp.output(expected_return_1=100 * mu_1)
        csp.output(expected_return_2=100 * mu_2)
        csp.output(vol_1=100 * sig_1)
        csp.output(vol_2=100 * sig_2)
        csp.output(return_premium=100 * (mu_1 - mu_2))
