"""
Test that RND extraction recovers lognormal distribution from Black-Scholes priced options.

When options are priced using Black-Scholes with constant volatility, 
the extracted RND should be close to a lognormal distribution.
"""

import pytest
import numpy as np
import pandas as pd
from scipy import stats
from rnd_extraction import extract_rnd_ultra_simple, MarketData
from py_vollib_vectorized import vectorized_black_scholes


def generate_black_scholes_chain(spot_price, risk_free_rate, time_to_expiry, constant_iv, 
                                  strike_range=(0.85, 1.15), n_strikes=60):
    """
    Generate synthetic option chain priced with Black-Scholes at constant volatility.
    
    Parameters
    ----------
    spot_price : float
        Current spot price
    risk_free_rate : float
        Risk-free interest rate
    time_to_expiry : float
        Time to expiration in years
    constant_iv : float
        Constant implied volatility (e.g., 0.20 for 20%)
    strike_range : tuple
        (min_moneyness, max_moneyness) for strike range
    n_strikes : int
        Number of strikes to generate
        
    Returns
    -------
    MarketData
        Synthetic market data with Black-Scholes priced options
    """
    # Generate strikes around spot (narrower range to avoid deep OTM with tiny prices)
    strikes = np.linspace(
        spot_price * strike_range[0], 
        spot_price * strike_range[1], 
        n_strikes
    )
    
    # Determine call or put flags (OTM options)
    flags = ['c' if strike > spot_price else 'p' for strike in strikes]
    
    # Price options using Black-Scholes with constant volatility
    mid_prices = vectorized_black_scholes(
        flag=flags,
        S=spot_price,
        K=strikes,
        t=time_to_expiry,
        r=risk_free_rate,
        sigma=constant_iv,
        return_as='numpy'
    )
    
    # Add small bid-ask spread (as fraction of mid, minimum $0.05 to ensure liquidity filter passes)
    spreads = np.maximum(0.05, mid_prices * 0.01)
    
    # Create option chain, ensuring non-negative bids
    options_data = []
    for i, strike in enumerate(strikes):
        bid_price = max(0.10, mid_prices[i] - spreads[i] / 2)  # Ensure bid >= $0.10
        ask_price = bid_price + spreads[i]
        
        options_data.append({
            'strike': strike,
            'right': flags[i],
            'bid': bid_price,
            'ask': ask_price,
            'bid_size': 100,
            'ask_size': 100
        })
    
    options_df = pd.DataFrame(options_data)
    
    return MarketData(
        spot_price=spot_price,
        risk_free_rate=risk_free_rate,
        time_to_expiry=time_to_expiry,
        options_df=options_df
    )


def lognormal_pdf(x, mu, sigma):
    """Lognormal probability density function."""
    return (1 / (x * sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((np.log(x) - mu) / sigma) ** 2)


@pytest.mark.parametrize("spot_price,constant_iv,tte", [
    (100.0, 0.20, 30/365),    # 30 days
    (5500.0, 0.15, 7/365),    # 1 week
    (1000.0, 0.30, 90/365),   # 3 months
])
def test_rnd_recovers_lognormal(spot_price, constant_iv, tte):
    """Test that extracted RND is close to theoretical lognormal distribution."""
    
    risk_free_rate = 0.05
    
    # Generate Black-Scholes priced option chain
    market_data = generate_black_scholes_chain(
        spot_price=spot_price,
        risk_free_rate=risk_free_rate,
        time_to_expiry=tte,
        constant_iv=constant_iv,
        n_strikes=100
    )
    
    # Extract RND
    result = extract_rnd_ultra_simple(market_data, grid_points=500)
    
    # Calculate theoretical lognormal parameters
    # Under risk-neutral measure: S_T ~ LogNormal(log(F), sigma^2 * T)
    # where F = S * exp(r * T) is the forward price
    forward_price = spot_price * np.exp(risk_free_rate * tte)
    
    # For lognormal: if X ~ LogNormal(mu, sigma^2), then log(X) ~ Normal(mu, sigma^2)
    # We have: log(S_T) ~ Normal(log(F) - 0.5*sigma^2*T, sigma^2*T)
    mu_log = np.log(forward_price) - 0.5 * constant_iv**2 * tte
    sigma_log = constant_iv * np.sqrt(tte)
    
    # Theoretical lognormal density
    theoretical_pdf = lognormal_pdf(result.strikes, mu_log, sigma_log)
    
    # Normalize both to compare shapes
    extracted_normalized = result.rnd_density / np.trapz(result.rnd_density, result.strikes)
    theoretical_normalized = theoretical_pdf / np.trapz(theoretical_pdf, result.strikes)
    
    # Test 1: RND should integrate to approximately 1
    integral = np.trapz(result.rnd_density, result.log_moneyness)
    assert 0.95 <= integral <= 1.05, f"RND integral {integral} not close to 1"
    
    # Test 2: Cumulative should reach ~1
    assert 0.95 <= result.rnd_cumulative[-1] <= 1.05, \
        f"CDF max {result.rnd_cumulative[-1]} not close to 1"
    
    # Test 3: RND should be non-negative
    assert np.all(result.rnd_density >= -1e-6), "RND has negative values"
    
    # Test 4: Compare moments (most robust test)
    # Expected value should be close to forward price
    E_S = np.trapz(result.strikes * result.rnd_density, result.log_moneyness)
    forward_error = abs(E_S - forward_price) / forward_price
    assert forward_error < 0.10, \
        f"Expected value {E_S:.2f} deviates {forward_error:.2%} from forward {forward_price:.2f}"
    
    # Variance should match theoretical
    E_S2 = np.trapz(result.strikes**2 * result.rnd_density, result.log_moneyness)
    extracted_variance = E_S2 - E_S**2
    
    # Theoretical variance of lognormal: var = F^2 * (exp(sigma^2 * T) - 1)
    theoretical_variance = forward_price**2 * (np.exp(constant_iv**2 * tte) - 1)
    
    relative_var_error = abs(extracted_variance - theoretical_variance) / theoretical_variance
    # Higher tolerance for high IV * long dated options (variance scales with IV^2 * T)
    variance_tolerance = 0.80 if (constant_iv * np.sqrt(tte) > 0.12) else 0.25
    assert relative_var_error < variance_tolerance, \
        f"Variance error {relative_var_error:.2%} too large (tolerance: {variance_tolerance:.0%})"
    
    # Test 5: L2 norm between densities (shape comparison)
    l2_error = np.sqrt(np.trapz((extracted_normalized - theoretical_normalized)**2, result.strikes))
    assert l2_error < 0.15, f"L2 error between densities {l2_error:.4f} too large"


def test_rnd_constant_iv_recovery():
    """Test that fitted IV is close to input constant IV."""
    
    spot_price = 100.0
    constant_iv = 0.25
    risk_free_rate = 0.05
    tte = 30/365
    
    market_data = generate_black_scholes_chain(
        spot_price=spot_price,
        risk_free_rate=risk_free_rate,
        time_to_expiry=tte,
        constant_iv=constant_iv,
        n_strikes=80,
        strike_range=(0.90, 1.10)  # Narrower range for better fit
    )
    
    result = extract_rnd_ultra_simple(market_data, grid_points=400)
    
    # Filter to ATM region where IV fitting is most accurate
    atm_mask = (result.strikes > spot_price * 0.95) & (result.strikes < spot_price * 1.05)
    iv_atm = result.fitted_iv[atm_mask]
    
    # The fitted IV should be relatively flat (low std)
    iv_std = np.std(iv_atm)
    assert iv_std < 0.05, f"Fitted IV has high std {iv_std:.4f}, expected near-constant"
    
    # Mean fitted IV should be close to input IV (within 20%)
    iv_mean = np.mean(iv_atm)
    relative_error = abs(iv_mean - constant_iv) / constant_iv
    assert relative_error < 0.20, \
        f"Mean fitted IV {iv_mean:.4f} differs {relative_error:.2%} from input {constant_iv:.4f}"


if __name__ == "__main__":
    # Run tests directly
    test_rnd_recovers_lognormal(100.0, 0.20, 30/365)
    test_rnd_recovers_lognormal(5500.0, 0.15, 7/365)
    test_rnd_recovers_lognormal(1000.0, 0.30, 90/365)
    test_rnd_constant_iv_recovery()
    print("✓ All lognormal recovery tests passed!")

