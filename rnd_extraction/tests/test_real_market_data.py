"""
Test RND extraction on real SPXW market data.

This test validates that the packaged rnd_extraction produces correct results
on real market data, replicating the development workflow from test_IV.py.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from rnd_extraction import extract_rnd_ultra_simple, MarketData


# Path to real market data
DATA_PATH = Path("/home/grego/Alphathon/data/options_2024/SPXW.parquet")


@pytest.fixture
def spxw_market_data():
    """Load SPXW market data for testing."""
    if not DATA_PATH.exists():
        pytest.skip(f"Market data not found at {DATA_PATH}")
    
    # Load data (exact same as test_IV.py)
    df = pd.read_parquet(DATA_PATH).set_index("ts")
    df.index = df.index.tz_localize("America/New_York")
    
    slice_time = pd.Timestamp("2024-09-04 13:42:00", tz="America/New_York")
    expiration_time = pd.Timestamp("2024-09-04 16:00:00", tz="America/New_York")
    tte_years = (expiration_time - slice_time).total_seconds() / (24 * 3600) / 365.25
    risk_free_rate = 0.05341
    
    data = df[df.index == slice_time]
    spot_price = data["spx"].values[0]
    
    # Prepare options dataframe
    vol = data[["strike", "right", "bid", "ask", "mid", "bid_size", "ask_size"]].reset_index(drop=True)
    
    # Calculate microprice
    vol["microprice"] = (
        (vol["ask_size"] * vol["bid"] + vol["bid_size"] * vol["ask"]) / 
        (vol["ask_size"] + vol["bid_size"])
    )
    
    # Filter: bid > 0.05
    vol = vol[vol.bid > 0.05].copy()
    vol["right"] = vol["right"].str.lower()
    
    # Create MarketData object
    options_df = vol[["strike", "right", "bid", "ask", "bid_size", "ask_size"]].reset_index(drop=True)
    
    market_data = MarketData(
        spot_price=spot_price,
        risk_free_rate=risk_free_rate,
        time_to_expiry=tte_years,
        options_df=options_df
    )
    
    return market_data, spot_price, tte_years, risk_free_rate


def test_spxw_rnd_extraction(spxw_market_data):
    """Test RND extraction on real SPXW data."""
    market_data, spot_price, tte_years, risk_free_rate = spxw_market_data
    
    # Extract RND using packaged version
    result = extract_rnd_ultra_simple(market_data, grid_points=300)
    
    # Calculate forward price
    forward_price = spot_price * np.exp(risk_free_rate * tte_years)
    
    # Validation 1: RND integral should be ~1
    integral = np.trapz(result.rnd_density, result.log_moneyness)
    assert 0.95 <= integral <= 1.05, f"RND integral {integral:.4f} not close to 1"
    
    # Validation 2: Cumulative should reach ~1
    assert 0.95 <= result.rnd_cumulative[-1] <= 1.05, \
        f"CDF max {result.rnd_cumulative[-1]:.4f} not close to 1"
    
    # Validation 3: RND should be non-negative
    assert np.all(result.rnd_density >= -1e-6), "RND has negative values"
    
    # Validation 4: Monotonic cumulative distribution
    diffs = np.diff(result.rnd_cumulative)
    assert np.all(diffs >= -1e-6), "CDF is not monotonic"
    
    # Validation 5: Forward price should be close to result.forward_price
    assert abs(result.forward_price - forward_price) / forward_price < 1e-3, \
        f"Forward price mismatch: {result.forward_price:.2f} vs {forward_price:.2f}"
    
    # Validation 6: Check characteristic function at u=0 (should be ~1)
    u_zero_idx = np.argmin(np.abs(result.characteristic_function_u))
    cf_at_zero = result.characteristic_function_values[u_zero_idx]
    assert abs(np.real(cf_at_zero) - 1.0) < 0.1, \
        f"CF(0) = {np.real(cf_at_zero):.4f}, expected ~1.0"
    assert abs(np.imag(cf_at_zero)) < 0.1, \
        f"CF(0) has imaginary part {np.imag(cf_at_zero):.4f}, expected ~0"
    
    # Validation 7: IV surface should be reasonable
    assert np.all(result.fitted_iv >= 0), "Fitted IV has negative values"
    assert np.all(result.fitted_iv < 2.0), "Fitted IV unreasonably high (>200%)"
    
    # Print summary
    print(f"\n=== SPXW RND Extraction Results ===")
    print(f"Spot: {spot_price:.2f}, Forward: {result.forward_price:.2f}")
    print(f"TTE: {tte_years:.6f} years ({tte_years*365:.1f} days)")
    print(f"Grid points: {len(result.strikes)}")
    print(f"Strike range: {result.strikes.min():.2f} to {result.strikes.max():.2f}")
    print(f"Log-moneyness range: {result.log_moneyness.min():.4f} to {result.log_moneyness.max():.4f}")
    print(f"RND integral: {integral:.6f}")
    print(f"CDF max: {result.rnd_cumulative[-1]:.6f}")
    print(f"Fitted IV - mean: {result.fitted_iv.mean():.4f}, std: {result.fitted_iv.std():.4f}")
    print(f"Fitted IV - range: {result.fitted_iv.min():.4f} to {result.fitted_iv.max():.4f}")


def test_spxw_rnd_moments(spxw_market_data):
    """Test RND moments match expected properties."""
    market_data, spot_price, tte_years, risk_free_rate = spxw_market_data
    
    result = extract_rnd_ultra_simple(market_data, grid_points=500)
    forward_price = spot_price * np.exp(risk_free_rate * tte_years)
    
    # Expected value (mean) should be close to forward price
    E_S = np.trapz(result.strikes * result.rnd_density, result.log_moneyness)
    forward_error = abs(E_S - forward_price) / forward_price
    
    # For real market data, allow larger tolerance due to:
    # - Bid-ask spreads
    # - Discrete strikes
    # - Market frictions
    assert forward_error < 0.15, \
        f"Expected value {E_S:.2f} deviates {forward_error:.2%} from forward {forward_price:.2f}"
    
    # Variance should be positive
    E_S2 = np.trapz(result.strikes**2 * result.rnd_density, result.log_moneyness)
    variance = E_S2 - E_S**2
    assert variance > 0, f"Negative variance: {variance}"
    
    # Implied volatility from moments
    implied_vol = np.sqrt(variance / E_S**2 / tte_years) if tte_years > 0 else 0
    
    print(f"\n=== RND Moments ===")
    print(f"Expected value: {E_S:.2f} (forward: {forward_price:.2f}, error: {forward_error:.2%})")
    print(f"Variance: {variance:.2f}")
    print(f"Implied vol (from moments): {implied_vol:.4f}")
    print(f"Mean fitted IV: {result.fitted_iv.mean():.4f}")
    
    # Sanity check: implied vol should be in reasonable range
    assert 0.05 <= implied_vol <= 1.0, \
        f"Implied volatility {implied_vol:.4f} outside reasonable range [0.05, 1.0]"


def test_spxw_grid_sizes(spxw_market_data):
    """Test RND extraction with different grid sizes."""
    market_data, _, _, _ = spxw_market_data
    
    grid_sizes = [50, 100, 300, 500]
    results = []
    
    for n_grid in grid_sizes:
        result = extract_rnd_ultra_simple(market_data, grid_points=n_grid)
        integral = np.trapz(result.rnd_density, result.log_moneyness)
        results.append({
            'grid_size': n_grid,
            'integral': integral,
            'cdf_max': result.rnd_cumulative[-1],
            'iv_mean': result.fitted_iv.mean()
        })
        
        # Each should integrate properly
        assert 0.90 <= integral <= 1.10, \
            f"Grid size {n_grid}: integral {integral:.4f} out of range"
    
    # Results should be relatively stable across grid sizes
    integrals = [r['integral'] for r in results]
    iv_means = [r['iv_mean'] for r in results]
    
    print(f"\n=== Grid Size Stability ===")
    for r in results:
        print(f"Grid {r['grid_size']:3d}: integral={r['integral']:.6f}, "
              f"CDF_max={r['cdf_max']:.6f}, IV_mean={r['iv_mean']:.4f}")
    
    # Integral std should be small
    integral_std = np.std(integrals)
    assert integral_std < 0.05, f"Integral varies too much across grids: std={integral_std:.4f}"


@pytest.mark.parametrize("slice_time_str", [
    "2024-09-04 10:05:00",  # Morning
    "2024-09-04 13:42:00",  # Afternoon
])
def test_spxw_multiple_slices(slice_time_str):
    """Test RND extraction at different time slices."""
    if not DATA_PATH.exists():
        pytest.skip(f"Market data not found at {DATA_PATH}")
    
    df = pd.read_parquet(DATA_PATH).set_index("ts")
    df.index = df.index.tz_localize("America/New_York")
    
    slice_time = pd.Timestamp(slice_time_str, tz="America/New_York")
    expiration_time = pd.Timestamp("2024-09-04 16:00:00", tz="America/New_York")
    tte_years = (expiration_time - slice_time).total_seconds() / (24 * 3600) / 365.25
    
    data = df[df.index == slice_time]
    
    if len(data) == 0:
        pytest.skip(f"No data at {slice_time_str}")
    
    spot_price = data["spx"].values[0]
    vol = data[["strike", "right", "bid", "ask", "bid_size", "ask_size"]].reset_index(drop=True)
    vol = vol[vol.bid > 0.05].copy()
    vol["right"] = vol["right"].str.lower()
    
    market_data = MarketData(
        spot_price=spot_price,
        risk_free_rate=0.05341,
        time_to_expiry=tte_years,
        options_df=vol
    )
    
    result = extract_rnd_ultra_simple(market_data, grid_points=300)
    
    integral = np.trapz(result.rnd_density, result.log_moneyness)
    assert 0.95 <= integral <= 1.05, \
        f"At {slice_time_str}: RND integral {integral:.4f} not close to 1"
    
    print(f"\n=== {slice_time_str} ===")
    print(f"Spot: {spot_price:.2f}, TTE: {tte_years*365:.1f} days")
    print(f"RND integral: {integral:.6f}")
    print(f"IV mean: {result.fitted_iv.mean():.4f}")


if __name__ == "__main__":
    # Run single test for quick validation
    import sys
    
    print("\n" + "="*80)
    print("RND EXTRACTION - REAL MARKET DATA TEST")
    print("="*80)
    
    # Create fixture manually
    if DATA_PATH.exists():
        df = pd.read_parquet(DATA_PATH).set_index("ts")
        df.index = df.index.tz_localize("America/New_York")
        
        slice_time = pd.Timestamp("2024-09-04 13:42:00", tz="America/New_York")
        expiration_time = pd.Timestamp("2024-09-04 16:00:00", tz="America/New_York")
        tte_years = (expiration_time - slice_time).total_seconds() / (24 * 3600) / 365.25
        risk_free_rate = 0.05341
        
        data = df[df.index == slice_time]
        spot_price = data["spx"].values[0]
        
        vol = data[["strike", "right", "bid", "ask", "mid", "bid_size", "ask_size"]].reset_index(drop=True)
        vol["microprice"] = (
            (vol["ask_size"] * vol["bid"] + vol["bid_size"] * vol["ask"]) / 
            (vol["ask_size"] + vol["bid_size"])
        )
        vol = vol[vol.bid > 0.05].copy()
        vol["right"] = vol["right"].str.lower()
        
        options_df = vol[["strike", "right", "bid", "ask", "bid_size", "ask_size"]].reset_index(drop=True)
        
        market_data = MarketData(
            spot_price=spot_price,
            risk_free_rate=risk_free_rate,
            time_to_expiry=tte_years,
            options_df=options_df
        )
        
        fixture_data = (market_data, spot_price, tte_years, risk_free_rate)
        
        print("\nRunning tests...")
        test_spxw_rnd_extraction(fixture_data)
        test_spxw_rnd_moments(fixture_data)
        test_spxw_grid_sizes(fixture_data)
        
        print("\n✓ All real market data tests passed!")
    else:
        print(f"⚠ Market data not found at {DATA_PATH}")
        print("Skipping tests.")

