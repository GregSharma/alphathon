"""Comprehensive tests and benchmarks for RND extraction."""
import time
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')


def load_test_data():
    """Load test data."""
    import os
    from rnd_extraction.types import MarketData
    
    data_path = os.path.join(os.path.dirname(__file__), "../../data/options_2024/SPXW.parquet")
    df = pd.read_parquet(data_path).set_index("ts")
    df.index = df.index.tz_localize("America/New_York")
    
    slice_time = pd.Timestamp("2024-09-04 10:05:00", tz="America/New_York")
    expiration_time = pd.Timestamp("2024-09-04 16:00:00", tz="America/New_York")
    tte_years = (expiration_time - slice_time).total_seconds() / (365.25 * 24 * 3600)
    
    data = df[df.index == slice_time]
    options_df = data[["strike", "right", "bid", "ask", "bid_size", "ask_size"]].reset_index(drop=True)
    
    return MarketData(spot_price=5524.19, risk_free_rate=0.05341,
                     time_to_expiry=tte_years, options_df=options_df)


def run_tests(verbose: bool = True):
    """Run all validation tests."""
    from rnd_extraction.core import extract_rnd
    from rnd_extraction.core_optimized import extract_rnd_optimized
    from rnd_extraction.core_ultra import extract_rnd_ultra_simple, extract_rnd_ultra
    
    market_data = load_test_data()
    tests_passed = []
    
    # Test 1: Original version
    if verbose: print("Testing original version...", end=" ")
    result = extract_rnd(market_data, grid_points=300)
    integral = np.trapz(result.rnd_density, result.log_moneyness)
    cumulative_max = result.rnd_cumulative[-1]
    assert 0.95 <= integral <= 1.05 and 0.95 <= cumulative_max <= 1.05
    assert np.all(result.rnd_density >= -1e-6) and np.all(np.diff(result.rnd_cumulative) >= -1e-6)
    if verbose: print("✓")
    tests_passed.append("original")
    
    # Test 2: Optimized version
    if verbose: print("Testing optimized version...", end=" ")
    result2 = extract_rnd_optimized(market_data, grid_points=300)
    integral = np.trapz(result2.rnd_density, result2.log_moneyness)
    cumulative_max = result2.rnd_cumulative[-1]
    assert 0.95 <= integral <= 1.05 and 0.95 <= cumulative_max <= 1.05
    if verbose: print("✓")
    tests_passed.append("optimized")
    
    # Test 3: Ultra version
    if verbose: print("Testing ultra version...", end=" ")
    result3 = extract_rnd_ultra_simple(market_data, grid_points=300)
    integral = np.trapz(result3.rnd_density, result3.log_moneyness)
    cumulative_max = result3.rnd_cumulative[-1]
    assert 0.95 <= integral <= 1.05 and 0.95 <= cumulative_max <= 1.05
    if verbose: print("✓")
    tests_passed.append("ultra")
    
    # Test 4: Incremental update
    if verbose: print("Testing incremental update...", end=" ")
    result4a, state = extract_rnd_ultra(market_data, grid_points=200)
    # Slightly perturb prices
    df_new = market_data.options_df.copy()
    df_new['bid'] = df_new['bid'] * 1.001
    df_new['ask'] = df_new['ask'] * 1.001
    from rnd_extraction.types import MarketData
    market_data_new = MarketData(
        spot_price=market_data.spot_price,
        risk_free_rate=market_data.risk_free_rate,
        time_to_expiry=market_data.time_to_expiry,
        options_df=df_new
    )
    result4b, _ = extract_rnd_ultra(market_data_new, grid_points=200, prev_state=state)
    integral = np.trapz(result4b.rnd_density, result4b.log_moneyness)
    assert 0.9 <= integral <= 1.1
    if verbose: print("✓")
    tests_passed.append("incremental")
    
    # Test 5: Consistency
    if verbose: print("Testing consistency...", end=" ")
    r1 = extract_rnd(market_data, grid_points=200)
    r2 = extract_rnd_optimized(market_data, grid_points=200)
    np.testing.assert_allclose(r1.rnd_density, r2.rnd_density, rtol=0.05, atol=1e-4)
    if verbose: print("✓")
    tests_passed.append("consistency")
    
    # Test 6: Different grid sizes
    if verbose: print("Testing grid sizes...", end=" ")
    for n in [50, 100, 500]:
        r = extract_rnd_ultra_simple(market_data, grid_points=n)
        integral = np.trapz(r.rnd_density, r.log_moneyness)
        assert 0.9 <= integral <= 1.1
    if verbose: print("✓")
    tests_passed.append("grid_sizes")
    
    if verbose: print(f"\n✓ All {len(tests_passed)} tests passed!")
    return True


def run_benchmarks(n_runs: int = 50):
    """Run performance benchmarks."""
    from rnd_extraction.core import extract_rnd
    from rnd_extraction.core_optimized import extract_rnd_optimized
    
    market_data = load_test_data()
    
    print("\n" + "="*80)
    print("PERFORMANCE BENCHMARK")
    print("="*80 + "\n")
    
    # Benchmark both versions
    for name, func in [("Original", extract_rnd), ("Optimized", extract_rnd_optimized)]:
        func(market_data, grid_points=300)  # Warmup
        times = [time.perf_counter() for _ in range(n_runs)]
        for i in range(n_runs):
            start = time.perf_counter()
            func(market_data, grid_points=300)
            times[i] = time.perf_counter() - start
        
        print(f"{name:15} Mean: {np.mean(times)*1000:6.2f}ms  Std: {np.std(times)*1000:5.2f}ms")
    
    # Scaling analysis
    print(f"\nScaling (Optimized):")
    print(f"{'Grid':<10} {'Mean (ms)':<12} {'μs/point':<12}")
    print("-" * 34)
    
    for n in [50, 100, 200, 300, 500, 1000]:
        times = []
        for _ in range(20):
            start = time.perf_counter()
            extract_rnd_optimized(market_data, grid_points=n)
            times.append(time.perf_counter() - start)
        mean_ms = np.mean(times) * 1000
        print(f"{n:<10} {mean_ms:<12.2f} {mean_ms*1000/n:<12.1f}")
    
    print(f"\n{'='*80}\n")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "bench":
        run_benchmarks()
    else:
        print("\n" + "="*80)
        print("RND EXTRACTION - TEST SUITE")
        print("="*80 + "\n")
        run_tests(verbose=True)
        if "--bench" in sys.argv or "-b" in sys.argv:
            run_benchmarks()
