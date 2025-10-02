"""Ultimate benchmark - all versions + incremental updates."""
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


def create_price_perturbation(market_data, price_shift_pct=0.01):
    """Create a slightly perturbed market data (prices shift, strikes same)."""
    from rnd_extraction.types import MarketData
    
    df = market_data.options_df.copy()
    df['bid'] = df['bid'] * (1 + price_shift_pct * np.random.randn(len(df)) * 0.1)
    df['ask'] = df['ask'] * (1 + price_shift_pct * np.random.randn(len(df)) * 0.1)
    df['bid'] = np.maximum(df['bid'], 0.05)  # Keep valid
    df['ask'] = np.maximum(df['ask'], df['bid'] + 0.01)
    
    return MarketData(
        spot_price=market_data.spot_price,
        risk_free_rate=market_data.risk_free_rate,
        time_to_expiry=market_data.time_to_expiry,
        options_df=df
    )


def benchmark_all_versions():
    """Benchmark all implementations."""
    from rnd_extraction.core import extract_rnd
    from rnd_extraction.core_optimized import extract_rnd_optimized
    from rnd_extraction.core_ultra import extract_rnd_ultra_simple, extract_rnd_ultra
    
    market_data = load_test_data()
    
    print("="*80)
    print("ULTIMATE PERFORMANCE BENCHMARK")
    print("="*80)
    
    implementations = [
        ("Original (Numba)", lambda md: extract_rnd(md, 300)),
        ("Optimized (Aggressive Numba)", lambda md: extract_rnd_optimized(md, 300)),
        ("Ultra (Cython + Smart Caching)", lambda md: extract_rnd_ultra_simple(md, 300)),
    ]
    
    results = []
    n_runs = 50
    
    for name, func in implementations:
        print(f"\nBenchmarking {name}...")
        
        # Warmup
        func(market_data)
        
        # Benchmark
        times = []
        for _ in range(n_runs):
            start = time.perf_counter()
            result = func(market_data)
            times.append(time.perf_counter() - start)
        
        # Validate
        integral = np.trapz(result.rnd_density, result.log_moneyness)
        cumulative_max = result.rnd_cumulative[-1]
        valid = 0.95 <= integral <= 1.05 and 0.95 <= cumulative_max <= 1.05
        
        results.append({
            'name': name,
            'mean_ms': np.mean(times) * 1000,
            'std_ms': np.std(times) * 1000,
            'median_ms': np.median(times) * 1000,
            'min_ms': np.min(times) * 1000,
            'valid': '✓' if valid else '✗'
        })
    
    # Print results
    print(f"\n{'='*80}")
    print(f"BENCHMARK RESULTS ({n_runs} runs, 300 grid points)")
    print(f"{'='*80}\n")
    
    print(f"{'Implementation':<40} {'Mean (ms)':<12} {'Min (ms)':<12} {'Valid':<8}")
    print("-" * 80)
    
    for r in results:
        print(f"{r['name']:<40} {r['mean_ms']:<12.2f} {r['min_ms']:<12.2f} {r['valid']:<8}")
    
    # Speedup analysis
    baseline = results[0]['mean_ms']
    print(f"\n{'='*80}")
    print("SPEEDUP vs ORIGINAL")
    print(f"{'='*80}\n")
    
    for r in results[1:]:
        speedup = baseline / r['mean_ms']
        reduction = (baseline - r['mean_ms']) / baseline * 100
        print(f"{r['name']:}")
        print(f"  Speedup: {speedup:.2f}x")
        print(f"  Time reduction: {reduction:.1f}%")
        print(f"  Absolute savings: {baseline - r['mean_ms']:.2f}ms per run")
        print()


def benchmark_incremental_updates():
    """Benchmark incremental update scenario."""
    from rnd_extraction.core_ultra import extract_rnd_ultra
    from rnd_extraction.core import extract_rnd
    
    market_data = load_test_data()
    
    print("="*80)
    print("INCREMENTAL UPDATE BENCHMARK")
    print("Scenario: Process same option chain every minute with price updates")
    print("="*80 + "\n")
    
    # Initial extraction
    print("Initial extraction (cold start)...")
    start = time.perf_counter()
    result, state = extract_rnd_ultra(market_data, 300, prev_state=None)
    initial_time = (time.perf_counter() - start) * 1000
    print(f"  Time: {initial_time:.2f}ms")
    
    # Simulate 10 minute updates
    n_updates = 10
    update_times = []
    full_recompute_times = []
    
    print(f"\nSimulating {n_updates} incremental updates...")
    
    for i in range(n_updates):
        # Perturb prices slightly
        market_data_new = create_price_perturbation(market_data, price_shift_pct=0.005)
        
        # Incremental update
        start = time.perf_counter()
        result_inc, state = extract_rnd_ultra(market_data_new, 300, prev_state=state)
        update_times.append((time.perf_counter() - start) * 1000)
        
        # Full recompute for comparison
        start = time.perf_counter()
        result_full = extract_rnd(market_data_new, 300)
        full_recompute_times.append((time.perf_counter() - start) * 1000)
    
    # Results
    print(f"\n{'='*80}")
    print("INCREMENTAL UPDATE RESULTS")
    print(f"{'='*80}\n")
    
    print(f"Initial extraction:     {initial_time:.2f}ms")
    print(f"\nIncremental updates ({n_updates} runs):")
    print(f"  Mean: {np.mean(update_times):.2f}ms ± {np.std(update_times):.2f}ms")
    print(f"  Min:  {np.min(update_times):.2f}ms")
    print(f"  Max:  {np.max(update_times):.2f}ms")
    
    print(f"\nFull recompute ({n_updates} runs):")
    print(f"  Mean: {np.mean(full_recompute_times):.2f}ms ± {np.std(full_recompute_times):.2f}ms")
    
    speedup = np.mean(full_recompute_times) / np.mean(update_times)
    savings = np.mean(full_recompute_times) - np.mean(update_times)
    
    print(f"\n{'='*80}")
    print(f"INCREMENTAL SPEEDUP: {speedup:.2f}x faster")
    print(f"TIME SAVED: {savings:.2f}ms per update")
    print(f"Over 1 hour (60 updates): {savings * 60 / 1000:.2f}s saved")
    print(f"{'='*80}\n")


def main():
    """Run all benchmarks."""
    benchmark_all_versions()
    print("\n")
    benchmark_incremental_updates()


if __name__ == "__main__":
    main()

