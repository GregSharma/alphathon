# RND Extraction Benchmarks

This directory contains performance benchmarking scripts.

## Benchmark Files

### `benchmark_ultra.py`
Comprehensive performance comparison:
- **Original** (`extract_rnd`): Baseline implementation
- **Optimized** (`extract_rnd_optimized`): Numba-accelerated
- **Ultra** (`extract_rnd_ultra`): Cython + incremental updates

**Run:**
```bash
python benchmarks/benchmark_ultra.py
```

**Output:**
- Timing statistics (mean, std, min, max)
- Speedup factors vs baseline
- Incremental update performance

## Benchmark Results

**Typical results** (real 0DTE SPY chain, 120 options):

| Version | Time (ms) | Speedup |
|---------|-----------|---------|
| Original | 22.55 | 1.00x |
| Optimized (Numba) | 19.55 | 1.15x |
| (Cython + Incremental Updating)** | **15.47** | **1.46x** |

**Incremental updates**:
- Full recompute: 28.95ms
- Incremental update: 17.48ms
- Speedup: 1.66x

## Running Benchmarks

### Standard benchmark
```bash
cd rnd_extraction
python benchmarks/benchmark_ultra.py
```

### With profiling
```bash
python -m cProfile -o benchmark.prof benchmarks/benchmark_ultra.py
python -m pstats benchmark.prof
```

### Memory profiling
```bash
python -m memory_profiler benchmarks/benchmark_ultra.py
```

## Benchmark Data

Benchmarks use:
- Real market data from Nov 5, 2024 (Election Day)
- SPY 0DTE option chain (~120 liquid options)

See `../docs/PERFORMANCE.md` for detailed analysis.


