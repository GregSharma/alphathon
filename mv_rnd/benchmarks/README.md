# CBRA Benchmarks

Quick performance reference.

## Current Performance (Numba-optimized)

| Assets | States | Time | Speedup vs Baseline |
|--------|--------|------|---------------------|
| d=6 | n=10,000 | 0.16s | 7.8x |
| d=20 | n=5,000 | 0.22s | ~10x |
| d=40 | n=5,000 | 1.46s | ~5x |

## Run Benchmarks

```bash
# Standard (d=6, paper validation)
python benchmarks/benchmark_cbra.py

# Real-time trading (d=40)
python benchmarks/benchmark_realtime.py

# Incremental updates (warm starts)
python benchmarks/benchmark_incremental.py

# Detailed profiling
python benchmarks/detailed_profile.py
```

## Recommended Settings

**For d=20:**
- n=5,000
- Result: 0.22s (4.6 Hz refresh rate)

**For d=40:**
- n=3,000 for real-time (~0.4s)
- n=5,000 for accuracy (~0.8s)

---

*See main README.md for full documentation*
