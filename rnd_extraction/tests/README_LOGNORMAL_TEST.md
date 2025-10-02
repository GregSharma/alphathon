# Lognormal RND Recovery Test

## Overview

This test validates that the RND extraction algorithm correctly recovers a lognormal distribution when given option prices generated from Black-Scholes with constant implied volatility.

## Theory

Under the Black-Scholes model with constant volatility σ:
- Options are priced using the Black-Scholes formula
- The risk-neutral density (RND) is lognormal
- The extracted RND should match the theoretical lognormal distribution

This provides a powerful validation:
1. **Consistency check**: If we price options with constant IV, the extracted RND should be lognormal
2. **Numerical accuracy**: Tests the entire pipeline (IV fitting → GP → RND extraction)
3. **Moment matching**: Expected value should equal forward price

## Test Cases

The test suite (`test_lognormal_rnd.py`) includes:

### 1. RND Recovery Tests
Three parameter combinations testing different regimes:
- **Short-dated, medium IV**: 100.0 spot, 20% IV, 30 days
- **Very short-dated, low IV**: 5500.0 spot, 15% IV, 7 days  
- **Medium-dated, high IV**: 1000.0 spot, 30% IV, 90 days

For each case, we verify:
- RND integrates to ~1
- Cumulative distribution reaches ~1
- RND is non-negative
- Expected value matches forward price (within 10%)
- Variance matches theoretical lognormal variance
- L2 distance between extracted and theoretical densities is small

### 2. IV Recovery Test
Validates that:
- Fitted IV surface is nearly flat (low std deviation)
- Mean fitted IV is close to input constant IV (within 20%)

## Key Fixes

During implementation, we discovered and fixed:
1. **Zero variance issue**: When input IVs are constant, `np.var(y) = 0`, causing GP kernel singularity. Fixed by adding minimum variance: `max(np.var(y), 1e-4)`
2. **Deep OTM filtering**: Options with bid < $0.05 are filtered out. Test data generation ensures all options have bid ≥ $0.10
3. **Strike range**: Narrower ranges (±15% moneyness) work better than wide ranges with deep OTM options

## Usage

Run tests:
```bash
cd rnd_extraction
pytest tests/test_lognormal_rnd.py -v
```

Generate synthetic test data:
```python
from tests.test_lognormal_rnd import generate_black_scholes_chain

market_data = generate_black_scholes_chain(
    spot_price=100.0,
    risk_free_rate=0.05,
    time_to_expiry=30/365,
    constant_iv=0.20,
    n_strikes=80
)
```

## References

- See `examples/basic_usage.py` for a complete example using Black-Scholes pricing
- The example now uses `py_vollib_vectorized` for accurate option pricing

