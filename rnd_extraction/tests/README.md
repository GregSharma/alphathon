# RND Extraction Tests

This directory contains the test suite for the RND extraction package.

## Test Files

### `tests.py`
Core test suite covering:
- RND extraction accuracy
- GP regression functionality
- Data preparation and filtering
- Numerical integration
- Characteristic function computation
- Edge cases and error handling

**Run:**
```bash
python -m pytest tests/tests.py -v
# or
python tests/tests.py
```

### `test_IV_legacy.py`
Legacy implied volatility tests:
- IV computation validation
- Historical test cases
- Regression tests for backward compatibility

## Test Coverage

The test suite validates:
- ✅ Arbitrage-free RND (non-negative)
- ✅ Normalization (integral = 1)
- ✅ Forward price consistency
- ✅ GP smoothness
- ✅ Incremental update accuracy
- ✅ Cython vs Numba equivalence

## Running Tests

### All tests
```bash
cd /home/grego/Alphathon/rnd_extraction
python -m pytest tests/ -v
```

### Specific test file
```bash
python tests/tests.py
```

### With coverage
```bash
pytest tests/ --cov=rnd_extraction --cov-report=html
```

## Test Data

Tests use synthetic option chains with known properties to validate:
- Numerical accuracy
- Edge case handling
- Performance characteristics

For benchmarking with real market data, see `../benchmarks/`.


