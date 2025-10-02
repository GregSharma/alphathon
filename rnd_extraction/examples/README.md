# RND Extraction Examples

This directory contains example scripts demonstrating various use cases of the RND extraction package.

## Available Examples

### `basic_usage.py`
Basic example showing how to:
- Create MarketData from option chain
- Extract RND using `extract_rnd_ultra_simple()`
- Visualize results (RND, CDF, IV surface, characteristic function)
- Compute statistics from RND (expected value, variance, probabilities)

**Run:**
```bash
cd examples
python basic_usage.py
```

## Future Examples

Additional examples will cover:
- **Incremental updates**: Using `extract_rnd_ultra()` with state caching for streaming data
- **CSP integration**: Wrapping RND extraction in CSP nodes
- **RND superimposition**: Decomposing RND into Kalshi-conditional densities
- **Multi-underlying analysis**: Processing option chains for multiple assets
- **Real-time streaming**: Integration with Parquet readers and live market data

## Data Requirements

All examples expect option data as a pandas DataFrame with columns:
- `strike`: Strike price
- `right`: Option type ('c' for call, 'p' for put)
- `bid`, `ask`: Bid/ask prices
- `bid_size`, `ask_size`: Bid/ask sizes

See `../README.md` for full API documentation.

