# Quick Reference: Using Pickled Models in CSP

## The Pattern

```python
import csp
import joblib

# 1. Load model ONCE (outside graph)
model = joblib.load('/path/to/model.joblib')

# 2. Pass model to node as parameter
@csp.node
def predict(features: csp.ts[dict], model: object) -> csp.ts[float]:
    if csp.ticked(features):
        X = np.array([[features['feat1'], features['feat2'], ...]])
        return model.predict(X)[0]

# 3. Use in graph
@csp.graph
def my_graph():
    features = compute_features()
    predictions = predict(features, model=model)  # Pass pre-loaded model
    csp.print("PRED", predictions)

# 4. Run
csp.run(my_graph, starttime=..., endtime=...)
```

## David's Model Example

```python
import csp
import joblib
from datetime import timedelta
from CSP_Options.microstructure import compute_microstructure_features_basket
from CSP_Options.utils.readers import get_taq_quotes, get_taq_trades

# Load David's model
model = joblib.load('david/il_artifacts/model_general_logit_binary.joblib')

@csp.graph
def realtime_leadership():
    # Get data
    quotes = get_taq_quotes()
    trades = get_taq_trades()
    
    # Compute features
    bars = compute_microstructure_features_basket(
        trades_basket=trades,
        quotes_basket=quotes,
        bar_interval=timedelta(minutes=1),
    )
    
    # Make predictions
    timer = csp.timer(timedelta(minutes=5))
    predictions = david_predictor(bars, timer, model=model)
    
    csp.print("PRED", predictions)
```

## Key Points

1. **Load ONCE**: `model = joblib.load(...)` happens BEFORE `@csp.graph`
2. **Pass as param**: `predict(features, model=model)`
3. **Never load per-tick**: Don't `joblib.load()` inside `if csp.ticked(...)`
4. **Feature order matters**: Match training order exactly
5. **Handle errors**: Wrap `model.predict()` in try/except

## Files Created

- `test_pickled_model_realtime.py` - General patterns and examples
- `test_david_model_realtime.py` - David's VECM model specific
- `REALTIME_MODEL_INFERENCE.md` - Full documentation

## Run Commands

```bash
# Demo with synthetic model
python graphing/test_david_model_realtime.py demo

# Real model (once David trains it)
python graphing/test_david_model_realtime.py real \
  --model david/il_artifacts/model_general_logit_binary.joblib \
  --lag_map david/il_artifacts/lag_map_per_prefix.joblib \
  --etf SPY
```

## Training → Inference Flow

```
OFFLINE (David's code):
  Historical Data → Features → Train Model → model.joblib
                                              lag_map.joblib

ONLINE (Your CSP code):
  Live Data → CSP Features → Load Model → Predict → Stream Output
              (microstructure)  (once)     (per tick)
```

## Common Mistakes

| ❌ Don't Do This | ✅ Do This Instead |
|-----------------|-------------------|
| Load model per tick | Load once at graph construction |
| `model.predict([f1, f2])` | `model.predict([[f1, f2]])` (2D array) |
| Ignore feature order | Match training feature order exactly |
| No error handling | Wrap predict in try/except |
| Load in `csp.state()` | Load before `@csp.graph` |

## David's Model Structure

```
Input Features (per ticker):
  - iso_flow_intensity_{TICKER}
  - total_flow_{TICKER}
  - total_flow_non_iso_{TICKER}
  - num_trades_{TICKER}
  - quote_updates_{TICKER}
  - avg_rsprd_{TICKER}
  - pct_trades_iso_{TICKER}

Model:
  Pipeline([
    StandardScaler(with_mean=False),
    LogisticRegression()  # or RandomForest
  ])

Output:
  Binary: 0/1 (is ETF leader?)
  Regression: float (ILS score)
```

