# Complete Integration Guide: CSP → RND → CBRA → Fama Factor Implied RND

```        
CSP Option Quotes 
    ↓ (nodes.py: quote_list_to_vector)
VectorizedOptionQuote
    ↓ (rnd_extraction: GPR)
Risk-Neutral Density (PDF, CDF)
    ↓ (mvrncd: CBRA pipe)
Joint Distribution (D+K assets)   // Not completed in time
    ↓ (Project onto Fama factor space)
Implied Fama Factor RND           // Not completed in time
```

------------------------------------------------------------------------

## Step 1: CSP Options → VectorizedOptionQuote

**Key function**: `quote_list_to_vector()`

**Output format**:

``` python
VectorizedOptionQuote(
    strike=np.array([...]),      # Filtered strikes
    right=np.array(['c','p',...]), # Call/put flags
    bid=np.array([...]),
    ask=np.array([...]),
    mid=np.array([...]),
    bid_size=np.array([...]),
    ask_size=np.array([...]),
    iv=np.array([...]),          # Implied volatilities
    tte=float                    # Time to expiry (years)
)
```

**CSP Integration**:

``` python
@csp.graph
def option_chain_processor(underlying: str, expiry: datetime) -> ts[VectorizedOptionQuote]:
    # Demux options by underlying+expiry
    option_stream = get_option_quotes(underlying, expiry)
    spot_price = get_spot_price(underlying)  # From equity mid
    tte = compute_tte(expiry)
    
    # Vectorize
    vectorized = quote_list_to_vector(
        list_of_quotes=list(option_stream.values()),
        spot_price=spot_price,
        tte_yrs=tte
    )
    
    return vectorized.vq  # Return vectorized quotes
```

------------------------------------------------------------------------

## Step 2: VectorizedOptionQuote → Risk-Neutral Density

**Key files**: - `src/gpr.py`: Gaussian Process Regression - `src/breeden_litzenberger.py`: RND extraction - `examples/basic_usage.py`: Complete example

**API** (from your existing code):

``` python
from rnd_extraction import extract_rnd_complete

# Input from CSP
strikes = vectorized_quote.strike
ivs = vectorized_quote.iv
tte = vectorized_quote.tte
spot = impl_spot_price  # Or use actual spot

# Extract RND
result = extract_rnd_complete(
    strikes=strikes,
    ivs=ivs,
    spot=spot,
    tte=tte,
    r=0.05,  # Risk-free rate
    n_grid=1000,  # Discretization points
)

# Outputs
strike_grid = result['strike_grid']
pdf = result['pdf']  # Risk-neutral density
cdf = result['cdf']  # Cumulative distribution
smooth_iv = result['smooth_iv']  # Fitted IV curve
```

**CSP Wrapper Node**:

``` python
@csp.node
def extract_rnd_csp(
    vectorized_chain: ts[VectorizedOptionQuote],
    use_impl_spot: bool = True,  # Toggle: put-call parity vs actual spot
) -> ts[RNDensity]:
    """
    Wrapper for Greg's existing RND extraction
    """
    with csp.state():
        # Import your existing package
        from rnd_extraction import extract_rnd_complete
    
    if csp.ticked(vectorized_chain):
        try:
            # Call your existing function
            result = extract_rnd_complete(
                strikes=vectorized_chain.strike,
                ivs=vectorized_chain.iv,
                spot=vectorized_chain.spot if use_impl_spot else actual_spot,
                tte=vectorized_chain.tte,
                r=0.05,
                n_grid=1000,
            )
            
            return RNDensity(
                underlying=underlying,  # Pass from graph
                expiry=expiry,
                timestamp=csp.now(),
                strike_grid=result['strike_grid'],
                pdf=result['pdf'],
                cdf=result['cdf'],
                smooth_iv=result['smooth_iv'],
                no_arbitrage=True,  # Your code guarantees this
                calibration_error=np.mean(np.abs(result['smooth_iv'] - vectorized_chain.iv))
            )
        except Exception as e:
            csp.log_error(f"RND extraction failed for {underlying}: {e}")
            return None
```

------------------------------------------------------------------------

## Step 3: Multiple RNDs → CBRA Joint Distribution

``` python
from cbrapipe import discretize_instruments, cbra_optimize, extract_joint_distribution

# Inputs: D+K inverse CDFs (from RND extraction)
# Convert your RND CDFs to inverse CDFs (quantile functions)

def cdf_to_inv_cdf(strike_grid, cdf):
    """Convert CDF to inverse CDF (quantile function)"""
    from scipy.interpolate import interp1d
    # Ensure CDF is strictly increasing
    F_inv = interp1d(cdf, strike_grid, bounds_error=False, fill_value='extrapolate')
    return lambda p: F_inv(p)

# Example: 6 assets (4 equities + 2 ETFs)
D = 4  # AAPL, MSFT, NVDA, GOOGL
K = 2  # SPY, QQQ

F_inv_list = []

# Add equity RNDs
for ticker in ['AAPL', 'MSFT', 'NVDA', 'GOOGL']:
    rnd = rnd_density[ticker]  # From Step 2
    F_inv = cdf_to_inv_cdf(rnd.strike_grid, rnd.cdf)
    F_inv_list.append(F_inv)

# Add ETF RNDs  
for ticker in ['SPY', 'QQQ']:
    rnd = rnd_density[ticker]
    F_inv = cdf_to_inv_cdf(rnd.strike_grid, rnd.cdf)
    F_inv_list.append(F_inv)

# Run CBRA (no constraints since factors are untraded)
n_states = 5000
X = discretize_instruments(n_states, F_inv_list)

# Randomize for max entropy
for j in range(X.shape[1]):
    np.random.shuffle(X[:, j])

# Optimize (BRA mode: K=0 constraints)
X_joint = cbra_optimize(X, tilde_alpha=None, blocks=None, max_iter=2000)

# X_joint is now n_states × (D+K) matrix
# Each row is an equiprobable state of the world
```

**CSP Integration**:

``` python
@csp.node
def compute_joint_rnd(
    rnd_basket: {str: ts[RNDensity]},  # Basket of all underlyings
    n_states: int = 5000,
) -> ts[JointRND]:
    """
    Compute joint distribution from marginal RNDs
    """
    with csp.state():
        from cbrapipe import discretize_instruments, cbra_optimize
        s_last_joint = None
    
    if csp.ticked(rnd_basket):
        # Get all valid RNDs
        valid_rnds = {k: v for k, v in rnd_basket.validitems()}
        
        if len(valid_rnds) < 2:
            return  # Need at least 2 assets
        
        # Convert to inverse CDFs
        F_inv_list = []
        tickers = sorted(valid_rnds.keys())
        
        for ticker in tickers:
            rnd = valid_rnds[ticker]
            F_inv = cdf_to_inv_cdf(rnd.strike_grid, rnd.cdf)
            F_inv_list.append(F_inv)
        
        # Run CBRA
        X = discretize_instruments(n_states, F_inv_list)
        
        # Randomize
        for j in range(X.shape[1]):
            np.random.shuffle(X[:, j])
        
        # Optimize (no constraints for max entropy)
        X_joint = cbra_optimize(X, tilde_alpha=None, blocks=None, max_iter=2000)
        
        return JointRND(
            tickers=tickers,
            n_states=n_states,
            joint_matrix=X_joint,  # Shape: (n_states, len(tickers))
            timestamp=csp.now()
        )
```

------------------------------------------------------------------------

## Step 4: Joint Distribution → Implied Fama Factor RND

**The Connection**: Use factor weights to project joint states onto factor space

**Implementation**:

``` python
def compute_implied_fama_rnd(
    joint_rnd: JointRND,
    factor_weights: dict,  # From factor_weights_tickers.json
    factor_name: str = 'MKT-RF'
) -> dict:
    """
    Project joint distribution onto Fama factor space
    
    For each state i (row of joint_matrix):
        F_i = Σ w_j × S_j_i
    
    Where:
        w_j = factor weight for ticker j
        S_j_i = asset j value in state i
    """
    # Get weights for this factor and these tickers
    weights_for_tickers = {}
    for ticker in joint_rnd.tickers:
        weights_for_tickers[ticker] = factor_weights[factor_name].get(ticker, 0.0)
    
    # Project each state onto factor
    n_states = joint_rnd.n_states
    factor_values = np.zeros(n_states)
    
    for i in range(n_states):
        # For this state, compute factor value
        factor_val = 0.0
        for j, ticker in enumerate(joint_rnd.tickers):
            factor_val += weights_for_tickers[ticker] * joint_rnd.joint_matrix[i, j]
        factor_values[i] = factor_val
    
    # Sort and create empirical CDF
    factor_sorted = np.sort(factor_values)
    prob_grid = np.linspace(0, 1, n_states)
    
    # Compute PDF via numerical differentiation
    factor_pdf = np.gradient(prob_grid, factor_sorted)
    
    return {
        'factor_name': factor_name,
        'factor_grid': factor_sorted,
        'pdf': factor_pdf,
        'cdf': prob_grid,
        'mean': np.mean(factor_values),
        'var': np.var(factor_values),
    }

# Compute for all 5 factors
@csp.node
def implied_fama_factor_rnds(
    joint_rnd: ts[JointRND],
    factor_weights: dict,
) -> ts[dict]:
    """
    Compute implied RND for all 5 Fama factors
    """
    if csp.ticked(joint_rnd):
        results = {}
        for factor in ['MKT-RF', 'SMB', 'HML', 'RMW', 'CMA']:
            results[factor] = compute_implied_fama_rnd(
                joint_rnd,
                factor_weights,
                factor
            )
        return results
```