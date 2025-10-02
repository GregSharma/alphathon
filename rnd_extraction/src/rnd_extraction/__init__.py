"""
RND Extraction Pipeline v2.0
=================================

Continuous, arbitrage-free risk-neutral density extraction from option chains.
Optimized with Cython kernels and incremental GP updates for real-time CSP streaming.

Purpose (Alphathon Q1)
----------------------
Extract option-implied probability distributions to analyze:
- Information flow from Kalshi election probabilities → Options → Equities
- RND regime switching conditional on macro events
- Forward-looking market expectations

Quick Start
-----------
>>> from rnd_extraction import extract_rnd_ultra_simple, MarketData
>>> 
>>> market_data = MarketData(
...     spot_price=5524.19,
...     risk_free_rate=0.05341,
...     time_to_expiry=0.00068,
...     options_df=df  # columns: strike, right, bid, ask, bid_size, ask_size
... )
>>> 
>>> result = extract_rnd_ultra_simple(market_data, grid_points=300)
>>> 
>>> # Outputs
>>> rnd_density = result.rnd_density           # Continuous RND
>>> rnd_cumulative = result.rnd_cumulative     # Cumulative RND
>>> iv_surface = result.fitted_iv              # Smoothed IV curve

Performance
-----------
- Ultra (Cython + Incremental): 15.47ms (1.46x faster)
- Incremental updates: 17.48ms vs 28.95ms = 1.66x faster

See Also
--------
- OVERVIEW.md: Alphathon Q1 system architecture
- MERMAID.md: Data flow diagrams (Layer 7: Advanced Models)
- math_docs.md: Mathematical methodology (Breeden-Litzenberger, GP regression)
"""

from .core import extract_rnd
from .core_optimized import extract_rnd_optimized
from .core_ultra import extract_rnd_ultra, extract_rnd_ultra_simple
from .types import GPHyperparameters, MarketData, RNDResult
from .incremental import GPState

__all__ = [
    'extract_rnd', 
    'extract_rnd_optimized', 
    'extract_rnd_ultra', 
    'extract_rnd_ultra_simple',  # Main interface - accepts numpy arrays directly
    'MarketData', 
    'RNDResult', 
    'GPHyperparameters', 
    'GPState'
]

__version__ = '2.0.0'


