"""
CSP_Options - Options processing and market analytics for CSP framework
"""

from CSP_Options.fama import (
    compute_fama_factors_graph,
    compute_fama_log_prices,
    compute_fama_log_prices_efficient,
    compute_fama_returns,
    load_factor_weights,
)
from CSP_Options.structs import FamaFactors, FamaReturns

# Panel dashboard (optional import)
try:
    from CSP_Options.panel_dashboard import BaseDashboard, RegimeDashboard

    __all_panel__ = ["BaseDashboard", "RegimeDashboard"]
except ImportError:
    __all_panel__ = []

__all__ = [
    "compute_fama_factors_graph",
    "compute_fama_log_prices",
    "compute_fama_log_prices_efficient",
    "compute_fama_returns",
    "load_factor_weights",
    "FamaFactors",
    "FamaReturns",
] + __all_panel__
