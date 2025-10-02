"""Type definitions for RND extraction pipeline."""
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd


@dataclass
class MarketData:
    """Input market data for RND extraction."""
    spot_price: float
    risk_free_rate: float
    time_to_expiry: float
    options_df: pd.DataFrame  # Columns: strike, right, bid, ask, bid_size, ask_size


@dataclass
class RNDResult:
    """Output from RND extraction pipeline."""
    log_moneyness: np.ndarray
    strikes: np.ndarray
    rnd_density: np.ndarray
    rnd_cumulative: np.ndarray
    fitted_iv: np.ndarray
    fitted_iv_std: np.ndarray
    characteristic_function_u: np.ndarray
    characteristic_function_values: np.ndarray
    forward_price: float
    
    
@dataclass
class GPHyperparameters:
    """Gaussian Process hyperparameters."""
    length_scale: float
    signal_variance: float
    method: Literal["exact", "lowrank"] = "exact"

