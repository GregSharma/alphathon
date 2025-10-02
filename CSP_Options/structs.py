import numpy as np

import csp


class IndexPrice(csp.Struct):
    symbol: str
    price: float


class EquityTrade(csp.Struct):
    symbol: str
    conditions: str
    exchange: int
    price: float
    size: int
    tape: int


class EquityQuoteWoSize(csp.Struct):
    symbol: str
    bid_price: float
    ask_price: float


class EquityQuoteWithSize(csp.Struct):
    symbol: str
    bid_size: int
    bid_price: float
    ask_size: int
    ask_price: float


class OptionQuote(csp.Struct):
    symbol: str
    underlying: str
    expiration: int
    strike: float
    right: str
    bid_size: int
    bid: float
    ask_size: int
    ask: float


class VectorizedOptionQuote(csp.Struct):
    underlying: str
    expiration: int
    strike: np.ndarray
    right: np.ndarray
    bid: np.ndarray
    ask: np.ndarray
    mid: np.ndarray
    bid_size: np.ndarray
    ask_size: np.ndarray
    iv: np.ndarray
    tte: float


class VolFit(csp.Struct):
    model_name: str
    params: np.ndarray
    strike: np.ndarray
    iv: np.ndarray
    fitted_iv: np.ndarray


class KalshiTrade(csp.Struct):
    symbol: str
    contracts_traded: int
    price: float


class EquityBar1m(csp.Struct):
    """
    One-minute aggregated equity bar with microstructure features.

    This struct matches the schema required for David's VECM model.
    All features are computed over a 1-minute window.
    """

    symbol: str

    # Price features
    log_mid: float  # Log of mid-price: np.log((bid + ask) / 2)

    # Flow features
    iso_flow_intensity: float  # ISO flow / total volume
    total_flow: float  # Total signed dollar flow
    total_flow_non_iso: float  # Non-ISO signed dollar flow

    # Count features
    num_trades: int  # Number of trades in period
    quote_updates: int  # Number of quote updates

    # Quote quality
    avg_rsprd: float  # Average relative spread: (ask - bid) / mid
    pct_trades_iso: float  # Percentage of trades that are ISO


class FamaFactors(csp.Struct):
    """
    Real-time Fama-French 5-factor values in log-price space.
    """

    HML: float  # High Minus Low (value factor)
    SMB: float  # Small Minus Big (size factor)
    RMW: float  # Robust Minus Weak (profitability factor)
    CMA: float  # Conservative Minus Aggressive (investment factor)
    MKT_RF: float  # Market Risk Premium (market factor minus risk-free rate)


class FamaReturns(csp.Struct):
    """
    Real-time Fama-French 5-factor returns (innovations/deltas).
    """

    HML: float
    SMB: float
    RMW: float
    CMA: float
    MKT_RF: float
