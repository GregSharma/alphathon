"""
Example: How to use the portfolio weights from the Fama-French notebook
to calculate DAILY factor returns using your own daily price data.

NOTE: For a complete WRDS-based solution, use Compute_daily.py instead!
This file is just a reference example showing the calculation logic.
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path

# Load the portfolio weights from the Fama-French notebook
WEIGHTS_FILE = Path('fama/factor_weights_tickers.json')

if WEIGHTS_FILE.exists():
    with open(WEIGHTS_FILE, 'r') as f:
        factor_weights = json.load(f)
else:
    print(f"⚠️  Weights file not found: {WEIGHTS_FILE}")
    print("Run '2. Fit Fama NB Version.ipynb' first to generate portfolio weights.")
    factor_weights = {}

# Example: Load your daily returns data (replace with your actual data source)
# This should have columns: ['date', 'ticker', 'return']
# daily_returns = pd.read_csv('your_daily_returns.csv')

def calculate_daily_factor_return(daily_returns, factor_name, weights_dict):
    """
    Calculate daily factor returns given a weights dictionary.
    
    Parameters:
    -----------
    daily_returns : pd.DataFrame
        DataFrame with columns ['date', 'ticker', 'return']
    factor_name : str
        Name of the factor (HML, SMB, RMW, CMA)
    weights_dict : dict
        Dictionary mapping ticker -> weight (from factor_weights_tickers.json)
    
    Returns:
    --------
    pd.DataFrame with columns ['date', f'{factor_name}_daily']
    """
    # Filter to only tickers in the factor portfolio
    portfolio_returns = daily_returns[daily_returns['ticker'].isin(weights_dict.keys())].copy()
    
    # Add weights
    portfolio_returns['weight'] = portfolio_returns['ticker'].map(weights_dict)
    
    # Calculate weighted return for each date
    daily_factor = (
        portfolio_returns
        .groupby('date')
        .apply(lambda x: np.average(x['return'], weights=x['weight'].abs()))
        .reset_index()
        .rename(columns={0: f'{factor_name}_daily'})
    )
    
    return daily_factor


# Example usage:
if __name__ == "__main__":
    # Dummy example data
    dates = pd.date_range('2024-01-01', '2024-01-10', freq='D')
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META']
    
    daily_returns_example = pd.DataFrame([
        {'date': date, 'ticker': ticker, 'return': np.random.randn() * 0.02}
        for date in dates
        for ticker in tickers
    ])
    
    print("Daily returns example:")
    print(daily_returns_example.head())
    
    # Calculate daily HML factor
    hml_daily = calculate_daily_factor_return(
        daily_returns_example, 
        'HML', 
        factor_weights['HML']
    )
    
    print("\nDaily HML factor:")
    print(hml_daily)
    
    print("\n" + "="*60)
    print("WORKFLOW:")
    print("="*60)
    print("1. Run the Fama notebook with MSF (monthly) to get portfolio weights")
    print("2. Load factor_weights_tickers.json (already created)")
    print("3. Get your daily price data from your preferred source")
    print("4. Use this script to calculate daily factor returns")
    print("="*60)
