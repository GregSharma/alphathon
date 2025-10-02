"""
Full-day Fama Factor + Microstructure + VECM Leadership Analysis

This test connects the dots:
1. Computes Fama-French 5 factors over full trading day
2. Loads TAQ data for 40 instruments (quotes + trades)
3. Computes microstructure features
4. Regresses factor returns on equity returns
5. Runs VECM leadership analysis on factors + ETFs + select equities
6. Visualizes everything to understand price discovery dynamics

Key Question: Do ETFs have more information share than individual equities?
If factors (derived from equities) are led by ETFs, that's evidence of
ETF-driven price discovery.
"""

import json
from pathlib import Path

import csp
import numpy as np
import pandas as pd
from csp import ts
from scipy import stats

from CSP_Options.csp_tqdm import run_with_progress
from CSP_Options.fama import compute_fama_factors_graph
from CSP_Options.microstructure import compute_microstructure_features_basket
from CSP_Options.structs import EquityBar1m, FamaFactors, FamaReturns
from CSP_Options.utils.readers import get_quotes_1s_wo_size, get_taq_quotes, get_taq_trades

DATE_OF_INTEREST = "2024-11-05"
FACTOR_WEIGHTS_PATH = "/home/grego/Alphathon/fama/factor_weights_tickers.json"

# Load factor weights
with open(FACTOR_WEIGHTS_PATH, "r") as f:
    factor_weights = json.load(f)

ALL_FACTOR_TICKERS = set()
for factor_dict in factor_weights.values():
    ALL_FACTOR_TICKERS.update(factor_dict.keys())

print(f"Fama Factor Full-Day Analysis")
print(f"=" * 80)
print(f"Total unique tickers in factors: {len(ALL_FACTOR_TICKERS)}")
print(f"Date: {DATE_OF_INTEREST}")
print(f"=" * 80)

# First 30 minutes for faster testing
start_ts = pd.Timestamp(f"{DATE_OF_INTEREST} 09:30:00", tz="America/New_York")
end_ts = pd.Timestamp(f"{DATE_OF_INTEREST} 16:00:00", tz="America/New_York")


# =============================================================================
# GRAPH 1: Fama Factors + TAQ Microstructure
# =============================================================================


@csp.graph
def full_day_analysis():
    """
    Complete analysis graph:
    - Fama factors from 1s quotes (3240 tickers)
    - Microstructure features from TAQ (40 tickers)
    - Writes to parquet files
    """
    from pathlib import Path

    from csp.adapters.parquet import ParquetOutputConfig

    from CSP_Options.parquet_writer import GregsParquetWriter

    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)

    # Fama factors from all tickers
    quotes_basket_fama = get_quotes_1s_wo_size(list(ALL_FACTOR_TICKERS))
    fama = compute_fama_factors_graph(
        quotes_basket=quotes_basket_fama,
        factor_weights_path=FACTOR_WEIGHTS_PATH,
        use_efficient=True,
    )

    # TAQ data for 40 instruments
    quotes_basket_taq = get_taq_quotes()
    trades_basket_taq = get_taq_trades()

    # Microstructure features (1-minute bars)
    bars_basket = compute_microstructure_features_basket(
        trades_basket=trades_basket_taq,
        quotes_basket=quotes_basket_taq,
        bar_interval=pd.Timedelta(minutes=1),
    )

    # Write Fama factors to parquet
    fama_writer = GregsParquetWriter(
        file_name=str(output_dir / "fama_factors.parquet"),
        timestamp_column_name="timestamp",
        config=ParquetOutputConfig(allow_overwrite=True),
        include_dimensions=False,
    )
    fama_writer.publish("HML_log", fama.log_prices.HML)
    fama_writer.publish("SMB_log", fama.log_prices.SMB)
    fama_writer.publish("RMW_log", fama.log_prices.RMW)
    fama_writer.publish("CMA_log", fama.log_prices.CMA)
    fama_writer.publish("MKT_RF_log", fama.log_prices.MKT_RF)
    fama_writer.publish("HML_ret", fama.returns.HML)
    fama_writer.publish("SMB_ret", fama.returns.SMB)
    fama_writer.publish("RMW_ret", fama.returns.RMW)
    fama_writer.publish("CMA_ret", fama.returns.CMA)
    fama_writer.publish("MKT_RF_ret", fama.returns.MKT_RF)

    # Write microstructure bars to parquet (one writer for all tickers)
    bars_writer = GregsParquetWriter(
        file_name=str(output_dir / "microstructure_bars.parquet"),
        timestamp_column_name="timestamp",
        config=ParquetOutputConfig(allow_overwrite=True),
        include_dimensions=False,
    )
    # Publish all bar fields for each ticker
    for ticker, bar_ts in bars_basket.items():
        bars_writer.publish(f"{ticker}_log_mid", bar_ts.log_mid)
        bars_writer.publish(f"{ticker}_iso_flow_intensity", bar_ts.iso_flow_intensity)
        bars_writer.publish(f"{ticker}_total_flow", bar_ts.total_flow)
        bars_writer.publish(f"{ticker}_num_trades", bar_ts.num_trades)
        bars_writer.publish(f"{ticker}_avg_rsprd", bar_ts.avg_rsprd)
        bars_writer.publish(f"{ticker}_pct_trades_iso", bar_ts.pct_trades_iso)

    # Still collect outputs for in-memory analysis
    csp.add_graph_output("fama_log_prices", fama.log_prices)
    csp.add_graph_output("fama_returns", fama.returns)

    # Collect bars for each ticker
    for ticker, bar_ts in bars_basket.items():
        csp.add_graph_output(f"bar_{ticker}", bar_ts)


print(f"\n{'=' * 80}")
print(f"Running full-day analysis from {start_ts} to {end_ts}")
print(f"This will take a few minutes...")
print(f"{'=' * 80}\n")

# Run CSP graph and collect outputs
results = run_with_progress(
    full_day_analysis,
    starttime=start_ts,
    endtime=end_ts,
    realtime=False,
)

print(f"✅ CSP run completed! Collected {len(results)} output streams")


# =============================================================================
# Convert to DataFrames
# =============================================================================


def results_to_dataframe(results_dict, key_prefix=None):
    """Convert CSP results to DataFrame."""
    dfs = []
    for key, data in results_dict.items():
        if key_prefix and not key.startswith(key_prefix):
            continue
        if data:
            times, values = zip(*data)
            df_temp = pd.DataFrame({"timestamp": pd.to_datetime(times)})

            # Handle different types
            if hasattr(values[0], "__dict__"):
                # Struct - extract fields
                for field in values[0].__dict__.keys():
                    df_temp[field] = [getattr(v, field) for v in values]
            else:
                df_temp["value"] = values

            df_temp["key"] = key
            dfs.append(df_temp)

    if dfs:
        df = pd.concat(dfs, ignore_index=True)
        if df.index.tz is None:
            df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_localize("UTC")
        df["timestamp"] = df["timestamp"].dt.tz_convert("America/New_York")
        return df
    return pd.DataFrame()


print("\n📊 Converting results to DataFrames...")

# Fama factors
fama_log_prices = []
fama_returns = []

for t, val in results.get("fama_log_prices", []):
    ts = pd.to_datetime(t)
    if ts.tz is None:
        ts = ts.tz_localize("UTC")
    ts = ts.tz_convert("America/New_York")
    fama_log_prices.append(
        {
            "timestamp": ts,
            "HML": val.HML,
            "SMB": val.SMB,
            "RMW": val.RMW,
            "CMA": val.CMA,
            "MKT_RF": val.MKT_RF,
        }
    )

for t, val in results.get("fama_returns", []):
    ts = pd.to_datetime(t)
    if ts.tz is None:
        ts = ts.tz_localize("UTC")
    ts = ts.tz_convert("America/New_York")
    fama_returns.append(
        {
            "timestamp": ts,
            "HML": val.HML,
            "SMB": val.SMB,
            "RMW": val.RMW,
            "CMA": val.CMA,
            "MKT_RF": val.MKT_RF,
        }
    )

df_fama_log = pd.DataFrame(fama_log_prices).set_index("timestamp")
df_fama_ret = pd.DataFrame(fama_returns).set_index("timestamp")

print(f"✅ Fama log prices: {df_fama_log.shape}")
print(f"✅ Fama returns: {df_fama_ret.shape}")

# Microstructure bars
bars_data = []
for key, data in results.items():
    if key.startswith("bar_"):
        ticker = key.replace("bar_", "")
        for t, bar in data:
            ts = pd.to_datetime(t)
            if ts.tz is None:
                ts = ts.tz_localize("UTC")
            ts = ts.tz_convert("America/New_York")
            bars_data.append(
                {
                    "timestamp": ts,
                    "ticker": ticker,
                    "log_mid": bar.log_mid,
                    "iso_flow_intensity": bar.iso_flow_intensity,
                    "total_flow": bar.total_flow,
                    "num_trades": bar.num_trades,
                    "avg_rsprd": bar.avg_rsprd,
                    "pct_trades_iso": bar.pct_trades_iso,
                }
            )

df_bars = pd.DataFrame(bars_data)
print(f"✅ Microstructure bars: {df_bars.shape}")

# Save to parquet files
print(f"\n💾 Saving data to parquet files...")
output_dir = Path("outputs")
output_dir.mkdir(exist_ok=True)

df_fama_log.to_parquet(output_dir / "fama_log_prices.parquet")
df_fama_ret.to_parquet(output_dir / "fama_returns.parquet")
df_bars.to_parquet(output_dir / "microstructure_bars.parquet")

print(f"✅ Saved fama_log_prices.parquet ({len(df_fama_log)} rows)")
print(f"✅ Saved fama_returns.parquet ({len(df_fama_ret)} rows)")
print(f"✅ Saved microstructure_bars.parquet ({len(df_bars)} rows)")


# =============================================================================
# Analysis 1: Regression - Fama Factors vs Equity Returns
# =============================================================================

print(f"\n{'=' * 80}")
print("ANALYSIS 1: Regressing Fama Factor Returns on Equity Returns")
print(f"{'=' * 80}\n")

# Pivot bars to wide format (log_mid by ticker)
df_log_mid = df_bars.pivot(index="timestamp", columns="ticker", values="log_mid")
df_log_mid_ret = df_log_mid.diff()  # Log returns

# Align timestamps with Fama returns
common_times = df_fama_ret.index.intersection(df_log_mid_ret.index)

if len(common_times) > 100:
    print(f"✅ Found {len(common_times)} common timestamps for regression")

    # Run regression for each factor
    regression_results = {}

    for factor in ["HML", "SMB", "RMW", "CMA", "MKT_RF"]:
        y = df_fama_ret.loc[common_times, factor].dropna()
        X = df_log_mid_ret.loc[y.index].dropna(axis=1, how="all")

        # Drop columns with too many NaNs
        X = X.loc[:, X.notna().sum() > len(X) * 0.5]

        # Align
        common_idx = y.index.intersection(X.index)
        if len(common_idx) > 50:
            y_aligned = y.loc[common_idx]
            X_aligned = X.loc[common_idx].fillna(0)

            # Simple correlation analysis (lightweight)
            correlations = {}
            for col in X_aligned.columns:
                if X_aligned[col].std() > 0:
                    corr, pval = stats.pearsonr(y_aligned, X_aligned[col])
                    correlations[col] = {"corr": corr, "pval": pval}

            # Top 5 most correlated
            top_corr = sorted(correlations.items(), key=lambda x: abs(x[1]["corr"]), reverse=True)[
                :5
            ]

            regression_results[factor] = {
                "n_obs": len(common_idx),
                "top_correlations": top_corr,
            }

            print(f"\n{factor}:")
            print(f"  Observations: {len(common_idx)}")
            print(f"  Top 5 correlated equities:")
            for ticker, stats_dict in top_corr:
                print(f"    {ticker}: r={stats_dict['corr']:.3f}, p={stats_dict['pval']:.4f}")
else:
    print(f"⚠️  Not enough common timestamps: {len(common_times)}")


# =============================================================================
# Analysis 2: VECM Leadership (Factors + ETFs + Select Equities)
# =============================================================================

print(f"\n{'=' * 80}")
print("ANALYSIS 2: VECM Leadership Analysis")
print(f"{'=' * 80}\n")

# Select instruments for VECM:
# - All 5 Fama factors
# - Key ETFs: SPY, QQQ, IWM
# - Top 5 most liquid equities

etfs_of_interest = ["SPY", "QQQ", "IWM"]
equities_for_vecm = ["AAPL", "MSFT", "NVDA", "TSLA", "AMZN"]  # Top liquid names

# Build combined dataset
vecm_data = df_fama_log.copy()
vecm_data.columns = [f"FAMA_{col}" for col in vecm_data.columns]

# Add equity log mids
for ticker in etfs_of_interest + equities_for_vecm:
    if ticker in df_log_mid.columns:
        vecm_data[ticker] = df_log_mid[ticker]

# Drop NaNs
vecm_data = vecm_data.dropna()

print(f"VECM dataset shape: {vecm_data.shape}")
print(f"Columns: {list(vecm_data.columns)}")

if vecm_data.shape[0] > 60 and vecm_data.shape[1] > 2:
    print(f"\n✅ Running VECM (using statsmodels)...")

    try:
        from statsmodels.tsa.vector_ar.vecm import VECM

        # Use last 60 minutes of data
        vecm_subset = vecm_data.iloc[-60:]

        vecm_model = VECM(vecm_subset, k_ar_diff=1, coint_rank=1)
        vecm_res = vecm_model.fit()

        alpha = vecm_res.alpha
        Omega = vecm_res.sigma_u

        # Compute alpha_perp for information shares
        U, S, Vt = np.linalg.svd(alpha, full_matrices=True)
        alpha_perp = U[:, -1]
        sgn = np.sign(alpha_perp[0]) if alpha_perp[0] != 0 else 1.0
        alpha_perp = sgn * alpha_perp
        denom = np.sum(np.abs(alpha_perp))
        alpha_perp = (
            alpha_perp / denom if denom > 1e-12 else np.ones(len(alpha_perp)) / len(alpha_perp)
        )

        # Information shares
        IS = np.abs(alpha_perp)
        IS = IS / IS.sum()

        leadership_scores = {col: float(IS[i]) for i, col in enumerate(vecm_subset.columns)}

        print(f"\n📊 INFORMATION SHARES (Leadership):")
        sorted_leadership = sorted(leadership_scores.items(), key=lambda x: x[1], reverse=True)
        for rank, (name, score) in enumerate(sorted_leadership, 1):
            print(f"  {rank}. {name}: {score:.4f}")

        # Key insight: Are ETFs leading?
        etf_share = sum(v for k, v in leadership_scores.items() if k in etfs_of_interest)
        factor_share = sum(v for k, v in leadership_scores.items() if k.startswith("FAMA_"))
        equity_share = sum(v for k, v in leadership_scores.items() if k in equities_for_vecm)

        print(f"\n🔍 LEADERSHIP BREAKDOWN:")
        print(f"  ETFs ({', '.join(etfs_of_interest)}): {etf_share:.2%}")
        print(f"  Fama Factors: {factor_share:.2%}")
        print(f"  Individual Equities: {equity_share:.2%}")

        if etf_share > equity_share:
            print(f"\n✅ ETFs have MORE information share than individual equities!")
            print(f"   This suggests price discovery happens in the ETF market.")

    except Exception as e:
        print(f"⚠️  VECM failed: {e}")
else:
    print(f"⚠️  Not enough data for VECM: {vecm_data.shape}")


# =============================================================================
# Visualization: Save Key Plots
# =============================================================================

print(f"\n{'=' * 80}")
print("VISUALIZATION: Creating plots...")
print(f"={'=' * 80}\n")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# Filter data to skip choppy open (start at 9:45 AM)
filter_start = pd.Timestamp(f"{DATE_OF_INTEREST} 09:45:00", tz="America/New_York")

# Apply filter to all dataframes
df_fama_log_filtered = df_fama_log[df_fama_log.index >= filter_start]
df_fama_ret_filtered = df_fama_ret[df_fama_ret.index >= filter_start]
df_log_mid_filtered = df_log_mid[df_log_mid.index >= filter_start]
df_bars_filtered = df_bars[df_bars["timestamp"] >= filter_start]

# Create vertical layout: N×1 subplots
num_plots = (
    5
    + len(df_fama_log.columns)
    + len(df_fama_ret.columns)
    + len(etfs_of_interest)
    + len(equities_for_vecm)
)
fig = plt.figure(figsize=(16, num_plots * 2))
fig.suptitle(
    f"Fama-French Factors + Market Microstructure Analysis | {DATE_OF_INTEREST}",
    fontsize=16,
    fontweight="bold",
    y=0.995,
)
plot_idx = 1

# ============================================================================
# SECTION 1: FAMA FACTOR LOG PRICES (Individual)
# ============================================================================
for i, factor in enumerate(df_fama_log_filtered.columns):
    ax = plt.subplot(num_plots, 1, plot_idx)
    normalized = df_fama_log_filtered[factor] - df_fama_log_filtered[factor].iloc[0]
    ax.plot(
        df_fama_log_filtered.index,
        normalized,
        label=factor,
        alpha=0.8,
        linewidth=1.5,
        color=f"C{i}",
    )
    ax.set_title(
        f"Fama Factor: {factor} (Log Price, Normalized to Start)", fontsize=10, fontweight="bold"
    )
    ax.set_ylabel("Log Price Change", fontsize=9)
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    if plot_idx == 1:
        ax.text(
            0.02,
            0.98,
            "FAMA FACTORS - LOG PRICES",
            transform=ax.transAxes,
            fontsize=11,
            fontweight="bold",
            va="top",
            ha="left",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8),
        )
    plot_idx += 1

# ============================================================================
# SECTION 2: FAMA FACTOR RETURNS (Individual)
# ============================================================================
for i, factor in enumerate(df_fama_ret_filtered.columns):
    ax = plt.subplot(num_plots, 1, plot_idx)
    cumsum = df_fama_ret_filtered[factor].cumsum()
    ax.plot(
        df_fama_ret_filtered.index, cumsum, label=factor, alpha=0.8, linewidth=1.5, color=f"C{i}"
    )
    ax.set_title(f"Fama Factor: {factor} (Cumulative Return)", fontsize=10, fontweight="bold")
    ax.set_ylabel("Cumulative Return", fontsize=9)
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    if plot_idx == len(df_fama_log_filtered.columns) + 1:
        ax.text(
            0.02,
            0.98,
            "FAMA FACTORS - RETURNS",
            transform=ax.transAxes,
            fontsize=11,
            fontweight="bold",
            va="top",
            ha="left",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgreen", alpha=0.8),
        )
    plot_idx += 1

# ============================================================================
# SECTION 3: ETF LOG PRICES (Individual)
# ============================================================================
for i, ticker in enumerate(etfs_of_interest):
    if ticker in df_log_mid_filtered.columns:
        ax = plt.subplot(num_plots, 1, plot_idx)
        pct_change = (df_log_mid_filtered[ticker] - df_log_mid_filtered[ticker].iloc[0]) * 100
        ax.plot(
            df_log_mid_filtered.index,
            pct_change,
            label=ticker,
            linewidth=1.5,
            alpha=0.8,
            color=f"C{i}",
        )
        ax.set_title(f"ETF: {ticker} (% Change from Start)", fontsize=10, fontweight="bold")
        ax.set_ylabel("% Change", fontsize=9)
        ax.legend(loc="upper left", fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        if i == 0:
            ax.text(
                0.02,
                0.98,
                "ETFs - LOG MID PRICES",
                transform=ax.transAxes,
                fontsize=11,
                fontweight="bold",
                va="top",
                ha="left",
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8),
            )
        plot_idx += 1

# ============================================================================
# SECTION 4: TOP EQUITY LOG PRICES (Individual)
# ============================================================================
for i, ticker in enumerate(equities_for_vecm):
    if ticker in df_log_mid_filtered.columns:
        ax = plt.subplot(num_plots, 1, plot_idx)
        pct_change = (df_log_mid_filtered[ticker] - df_log_mid_filtered[ticker].iloc[0]) * 100
        ax.plot(
            df_log_mid_filtered.index,
            pct_change,
            label=ticker,
            linewidth=1.5,
            alpha=0.8,
            color=f"C{i}",
        )
        ax.set_title(f"Equity: {ticker} (% Change from Start)", fontsize=10, fontweight="bold")
        ax.set_ylabel("% Change", fontsize=9)
        ax.legend(loc="upper left", fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        if i == 0:
            ax.text(
                0.02,
                0.98,
                "EQUITIES - LOG MID PRICES",
                transform=ax.transAxes,
                fontsize=11,
                fontweight="bold",
                va="top",
                ha="left",
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightcoral", alpha=0.8),
            )
        plot_idx += 1

# ============================================================================
# SECTION 5: MICROSTRUCTURE FEATURES
# ============================================================================
# ISO Flow Intensity (SPY)
if "SPY" in df_bars_filtered["ticker"].values:
    ax = plt.subplot(num_plots, 1, plot_idx)
    spy_data = df_bars_filtered[df_bars_filtered["ticker"] == "SPY"].set_index("timestamp")
    ax.plot(
        spy_data.index, spy_data["iso_flow_intensity"], color="orange", alpha=0.8, linewidth=1.5
    )
    ax.set_title("SPY: ISO Flow Intensity", fontsize=10, fontweight="bold")
    ax.set_ylabel("ISO Flow Intensity", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.text(
        0.02,
        0.98,
        "MICROSTRUCTURE FEATURES",
        transform=ax.transAxes,
        fontsize=11,
        fontweight="bold",
        va="top",
        ha="left",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="lavender", alpha=0.8),
    )
    plot_idx += 1

    # Average Relative Spread (SPY)
    ax = plt.subplot(num_plots, 1, plot_idx)
    ax.plot(spy_data.index, spy_data["avg_rsprd"] * 10000, color="red", alpha=0.8, linewidth=1.5)
    ax.set_title("SPY: Average Relative Spread", fontsize=10, fontweight="bold")
    ax.set_ylabel("Spread (bps)", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    plot_idx += 1

    # Total Flow (SPY)
    ax = plt.subplot(num_plots, 1, plot_idx)
    ax.plot(spy_data.index, spy_data["total_flow"], color="green", alpha=0.8, linewidth=1.5)
    ax.set_title("SPY: Total Flow", fontsize=10, fontweight="bold")
    ax.set_ylabel("Total Flow", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    plot_idx += 1

    # Num Trades (SPY)
    ax = plt.subplot(num_plots, 1, plot_idx)
    ax.plot(spy_data.index, spy_data["num_trades"], color="purple", alpha=0.8, linewidth=1.5)
    ax.set_title("SPY: Number of Trades per Minute", fontsize=10, fontweight="bold")
    ax.set_ylabel("Num Trades", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    plot_idx += 1

    # % Trades ISO (SPY)
    ax = plt.subplot(num_plots, 1, plot_idx)
    ax.plot(
        spy_data.index, spy_data["pct_trades_iso"] * 100, color="brown", alpha=0.8, linewidth=1.5
    )
    ax.set_title("SPY: % of Trades that are ISO", fontsize=10, fontweight="bold")
    ax.set_ylabel("% ISO Trades", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    plot_idx += 1

plt.tight_layout()
output_path = Path("outputs/fama_full_analysis.pdf")
output_path.parent.mkdir(exist_ok=True)
plt.savefig(output_path, dpi=300, bbox_inches="tight")
print(f"✅ Plot saved to: {output_path}")
plt.close()


# =============================================================================
# Summary
# =============================================================================

print(f"\n{'=' * 80}")
print("SUMMARY")
print(f"={'=' * 80}\n")

print(f"📊 Data Collection:")
print(f"  - Fama log prices: {len(df_fama_log)} ticks")
print(f"  - Fama returns: {len(df_fama_ret)} ticks")
print(
    f"  - Microstructure bars: {len(df_bars)} observations across {df_bars['ticker'].nunique()} tickers"
)

print(f"\n🔬 Regression Analysis:")
print(f"  - Tested correlation between Fama factors and {len(df_log_mid.columns)} equity returns")
print(f"  - Results show which equities are most aligned with factor movements")

print(f"\n📈 VECM Leadership:")
print(f"  - Analyzed {vecm_data.shape[1]} instruments: {vecm_data.shape[0]} periods")
if "sorted_leadership" in locals():
    print(f"  - Top leader: {sorted_leadership[0][0]} (IS={sorted_leadership[0][1]:.2%})")

print(f"\n✅ Analysis complete! All data saved and visualized.")
print(f"{'=' * 80}\n")
