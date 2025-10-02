"""
Batch RND Extraction for Multiple Tickers
==========================================

Processes all tickers from instrument_selected_instruments.json:
- Loads options data for each ticker
- Infers spot price from put-call parity
- Extracts RND with runtime tracking
- Saves plots and runtime summary
"""

import json

import matplotlib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

matplotlib.use("Agg")  # Non-interactive backend
import gc
import pickle
import time
import warnings
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import pyarrow.parquet as pq
from py_vollib_vectorized import vectorized_implied_volatility as viv

from rnd_extraction import extract_rnd_ultra_simple

# Suppress all warnings
warnings.filterwarnings("ignore")

# ==================== CONFIGURATION ====================
INSTRUMENTS_JSON = Path("/home/grego/Alphathon/instrument_selected_instruments.json")
DATA_DIR = Path("/home/grego/Alphathon/AlphathonDataSets/options_selected_formatted")
OUTPUT_DIR = Path("/home/grego/Alphathon/example_RND_10am_1105")

slice_time = pd.Timestamp("2024-11-05 10:00:00", tz="America/New_York")
expiration_time = pd.Timestamp("2024-11-15 16:00:00", tz="America/New_York")
RISK_FREE_RATE = 0.05341
MIN_BID = 0.05
USE_LOWRANK = True  # Low-rank GP for ~10-64x speedup (uses n//4 inducing points)
# ========================================================

# Create output directories
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
PICKLE_DIR = OUTPUT_DIR / "rnd_pickles"
PICKLE_DIR.mkdir(exist_ok=True, parents=True)
print(f"Output directory: {OUTPUT_DIR}")
print(f"Pickle directory: {PICKLE_DIR}")

# Load instruments
with open(INSTRUMENTS_JSON, "r") as f:
    instruments = json.load(f)

all_tickers = instruments["selected_stocks"] + instruments["selected_ETFs"]
print(
    f"\nProcessing {len(all_tickers)} tickers: {len(instruments['selected_stocks'])} stocks + {len(instruments['selected_ETFs'])} ETFs"
)
print(f"Mode: {'Low-Rank GP (fast)' if USE_LOWRANK else 'Exact GP (accurate)'}")

# Track results
results = []
errors = []

# Process all tickers
for idx, ticker in enumerate(all_tickers, 1):
    print(f"\n[{idx}/{len(all_tickers)}] Processing {ticker}...", end=" ", flush=True)

    try:
        if ticker == "BRK.B":
            ticker = "BRKB"
        # Construct data path
        data_path = DATA_DIR / f"{ticker}_20241115_20241105_20241105_1000.parquet"

        if not data_path.exists():
            print(f"SKIP (file not found)")
            errors.append({"ticker": ticker, "error": "File not found"})
            continue

        # Load data at exact slice time for memory efficiency
        t0_load = time.perf_counter()
        # Use pyarrow to query only the timestamp == slice_time

        table = pq.read_table(data_path)
        df = table.to_pandas()

        # Filter for exact slice time (assuming timestamp column is already in NY timezone)
        data = df[df["timestamp"] == slice_time].copy()
        if len(data) == 0:
            # Find nearest time if exact doesn't exist
            unique_times = df["timestamp"].unique()
            nearest_idx = np.argmin(np.abs(unique_times - slice_time))
            actual_slice_time = unique_times[nearest_idx]
            data = df[df["timestamp"] == actual_slice_time].copy()
        else:
            actual_slice_time = slice_time

        # Clean up to save memory
        del df
        gc.collect()

        load_time = time.perf_counter() - t0_load
        print(f"loaded,", end=" ", flush=True)

        # Calculate TTE
        TTE_YEARS = (
            (expiration_time - actual_slice_time).total_seconds() / (24 * 3600) / 365.25
        )

        # Infer spot price from put-call parity
        temp_mid = (data["bid"] + data["ask"]) / 2
        temp_strike = data["strike"].values
        temp_right = data["right"].str.lower().values

        unique_strikes = np.unique(temp_strike)
        median_strike = np.median(unique_strikes)
        closest_idx = np.argmin(np.abs(unique_strikes - median_strike))
        atm_strike = unique_strikes[closest_idx]

        call_mask = (temp_right == "c") & (temp_strike == atm_strike)
        put_mask = (temp_right == "p") & (temp_strike == atm_strike)

        if call_mask.sum() > 0 and put_mask.sum() > 0:
            call_mid = temp_mid[call_mask].iloc[0]
            put_mid = temp_mid[put_mask].iloc[0]
            spot_price = (
                call_mid - put_mid + atm_strike * np.exp(-RISK_FREE_RATE * TTE_YEARS)
            )
        else:
            print(f"SKIP (no spot price)")
            errors.append({"ticker": ticker, "error": "Could not infer spot price"})
            continue

        # Prepare options data
        vol = data[
            ["strike", "right", "bid", "ask", "bid_size", "ask_size"]
        ].reset_index(drop=True)
        vol["mid"] = (vol["bid"] + vol["ask"]) / 2
        vol["microprice"] = (
            vol["ask_size"] * vol["bid"] + vol["bid_size"] * vol["ask"]
        ) / (vol["ask_size"] + vol["bid_size"])
        vol = vol[vol.bid > MIN_BID].copy()
        vol["right"] = vol["right"].str.lower()

        # Clean up original data
        del data
        gc.collect()

        if len(vol) < 10:
            print(f"SKIP (only {len(vol)} options)")
            errors.append({"ticker": ticker, "error": f"Too few options ({len(vol)})"})
            continue

        # ===== RND EXTRACTION (TIMED) =====
        t0_rnd = time.perf_counter()
        result = extract_rnd_ultra_simple(
            strikes=vol["strike"].values,
            rights=vol["right"].values,
            bids=vol["bid"].values,
            asks=vol["ask"].values,
            bid_sizes=vol["bid_size"].values,
            ask_sizes=vol["ask_size"].values,
            spot_price=spot_price,
            risk_free_rate=RISK_FREE_RATE,
            time_to_expiry=TTE_YEARS,
            grid_points=300,
            min_bid=MIN_BID,
            use_lowrank=USE_LOWRANK,
        )
        rnd_time = time.perf_counter() - t0_rnd
        # ==================================
        print(f"RND done ({rnd_time*1000:.1f}ms),", end=" ", flush=True)

        # Calculate statistics
        integral = np.trapz(result.rnd_density, result.log_moneyness)
        E_S = np.trapz(result.strikes * result.rnd_density, result.log_moneyness)
        forward = spot_price * np.exp(RISK_FREE_RATE * TTE_YEARS)

        # Calculate market IVs for plotting
        print(f"calc IV,", end=" ", flush=True)
        vol["market_iv"] = viv(
            price=vol["microprice"],
            S=spot_price,
            K=vol["strike"],
            t=TTE_YEARS,
            r=RISK_FREE_RATE,
            flag=vol["right"],
            model="black_scholes",
            return_as="numpy",
        )
        print(f"IV done,", end=" ", flush=True)

        vol_otm = vol[
            ((vol["right"] == "c") & (vol["strike"] > forward))
            | ((vol["right"] == "p") & (vol["strike"] < forward))
        ].copy()

        # Create plot
        print(f"plotting,", end=" ", flush=True)
        t0_plot = time.perf_counter()
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        colors = ["purple" if r == "c" else "blue" for r in vol_otm["right"]]
        print(f"fig created,", end=" ", flush=True)

        # 1. IV Surface
        ax1 = axes[0]
        ax1.scatter(
            vol_otm["strike"],
            vol_otm["market_iv"],
            c=colors,
            s=40,
            alpha=0.6,
            label="Market IV",
            edgecolors="black",
            linewidths=0.5,
        )
        ax1.plot(
            result.strikes, result.fitted_iv, "r-", linewidth=2.5, label="Fitted IV"
        )
        ax1.fill_between(
            result.strikes,
            result.fitted_iv - 1.96 * result.fitted_iv_std,
            result.fitted_iv + 1.96 * result.fitted_iv_std,
            alpha=0.2,
            color="red",
            label="95% CI",
        )
        ax1.axvline(
            forward,
            color="k",
            linestyle="--",
            alpha=0.5,
            linewidth=1.5,
            label="Forward",
        )
        ax1.axvline(
            spot_price,
            color="g",
            linestyle="--",
            alpha=0.5,
            linewidth=1.5,
            label="Spot",
        )
        ax1.set_xlabel("Strike", fontsize=11)
        ax1.set_ylabel("Implied Volatility", fontsize=11)
        ax1.set_title("IV Surface", fontsize=12, fontweight="bold")
        ax1.legend(fontsize=9, loc="best")
        ax1.grid(True, alpha=0.3)

        # 2. RND
        ax2 = axes[1]
        ax2.bar(
            result.log_moneyness,
            result.rnd_density,
            width=(result.log_moneyness[1] - result.log_moneyness[0]),
            alpha=0.6,
            color="steelblue",
            edgecolor="navy",
            linewidth=0.5,
        )
        ax2.axvline(0, color="k", linestyle="--", alpha=0.6, linewidth=1.5, label="ATM")
        ax2.set_xlabel("Log-Moneyness", fontsize=11)
        ax2.set_ylabel("Density", fontsize=11)
        ax2.set_title(
            f"Risk-Neutral Density (integral={integral:.4f})",
            fontsize=12,
            fontweight="bold",
        )
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3, axis="y")

        # 3. CDF
        ax3 = axes[2]
        ax3.plot(result.strikes, result.rnd_cumulative, "g-", linewidth=2.5)
        ax3.axvline(
            forward,
            color="k",
            linestyle="--",
            alpha=0.5,
            linewidth=1.5,
            label="Forward",
        )
        ax3.axvline(
            spot_price,
            color="r",
            linestyle="--",
            alpha=0.5,
            linewidth=1.5,
            label="Spot",
        )
        ax3.axhline(0.5, color="gray", linestyle=":", alpha=0.5, linewidth=1.5)
        ax3.set_xlabel("Strike", fontsize=11)
        ax3.set_ylabel("Cumulative Probability", fontsize=11)
        ax3.set_title(
            f"Cumulative RND (max={result.rnd_cumulative[-1]:.4f})",
            fontsize=12,
            fontweight="bold",
        )
        ax3.legend(fontsize=9)
        ax3.grid(True, alpha=0.3)

        plt.suptitle(
            f'{ticker} RND Extraction - {actual_slice_time.strftime("%Y-%m-%d %H:%M")} (Exp: {expiration_time.strftime("%Y-%m-%d")})',
            fontsize=14,
            fontweight="bold",
        )
        plt.tight_layout()

        # Save plot
        plot_path = OUTPUT_DIR / f"{ticker.lower()}_rnd.png"
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close()
        plot_time = time.perf_counter() - t0_plot
        print(f"plot saved,", end=" ", flush=True)

        # Save RND data as pickle for CBRA/portfolio analysis
        rnd_data = {
            "ticker": ticker,
            "spot_price": spot_price,
            "forward_price": forward,
            "tte_years": TTE_YEARS,
            "strikes": result.strikes,
            "log_moneyness": result.log_moneyness,
            "rnd_density": result.rnd_density,
            "rnd_cumulative": result.rnd_cumulative,
            "fitted_iv": result.fitted_iv,
            "slice_time": actual_slice_time,
            "expiration_time": expiration_time,
        }
        pickle_path = PICKLE_DIR / f"{ticker.lower()}_rnd.pkl"
        with open(pickle_path, "wb") as f:
            pickle.dump(rnd_data, f)
        print(f"pickle saved,", end=" ", flush=True)

        # Record results
        results.append(
            {
                "ticker": ticker,
                "spot_price": spot_price,
                "n_options": len(vol),
                "rnd_extraction_ms": rnd_time * 1000,
                "load_time_ms": load_time * 1000,
                "plot_time_ms": plot_time * 1000,
                "integral": integral,
                "mean_iv": result.fitted_iv.mean(),
                "success": True,
            }
        )

        print(f"✓ COMPLETE")

        # Clean up memory
        del df, data, vol, vol_otm, fig
        gc.collect()

    except Exception as e:
        print(f"ERROR: {str(e)}")
        errors.append({"ticker": ticker, "error": str(e)})
        results.append(
            {
                "ticker": ticker,
                "spot_price": np.nan,
                "n_options": 0,
                "rnd_extraction_ms": np.nan,
                "load_time_ms": np.nan,
                "plot_time_ms": np.nan,
                "integral": np.nan,
                "mean_iv": np.nan,
                "success": False,
            }
        )

# Save results
results_df = pd.DataFrame(results)
results_df.to_csv(OUTPUT_DIR / "rnd_extraction_runtimes.csv", index=False)

# Print summary
print("\n" + "=" * 80)
print("BATCH RND EXTRACTION SUMMARY")
print("=" * 80)

successful = results_df[results_df["success"]].copy()
print(f"\nTotal tickers: {len(all_tickers)}")
print(f"Successful: {len(successful)}")
print(f"Failed: {len(all_tickers) - len(successful)}")

if len(successful) > 0:
    print(f"\n--- RND Extraction Runtime Statistics (successful only) ---")
    print(f"Mean:   {successful['rnd_extraction_ms'].mean():.2f} ms")
    print(f"Median: {successful['rnd_extraction_ms'].median():.2f} ms")
    print(f"Min:    {successful['rnd_extraction_ms'].min():.2f} ms")
    print(f"Max:    {successful['rnd_extraction_ms'].max():.2f} ms")
    print(f"Std:    {successful['rnd_extraction_ms'].std():.2f} ms")

    print(f"\n--- Top 5 Fastest RND Extractions ---")
    print(
        successful.nsmallest(5, "rnd_extraction_ms")[
            ["ticker", "rnd_extraction_ms", "n_options"]
        ]
    )

    print(f"\n--- Top 5 Slowest RND Extractions ---")
    print(
        successful.nlargest(5, "rnd_extraction_ms")[
            ["ticker", "rnd_extraction_ms", "n_options"]
        ]
    )

if len(errors) > 0:
    print(f"\n--- Errors ({len(errors)}) ---")
    for err in errors:
        print(f"  {err['ticker']}: {err['error']}")

print(f"\nResults saved to: {OUTPUT_DIR / 'rnd_extraction_runtimes.csv'}")
print(f"Plots saved to: {OUTPUT_DIR}/")
print("\n" + "=" * 80)
