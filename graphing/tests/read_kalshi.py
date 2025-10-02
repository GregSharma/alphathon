import logging
from datetime import timedelta

import pandas as pd
import pytz
from csp.basketlib import sample_dict

import csp
from CSP_Options.csp_tqdm import run_with_progress
from CSP_Options.nodes import bivariate_kalshi_filter
from CSP_Options.utils.readers import get_kalshi_trades

logging.basicConfig(level=logging.INFO)

DATE_OF_INTEREST = "2024-11-05"

"""
Kalshi presidential election trades (PRES-2024-KH and PRES-2024-DJT)
Testing bivariate Kalman filter (Joint-UCBM)
"""

start_ts = pd.Timestamp(f"{DATE_OF_INTEREST} 09:30:00", tz="America/New_York")
end_ts = pd.Timestamp(f"{DATE_OF_INTEREST} 16:00:00", tz="America/New_York")  # Full trading day


@csp.graph
def kalshi_raw_prices():
    """Test 1: Raw price logging (original functionality)"""
    kalshi_basket = get_kalshi_trades()
    test_timer = csp.timer(timedelta(milliseconds=100))
    sampled_kalshi = sample_dict(test_timer, kalshi_basket)
    csp.log(
        logging.INFO,
        "kalshi_raw",
        csp.get_basket_field(sampled_kalshi, "price"),
        logger_tz=pytz.timezone("America/New_York"),
        use_thread=True,
    )


@csp.graph
def kalshi_filtered():
    """Test 2: Bivariate Kalman filter outputs - collect for plotting"""
    kalshi_basket = get_kalshi_trades()

    # Extract individual contract streams
    djt_stream = kalshi_basket["PRES-2024-DJT"]
    kh_stream = kalshi_basket["PRES-2024-KH"]

    # Apply bivariate filter
    filter_outputs = bivariate_kalshi_filter(
        djt_trades=djt_stream,
        kh_trades=kh_stream,
        obs_noise_std=0.005,  # 0.5¢ trade noise
        process_var_base=1e-5,
        correlation_prior=-0.97,
        window_size=60,
    )

    # Collect outputs using add_graph_output (for DataFrame conversion)
    csp.add_graph_output("b_djt", filter_outputs.b_djt)
    csp.add_graph_output("b_kh", filter_outputs.b_kh)
    csp.add_graph_output("u_djt", filter_outputs.u_djt)
    csp.add_graph_output("u_kh", filter_outputs.u_kh)
    csp.add_graph_output("nu_squared", filter_outputs.nu_squared)
    csp.add_graph_output("rho", filter_outputs.rho)
    csp.add_graph_output("innovation_djt", filter_outputs.innovation_djt)
    csp.add_graph_output("innovation_kh", filter_outputs.innovation_kh)
    csp.add_graph_output("vig", filter_outputs.vig)  # Track vig before normalization

    # Also collect raw trade prices for comparison
    csp.add_graph_output(
        "raw_price_djt", csp.sample(csp.timer(timedelta(seconds=1)), djt_stream.price)
    )
    csp.add_graph_output(
        "raw_price_kh", csp.sample(csp.timer(timedelta(seconds=1)), kh_stream.price)
    )


def convert_to_dataframe(results_dict):
    """Convert CSP output dictionary to pandas DataFrame"""
    import numpy as np

    # Convert each output to a dataframe
    dfs = {}
    for key, data in results_dict.items():
        if data:  # Check if data exists
            times, values = zip(*data)
            df_temp = pd.DataFrame({"timestamp": pd.to_datetime(times), key: values})
            # Group by timestamp and take the last value (most recent)
            df_temp = df_temp.groupby("timestamp").last()
            dfs[key] = df_temp

    # Merge all dataframes on timestamp (outer join to keep all timestamps)
    if dfs:
        df = pd.concat(dfs.values(), axis=1, join="outer")
        # Forward fill any NaN values (from misaligned timestamps)
        df = df.ffill()
        # Localize to UTC first, then convert to ET for plotting
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        df.index = df.index.tz_convert("America/New_York")
        return df
    return pd.DataFrame()


def plot_kalshi_analysis(df, output_path="outputs/kalshi_filter_analysis.pdf"):
    """Create comprehensive visualization of Kalshi filter outputs"""
    from pathlib import Path

    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    # Create output directory if needed
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Set up the plot style
    plt.style.use("seaborn-v0_8-darkgrid")
    plt.rcParams["figure.figsize"] = (16, 20)
    plt.rcParams["font.size"] = 10
    plt.rcParams["axes.labelsize"] = 11
    plt.rcParams["axes.titlesize"] = 12
    plt.rcParams["legend.fontsize"] = 9

    # Create figure with subplots
    fig, axes = plt.subplots(5, 2, figsize=(16, 20))
    fig.suptitle(
        "Bivariate Kalman Filter Analysis: Kalshi Election Contracts\n(Joint-UCBM Model: Nov 5, 2024)",
        fontsize=16,
        y=0.995,
        fontweight="bold",
    )

    # Time formatter for x-axis
    time_fmt = mdates.DateFormatter("%H:%M", tz=pytz.timezone("America/New_York"))

    # 1. Filtered Probabilities (Top Left)
    ax = axes[0, 0]
    if "b_djt" in df.columns and "b_kh" in df.columns:
        ax.plot(
            df.index, df["b_djt"], label="DJT (Filtered)", color="#d62728", linewidth=1.5, alpha=0.9
        )
        ax.plot(
            df.index, df["b_kh"], label="KH (Filtered)", color="#1f77b4", linewidth=1.5, alpha=0.9
        )
        ax.axhline(y=0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
        ax.set_ylabel("Probability")
        ax.set_title("Filtered Probabilities", fontweight="bold")
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(time_fmt)
        ax.set_ylim([0.35, 0.65])

    # 2. Raw vs Filtered (Top Right)
    ax = axes[0, 1]
    if "raw_price_djt" in df.columns and "b_djt" in df.columns:
        ax.plot(
            df.index,
            df["raw_price_djt"] / 100,
            label="DJT Raw",
            color="#d62728",
            linewidth=0.8,
            alpha=0.4,
            linestyle=":",
        )
        ax.plot(
            df.index, df["b_djt"], label="DJT Filtered", color="#d62728", linewidth=1.5, alpha=0.9
        )
        ax.set_ylabel("Probability (DJT)")
        ax.set_title("Raw vs Filtered: DJT Contract", fontweight="bold")
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(time_fmt)

    # 3. Probit Gauge States (Second Row Left)
    ax = axes[1, 0]
    if "u_djt" in df.columns and "u_kh" in df.columns:
        ax.plot(df.index, df["u_djt"], label="U_DJT", color="#d62728", linewidth=1.2, alpha=0.9)
        ax.plot(df.index, df["u_kh"], label="U_KH", color="#1f77b4", linewidth=1.2, alpha=0.9)
        ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
        ax.set_ylabel("Probit Gauge U_t")
        ax.set_title("Latent States (Probit Gauge)", fontweight="bold")
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(time_fmt)

    # 4. Information Clock (Second Row Right)
    ax = axes[1, 1]
    if "nu_squared" in df.columns:
        ax.plot(
            df.index,
            df["nu_squared"],
            label="ν² (Info Clock)",
            color="#2ca02c",
            linewidth=1.2,
            alpha=0.9,
        )
        ax.set_ylabel("ν²")
        ax.set_title("Information Flow Rate", fontweight="bold")
        ax.set_yscale("log")
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3, which="both")
        ax.xaxis.set_major_formatter(time_fmt)

    # 5. Correlation (Third Row Left)
    ax = axes[2, 0]
    if "rho" in df.columns:
        ax.plot(
            df.index, df["rho"], label="ρ (Correlation)", color="#9467bd", linewidth=1.2, alpha=0.9
        )
        ax.axhline(
            y=-1.0,
            color="gray",
            linestyle="--",
            linewidth=0.8,
            alpha=0.5,
            label="ρ = -1 (Perfect Anti-Corr)",
        )
        ax.set_ylabel("ρ_t")
        ax.set_title("Cross-Correlation", fontweight="bold")
        ax.set_ylim([-1.05, -0.4])
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(time_fmt)

    # 6. Vig Evolution (Third Row Right)
    ax = axes[2, 1]
    if "vig" in df.columns:
        # Plot vig in percentage
        ax.plot(
            df.index,
            df["vig"] * 100,
            label="Vig (Pre-Normalization)",
            color="#ff7f0e",
            linewidth=1.2,
            alpha=0.9,
        )
        ax.axhline(y=0.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5, label="No Vig")
        ax.fill_between(df.index, 0, df["vig"] * 100, alpha=0.2, color="orange")
        ax.set_ylabel("Vig (%)")
        ax.set_title("Market Vig Evolution (Removed by Normalization)", fontweight="bold")
        ax.set_ylim([-0.5, 3.0])
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(time_fmt)
    elif "b_djt" in df.columns and "b_kh" in df.columns:
        # Fallback: show normalized sum (should be exactly 1.0 now)
        prob_sum = df["b_djt"] + df["b_kh"]
        ax.plot(
            df.index,
            prob_sum,
            label="DJT + KH (Normalized)",
            color="#8c564b",
            linewidth=1.2,
            alpha=0.9,
        )
        ax.axhline(y=1.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5, label="Sum = 1")
        ax.set_ylabel("Probability Sum")
        ax.set_title("Complementarity Check (Post-Normalization)", fontweight="bold")
        ax.set_ylim([0.998, 1.002])
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(time_fmt)

    # 7. DJT Innovations (Fourth Row Left)
    ax = axes[3, 0]
    if "innovation_djt" in df.columns:
        innovations = df["innovation_djt"]
        colors = ["#d62728" if x > 0 else "#1f77b4" for x in innovations]
        ax.scatter(df.index, innovations, c=colors, s=10, alpha=0.6, label="Innovations")
        ax.axhline(y=0, color="gray", linestyle="-", linewidth=0.8, alpha=0.5)
        # Mark large jumps
        large_jumps = innovations.abs() > 0.01
        if large_jumps.any():
            ax.scatter(
                df.index[large_jumps],
                innovations[large_jumps],
                color="red",
                s=50,
                marker="x",
                linewidth=2,
                label="Jumps (|ν| > 0.01)",
                zorder=5,
            )
        ax.set_ylabel("Innovation")
        ax.set_title("DJT Innovations (Jump Detection)", fontweight="bold")
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(time_fmt)

    # 8. KH Innovations (Fourth Row Right)
    ax = axes[3, 1]
    if "innovation_kh" in df.columns:
        innovations = df["innovation_kh"]
        colors = ["#d62728" if x > 0 else "#1f77b4" for x in innovations]
        ax.scatter(df.index, innovations, c=colors, s=10, alpha=0.6, label="Innovations")
        ax.axhline(y=0, color="gray", linestyle="-", linewidth=0.8, alpha=0.5)
        # Mark large jumps
        large_jumps = innovations.abs() > 0.01
        if large_jumps.any():
            ax.scatter(
                df.index[large_jumps],
                innovations[large_jumps],
                color="red",
                s=50,
                marker="x",
                linewidth=2,
                label="Jumps (|ν| > 0.01)",
                zorder=5,
            )
        ax.set_ylabel("Innovation")
        ax.set_title("KH Innovations (Jump Detection)", fontweight="bold")
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(time_fmt)

    # 9. Phase Portrait (Fifth Row Left)
    ax = axes[4, 0]
    if "u_djt" in df.columns and "u_kh" in df.columns:
        # Color by time
        scatter = ax.scatter(
            df["u_djt"], df["u_kh"], c=range(len(df)), cmap="viridis", s=5, alpha=0.6
        )
        ax.plot(
            df["u_djt"].iloc[0], df["u_kh"].iloc[0], "go", markersize=10, label="Start", zorder=5
        )
        ax.plot(
            df["u_djt"].iloc[-1], df["u_kh"].iloc[-1], "ro", markersize=10, label="End", zorder=5
        )
        # Complementarity line (U^D ≈ -U^H)
        u_range = [df["u_djt"].min(), df["u_djt"].max()]
        ax.plot(u_range, [-x for x in u_range], "k--", linewidth=1, alpha=0.5, label="U_D = -U_H")
        ax.set_xlabel("U_t (DJT)")
        ax.set_ylabel("U_t (KH)")
        ax.set_title("Phase Portrait: Probit Gauge Dynamics", fontweight="bold")
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label("Time Index")

    # 10. Summary Statistics (Fifth Row Right)
    ax = axes[4, 1]
    ax.axis("off")

    # Compute summary stats
    stats_text = "SUMMARY STATISTICS\n" + "=" * 40 + "\n\n"

    if "b_djt" in df.columns and "b_kh" in df.columns:
        mean_djt = df["b_djt"].mean()
        mean_kh = df["b_kh"].mean()
        std_djt = df["b_djt"].std()
        std_kh = df["b_kh"].std()
        mean_sum = (df["b_djt"] + df["b_kh"]).mean()

        stats_text += f"DJT: μ = {mean_djt:.3f}, σ = {std_djt:.4f}\n"
        stats_text += f"KH:  μ = {mean_kh:.3f}, σ = {std_kh:.4f}\n\n"
        stats_text += f"Post-Norm Sum: {mean_sum:.6f}\n"

        # Show vig stats if available
        if "vig" in df.columns:
            mean_vig = df["vig"].mean() * 100
            std_vig = df["vig"].std() * 100
            max_vig = df["vig"].max() * 100
            stats_text += f"Mean Vig: {mean_vig:.2f}%\n"
            stats_text += f"Std Vig:  {std_vig:.2f}%\n"
            stats_text += f"Max Vig:  {max_vig:.2f}%\n\n"
        else:
            stats_text += f"Vig (calc): {(mean_sum - 1.0) * 100:.2f}%\n\n"

    if "rho" in df.columns:
        mean_rho = df["rho"].mean()
        stats_text += f"Mean ρ: {mean_rho:.3f}\n\n"

    if "nu_squared" in df.columns:
        mean_nu = df["nu_squared"].mean()
        max_nu = df["nu_squared"].max()
        stats_text += f"Mean ν²: {mean_nu:.2e}\n"
        stats_text += f"Max ν²:  {max_nu:.2e}\n\n"

    if "innovation_djt" in df.columns:
        n_jumps_djt = (df["innovation_djt"].abs() > 0.01).sum()
        max_jump_djt = df["innovation_djt"].abs().max()
        stats_text += f"DJT Jumps (|ν| > 0.01): {n_jumps_djt}\n"
        stats_text += f"Max Jump: {max_jump_djt:.4f}\n\n"

    if "innovation_kh" in df.columns:
        n_jumps_kh = (df["innovation_kh"].abs() > 0.01).sum()
        max_jump_kh = df["innovation_kh"].abs().max()
        stats_text += f"KH Jumps (|ν| > 0.01): {n_jumps_kh}\n"
        stats_text += f"Max Jump: {max_jump_kh:.4f}\n\n"

    # Filter info
    stats_text += "=" * 40 + "\n"
    stats_text += "FILTER PARAMETERS\n"
    stats_text += "=" * 40 + "\n"
    stats_text += "Obs Noise: 0.005\n"
    stats_text += "Prior ρ₀: -0.97\n"
    stats_text += "Window: 60s\n"
    stats_text += "Normalization: Inverse Vig-Weighted"

    ax.text(
        0.1,
        0.95,
        stats_text,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        family="monospace",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.3),
    )

    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"\n✅ Plot saved to: {output_path}")
    plt.close()


print(f"\nRunning from {start_ts} to {end_ts}...")
print("Collecting bivariate Kalman filter outputs for full trading day...")

# Run filtered version and collect outputs
results = run_with_progress(
    kalshi_filtered,
    starttime=start_ts,
    endtime=end_ts,
    realtime=False,
)

print(f"✅ CSP run completed! Processing {len(results)} output streams...")

# Convert to DataFrame
df = convert_to_dataframe(results)
print(f"✅ DataFrame created with shape: {df.shape}")
print(f"   Columns: {list(df.columns)}")
print(f"   Time range: {df.index[0]} to {df.index[-1]}")

# Save to parquet for downstream analysis
from pathlib import Path

output_dir = Path("outputs")
output_dir.mkdir(exist_ok=True)
df.to_parquet(output_dir / "kalshi_filtered_data.parquet")
print(f"✅ Saved kalshi_filtered_data.parquet ({len(df)} rows)")

# Create visualizations
print("\n📊 Generating comprehensive visualization...")
plot_kalshi_analysis(df, output_path="outputs/kalshi_filter_analysis.pdf")

print("\n🎉 Analysis complete!")
