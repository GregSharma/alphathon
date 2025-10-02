import os
from typing import Tuple

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

from cbrapipe import (
    discretize_instruments,
    discretize_constraints,
    build_initial_matrix,
    expand_coefficients,
    identify_admissible_blocks,
    cbra_optimize,
    extract_joint_distribution,
)


def build_funky_non_normal_env(n: int = 10000, seed: int | None = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build a FUNKY environment with fat-tailed, skewed distributions.
    
    Marginals:
    - X1, X2: Student-t with df=3 (fat tails, symmetric)
    - X3, X4: Gamma(shape=2, scale=1) shifted to have mean 0 (right-skewed)
    - X5: Lognormal (heavy right tail)
    - X6: Gumbel (left-skewed, extreme value)
    
    Constraints: We'll still use sums but make them interesting
    
    Returns
    -------
    X : (n, d)
    S : (n, K)  
    A : (d, K)
    tilde_alpha : (K, d+K)
    """
    if seed is not None:
        np.random.seed(seed)

    d = 6
    K = 2  # Fewer constraints for this funky example

    # FUNKY Marginals - all centered at 0 but with different shapes
    F_inv_stocks = [
        # X1, X2: Student-t(df=3) - fat tails, symmetric
        lambda p: stats.t.ppf(p, df=3, loc=0, scale=1.0),
        lambda p: stats.t.ppf(p, df=3, loc=0, scale=1.0),
        
        # X3, X4: Gamma shifted to mean 0 (right-skewed with fat right tail)
        lambda p: stats.gamma.ppf(p, a=2, scale=1.5) - 3.0,  # shift so mean ~ 0
        lambda p: stats.gamma.ppf(p, a=2, scale=1.5) - 3.0,
        
        # X5: Lognormal shifted (heavy right tail)
        lambda p: stats.lognorm.ppf(p, s=0.8, scale=1.0) - np.exp(0.8**2 / 2),
        
        # X6: Gumbel (left-skewed extreme value distribution)
        lambda p: stats.gumbel_r.ppf(p, loc=-0.5772, scale=1.0),  # loc is -Euler's constant for mean 0
    ]

    # Constraints: Create OVERLAPPING sums for extra complexity!
    # We'll compute these empirically via Monte Carlo to get true distributions
    # Constraint 1: X1 + X2 + X3 (t + t + gamma)
    # Constraint 2: X3 + X4 + X5 + X6 (gamma + gamma + lognormal + gumbel)
    # NOTE: X3 appears in BOTH constraints - this creates complex dependencies!
    
    # Generate large MC sample to get constraint distributions
    mc_n = 100000
    mc_X = np.zeros((mc_n, d))
    for j in range(d):
        mc_X[:, j] = F_inv_stocks[j](np.random.uniform(0.001, 0.999, mc_n))
    
    # Compute constraint sums
    S1_mc = mc_X[:, 0] + mc_X[:, 1] + mc_X[:, 2]  # X1 + X2 + X3
    S2_mc = mc_X[:, 2] + mc_X[:, 3] + mc_X[:, 4] + mc_X[:, 5]  # X3 + X4 + X5 + X6
    
    # Create empirical quantile functions from MC samples
    S1_sorted = np.sort(S1_mc)
    S2_sorted = np.sort(S2_mc)
    
    def make_empirical_ppf(sorted_samples):
        """Create an empirical PPF (inverse CDF) from sorted samples."""
        def ppf(p):
            idx = int(np.clip(p * len(sorted_samples), 0, len(sorted_samples) - 1))
            return sorted_samples[idx]
        return ppf
    
    F_inv_constraints = [
        make_empirical_ppf(S1_sorted),
        make_empirical_ppf(S2_sorted),
    ]

    # Weight matrix A (d x K)
    # S1 = X1 + X2 + X3
    # S2 = X3 + X4 + X5 + X6  (NOTE: X3 appears in BOTH!)
    A = np.array([
        [1.0, 0.0],  # X1 in S1 only
        [1.0, 0.0],  # X2 in S1 only
        [1.0, 1.0],  # X3 in BOTH S1 and S2 ← OVERLAPPING!
        [0.0, 1.0],  # X4 in S2 only
        [0.0, 1.0],  # X5 in S2 only
        [0.0, 1.0],  # X6 in S2 only
    ])

    X = discretize_instruments(n, F_inv_stocks)
    S = discretize_constraints(n, F_inv_constraints)

    tilde_alpha = expand_coefficients(A)
    
    # Store MC samples for computing true portfolio distribution
    return X, S, A, tilde_alpha


def run_cbra_funky(n: int = 10000, seed: int | None = 42, max_iter: int = 20000) -> tuple[np.ndarray, np.ndarray]:
    """
    Run CBRA for the funky non-normal example and return final X and Y.
    """
    X, S, A, tilde_alpha = build_funky_non_normal_env(n=n, seed=seed)
    Y = build_initial_matrix(X, S)

    # Randomize columns to avoid comonotonic start
    for j in range(Y.shape[1]):
        np.random.shuffle(Y[:, j])

    blocks = identify_admissible_blocks(tilde_alpha)
    Y_final = cbra_optimize(Y, tilde_alpha, blocks, max_iter=max_iter, verbose=True)
    X_final = extract_joint_distribution(Y_final, d=6)
    return X_final, Y_final


def compute_true_portfolio_distribution_mc(seed: int = 999) -> np.ndarray:
    """
    Compute TRUE distribution of equal-weighted portfolio P via Monte Carlo.
    
    We generate independent samples from each marginal and compute P = mean(X_i).
    This is the "ground truth" that CBRA should approximate (but without knowing
    the joint structure).
    
    Returns
    -------
    P_true_samples : np.ndarray
        Large MC sample from the true distribution of P
    """
    np.random.seed(seed)
    
    mc_n = 200000  # Large sample for smooth true distribution
    d = 6
    
    # Define the same funky marginals as in build_funky_non_normal_env
    marginal_rvs = [
        # X1, X2: Student-t(df=3)
        lambda size: stats.t.rvs(df=3, loc=0, scale=1.0, size=size),
        lambda size: stats.t.rvs(df=3, loc=0, scale=1.0, size=size),
        
        # X3, X4: Gamma shifted
        lambda size: stats.gamma.rvs(a=2, scale=1.5, size=size) - 3.0,
        lambda size: stats.gamma.rvs(a=2, scale=1.5, size=size) - 3.0,
        
        # X5: Lognormal shifted
        lambda size: stats.lognorm.rvs(s=0.8, scale=1.0, size=size) - np.exp(0.8**2 / 2),
        
        # X6: Gumbel
        lambda size: stats.gumbel_r.rvs(loc=-0.5772, scale=1.0, size=size),
    ]
    
    # Generate independent samples (this ignores constraints, gives marginal-only truth)
    mc_X = np.zeros((mc_n, d))
    for j in range(d):
        mc_X[:, j] = marginal_rvs[j](mc_n)
    
    # Compute equal-weighted portfolio
    P_true_samples = mc_X.mean(axis=1)
    
    return P_true_samples


def make_pdf_cdf_plot_funky(P_cbra: np.ndarray, P_true_mc: np.ndarray, out_path: str) -> None:
    """
    Plot true vs. CBRA-estimated PDF and CDF for portfolio P (FUNKY VERSION).
    
    Parameters
    ----------
    P_cbra : np.ndarray
        Portfolio values from CBRA (with constraints)
    P_true_mc : np.ndarray
        Portfolio values from Monte Carlo (independent marginals, no constraints)
    out_path : str
        Path to save figure
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # Compute range for plotting
    x_min = min(P_cbra.min(), P_true_mc.min())
    x_max = max(P_cbra.max(), P_true_mc.max())
    
    # For PDF: use KDE for smooth curves
    from scipy.stats import gaussian_kde
    
    # True distribution (from MC)
    kde_true = gaussian_kde(P_true_mc)
    # CBRA distribution
    kde_cbra = gaussian_kde(P_cbra)
    
    x_grid = np.linspace(x_min, x_max, 1000)
    pdf_true = kde_true(x_grid)
    pdf_cbra = kde_cbra(x_grid)

    # Empirical CDFs
    P_true_sorted = np.sort(P_true_mc)
    P_cbra_sorted = np.sort(P_cbra)
    cdf_true_y = np.arange(1, P_true_sorted.size + 1) / P_true_sorted.size
    cdf_cbra_y = np.arange(1, P_cbra_sorted.size + 1) / P_cbra_sorted.size

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # PDF subplot: KDE comparison
    ax_pdf = axes[0]
    ax_pdf.fill_between(x_grid, pdf_cbra, alpha=0.3, color='blue', label='CBRA (with constraints)')
    ax_pdf.plot(x_grid, pdf_cbra, 'b-', lw=2)
    ax_pdf.plot(x_grid, pdf_true, 'r--', lw=2.5, label='True (independent marginals)', alpha=0.8)
    ax_pdf.set_xlim(-5, 5)
    ax_pdf.set_title("PDF: Equal-Weighted Portfolio P\n(Funky Non-Normal Marginals)", fontsize=12, fontweight='bold')
    ax_pdf.set_xlabel("Portfolio Value P", fontsize=11)
    ax_pdf.set_ylabel("Density", fontsize=11)
    ax_pdf.legend(fontsize=10)
    ax_pdf.grid(True, alpha=0.3)

    # CDF subplot: empirical step vs. true cdf
    ax_cdf = axes[1]
    ax_cdf.plot(P_cbra_sorted, cdf_cbra_y, 'b-', lw=2, label='CBRA (with constraints)', alpha=0.7)
    ax_cdf.plot(P_true_sorted, cdf_true_y, 'r--', lw=2.5, label='True (independent marginals)', alpha=0.8)
    ax_cdf.set_xlim(-5, 5)
    ax_cdf.set_title("CDF: Equal-Weighted Portfolio P\n(Funky Non-Normal Marginals)", fontsize=12, fontweight='bold')
    ax_cdf.set_xlabel("Portfolio Value P", fontsize=11)
    ax_cdf.set_ylabel("Cumulative Probability", fontsize=11)
    ax_cdf.legend(fontsize=10)
    ax_cdf.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    
    # Print some stats
    print("\n" + "="*70)
    print("DISTRIBUTION COMPARISON STATISTICS")
    print("="*70)
    print(f"True (MC) - Mean: {P_true_mc.mean():.4f}, Std: {P_true_mc.std():.4f}")
    print(f"          - Skew: {stats.skew(P_true_mc):.4f}, Kurt: {stats.kurtosis(P_true_mc):.4f}")
    print(f"CBRA      - Mean: {P_cbra.mean():.4f}, Std: {P_cbra.std():.4f}")
    print(f"          - Skew: {stats.skew(P_cbra):.4f}, Kurt: {stats.kurtosis(P_cbra):.4f}")
    print("="*70 + "\n")


def main() -> None:
    print("\n" + "="*70)
    print("FUNKY NON-NORMAL PORTFOLIO DISTRIBUTION EXPERIMENT")
    print("="*70)
    print("\nMarginals:")
    print("  X1, X2: Student-t(df=3) - fat tails, symmetric")
    print("  X3, X4: Gamma(2, 1.5) shifted - right-skewed")
    print("  X5:     Lognormal(0.8) shifted - heavy right tail")
    print("  X6:     Gumbel - left-skewed extreme value")
    print("\nConstraints (OVERLAPPING!):")
    print("  S1 = X1 + X2 + X3")
    print("  S2 = X3 + X4 + X5 + X6")
    print("  ⚡ NOTE: X3 appears in BOTH constraints!")
    print("="*70 + "\n")
    
    # 1) Run CBRA to obtain joint sample for X (with constraints)
    print("Running CBRA optimization...")
    X_final, _ = run_cbra_funky(n=10000, seed=42, max_iter=20000)

    # 2) Compute equal-weighted portfolio P from CBRA sample
    P_cbra = X_final.mean(axis=1)  # (1/d) * sum_j X_j
    print(f"\nCBRA converged! Generated {len(P_cbra)} portfolio samples.")

    # 3) Compute TRUE distribution via MC (independent marginals, no constraints)
    print("\nGenerating TRUE distribution via Monte Carlo (200k samples)...")
    P_true_mc = compute_true_portfolio_distribution_mc(seed=999)
    print(f"Generated {len(P_true_mc)} true portfolio samples.")

    # 4) Plot true vs CBRA (with constraints) PDF and CDF
    out_path = os.path.join("figures", "portfolio_funky_pdf_cdf.png")
    make_pdf_cdf_plot_funky(P_cbra, P_true_mc, out_path)
    print(f"\n✅ Saved figure to: {out_path}")
    print("\nDone! Check out the funky distributions! 🎉\n")


if __name__ == "__main__":
    main()


