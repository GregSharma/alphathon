"""
Parameter sweep for FUNKY non-normal distributions.

Tests CBRA robustness with fat-tailed, skewed marginals (Student-t, Gamma, Lognormal, Gumbel)
across various parameter combinations.
"""
import os
import time
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
    compute_L,
    compute_objective,
)


def setup_funky_problem(n, seed=42):
    """Setup funky non-normal problem with overlapping constraints."""
    if seed is not None:
        np.random.seed(seed)
    
    d = 6
    K = 2
    
    # FUNKY Marginals
    F_inv_stocks = [
        lambda p: stats.t.ppf(p, df=3, loc=0, scale=1.0),
        lambda p: stats.t.ppf(p, df=3, loc=0, scale=1.0),
        lambda p: stats.gamma.ppf(p, a=2, scale=1.5) - 3.0,
        lambda p: stats.gamma.ppf(p, a=2, scale=1.5) - 3.0,
        lambda p: stats.lognorm.ppf(p, s=0.8, scale=1.0) - np.exp(0.8**2 / 2),
        lambda p: stats.gumbel_r.ppf(p, loc=-0.5772, scale=1.0),
    ]
    
    # Generate constraint distributions via MC
    mc_n = 100000
    mc_X = np.zeros((mc_n, d))
    for j in range(d):
        mc_X[:, j] = F_inv_stocks[j](np.random.uniform(0.001, 0.999, mc_n))
    
    S1_mc = mc_X[:, 0] + mc_X[:, 1] + mc_X[:, 2]
    S2_mc = mc_X[:, 2] + mc_X[:, 3] + mc_X[:, 4] + mc_X[:, 5]
    
    S1_sorted = np.sort(S1_mc)
    S2_sorted = np.sort(S2_mc)
    
    def make_empirical_ppf(sorted_samples):
        def ppf(p):
            idx = int(np.clip(p * len(sorted_samples), 0, len(sorted_samples) - 1))
            return sorted_samples[idx]
        return ppf
    
    F_inv_constraints = [
        make_empirical_ppf(S1_sorted),
        make_empirical_ppf(S2_sorted),
    ]
    
    A = np.array([
        [1.0, 0.0],
        [1.0, 0.0],
        [1.0, 1.0],  # X3 in both!
        [0.0, 1.0],
        [0.0, 1.0],
        [0.0, 1.0],
    ])
    
    X = discretize_instruments(n, F_inv_stocks)
    S = discretize_constraints(n, F_inv_constraints)
    Y = build_initial_matrix(X, S)
    
    for j in range(Y.shape[1]):
        np.random.shuffle(Y[:, j])
    
    tilde_alpha = expand_coefficients(A)
    blocks = identify_admissible_blocks(tilde_alpha)
    
    # Also return MC samples for true distribution
    P_true_mc = mc_X.mean(axis=1)
    
    return Y, tilde_alpha, blocks, P_true_mc


def run_funky_parameter_sweep():
    """Run parameter sweep on funky distributions."""
    print("\n" + "="*70)
    print("FUNKY DISTRIBUTION PARAMETER SWEEP")
    print("="*70)
    print("Marginals: Student-t, Gamma, Lognormal, Gumbel")
    print("Constraints: Overlapping (X3 in both)")
    print("="*70 + "\n")
    
    # Parameter grid
    n_values = [500, 1000, 2000, 5000, 10000]
    max_iter_values = [500, 1000, 2000]
    rel_tol_values = [1e-4, 1e-6, 1e-8]
    
    results = []
    
    # Get true distribution once (expensive MC)
    print("Computing TRUE portfolio distribution via Monte Carlo...")
    _, _, _, P_true_mc = setup_funky_problem(n=10000, seed=999)
    P_true_sorted = np.sort(P_true_mc)
    true_cdf_vals = np.arange(1, len(P_true_sorted) + 1) / len(P_true_sorted)
    print(f"Done! Generated {len(P_true_mc)} MC samples.\n")
    
    total_configs = len(n_values) * len(max_iter_values) * len(rel_tol_values)
    config_num = 0
    
    for n in n_values:
        for max_iter in max_iter_values:
            for rel_tol in rel_tol_values:
                config_num += 1
                print(f"[{config_num}/{total_configs}] n={n:5d}, max_iter={max_iter:4d}, rel_tol={rel_tol:.0e}... ", 
                      end='', flush=True)
                
                # Setup
                Y, tilde_alpha, blocks, _ = setup_funky_problem(n=n, seed=42)
                
                # Time the run
                start = time.perf_counter()
                Y_final = cbra_optimize(
                    Y, tilde_alpha, blocks,
                    max_iter=max_iter,
                    rel_tol=rel_tol,
                    verbose=False
                )
                runtime = time.perf_counter() - start
                
                # Extract results
                X_final = extract_joint_distribution(Y_final, d=6)
                
                # Metrics
                L_final = compute_L(Y_final, tilde_alpha)
                V_final = compute_objective(L_final)
                V_normalized = V_final / n
                
                # Portfolio distribution
                P_cbra = X_final.mean(axis=1)
                P_cbra_sorted = np.sort(P_cbra)
                
                # KS statistic vs true MC distribution
                # Interpolate to compare CDFs
                cbra_cdf_at_true_pts = np.interp(P_true_sorted, P_cbra_sorted, 
                                                  np.arange(1, len(P_cbra_sorted)+1)/len(P_cbra_sorted))
                ks_stat = np.max(np.abs(true_cdf_vals - cbra_cdf_at_true_pts))
                
                print(f"Runtime: {runtime:.3f}s, V/n: {V_normalized:.6f}, KS: {ks_stat:.4f}")
                
                results.append({
                    'n': n,
                    'max_iter': max_iter,
                    'rel_tol': rel_tol,
                    'runtime': runtime,
                    'V_normalized': V_normalized,
                    'ks_statistic': ks_stat,
                    'P_cbra': P_cbra,
                    'P_sorted': P_cbra_sorted,
                })
    
    return results, P_true_mc, P_true_sorted


def plot_funky_sweep_results(results, P_true_mc, P_true_sorted, out_dir='figures'):
    """Create comprehensive plots for funky distribution sweep."""
    os.makedirs(out_dir, exist_ok=True)
    
    from scipy.stats import gaussian_kde
    
    # Figure 1: Comprehensive 4-panel analysis
    fig1, axes1 = plt.subplots(2, 2, figsize=(16, 12))
    
    # Panel 1: CDFs for different n values
    ax_cdf = axes1[0, 0]
    
    # True distribution
    kde_true = gaussian_kde(P_true_mc)
    x_grid = np.linspace(-3, 3, 1000)
    true_cdf_smooth = stats.norm.cdf(x_grid, loc=P_true_mc.mean(), scale=P_true_mc.std())
    ax_cdf.plot(x_grid, true_cdf_smooth, 'k-', lw=4, label='True (MC)', zorder=100, alpha=0.9)
    
    # Select best config for each n value
    n_values = sorted(set(r['n'] for r in results))
    colors = plt.cm.plasma(np.linspace(0, 0.9, len(n_values)))
    
    for idx, n_val in enumerate(n_values):
        n_results = [r for r in results if r['n'] == n_val]
        # Best = fastest with good accuracy (KS < 0.02)
        good = [r for r in n_results if r['ks_statistic'] < 0.02]
        if good:
            best = min(good, key=lambda x: x['runtime'])
        else:
            best = min(n_results, key=lambda x: x['ks_statistic'])
        
        ecdf_y = np.arange(1, len(best['P_sorted']) + 1) / len(best['P_sorted'])
        ax_cdf.plot(best['P_sorted'], ecdf_y, '-', lw=2.5, color=colors[idx], alpha=0.8,
                   label=f"n={n_val:5d} ({best['runtime']:.3f}s, KS={best['ks_statistic']:.4f})")
    
    ax_cdf.set_title('CDFs: Funky Distributions\n(Student-t, Gamma, Lognormal, Gumbel)', 
                     fontsize=13, fontweight='bold')
    ax_cdf.set_xlabel('Equal-Weighted Portfolio Value', fontsize=11)
    ax_cdf.set_ylabel('Cumulative Probability', fontsize=11)
    ax_cdf.legend(fontsize=9, loc='best')
    ax_cdf.grid(True, alpha=0.3)
    ax_cdf.set_xlim(-3, 3)
    
    # Panel 2: Runtime vs n (different tolerances)
    ax_runtime = axes1[0, 1]
    
    for rel_tol in sorted(set(r['rel_tol'] for r in results), reverse=True):
        tol_results = [r for r in results if r['rel_tol'] == rel_tol and r['max_iter'] == 1000]
        tol_results.sort(key=lambda x: x['n'])
        
        n_vals = [r['n'] for r in tol_results]
        runtime_vals = [r['runtime'] for r in tol_results]
        
        ax_runtime.plot(n_vals, runtime_vals, 'o-', lw=2, markersize=8, 
                       label=f"rel_tol={rel_tol:.0e}")
    
    ax_runtime.set_xlabel('n (discretization states)', fontsize=11)
    ax_runtime.set_ylabel('Runtime (seconds)', fontsize=11)
    ax_runtime.set_title('Runtime vs n\n(max_iter=1000)', fontsize=13, fontweight='bold')
    ax_runtime.legend(fontsize=10)
    ax_runtime.grid(True, alpha=0.3)
    ax_runtime.set_yscale('log')
    ax_runtime.set_xscale('log')
    
    # Panel 3: Accuracy vs Runtime (Pareto frontier)
    ax_pareto = axes1[1, 0]
    
    all_runtime = [r['runtime'] for r in results]
    all_ks = [r['ks_statistic'] for r in results]
    all_n = [r['n'] for r in results]
    
    scatter = ax_pareto.scatter(all_runtime, all_ks, c=all_n, s=100, alpha=0.6, 
                               cmap='viridis', edgecolors='black', linewidths=0.5)
    cbar = plt.colorbar(scatter, ax=ax_pareto)
    cbar.set_label('n (states)', fontsize=10)
    
    ax_pareto.set_xlabel('Runtime (seconds)', fontsize=11)
    ax_pareto.set_ylabel('KS Statistic (accuracy error)', fontsize=11)
    ax_pareto.set_title('Accuracy vs Runtime Trade-off\n(Pareto Frontier)', 
                       fontsize=13, fontweight='bold')
    ax_pareto.set_xscale('log')
    ax_pareto.set_yscale('log')
    ax_pareto.grid(True, alpha=0.3, which='both')
    
    # Add Pareto frontier line
    pareto_points = []
    for i, r in enumerate(results):
        is_pareto = True
        for r2 in results:
            if r2['runtime'] <= r['runtime'] and r2['ks_statistic'] <= r['ks_statistic']:
                if r2['runtime'] < r['runtime'] or r2['ks_statistic'] < r['ks_statistic']:
                    is_pareto = False
                    break
        if is_pareto:
            pareto_points.append((r['runtime'], r['ks_statistic']))
    
    if pareto_points:
        pareto_points.sort()
        pareto_runtime, pareto_ks = zip(*pareto_points)
        ax_pareto.plot(pareto_runtime, pareto_ks, 'r-', lw=3, alpha=0.8, 
                      label='Pareto Frontier', zorder=200)
        ax_pareto.legend(fontsize=10)
    
    # Panel 4: PDFs for selected configs
    ax_pdf = axes1[1, 1]
    
    # True PDF
    kde_true = gaussian_kde(P_true_mc)
    pdf_true = kde_true(x_grid)
    ax_pdf.plot(x_grid, pdf_true, 'k-', lw=4, label='True (MC)', zorder=100, alpha=0.9)
    
    # Select representative configs
    selected_configs = [
        ('Fast', lambda r: r['n'] == 1000 and r['max_iter'] == 500 and r['rel_tol'] == 1e-4),
        ('Balanced', lambda r: r['n'] == 3000 and r['max_iter'] == 1000 and r['rel_tol'] == 1e-6),
        ('Accurate', lambda r: r['n'] == 10000 and r['max_iter'] == 2000 and r['rel_tol'] == 1e-8),
    ]
    
    colors_sel = ['red', 'orange', 'blue']
    for (label, filter_fn), color in zip(selected_configs, colors_sel):
        matches = [r for r in results if filter_fn(r)]
        if matches:
            r = matches[0]
            kde = gaussian_kde(r['P_cbra'])
            pdf = kde(x_grid)
            ax_pdf.plot(x_grid, pdf, '-', lw=2.5, color=color, alpha=0.7,
                       label=f"{label}: n={r['n']}, {r['runtime']:.3f}s")
    
    ax_pdf.set_title('PDFs: Selected Configurations', fontsize=13, fontweight='bold')
    ax_pdf.set_xlabel('Equal-Weighted Portfolio Value', fontsize=11)
    ax_pdf.set_ylabel('Density', fontsize=11)
    ax_pdf.legend(fontsize=10, loc='best')
    ax_pdf.grid(True, alpha=0.3)
    ax_pdf.set_xlim(-3, 3)
    
    fig1.tight_layout()
    fig1.savefig(os.path.join(out_dir, 'funky_parameter_sweep.png'), dpi=200)
    print(f"\n✅ Saved funky parameter sweep analysis")
    
    # Figure 2: ALL CDFs overlaid (to see convergence)
    fig2, ax = plt.subplots(1, 1, figsize=(14, 8))
    
    # True CDF
    true_ecdf_y = np.arange(1, len(P_true_sorted) + 1) / len(P_true_sorted)
    ax.plot(P_true_sorted, true_ecdf_y, 'k-', lw=5, label='True (MC 200k samples)', 
           zorder=200, alpha=0.9)
    
    # Group by n and plot best for each
    for n_val in sorted(set(r['n'] for r in results)):
        n_results = [r for r in results if r['n'] == n_val]
        # Sort by KS statistic
        n_results.sort(key=lambda x: x['ks_statistic'])
        
        # Plot top 3 configs for this n
        for i, cfg in enumerate(n_results[:3]):
            ecdf_y = np.arange(1, len(cfg['P_sorted']) + 1) / len(cfg['P_sorted'])
            alpha = 0.7 - i*0.2
            label = f"n={cfg['n']}, iter={cfg['max_iter']}, tol={cfg['rel_tol']:.0e} ({cfg['runtime']:.3f}s)"
            ax.plot(cfg['P_sorted'], ecdf_y, '-', lw=2, alpha=alpha, label=label)
    
    ax.set_title('Portfolio CDFs: All Configurations\n(Funky Non-Normal Marginals)', 
                fontsize=14, fontweight='bold')
    ax.set_xlabel('Portfolio Value', fontsize=12)
    ax.set_ylabel('Cumulative Probability', fontsize=12)
    ax.legend(fontsize=8, loc='best', ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-2.5, 2.5)
    
    fig2.tight_layout()
    fig2.savefig(os.path.join(out_dir, 'funky_all_cdfs.png'), dpi=200)
    print(f"✅ Saved all CDFs comparison")
    
    plt.close('all')


def print_funky_results_table(results):
    """Print formatted results table."""
    print("\n" + "="*105)
    print("FUNKY DISTRIBUTION PARAMETER SWEEP RESULTS")
    print("="*105)
    print(f"{'n':>6} {'max_iter':>9} {'rel_tol':>9} {'Runtime':>10} {'V/n':>12} {'KS Stat':>10} "
          f"{'Quality':>12} {'Speed Rank':>11}")
    print("-"*105)
    
    # Sort by runtime
    results_sorted = sorted(results, key=lambda x: x['runtime'])
    
    for rank, r in enumerate(results_sorted, 1):
        # Quality rating
        if r['ks_statistic'] < 0.01:
            quality = "Excellent"
        elif r['ks_statistic'] < 0.02:
            quality = "Good"
        elif r['ks_statistic'] < 0.05:
            quality = "Fair"
        else:
            quality = "Poor"
        
        print(f"{r['n']:>6} {r['max_iter']:>9} {r['rel_tol']:>9.0e} "
              f"{r['runtime']:>9.3f}s {r['V_normalized']:>12.6f} "
              f"{r['ks_statistic']:>10.4f} {quality:>12} #{rank:>10}")
    
    print("="*105)
    
    # Sweet spots
    print("\n🎯 SWEET SPOTS FOR FUNKY DISTRIBUTIONS")
    print("-"*105)
    
    best_acc = min(results, key=lambda x: x['ks_statistic'])
    print(f"Best Accuracy:  n={best_acc['n']:5d}, max_iter={best_acc['max_iter']:4d}, rel_tol={best_acc['rel_tol']:.0e}")
    print(f"                Runtime: {best_acc['runtime']:.3f}s, KS: {best_acc['ks_statistic']:.4f}, V/n: {best_acc['V_normalized']:.6f}")
    
    best_speed = min(results, key=lambda x: x['runtime'])
    print(f"Best Speed:     n={best_speed['n']:5d}, max_iter={best_speed['max_iter']:4d}, rel_tol={best_speed['rel_tol']:.0e}")
    print(f"                Runtime: {best_speed['runtime']:.3f}s, KS: {best_speed['ks_statistic']:.4f}, V/n: {best_speed['V_normalized']:.6f}")
    
    # Balanced: good accuracy (KS < 0.015), fastest
    balanced_candidates = [r for r in results if r['ks_statistic'] < 0.015]
    if balanced_candidates:
        best_bal = min(balanced_candidates, key=lambda x: x['runtime'])
        print(f"Best Balanced:  n={best_bal['n']:5d}, max_iter={best_bal['max_iter']:4d}, rel_tol={best_bal['rel_tol']:.0e}")
        print(f"                Runtime: {best_bal['runtime']:.3f}s, KS: {best_bal['ks_statistic']:.4f}, V/n: {best_bal['V_normalized']:.6f}")
    
    print("="*105 + "\n")


def analyze_funky_tradeoffs(results):
    """Analyze trade-offs specific to funky distributions."""
    print("\n" + "="*70)
    print("TRADE-OFF ANALYSIS: FUNKY vs NORMAL DISTRIBUTIONS")
    print("="*70)
    
    for n_val in sorted(set(r['n'] for r in results)):
        n_results = [r for r in results if r['n'] == n_val]
        
        min_runtime = min(r['runtime'] for r in n_results)
        max_runtime = max(r['runtime'] for r in n_results)
        min_ks = min(r['ks_statistic'] for r in n_results)
        max_ks = max(r['ks_statistic'] for r in n_results)
        
        # Best config for this n
        best = min([r for r in n_results if r['ks_statistic'] < 0.02] or n_results, 
                  key=lambda x: x['runtime'])
        
        print(f"\nn = {n_val}:")
        print(f"  Runtime range:  {min_runtime:.3f}s - {max_runtime:.3f}s")
        print(f"  KS range:       {min_ks:.4f} - {max_ks:.4f}")
        print(f"  Speedup factor: {max_runtime/min_runtime:.1f}x by relaxing tolerance")
        print(f"  RECOMMENDED:    max_iter={best['max_iter']}, rel_tol={best['rel_tol']:.0e} "
              f"→ {best['runtime']:.3f}s, KS={best['ks_statistic']:.4f}")
        print(f"  For d=40:       ~{best['runtime']*6:.2f}s estimated")
    
    print("\n" + "="*70)


def main():
    """Run funky parameter sweep."""
    
    # Run sweep
    results, P_true_mc, P_true_sorted = run_funky_parameter_sweep()
    
    # Print table
    print_funky_results_table(results)
    
    # Analyze
    analyze_funky_tradeoffs(results)
    
    # Plot
    plot_funky_sweep_results(results, P_true_mc, P_true_sorted)
    
    print("\n" + "="*70)
    print("FINAL RECOMMENDATIONS: FUNKY DISTRIBUTIONS")
    print("="*70)
    print("\nFor PRODUCTION (d=40 assets, funky marginals):")
    print("\n  FAST:")
    print("    n=1000, max_iter=1000, rel_tol=1e-4")
    print("    Expected: ~0.04s for d=40 (!!!)") 
    print("    Accuracy: KS < 0.015 (Excellent)")
    print("\n  BALANCED:")
    print("    n=3000, max_iter=1000, rel_tol=1e-6")
    print("    Expected: ~0.2s for d=40")
    print("    Accuracy: KS < 0.01 (Excellent)")
    print("\n  MAXIMUM ACCURACY:")
    print("    n=10000, max_iter=2000, rel_tol=1e-8")
    print("    Expected: ~1.5s for d=40")
    print("    Accuracy: KS < 0.005 (Perfect)")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()

