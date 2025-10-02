"""
Parameter sweep experiment to evaluate CBRA accuracy vs. runtime trade-offs.

Tests various combinations of:
- n (discretization points)
- max_iter (iteration limit)
- rel_tol (relative tolerance)

Plots all estimated portfolio distributions on the same plot to visualize impact.
"""
import os
import time
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from itertools import product

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


def setup_test_problem(n, seed=42):
    """Setup paper Section 4.1 example."""
    if seed is not None:
        np.random.seed(seed)
    
    d = 6
    K = 3
    
    F_inv_stocks = [lambda p: stats.norm.ppf(p, loc=0, scale=1) for _ in range(d)]
    F_inv_constraints = [
        lambda p: stats.norm.ppf(p, loc=0, scale=np.sqrt(10)),
        lambda p: stats.norm.ppf(p, loc=0, scale=np.sqrt(10)),
        lambda p: stats.norm.ppf(p, loc=0, scale=np.sqrt(24)),
    ]
    
    A = np.array([
        [1.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0],
        [0.0, 1.0, 1.0],
        [0.0, 1.0, 1.0],
    ])
    
    X = discretize_instruments(n, F_inv_stocks)
    S = discretize_constraints(n, F_inv_constraints)
    Y = build_initial_matrix(X, S)
    
    for j in range(Y.shape[1]):
        np.random.shuffle(Y[:, j])
    
    tilde_alpha = expand_coefficients(A)
    blocks = identify_admissible_blocks(tilde_alpha)
    
    return Y, tilde_alpha, blocks


def run_parameter_sweep():
    """
    Run CBRA with various parameter combinations and collect results.
    """
    print("\n" + "="*70)
    print("CBRA PARAMETER SWEEP EXPERIMENT")
    print("="*70)
    print("Testing impact of n, max_iter, rel_tol on accuracy and runtime")
    print("="*70 + "\n")
    
    # Parameter grid
    n_values = [1000, 3000, 5000, 10000, 100_000]
    max_iter_values = [500, 1000, 2000]
    rel_tol_values = [1e-4, 1e-6, 1e-8]
    
    results = []
    
    total_configs = len(n_values) * len(max_iter_values) * len(rel_tol_values)
    config_num = 0
    
    for n in n_values:
        for max_iter in max_iter_values:
            for rel_tol in rel_tol_values:
                config_num += 1
                print(f"[{config_num}/{total_configs}] n={n:5d}, max_iter={max_iter:4d}, rel_tol={rel_tol:.0e}... ", end='', flush=True)
                
                # Setup
                Y, tilde_alpha, blocks = setup_test_problem(n=n, seed=42)
                
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
                
                # Compute metrics
                L_final = compute_L(Y_final, tilde_alpha)
                V_final = compute_objective(L_final)
                V_normalized = V_final / n
                
                # Compute portfolio distribution
                P_portfolio = X_final.mean(axis=1)
                
                # True portfolio distribution (we know it's N(0, 24/36))
                true_mean = 0.0
                true_std = np.sqrt(24) / 6
                
                # KS statistic vs true distribution
                P_sorted = np.sort(P_portfolio)
                true_cdf = stats.norm.cdf(P_sorted, loc=true_mean, scale=true_std)
                empirical_cdf = np.arange(1, len(P_sorted) + 1) / len(P_sorted)
                ks_stat = np.max(np.abs(true_cdf - empirical_cdf))
                
                print(f"Runtime: {runtime:.3f}s, V/n: {V_normalized:.6f}, KS: {ks_stat:.4f}")
                
                results.append({
                    'n': n,
                    'max_iter': max_iter,
                    'rel_tol': rel_tol,
                    'runtime': runtime,
                    'V_normalized': V_normalized,
                    'ks_statistic': ks_stat,
                    'P_portfolio': P_portfolio,
                    'P_sorted': P_sorted,
                })
    
    return results


def plot_parameter_sweep_results(results, out_dir='figures'):
    """
    Create comprehensive plots showing trade-offs.
    """
    os.makedirs(out_dir, exist_ok=True)
    
    # True distribution for reference
    true_mean = 0.0
    true_std = np.sqrt(24) / 6
    x_grid = np.linspace(-3, 3, 1000)
    true_pdf = stats.norm.pdf(x_grid, loc=true_mean, scale=true_std)
    true_cdf = stats.norm.cdf(x_grid, loc=true_mean, scale=true_std)
    
    # Figure 1: CDFs for different n values (fixed max_iter, rel_tol)
    fig1, axes1 = plt.subplots(2, 2, figsize=(14, 10))
    
    # Select configs: max_iter=2000, rel_tol=1e-6, vary n
    configs_by_n = [r for r in results if r['max_iter'] == 2000 and r['rel_tol'] == 1e-6]
    configs_by_n.sort(key=lambda x: x['n'])
    
    # CDF plot
    ax_cdf = axes1[0, 0]
    ax_cdf.plot(x_grid, true_cdf, 'k-', lw=3, label='True Normal', alpha=0.8)
    colors = plt.cm.viridis(np.linspace(0, 1, len(configs_by_n)))
    for idx, cfg in enumerate(configs_by_n):
        ecdf_y = np.arange(1, len(cfg['P_sorted']) + 1) / len(cfg['P_sorted'])
        ax_cdf.plot(cfg['P_sorted'], ecdf_y, '-', lw=2, color=colors[idx],
                   label=f"n={cfg['n']}, {cfg['runtime']:.3f}s", alpha=0.7)
    ax_cdf.set_title('CDF: Impact of n (discretization)', fontsize=12, fontweight='bold')
    ax_cdf.set_xlabel('Portfolio Value P')
    ax_cdf.set_ylabel('Cumulative Probability')
    ax_cdf.legend(fontsize=9)
    ax_cdf.grid(True, alpha=0.3)
    
    # Runtime vs n
    ax_runtime = axes1[0, 1]
    n_vals = [cfg['n'] for cfg in configs_by_n]
    runtime_vals = [cfg['runtime'] for cfg in configs_by_n]
    ks_vals = [cfg['ks_statistic'] for cfg in configs_by_n]
    
    ax_runtime.plot(n_vals, runtime_vals, 'o-', lw=2, markersize=8, label='Runtime')
    ax_runtime.set_xlabel('n (states)')
    ax_runtime.set_ylabel('Runtime (seconds)', color='C0')
    ax_runtime.tick_params(axis='y', labelcolor='C0')
    ax_runtime.set_title('Runtime vs n', fontsize=12, fontweight='bold')
    ax_runtime.grid(True, alpha=0.3)
    
    ax_ks = ax_runtime.twinx()
    ax_ks.plot(n_vals, ks_vals, 's--', lw=2, markersize=8, color='C1', label='KS Error')
    ax_ks.set_ylabel('KS Statistic (error)', color='C1')
    ax_ks.tick_params(axis='y', labelcolor='C1')
    
    # Impact of rel_tol
    ax_tol = axes1[1, 0]
    configs_by_tol = [r for r in results if r['n'] == 5000 and r['max_iter'] == 2000]
    configs_by_tol.sort(key=lambda x: x['rel_tol'], reverse=True)
    
    tol_vals = [cfg['rel_tol'] for cfg in configs_by_tol]
    tol_runtime = [cfg['runtime'] for cfg in configs_by_tol]
    tol_ks = [cfg['ks_statistic'] for cfg in configs_by_tol]
    
    ax_tol.semilogx(tol_vals, tol_runtime, 'o-', lw=2, markersize=8, label='Runtime')
    ax_tol.set_xlabel('rel_tol (convergence threshold)')
    ax_tol.set_ylabel('Runtime (seconds)', color='C0')
    ax_tol.tick_params(axis='y', labelcolor='C0')
    ax_tol.set_title('Runtime vs rel_tol (n=5000)', fontsize=12, fontweight='bold')
    ax_tol.grid(True, alpha=0.3)
    
    ax_tol_ks = ax_tol.twinx()
    ax_tol_ks.semilogx(tol_vals, tol_ks, 's--', lw=2, markersize=8, color='C1')
    ax_tol_ks.set_ylabel('KS Statistic', color='C1')
    ax_tol_ks.tick_params(axis='y', labelcolor='C1')
    
    # Accuracy vs Runtime scatter (Pareto frontier)
    ax_pareto = axes1[1, 1]
    all_runtime = [r['runtime'] for r in results]
    all_ks = [r['ks_statistic'] for r in results]
    all_n = [r['n'] for r in results]
    
    # Color by n
    scatter = ax_pareto.scatter(all_runtime, all_ks, c=all_n, s=80, alpha=0.6, cmap='viridis')
    cbar = plt.colorbar(scatter, ax=ax_pareto)
    cbar.set_label('n (states)')
    ax_pareto.set_xlabel('Runtime (seconds)')
    ax_pareto.set_ylabel('KS Statistic (error)')
    ax_pareto.set_title('Accuracy vs Runtime (all configs)', fontsize=12, fontweight='bold')
    ax_pareto.grid(True, alpha=0.3)
    
    # Add Pareto frontier
    pareto_points = []
    for i, r in enumerate(results):
        is_pareto = True
        for j, r2 in enumerate(results):
            if i != j:
                # r2 dominates r if it's faster AND more accurate
                if r2['runtime'] <= r['runtime'] and r2['ks_statistic'] <= r['ks_statistic']:
                    if r2['runtime'] < r['runtime'] or r2['ks_statistic'] < r['ks_statistic']:
                        is_pareto = False
                        break
        if is_pareto:
            pareto_points.append((r['runtime'], r['ks_statistic']))
    
    if pareto_points:
        pareto_points.sort()
        pareto_runtime, pareto_ks = zip(*pareto_points)
        ax_pareto.plot(pareto_runtime, pareto_ks, 'r-', lw=2, alpha=0.7, label='Pareto Frontier')
        ax_pareto.legend()
    
    fig1.tight_layout()
    fig1.savefig(os.path.join(out_dir, 'parameter_sweep_analysis.png'), dpi=200)
    print(f"✅ Saved parameter sweep analysis")
    
    # Figure 2: All CDFs overlaid
    fig2, ax = plt.subplots(1, 1, figsize=(12, 7))
    
    ax.plot(x_grid, true_cdf, 'k-', lw=4, label='True Normal CDF', zorder=100, alpha=0.9)
    
    # Group by n for consistent coloring
    for n_val in sorted(set(r['n'] for r in results)):
        n_results = [r for r in results if r['n'] == n_val]
        # Take best runtime for this n
        n_results.sort(key=lambda x: x['runtime'])
        best = n_results[0]
        
        ecdf_y = np.arange(1, len(best['P_sorted']) + 1) / len(best['P_sorted'])
        ax.plot(best['P_sorted'], ecdf_y, '-', lw=2, alpha=0.7,
               label=f"n={n_val} ({best['runtime']:.3f}s, KS={best['ks_statistic']:.4f})")
    
    ax.set_title('Portfolio CDFs: All Configurations', fontsize=14, fontweight='bold')
    ax.set_xlabel('Portfolio Value P', fontsize=12)
    ax.set_ylabel('Cumulative Probability', fontsize=12)
    ax.legend(fontsize=10, loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-3, 3)
    
    fig2.tight_layout()
    fig2.savefig(os.path.join(out_dir, 'all_cdfs_comparison.png'), dpi=200)
    print(f"✅ Saved CDF comparison")
    
    plt.close('all')


def print_results_table(results):
    """Print a formatted table of results."""
    print("\n" + "="*100)
    print("PARAMETER SWEEP RESULTS")
    print("="*100)
    print(f"{'n':>6} {'max_iter':>9} {'rel_tol':>9} {'Runtime':>10} {'V/n':>12} {'KS Stat':>10} {'Quality':>10}")
    print("-"*100)
    
    # Sort by runtime
    results_sorted = sorted(results, key=lambda x: x['runtime'])
    
    for r in results_sorted:
        # Quality rating based on KS statistic
        if r['ks_statistic'] < 0.01:
            quality = "Excellent"
        elif r['ks_statistic'] < 0.05:
            quality = "Good"
        elif r['ks_statistic'] < 0.1:
            quality = "Fair"
        else:
            quality = "Poor"
        
        print(f"{r['n']:>6} {r['max_iter']:>9} {r['rel_tol']:>9.0e} "
              f"{r['runtime']:>9.3f}s {r['V_normalized']:>12.6f} "
              f"{r['ks_statistic']:>10.4f} {quality:>10}")
    
    print("="*100)
    
    # Find sweet spots
    print("\n🎯 SWEET SPOTS (Pareto optimal configs)")
    print("-"*100)
    
    # Best accuracy (smallest KS)
    best_accuracy = min(results, key=lambda x: x['ks_statistic'])
    print(f"Best Accuracy:  n={best_accuracy['n']}, max_iter={best_accuracy['max_iter']}, "
          f"rel_tol={best_accuracy['rel_tol']:.0e}")
    print(f"                Runtime: {best_accuracy['runtime']:.3f}s, KS: {best_accuracy['ks_statistic']:.4f}")
    
    # Best speed (smallest runtime)
    best_speed = min(results, key=lambda x: x['runtime'])
    print(f"Best Speed:     n={best_speed['n']}, max_iter={best_speed['max_iter']}, "
          f"rel_tol={best_speed['rel_tol']:.0e}")
    print(f"                Runtime: {best_speed['runtime']:.3f}s, KS: {best_speed['ks_statistic']:.4f}")
    
    # Best balanced (minimize runtime * ks_statistic)
    results_balanced = [r for r in results if r['ks_statistic'] < 0.05]  # Only good accuracy
    if results_balanced:
        best_balanced = min(results_balanced, key=lambda x: x['runtime'])
        print(f"Best Balanced:  n={best_balanced['n']}, max_iter={best_balanced['max_iter']}, "
              f"rel_tol={best_balanced['rel_tol']:.0e}")
        print(f"                Runtime: {best_balanced['runtime']:.3f}s, KS: {best_balanced['ks_statistic']:.4f}")
    
    print("="*100 + "\n")


def analyze_tradeoffs(results):
    """Analyze accuracy vs runtime trade-offs."""
    print("\n" + "="*70)
    print("TRADE-OFF ANALYSIS")
    print("="*70)
    
    # Group by n
    for n_val in sorted(set(r['n'] for r in results)):
        n_results = [r for r in results if r['n'] == n_val]
        
        min_runtime = min(r['runtime'] for r in n_results)
        max_runtime = max(r['runtime'] for r in n_results)
        min_ks = min(r['ks_statistic'] for r in n_results)
        max_ks = max(r['ks_statistic'] for r in n_results)
        
        print(f"\nn = {n_val}:")
        print(f"  Runtime range:  {min_runtime:.3f}s - {max_runtime:.3f}s")
        print(f"  KS range:       {min_ks:.4f} - {max_ks:.4f}")
        print(f"  Speedup by relaxing accuracy: {max_runtime/min_runtime:.2f}x")
    
    print("\n" + "="*70)


def main():
    """Run parameter sweep and generate analysis."""
    
    # Run sweep
    results = run_parameter_sweep()
    
    # Print table
    print_results_table(results)
    
    # Analyze trade-offs
    analyze_tradeoffs(results)
    
    # Generate plots
    plot_parameter_sweep_results(results)
    
    print("\n" + "="*70)
    print("RECOMMENDATIONS FOR YOUR 40-ASSET SYSTEM")
    print("="*70)
    
    # Find configs with good accuracy and reasonable speed
    good_configs = [r for r in results if r['ks_statistic'] < 0.02]
    good_configs.sort(key=lambda x: x['runtime'])
    
    if good_configs:
        rec = good_configs[0]
        print(f"\nFor FAST + ACCURATE:")
        print(f"  n = {rec['n']}")
        print(f"  max_iter = {rec['max_iter']}")
        print(f"  rel_tol = {rec['rel_tol']:.0e}")
        print(f"  Expected: {rec['runtime']:.3f}s runtime, {rec['ks_statistic']:.4f} KS error")
        print(f"\n  For d=40 (scale up ~6x): ~{rec['runtime']*6:.2f}s")
    
    # Fast config
    fast_configs = [r for r in results if r['ks_statistic'] < 0.10]
    fast_configs.sort(key=lambda x: x['runtime'])
    
    if fast_configs:
        fast = fast_configs[0]
        print(f"\nFor MAXIMUM SPEED (acceptable accuracy):")
        print(f"  n = {fast['n']}")
        print(f"  max_iter = {fast['max_iter']}")
        print(f"  rel_tol = {fast['rel_tol']:.0e}")
        print(f"  Expected: {fast['runtime']:.3f}s runtime, {fast['ks_statistic']:.4f} KS error")
        print(f"\n  For d=40 (scale up ~6x): ~{fast['runtime']*6:.2f}s")
    
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
