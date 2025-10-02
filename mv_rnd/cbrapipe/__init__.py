"""
CBRA Pipeline - Constrained Block Rearrangement Algorithm

This package implements the CBRA algorithm for computing joint multivariate
distributions given risk-neutral cumulative distributions and constraints.
"""
from .discretize import (
    discretize_instruments,
    discretize_constraints,
    build_initial_matrix
)
from .blocks import (
    expand_coefficients,
    identify_admissible_blocks
)
from .optimize import (
    compute_L,
    compute_objective,
    block_rearrangement,
    cbra_optimize,
    extract_joint_distribution
)
from .incremental import (
    CBRAState,
    cbra_optimize_stateful,
    cbra_update_incremental,
    detect_marginal_changes
)

__all__ = [
    'discretize_instruments',
    'discretize_constraints',
    'build_initial_matrix',
    'expand_coefficients',
    'identify_admissible_blocks',
    'compute_L',
    'compute_objective',
    'block_rearrangement',
    'cbra_optimize',
    'extract_joint_distribution',
    'CBRAState',
    'cbra_optimize_stateful',
    'cbra_update_incremental',
    'detect_marginal_changes',
]
