This assignment outlines the process for computing the joint multivariate distribution using the Constrained Block Rearrangement Algorithm (CBRA), given the risk-neutral cumulative distributions (RNCDs) for the underlying instruments and indices.

The worker must implement the following pipeline steps precisely.

### Job Assignment: Constrained Block Rearrangement Algorithm (CBRA) Pipeline

**Inputs Required from Function Caller:**

1.  **$n$ (Integer):** The number of equiprobable states for discretization (e.g., $n=1000$).

2.  **$d$ (Integer):** The number of individual equity/ETF instruments ($X_1, \ldots, X_d$).

3.  **$K$ (Integer):** The number of indices/constraints ($S_1, \ldots, S_K$).

4.  **$F_{j}^{-1}$ ($j=1, \ldots, d$):** The inverse RNCD (quantile function) for each individual instrument $X_j$.

5.  **$F_{\alpha^{k}}^{-1}$ ($k=1, \ldots, K$):** The inverse RNCD (quantile function) for each index $S_k$.

6.  **$\mathbf{A}$ (Matrix $d \times K$):** The matrix of weights $\alpha_{j}^{k}$ where $S_{k}=\sum_{i=1}^{d} \alpha_{i}^{k} X_{i}$.

---

### Step 1: Initialize the Input Matrix $\mathbf{Y}$

Construct the initial $n \times (d+K)$ matrix $\mathbf{Y}$ by sampling $n$ equiprobable values from all $d$ stock RNCDs and $K$ index RNCDs.

1.1. **Discretize Instruments ($\mathbf{X}$ matrix):** For each stock $j=1, \ldots, d$, calculate $n$ values $x_{ij}$:

$$

x_{i j}=F_{j}^{-1}\left(\frac{i-0.5}{n}\right), \quad i=1, \ldots, n

$$

The resulting $n \times d$ matrix is $\mathbf{X}$.

1.2. **Discretize Constraints ($\mathbf{S}$ matrix):** For each index $k=1, \ldots, K$, calculate $n$ values $s_{ik}$:

$$

s_{i k}=F_{\alpha^{k}}^{-1}\left(\frac{i-0.5}{n}\right), \quad i=1, \ldots, n

$$

The resulting $n \times K$ matrix is $\mathbf{S}$.

1.3. **Combine Matrices:** Form the full $n \times (d+K)$ initial matrix $\mathbf{Y}$:

$$

\mathbf{Y}=[\mathbf{X} ; \mathbf{S}]

$$

where the first $d$ columns represent the stocks and the last $K$ columns represent the indices.

### Step 2: Define Constraint Coefficients $\widetilde{\alpha}$ and Admissible Blocks $\mathcal{B}$

2.1. **Define Expanded Coefficients $\widetilde{\alpha}$:** Construct a set of $K$ coefficient vectors (one for each constraint $k$) of size $(d+K)$. Let $Y_j$ be the $j$-th column of $\mathbf{Y}$. The constraint $L_k$ is designed to measure the deviation from the required index sum distribution.

$$

L_{k}^{\pi}:=\sum_{j=1}^{d+K} \tilde{\alpha}_{j}^{k} Y_{j}^{\pi}, \quad k=1, \ldots, K

$$

Define the coefficients $\widetilde{\alpha}_{j}^{k}$ as follows:

*   For $j=1, \ldots, d$ (Stock columns): $\widetilde{\alpha}_{j}^{k} = \alpha_{j}^{k}$ (the weight of stock $j$ in index $k$).

*   For $j=d+k$ (The $k$-th Index column, matching constraint $k$): $\widetilde{\alpha}_{j}^{k} = -1$.

*   For all other index columns (i.e., $j=d+m$ where $m \neq k$): $\widetilde{\alpha}_{j}^{k} = 0$.

2.2. **Identify Admissible Blocks $\mathcal{B}$:** Determine the set $\mathcal{B}$ of all binary vectors $\delta \in \{0,1\}^{d+K}$ (blocks) that are *admissible*. A block $\delta$ is admissible if, for every constraint $k=1, \ldots, K$, the coefficients $\widetilde{\alpha}_{j}^{k}$ are constant for all columns $j$ belonging to the block $I_{\delta}$ (i.e., $\widetilde{\alpha}_{j}^{k}:=\widetilde{\alpha}^{k}$, for $j \in I_{\delta}$).

*   *Note:* The set $\mathcal{B}$ always includes all singleton blocks (blocks consisting of a single column).

### Step 3: Execute the Constrained Block Rearrangement Algorithm (CBRA)

Iteratively rearrange the rows of $\mathbf{Y}$ (by swapping elements within columns belonging to an admissible block) until the objective function $V$ converges (shows no further improvement).

The objective function $V$ to be minimized is the sum of the variances of the constraint vectors $L_k^{\pi}$:

$$

V:=V\left(\mathbf{Y}^{\pi}\right)=\sum_{k=1}^{K} \operatorname{var}\left(L_{k}^{\pi}\right)

$$

3.1. **Iteration Loop:** Start with the initial matrix $\mathbf{Y}^{\pi} = \mathbf{Y}$ and repeat the following steps:

    a. Iterate through every admissible block $\delta \in \mathcal{B}$.

    b. For the chosen block $\delta$, partition the columns into $I_{\delta}$ (columns in the block) and $I_{\delta}^{c}$ (columns outside the block).

    c. Calculate the current value of the variables outside the block: $Z = \sum_{j \in I_{\delta}^{c}} \tilde{\alpha}_{j}^{k} Y_{j}^{\pi}$ (this sum results in $K$ vectors, one for each constraint $k$).

    d. Apply a **Block Rearrangement $\pi^{\delta}$** to the columns $j \in I_{\delta}$ such that the combined variables within the block ($W = \sum_{j \in I_{\delta}} \tilde{\alpha}^{k} Y_{j}^{\pi^{\delta}}$) are **anti-monotonic** with the variables outside the block ($Z$).

        *   *To achieve anti-monotonicity:* Compute the sum $Z$ (or a linear combination of $Z$ vectors, aggregated across $k=1, \ldots, K$) and sort the rows of the current matrix $\mathbf{Y}^{\pi}$ within the block $I_{\delta}$ such that the elements in $W$ are counter-sorted relative to the corresponding elements in $Z$.

    e. Update $\mathbf{Y}^{\pi}$ with the newly rearranged columns.

    f. Compute the new objective value $V_{new}$.

3.2. **Convergence:** If the new objective value $V_{new}$ is not smaller than the objective value from the previous iteration, stop the process and output the current matrix $\widetilde{\mathbf{Y}} = \mathbf{Y}^{\pi}$.

### Step 4: Return the Joint Distribution

The final converged matrix $\widetilde{\mathbf{Y}}$ is an $n \times (d+K)$ matrix representing the $n$-state joint model.

4.1. **Extract Joint Distribution:** Return only the first $d$ columns of the matrix $\widetilde{\mathbf{Y}}$. This $n \times d$ matrix $\widetilde{\mathbf{X}}$ is the joint distribution for $(X_1, \ldots, X_d)$, where each row represents a joint realization occurring with probability $1/n$.

This matrix $\widetilde{\mathbf{X}}$ can then be used to price any path-independent multivariate derivative $G\left(X_{1}(T), \ldots, X_{d}(T)\right)$ by calculating the discounted average of the payoff across the $n$ states.

────────────────────────────────
1.  Python Module Layout
────────────────────────────────
• `cbrapipe/__init__.py`
• `cbrapipe/discretize.py`                ← Step 1 utilities  
• `cbrapipe/coefficients.py`              ← Step 2 utilities  
• `cbrapipe/blocks.py`                    ← admissible-block logic  
• `cbrapipe/rearrange.py`                 ← single block rearrangement  
• `cbrapipe/core.py`                      ← CBRA iteration / orchestration  
• `tests/`                                ← PyTest suite  
• `README.md`, `pyproject.toml` (NumPy, numba optional)

────────────────────────────────
2.  Function-level Plan & Type Hints
────────────────────────────────
discretize.py
1. `discretize_instruments(n: int, F_inv_list: Sequence[Callable[[float], float]]) -> NDArray[np.float64]`
2. `discretize_constraints(n: int, F_inv_list: Sequence[Callable[[float], float]]) -> NDArray[np.float64]`
3. `build_initial_matrix(X: NDArray, S: NDArray) -> NDArray`  # shape (n, d+K)

coefficients.py
4. `expand_coefficients(A: NDArray[np.float64]) -> NDArray[np.float64]`
   • Input A shape (d, K) → output coef shape (K, d+K)

blocks.py
5. `identify_admissible_blocks(tilde_alpha: NDArray[np.float64]) -> list[NDArray[np.int8]]`
   • Return list of binary masks length (d+K)

rearrange.py
6. `block_rearrangement(Y: NDArray, block_mask: NDArray[np.bool_], coef_block: NDArray[np.float64], coef_out: NDArray[np.float64]) -> None`
   • In-place row swap to achieve anti-monotonicity

core.py
7. `compute_objective(L: NDArray[np.float64]) -> float`
8. `cbra_optimize(Y: NDArray, tilde_alpha: NDArray, blocks: list[NDArray]) -> NDArray`
   • Loops until no objective decrease
9. `joint_distribution(Y_final: NDArray, d: int) -> NDArray`  # Extract first d cols

Optional perf helpers
10. `jit_if_available(func: F) -> F`  # decorator toggling numba

────────────────────────────────
3.  Testing Strategy (PyTest)
────────────────────────────────
Unit tests
• `test_discretize.py`
  – Check shapes and monotonicity of each column.  
• `test_coefficients.py`
  – Verify expanded matrix matches spec for random d, K.  
• `test_blocks.py`
  – Ensure singleton blocks always present, invariance property holds.  
• `test_rearrange.py`
  – After rearrangement, correlation of W & Z ≤ 0 (anti-monotone).  
• `test_objective.py`
  – Objective non-negative; rearrangement doesn’t increase variance of each constraint.  

Integration
• `test_cbra_end_to_end.py`
  – Small synthetic example (n=100, d=2, K=1) with analytic optimum; assert objective decreases then stabilizes; final joint matrix has correct marginals (Kolmogorov–Smirnov statistic).

────────────────────────────────
4.  Developer Instructions
────────────────────────────────
1.  Create virtual env; `pip install -r requirements.txt`  
   requirements: `numpy>=1.24`, `numba>=0.59` (optional), `pytest>=8`  
2.  Follow file layout above; start with `discretize.py` and work downward.  
3.  Keep all heavy loops in NumPy space; where unavoidable, wrap in `@jit_if_available`.  
4.  Run `pytest -q` often; aim for 100 % branch coverage.  
5.  Use `flake8` + `mypy` for lint/type checks.