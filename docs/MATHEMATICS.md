# Mathematical Methodology - Complete Technical Reference

**Alphathon Q1**: Real-Time Macro Information Diffusion System

---

## 1. Kalshi Probability Denoising (Kalman Filter)

### 1.1 State-Space Model

**Purpose**: Extract normalized innovations from noisy Kalshi probability trades

**Model**:

$$
\begin{align}
p_t &= p_{t-1} + w_t, \quad w_t \sim \mathcal{N}(0, Q) \tag{State Transition}\\
y_t &= p_t + v_t, \quad v_t \sim \mathcal{N}(0, R) \tag{Observation}
\end{align}
$$

Where:
- $p_t$: True underlying probability (latent state)
- $y_t$: Observed trade price from Kalshi
- $Q$: Process noise variance (≈ $10^{-5}$)
- $R$: Observation noise variance (≈ $10^{-3}$)

### 1.2 Kalman Filter Equations

**Prediction Step**:

$$
\begin{align}
\hat{p}_{t|t-1} &= \hat{p}_{t-1|t-1}\\
P_{t|t-1} &= P_{t-1|t-1} + Q
\end{align}
$$

**Update Step**:

$$
\begin{align}
K_t &= \frac{P_{t|t-1}}{P_{t|t-1} + R} \tag{Kalman Gain}\\
\hat{p}_{t|t} &= \hat{p}_{t|t-1} + K_t(y_t - \hat{p}_{t|t-1})\\
P_{t|t} &= (1 - K_t)P_{t|t-1}
\end{align}
$$

### 1.3 Innovation Extraction

**Raw Innovation**:
$$
\nu_t = y_t - \hat{p}_{t|t-1}
$$

**Normalized Innovation** (for regime detection):
$$
z_t = \frac{\nu_t}{\sqrt{P_{t|t-1} + R}} \sim \mathcal{N}(0, 1)
$$

**Regime Classification**:
- High shock: $|z_t| > 2$ (≈ 95th percentile)
- Moderate shock: $1 < |z_t| \leq 2$
- Baseline: $|z_t| \leq 1$

---

## 2. Microstructure Feature Engineering

### 2.1 Lee-Ready Directional Indicator

**Purpose**: Classify trades as buyer- or seller-initiated

**Rule**:

$$
D_t = \begin{cases}
+1 & \text{if } P_t > M_t \text{ (buyer-initiated)}\\\\
-1 & \text{if } P_t < M_t \text{ (seller-initiated)}\\\\
0 & \text{if } P_t = M_t \text{ (at mid)}
\end{cases}
$$

Where:
- $P_t$: Trade price
- $M_t = (B_t + A_t)/2$: Mid-price from NBBO

### 2.2 Flow Metrics

**Total Signed Flow**:
$$
F_t^{total} = \sum_{i=1}^{N_t} D_i \cdot Q_i \cdot P_i
$$

Where:
- $N_t$: Number of trades in time window $[t, t+\Delta t]$
- $Q_i$: Trade size
- $P_i$: Trade price

**ISO Flow** (Intermarket Sweep Orders):
$$
F_t^{ISO} = \sum_{i: C_i = 53} D_i \cdot Q_i \cdot P_i
$$

Where $C_i$ is the trade condition code (53 = ISO)

**ISO Flow Intensity**:
$$
I_t^{ISO} = \frac{F_t^{ISO}}{\sum_{i=1}^{N_t} Q_i}
$$

### 2.3 Slippage Metrics

**Definition**: Cost paid by liquidity takers

$$
S_i = D_i \cdot (P_i - M_i)
$$

**Interpretation**:
- $D_i = +1$: Buy at $P_i > M_i$ → positive slippage
- $D_i = -1$: Sell at $P_i < M_i$ → positive slippage

**Aggregate Slippage**:
$$
S_t^{total} = \sum_{i=1}^{N_t} D_i \cdot (P_i - M_i)
$$

**Normalized by Spread**:
$$
S_t^{norm} = \frac{S_t^{total}}{\overline{(A_t - B_t)}/2}
$$

---

## 3. Fama-French Five-Factor Model

### 3.1 Factor Construction

**Real-Time Factor Returns**:
$$
F_{k,t} = \sum_{i=1}^{N} w_{i,k} \cdot r_{i,t}
$$

Where:
- $F_{k,t}$: Factor $k$ return at time $t$ (MKT-RF, SMB, HML, RMW, CMA)
- $w_{i,k}$: Weight of stock $i$ in factor $k$ portfolio (from `factor_weights_tickers.json`)
- $r_{i,t} = \log(P_{i,t}/P_{i,t-1})$: Log return

**Constraints**:
$$
\sum_{i=1}^{N} w_{i,k} = 1 \quad \text{(weights sum to 1)}
$$

Some weights may be negative (long-short portfolios).

### 3.2 Beta Decomposition

**Model**:
$$
r_{i,t} = \alpha_i + \sum_{k=1}^{5} \beta_{i,k} F_{k,t} + \epsilon_{i,t}
$$

**Systematic Return**:
$$
r_{i,t}^{sys} = \sum_{k=1}^{5} \beta_{i,k} F_{k,t}
$$

**Idiosyncratic Return**:
$$
r_{i,t}^{idio} = r_{i,t} - r_{i,t}^{sys}
$$

### 3.3 Rolling Beta Estimation

**Simple approach** (for time constraint):
$$
\hat{\beta}_i = (\mathbf{F}^\top \mathbf{F})^{-1} \mathbf{F}^\top \mathbf{r}_i
$$

Using last $W$ observations (e.g., $W = 300$ seconds).

**Kalman filter approach** (if time):
- State: $[\beta_{MKT}, \beta_{SMB}, \beta_{HML}, \beta_{RMW}, \beta_{CMA}, \mu]^\top$
- Transition: $\beta_t = \beta_{t-1} + w_t$ (random walk)
- Observation: $r_{i,t} = \beta_t^\top F_t + v_t$

---

## 4. Information Leadership (Hasbrouck)

### 4.1 Vector Error Correction Model (VECM)

**Representation**:
$$
\Delta \mathbf{y}_t = \mathbf{\Pi} \mathbf{y}_{t-1} + \sum_{i=1}^{k-1} \mathbf{\Gamma}_i \Delta \mathbf{y}_{t-i} + \boldsymbol{\epsilon}_t
$$

Where:
- $\mathbf{y}_t$: Vector of log-prices (equity, ETF, ...)
- $\mathbf{\Pi} = \boldsymbol{\alpha} \boldsymbol{\beta}^\top$: Error correction term
- $\boldsymbol{\alpha}$: Adjustment speeds ($n \times r$)
- $\boldsymbol{\beta}$: Cointegration vectors ($n \times r$)
- $r$: Cointegration rank (typically $r = n-1$ for $n$ assets)

### 4.2 Information Share (Hasbrouck, 1995)

**Permanent Component Impact**:

The long-run impact matrix:
$$
\boldsymbol{\Psi}_\infty = \boldsymbol{\beta}_\perp (\boldsymbol{\alpha}_\perp^\top \boldsymbol{\beta}_\perp)^{-1} \boldsymbol{\alpha}_\perp^\top
$$

**Information Share**:
$$
IS_i = \frac{[\boldsymbol{\Psi}_\infty \boldsymbol{\Omega} \boldsymbol{\Psi}_\infty^\top]_{ii}}{\text{trace}(\boldsymbol{\Psi}_\infty \boldsymbol{\Omega} \boldsymbol{\Psi}_\infty^\top)}
$$

**Simplified Approximation** (for $r = n-1$):
$$
IS_i \approx \frac{\psi_i^2 \Omega_{ii}}{\boldsymbol{\psi}^\top \boldsymbol{\Omega} \boldsymbol{\psi}}
$$

Where:
- $\boldsymbol{\psi} = \boldsymbol{\alpha}_\perp$ (orthogonal complement via SVD)
- $\boldsymbol{\Omega}$: Residual covariance matrix

### 4.3 Information Leadership Share (Yan & Zivot, 2010)

**Component Share**:
$$
CS_i = \frac{|\psi_i|}{\sum_j |\psi_j|}
$$

**Information Leadership Share**:
$$
ILS_i = \frac{\beta_i^2}{\sum_j \beta_j^2}, \quad \beta_i = \frac{IS_i}{CS_i}
$$

**Interpretation**: Assets with high ILS lead price discovery.

---

## 5. Risk-Neutral Density Extraction

### 5.1 Gaussian Process Regression

**Model**:
$$
\sigma_{IV}(k) \sim \mathcal{GP}(0, K(k, k'))
$$

**RBF Kernel**:
$$
K(k, k') = \sigma_f^2 \exp\left(-\frac{(k - k')^2}{2\ell^2}\right)
$$

**Log-Moneyness**:
$$
k = \log(K/F), \quad F = S e^{r\tau}
$$

**Posterior Mean**:
$$
\mu(k_*) = \mathbf{K}(k_*, \mathbf{k})[\mathbf{K}(\mathbf{k}, \mathbf{k}) + \sigma^2 \mathbf{I}]^{-1} \mathbf{y}
$$

### 5.2 Breeden-Litzenberger Identity

**Risk-Neutral Density**:
$$
q(K) = e^{r\tau} \frac{\partial^2 C(K)}{\partial K^2}
$$

**Numerical Implementation**:
1. Compute call prices: $C(K) = BS(S, K, \tau, r, \mu(k), \text{call})$
2. First derivative: $\frac{\partial C}{\partial K} \approx \frac{C_{i+1} - C_{i-1}}{K_{i+1} - K_{i-1}}$
3. Second derivative: $\frac{\partial^2 C}{\partial K^2} \approx \frac{(\partial C/\partial K)_{i+1} - (\partial C/\partial K)_{i-1}}{K_{i+1} - K_{i-1}}$
4. Density: $q(K) = e^{r\tau} \cdot \frac{\partial^2 C}{\partial K^2}$

**Normalization**:
$$
\int_{0}^{\infty} q(K) dK = 1
$$

### 5.3 Cumulative Distribution

$$
Q(K) = \int_{0}^{K} q(s) ds
$$

**Properties**:
- $Q(0) = 0$
- $Q(\infty) = 1$
- Monotonically increasing: $\frac{\partial Q}{\partial K} = q(K) \geq 0$

---

## 6. RND Superimposition (Kalshi-Conditional Densities)

### 6.1 Mixture Model

**Observed RND as mixture**:
$$
f(S) = p \cdot f(S | K=1) + (1-p) \cdot f(S | K=0)
$$

Where:
- $f(S)$: Observed risk-neutral density from options
- $p$: Kalshi probability (filtered)
- $f(S|K=1)$: Density conditional on Kalshi event occurs
- $f(S|K=0)$: Density conditional on Kalshi event doesn't occur

### 6.2 Estimation via Maximum Entropy

**Objective**: Minimize KL divergence subject to moment constraints

$$
\min_{f_1, f_0} \mathbb{E}_{f_1}[\log f_1] + \mathbb{E}_{f_0}[\log f_0]
$$

**Constraints**:

$$
\begin{align}
p \cdot f_1(S) + (1-p) \cdot f_0(S) &= f(S) \tag{Consistency}\\\\
\int f_1(S) dS &= 1, \quad \int f_0(S) dS = 1 \tag{Normalization}\\\\
\mathbb{E}_{f_1}[S] &= \mu_1, \quad \mathbb{E}_{f_0}[S] = \mu_0 \tag{Moment Matching}
\end{align}
$$

### 6.3 Simplified Approach (Time-Constrained)

**Assumption**: Conditional densities are shifts of base distribution

$$
f(S | K=1) \approx f(S - \delta_1)\\\\
f(S | K=0) \approx f(S - \delta_0)
$$

Solve for $\delta_1, \delta_0$ via moment matching:
$$
p \cdot (\mu + \delta_1) + (1-p) \cdot (\mu + \delta_0) = \mu
$$

Gives: $\delta_1 = -\frac{(1-p)}{p} \delta_0$

---

## 7. CBRA Joint Distribution

### 7.1 Block Rearrangement Algorithm

**Purpose**: Infer joint distribution from marginals

**Problem Setup**:
- Given: $D + K$ marginal distributions $F_1, \ldots, F_{D+K}$
- Find: Joint distribution consistent with marginals

**Maximum Entropy Principle**:

Without constraints, choose joint distribution that:
1. Matches all marginals exactly
2. Maximizes entropy (least informative dependence structure)

### 7.2 Discretization

**Quantile Sampling**:
$$
X_{ij} = F_j^{-1}\left(\frac{i}{n}\right), \quad i = 1, \ldots, n, \quad j = 1, \ldots, D+K
$$

Creates $n \times (D+K)$ matrix where each column has correct marginal.

### 7.3 Rearrangement

**Initialize**: Random permutation of each column

**Objective**: None (for maximum entropy case)

**Algorithm**: Block rearrangement to improve dependence structure
- For unconstrained case (K=0), any rearrangement is valid
- Final distribution is equiprobable: $p_i = 1/n$ for each state $i$

### 7.4 Joint Distribution Properties

**Marginal Preservation**:
$$
\mathbb{P}(X_j \leq x) = F_j(x) \quad \forall j
$$

**Independence** (if no constraints):
$$
\mathbb{P}(X_1 \leq x_1, \ldots, X_{D+K} \leq x_{D+K}) = \prod_{j=1}^{D+K} F_j(x_j)
$$

---

## 8. Implied Fama Factor RND

### 8.1 Factor Projection

**Factor Definition**:
$$
F_k = \sum_{j=1}^{D+K} w_{j,k} \cdot S_j
$$

Where:
- $F_k$: Fama factor $k$ (MKT-RF, SMB, HML, RMW, CMA)
- $w_{j,k}$: Factor loading from `factor_weights_tickers.json`
- $S_j$: Asset $j$ price

### 8.2 Implied Factor Distribution

**From Joint States**:

For each state $i = 1, \ldots, n$:
$$
F_{k,i} = \sum_{j=1}^{D+K} w_{j,k} \cdot X_{ij}
$$

Where $X_{ij}$ is asset $j$ value in state $i$ from CBRA.

**Empirical Distribution**:
- States: $\{F_{k,1}, F_{k,2}, \ldots, F_{k,n}\}$
- Probability: $1/n$ each (equiprobable)

### 8.3 Moments

**Mean**:
$$
\mathbb{E}[F_k] = \frac{1}{n} \sum_{i=1}^{n} F_{k,i} = \sum_{j=1}^{D+K} w_{j,k} \cdot \mathbb{E}[S_j]
$$

**Variance**:
$$
\text{Var}(F_k) = \frac{1}{n} \sum_{i=1}^{n} (F_{k,i} - \mathbb{E}[F_k])^2
$$

**Implied Volatility**:
$$
\sigma_{F_k} = \sqrt{\text{Var}(F_k)}
$$

### 8.4 Probability Density Function

**Via Histogram**:
$$
\hat{f}_{F_k}(x) = \frac{1}{n \cdot h} \sum_{i=1}^{n} \mathbb{1}_{[x - h/2, x + h/2]}(F_{k,i})
$$

Where $h$ is the bin width.

**Via Kernel Density Estimation** (smoother):
$$
\hat{f}_{F_k}(x) = \frac{1}{n} \sum_{i=1}^{n} K_h(x - F_{k,i})
$$

With Gaussian kernel: $K_h(u) = \frac{1}{\sqrt{2\pi h^2}} \exp\left(-\frac{u^2}{2h^2}\right)$

---

## 9. Quote-Driven vs Trade-Driven Attribution

### 9.1 Deterministic Decomposition

**Price Change Decomposition**:
$$
\Delta M_t = \Delta M_t^{quote} + \Delta M_t^{trade}
$$

**Quote-Only Change**:
$$
\Delta M_t^{quote} = \sum_{k: \text{no trade in } [t_k - \tau, t_k + \tau]} (M_{t_k} - M_{t_k^-})
$$

**Trade-Linked Change**:
$$
\Delta M_t^{trade} = \sum_{k: \text{trade in } [t_k - \tau, t_k + \tau]} (M_{t_k} - M_{t_k^-})
$$

**Variance Attribution**:
$$
\text{Var}(\Delta M) = \text{Var}(\Delta M^{quote}) + \text{Var}(\Delta M^{trade}) + 2\text{Cov}(\Delta M^{quote}, \Delta M^{trade})
$$

**Shares**:
$$
\omega^{quote} = \frac{\text{Var}(\Delta M^{quote})}{\text{Var}(\Delta M)}, \quad \omega^{trade} = 1 - \omega^{quote}
$$

---

## 10. Cross-Correlation & Lead-Lag Analysis

### 10.1 Cross-Correlation Function

**Definition**:
$$
\rho_{XY}(\tau) = \frac{\mathbb{E}[(X_t - \mu_X)(Y_{t+\tau} - \mu_Y)]}{\sigma_X \sigma_Y}
$$

**Interpretation**:
- $\tau > 0$: $X$ leads $Y$ by $\tau$ periods
- $\tau < 0$: $Y$ leads $X$ by $|\tau|$ periods
- $\tau = 0$: Contemporaneous correlation

### 10.2 Leadership Detection

**For stock-ETF pair**:

$$
\begin{align}
\rho_{stock \to ETF} &= \max_{\tau > 0} \rho_{stock, ETF}(\tau)\\\\
\rho_{ETF \to stock} &= \max_{\tau < 0} \rho_{stock, ETF}(\tau)
\end{align}
$$

**Leadership indicator**:
$$
L = \rho_{stock \to ETF} - \rho_{ETF \to stock}
$$

- $L > 0$: Stock leads ETF
- $L < 0$: ETF leads stock

---

## 11. Option Greeks (For GEX Calculation)

### 11.1 Black-Scholes Greeks

**Delta**:

$$
\Delta = \frac{\partial C}{\partial S} = \begin{cases}
\Phi(d_1) & \text{call}\\\\
\Phi(d_1) - 1 & \text{put}
\end{cases}
$$

**Gamma**:
$$
\Gamma = \frac{\partial^2 C}{\partial S^2} = \frac{\phi(d_1)}{S \sigma \sqrt{\tau}}
$$

**Vega**:
$$
\nu = \frac{\partial C}{\partial \sigma} = S \phi(d_1) \sqrt{\tau}
$$

Where:
$$
d_1 = \frac{\log(S/K) + (r + \sigma^2/2)\tau}{\sigma \sqrt{\tau}}, \quad \phi(x) = \frac{1}{\sqrt{2\pi}} e^{-x^2/2}
$$

### 11.2 Gamma Exposure (GEX)

**Dealer Gamma Position**:
$$
GEX_K = -\Gamma(K) \cdot OI_K \cdot 100
$$

Where:
- $\Gamma(K)$: Gamma at strike $K$
- $OI_K$: Open interest at strike $K$
- Negative sign: Dealers are short gamma (sold options)

**Aggregate GEX**:
$$
GEX_{total} = \sum_K GEX_K
$$

**Net GEX** (calls - puts):
$$
GEX_{net} = \sum_K \left[\Gamma_K^{call} \cdot OI_K^{call} - \Gamma_K^{put} \cdot OI_K^{put}\right] \cdot 100
$$

---

## 12. Numerical Methods & Optimizations

### 12.1 Numerical Differentiation

**Central Difference** (2nd order accurate):
$$
f'(x_i) \approx \frac{f(x_{i+1}) - f(x_{i-1})}{x_{i+1} - x_{i-1}} + O(h^2)
$$

**Second Derivative**:
$$
f''(x_i) \approx \frac{f(x_{i+1}) - 2f(x_i) + f(x_{i-1})}{h^2} + O(h^2)
$$

### 12.2 Trapezoidal Integration

**Cumulative Integral**:
$$
F(x_i) = F(x_{i-1}) + \frac{1}{2}(f(x_{i-1}) + f(x_i))(x_i - x_{i-1})
$$

**Numba Optimization**: JIT-compiled for 10-100× speedup

### 12.3 Matrix Inversion (Cholesky Decomposition)

**For GP**:
$$
\mathbf{K} + \sigma^2 \mathbf{I} = \mathbf{L} \mathbf{L}^\top
$$

Solve $\mathbf{L} \mathbf{y} = \mathbf{b}$ via forward substitution (O(n²) vs O(n³)).

---

## 13. Model Validation & Diagnostics

### 13.1 Arbitrage-Free Checks

**Call Price Monotonicity**:
$$
\frac{\partial C(K)}{\partial K} \leq 0
$$

**Density Non-Negativity**:
$$
q(K) \geq 0 \quad \forall K
$$

**Put-Call Parity**:
$$
C(K) - P(K) = S - K e^{-r\tau}
$$

### 13.2 Calibration Error

**IV Fit Quality**:
$$
RMSE = \sqrt{\frac{1}{N} \sum_{i=1}^{N} (\sigma_i^{obs} - \sigma_i^{fitted})^2}
$$

**Density Normalization**:
$$
\left| \int q(K) dK - 1 \right| < \epsilon \quad (\epsilon = 10^{-6})
$$

---

## 14. Statistical Tests

### 14.1 Granger Causality

**Null Hypothesis**: $X$ does not Granger-cause $Y$

**Test Statistic**:
$$
F = \frac{(RSS_0 - RSS_1)/p}{RSS_1/(n - k)}
$$

Where:
- $RSS_0$: Residual sum of squares without $X$ lags
- $RSS_1$: Residual sum of squares with $X$ lags
- $p$: Number of lags
- $k$: Total parameters

**Reject if**: $F > F_{crit}(\alpha, p, n-k)$

### 14.2 Johansen Cointegration Test

**Trace Statistic**:
$$
LR_{trace}(r) = -T \sum_{i=r+1}^{n} \log(1 - \hat{\lambda}_i)
$$

Where $\hat{\lambda}_i$ are eigenvalues from VECM estimation.

**Null**: At most $r$ cointegrating relationships

---

## References

**Information Leadership**:
- Hasbrouck, J. (1995). "One security, many markets." *Journal of Finance*.
- Yan, B., & Zivot, E. (2010). "A structural analysis of price discovery." *Journal of Financial Markets*.
- Gonzalo, J., & Granger, C. (1995). "Estimation of common long-memory components." *Journal of Business & Economic Statistics*.

**Risk-Neutral Densities**:
- Breeden, D. T., & Litzenberger, R. H. (1978). "Prices of state-contingent claims." *Journal of Business*.
- Rasmussen, C. E., & Williams, C. K. I. (2006). *Gaussian Processes for Machine Learning*.

**Multivariate Option Pricing**:
- Bernard, C., Bondarenko, O., & Vanduffel, S. (2020). "A model-free approach to multivariate option pricing." *Annals of Operations Research*.

**Microstructure**:
- Lee, C., & Ready, M. J. (1991). "Inferring trade direction." *Journal of Finance*.
- Hasbrouck, J. (2007). *Empirical Market Microstructure*. Oxford University Press.

**Fama-French**:
- Fama, E. F., & French, K. R. (2015). "A five-factor asset pricing model." *Journal of Financial Economics*.

---

**APPENDIX A: Notation Table**

| Symbol | Meaning |
|--------|---------|
| $S_t$ | Spot price at time $t$ |
| $K$ | Strike price |
| $\tau$ | Time to expiry (years) |
| $r$ | Risk-free rate |
| $\sigma$ | Volatility |
| $q(S)$ | Risk-neutral density (PDF) |
| $Q(S)$ | Risk-neutral CDF |
| $F_k$ | Fama factor $k$ |
| $\beta_{i,k}$ | Beta of asset $i$ to factor $k$ |
| $IS_i$ | Information Share of asset $i$ |
| $D_t$ | Directional indicator (+1/-1/0) |
| $p_t$ | Kalshi probability |
| $z_t$ | Normalized innovation |
