# Risk-Neutral Density Extraction: Mathematical Methodology

## Overview

This pipeline extracts a continuous, arbitrage-free **risk-neutral density (RND)** and its **cumulative distribution** from an observed option chain using Gaussian Process regression and the Breeden-Litzenberger identity.

---

## 1. Input Processing

### 1.1 Implied Volatility Calculation
For each option with price $P$, we invert the Black-Scholes formula to obtain implied volatility $\sigma_{imp}$:

$$
P = BS(S, K, \tau, r, \sigma_{imp}, \phi)
$$

where:
- $S$: spot price
- $K$: strike price
- $\tau$: time to expiry
- $r$: risk-free rate
- $\phi \in \{c, p\}$: call/put flag

### 1.2 Microprice and Uncertainty
To account for bid-ask spread, we compute the **microprice** (size-weighted midpoint):

$$
P_{micro} = \frac{Q_{ask} \cdot P_{bid} + Q_{bid} \cdot P_{ask}}{Q_{bid} + Q_{ask}}
$$

Price uncertainty (for GP weighting) uses a Beta distribution with variance:

$$
\sigma_P^2 = \frac{(P_{ask} - P_{bid})^2}{12 \cdot (Q_{bid} + Q_{ask})}
$$

IV uncertainty via vega transformation:

$$
\sigma_{IV}^2 = \frac{\sigma_P^2}{\mathcal{V}^2}, \quad \mathcal{V} = S\sqrt{\tau} \cdot \phi(d_1)
$$

---

## 2. Gaussian Process Regression

### 2.1 Log-Moneyness Transformation
We work in **log-moneyness space** for numerical stability:

$$
k = \log(K/F), \quad F = S e^{r\tau}
$$

### 2.2 GP Model
Fit a GP to observed IVs $\{k_i, \sigma_i\}$ with RBF kernel:

$$
\sigma(k) \sim \mathcal{GP}(0, K(k, k'))
$$

$$
K(k, k') = \sigma_f^2 \exp\left(-\frac{(k - k')^2}{2\ell^2}\right)
$$

Hyperparameters $\ell$ (length scale) and $\sigma_f^2$ (signal variance) are estimated heuristically:
- $\ell = \text{median}(|k_i - k_j|)$
- $\sigma_f^2 = \text{var}(\sigma_i)$

### 2.3 Posterior Prediction
Given observations $\mathbf{y}$ with noise $\sigma_{IV}^2$, the posterior mean on grid $k_*$ is:

$$
\mu(k_*) = K(k_*, k) [K(k, k) + \Sigma]^{-1} \mathbf{y}
$$

where $\Sigma = \text{diag}(\sigma_{IV}^2)$.

---

## 3. RND Extraction (Breeden-Litzenberger)

### 3.1 Identity
The risk-neutral density $q(K)$ is the second derivative of the call price w.r.t. strike:

$$
q(K) = e^{r\tau} \frac{\partial^2 C(K)}{\partial K^2}
$$

### 3.2 Numerical Implementation
1. Compute call prices on grid using fitted IV: $C(K) = BS(S, K, \tau, r, \mu(k))$
2. Numerical differentiation (central differences):
   $$
   \frac{\partial C}{\partial K} \approx \frac{C_{i+1} - C_{i-1}}{K_{i+1} - K_{i-1}}
   $$
3. Density in strike space: $q(K) = e^{r\tau} \frac{\partial^2 C}{\partial K^2}$
4. Transform to log-moneyness: $q(k) = q(K) \cdot K$ (Jacobian)
5. Normalize: $\int q(k) \, dk = 1$

---

## 4. Cumulative RND

The **cumulative risk-neutral distribution** is:

$$
Q(k) = \int_{-\infty}^k q(s) \, ds
$$

Computed via trapezoidal integration with Numba optimization:

$$
Q(k_i) = Q(k_{i-1}) + \frac{1}{2}(q_{i-1} + q_i)(k_i - k_{i-1})
$$

Properties:
- $Q(-\infty) = 0$
- $Q(+\infty) = 1$
- Monotonically increasing

---

## 5. Characteristic Function

The **characteristic function** of the RND is its Fourier transform:

$$
\varphi(u) = \int_{-\infty}^{\infty} e^{\mathrm{i}uk} q(k) \, dk
$$

Computed numerically via trapezoidal rule:

$$
\varphi(u) \approx \Delta k \sum_j e^{\mathrm{i}u k_j} q(k_j)
$$

Properties:
- $\varphi(0) = 1$ (normalization check)
- Real part: $\text{Re}[\varphi(u)] = \int \cos(uk) q(k) \, dk$

---

## 6. Arbitrage-Free Guarantees

The pipeline ensures no-arbitrage through:

1. **Monotonic call prices**: GP smoothness prevents price crossings
2. **Non-negative density**: Clipping $q(k) = \max(q(k), 0)$ after differentiation
3. **Proper normalization**: $\int q(k) \, dk = 1$
4. **Put-call parity**: Implicit through forward price $F = Se^{r\tau}$

---

## 7. Computational Optimizations

- **Numba JIT compilation**: RBF kernel, integration, CF computation
- **Sparse GP**: Low-rank approximation for large chains (not yet implemented)
- **Vectorization**: NumPy broadcasting for BS pricing and differentiation

---

## References

- Breeden, D. T., & Litzenberger, R. H. (1978). "Prices of state-contingent claims implicit in option prices." *Journal of Business*.
- Rasmussen, C. E., & Williams, C. K. I. (2006). *Gaussian Processes for Machine Learning*.

