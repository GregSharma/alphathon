# Implied Factor Risk-Neutral Distributions and Information Share Analysis

**Gregory Sharma**

## Abstract

This paper develops a methodology for extracting risk-neutral joint distributions (RND) for synthetic factor portfolios using the Constrained Block Rearrangement Algorithm (CBRA) and measuring their information share dynamics through Vector Error Correction Models (VECM). The approach addresses the challenge of analyzing untraded factor portfolios by leveraging maximum entropy principles and constituent equity option prices to derive implied factor distributions under the risk-neutral measure, while simultaneously quantifying real-world price discovery mechanisms.

## 1. Introduction

Factor portfolios constructed as weighted combinations of individual equities play a central role in asset pricing models, yet their risk-neutral distributions cannot be directly observed due to the absence of traded options on these synthetic instruments. This paper develops a two-stage framework that combines risk-neutral distribution extraction with information share analysis to provide comprehensive factor risk characterization.

The core challenge: we need to extract **Risk-Neutral Joint Distributions (RND)** for factor portfolios that aren't actually traded—they're just synthetic weighted combinations of equities. The trick is doing this while still being able to measure **Information Share (IS)** in the real-world ($\mathbb{P}$-measure) using log-price data.

The solution adapts the **Constrained Block Rearrangement Algorithm (CBRA)** for these untraded factors by exploiting its built-in maximum entropy principle. Since we don't have option prices on the factors themselves, we have to be clever about it.

## 2. Methodology

The methodology consists of two complementary procedures:

**Stage I ($\mathbb{Q}$-Measure Analysis):** Extract the joint RND from the full cross-section of individual equity options, then back out the implied factor RND using the known portfolio weights. Since we don't have options on the factors themselves, I use **CBRA/BRA** under maximum entropy—basically letting the algorithm pick the "most agnostic" joint distribution consistent with what we observe.

**Stage II ($\mathbb{P}$-Measure Analysis):** Track information share in real-time using intraday data and a **VECM** framework. This tells us which equities are actually leading price discovery for each factor's permanent component.

### 2.1 Risk-Neutral Distribution Extraction ($\mathbb{Q}$-Measure)

Here's the problem: the factors $S_k$ aren't traded, so there are no options on them. Can't directly enforce a marginal constraint like we normally would with CBRA. Instead, I work with what we *do* have—options on all $d$ individual equities. The strategy is to construct the maximum entropy joint RND for the equities first, then derive the factor distributions as weighted combinations of that joint.

#### 2.1.1 Marginal Risk-Neutral Distribution Estimation

Start with the full cross-section of $d$ equities ($X_1, \ldots, X_d$) that make up the factors.

**Step 1: Risk-Neutral Cumulative Distribution Extraction**

For each stock $X_j$, use the observed option prices (calls $C_j(L)$ or puts $P_j(L)$) across strikes to back out the risk-neutral cumulative distribution $F_j(x)$. Standard model-free result from Breeden-Litzenberger:

$$
F_{j}(x)=\left.e^{r T} \frac{\partial C(L)}{\partial L}\right|_{L=x}=\left.e^{r T} \frac{\partial P(L)}{\partial L}\right|_{L=x} \tag{1}
$$

Working with the RNCD instead of the density is cleaner—first derivative instead of second, so less numerical noise.

**Step 2: Discretization**

Pick $n$ quantile points and draw equiprobable samples $x_{ij}$ from each marginal $F_j$. This gives the initial $n \times d$ matrix $\mathbf{X}$.

#### 2.1.2 Joint Distribution Construction via Maximum Entropy

Now the key step: we don't know the true factor distribution $F_{\alpha^k}$, so we have effectively **zero constraints** from factor option prices. The goal is to find the joint distribution $\widetilde{\mathbf{X}}$ that's consistent with the equity marginals $F_j$ but makes the fewest additional assumptions.

**Algorithm Implementation**

Run **BRA** (Block Rearrangement Algorithm), which is just CBRA with $K=0$ constraints. Without external restrictions, it gravitates toward the maximum entropy solution.

**Mathematical Framework**

The algorithm rearranges the rows of $\mathbf{X}$ to find $\widetilde{\mathbf{X}}$ that preserves all the marginals but is otherwise "maximally agnostic" about dependence. We're not imposing structure we don't observe. The output $\widetilde{\mathbf{X}}$ (still $n \times d$) is the discretized joint RND for $(X_1, \ldots, X_d)$ under $\mathbb{Q}$.

#### 2.1.3 Factor Risk-Neutral Distribution Derivation

With the joint equity distribution $\widetilde{\mathbf{X}}$ in hand, we can finally get the implied RND for the factors $S_k$ by just applying the known portfolio weights $\alpha_j^k$.

**Factor Portfolio Definition**

Each factor $k$ (e.g., MKT, SMB, HML) is a weighted sum of the equities:

$$
S_{k}=\sum_{i=1}^{d} \alpha_{i}^{k} X_{i} \tag{2}
$$

**Implied Distribution Computation**

For each scenario $i=1, \ldots, n$ in the discretized joint, compute the factor value by applying the weights to that row of $\widetilde{\mathbf{X}}$:

$$
S_{k, i}^{\mathbb{Q}} = \sum_{j=1}^d \alpha_j^k \tilde{x}_{ij} \tag{3}
$$

The collection $\{S_{k, i}^{\mathbb{Q}}\}_{i=1}^n$ is the implied risk-neutral distribution for factor $S_k$—what the market is pricing given the constituent option prices and the maximum entropy joint structure.

### 2.2 Information Share Analysis ($\mathbb{P}$-Measure)

Now we flip to the real-world measure. The question here: which equities are actually *leading* the factor's permanent component? Who's driving price discovery?

This uses intraday data in log-price levels (I(1) series) and applies a VECM framework. I run this **separately for each factor $f$**, focusing on a small system with just the factor and its most relevant equities.

#### 2.2.1 System Specification

**Factor Series Construction**

Build the factor series: Construct the cumulative log-return series for factor $F_{f,t}$ (the weighted dot product in I(1) space). Then pick the 5 equities with the strongest relationship to that factor—say, the highest absolute $\beta_{j,f}$ loadings.

**Multivariate System Definition**

The $N=6$ dimensional system is just these log-prices stacked together:

$$
\mathbf{Y}_t = [\log(P_{1,t}), \dots, \log(P_{5,t}), F_{f,t}]^\top \tag{4}
$$

#### 2.2.2 Vector Error Correction Model Estimation

**Model Specification**

Fit the VECM: Estimate the error correction model on $\mathbf{Y}_t$ to separate short-run noise from long-run equilibrium relationships. Use Johansen to determine the cointegrating rank $r$.

$$
\Delta \mathbf{Y}_t = \boldsymbol{\Pi} \mathbf{Y}_{t-1} + \sum_{i=1}^{k-1} \boldsymbol{\Gamma}_i \Delta \mathbf{Y}_{t-i} + \boldsymbol{\varepsilon}_t \tag{5}
$$

**Innovation Orthogonalization**

Orthogonalize the shocks: Take the VECM residuals $\boldsymbol{\varepsilon}_t$ and Cholesky decompose the covariance ($\boldsymbol{\Omega} = \mathbf{C} \mathbf{C}^\top$) to get uncorrelated innovations $\boldsymbol{\eta}_t$.

**Permanent Component Isolation**

Isolate permanent impacts: The information share calculation needs the long-run variance, which comes from the infinite MA representation. Approximate the long-run impact matrix $\boldsymbol{\Xi}$ by summing the MA coefficient matrices $\boldsymbol{\Psi}_m$ weighted by the common trends basis $\mathbf{B}_\perp$:

$$
\boldsymbol{\Xi} = \sum_{m=0}^M \boldsymbol{\Psi}_m \mathbf{B}_\perp \tag{6}
$$

#### 2.2.3 Information Share Computation

Finally, compute the information share $s_j$ for each component (the 5 equities and the factor). This is just the fraction of permanent variance explained by each series' innovations:

$$
s_j = \frac{ [\boldsymbol{\Xi} \boldsymbol{\Xi}^\top]_{j,j} }{ \trace(\boldsymbol{\Xi} \boldsymbol{\Xi}^\top) } \tag{7}
$$

The output is a vector $\mathbf{s}$ where $s_1, \ldots, s_5$ are the equity shares and $s_f$ is the factor's own share. This tells you who's actually *leading* price discovery for that factor's permanent component—which equities are moving first versus which are just following.

## 3. Results

[Results section to be completed based on empirical implementation]

## 4. Conclusion

This framework provides a comprehensive approach to factor risk analysis by combining risk-neutral distribution extraction with information share measurement. The maximum entropy approach to joint distribution construction offers a principled method for handling untraded factor portfolios, while the VECM-based information share analysis reveals the real-world price discovery dynamics that drive factor movements.

## References

[References to be added]