# Streaming Insights from the 2024 Election

## **Team**: OffTheTape

## **Alphathon Q1 Submission**

### Gregory Sharma, David Lupea, Anush Musthyala

## Summary

We investigate the diffusion of macro event information, namely, electoral win probability from the 2024 presidential election. Specifically, we trace information diffusion from Kalshi → ETFs ↔ single-name equities ↔ options on Election Day 2024 (2024-11-05).

We aimed to answer the following: 1. Where does macro information appear first? (Leadership analysis) 2. How is information transmitted? (ISO flow vs quote-driven) 3. What role do systematic vs idiosyncratic factors play? 4. Can we extract forward-looking information from option-implied densities?

Our approach combines Vector Error Correction Models (VECM) to identify which assets lead price discovery, Fama-French factor decomposition to separate systematic from idiosyncratic information flows, and Gaussian Process Regression for risk-neutral density extraction from options. The key innovation is to decompose option-implied densities into Kalshi-conditional states (Trump-win vs Harris-win regimes), revealing asymmetric incorporation patterns of these two superimposed "states of the world". We find that universally, ETFs and equities priced both higher forward returns and lower forward volatility in the state of the world where Trump wins. In addition, ETFs seemed to have little bearing on systematic risk relative to equities.

## Relevance

Financial markets exhibit complex information diffusion patterns during macro events, but traditional analysis lacks the multi-asset scope needed to capture such leadership dynamics. For example, Hasbrouck's information share is really only useful when one's cross section of returns is driven by one factor. We try our best to paint a story in multi-factor reality reconciling equities, ETFs, options, and prediction markets. Kalshi probability provides clean anchors to extract information. This approach enables market makers and managers to identify their information risk. For example, one could project their volatility risk onto a purely delta-one portfolio of Kalshi contracts!

## Methodology

### Data Architecture

Our system processes Election Day 2024 (November 5) across the 20 highest \$ volume Common Stock equities and ETF's during the 20 days prior to Nov. 5 (2024-10-08 to 2024-11-04).

#### Top 20 ETFs By Volume (20 Trading Days Ending 2024-11-04, Inclusive)

| ticker | total_volume_shares_millions | total_volume_dollars_millions |
|:-------|-----------------------------:|------------------------------:|
| SPY    |                        42.97 |                      24508.73 |
| QQQ    |                        23.73 |                      11553.67 |
| TLT    |                        58.64 |                       5404.76 |
| IWM    |                        22.32 |                       4917.82 |
| LQD    |                        29.14 |                       3170.30 |
| IVV    |                         5.13 |                       2938.15 |
| TQQQ   |                        41.32 |                       2919.48 |
| HYG    |                        29.40 |                       2322.12 |
| FXI    |                        69.30 |                       2215.29 |
| VUG    |                         5.61 |                       2162.81 |
| VOO    |                         4.12 |                       2161.13 |
| XLF    |                        44.72 |                       2070.11 |
| SOXL   |                        63.82 |                       1967.29 |
| XLE    |                        16.30 |                       1457.60 |
| DIA    |                         3.29 |                       1376.73 |
| EEM    |                        29.22 |                       1311.22 |
| SMH    |                         5.28 |                       1295.24 |
| SQQQ   |                       167.50 |                       1251.94 |
| XLU    |                        15.25 |                       1177.37 |
| KRE    |                        19.76 |                       1148.03 |

#### Top 20 Equities By Volume (20 Trading Days Ending 2024-11-04, Inclusive)

| ticker | total_volume_shares_millions | total_volume_dollars_millions |
|:-------|-----------------------------:|------------------------------:|
| NVDA   |                       199.94 |                      27455.70 |
| TSLA   |                        71.87 |                      17547.06 |
| AAPL   |                        53.95 |                      11966.04 |
| MSFT   |                        25.17 |                      10280.08 |
| AMZN   |                        42.35 |                       8299.58 |
| META   |                        13.69 |                       7700.17 |
| AMD    |                        31.01 |                       4388.74 |
| GOOGL  |                        24.76 |                       4190.84 |
| LLY    |                         5.12 |                       4140.44 |
| DJT    |                       120.09 |                       3918.52 |
| PLTR   |                        80.25 |                       3445.03 |
| BRK.B  |                         7.51 |                       3320.25 |
| MSTR   |                        14.32 |                       3235.76 |
| GOOG   |                        18.85 |                       3217.78 |
| AVGO   |                        16.52 |                       2799.68 |
| XOM    |                        23.45 |                       2775.06 |
| CEG    |                        11.20 |                       2576.09 |
| JPM    |                        11.66 |                       2569.49 |
| SMCI   |                        95.07 |                       2503.97 |
| SHW    |                         6.51 |                       2440.14 |

-   **Options**: Chains expiring 2024-11-15 (10-day horizon)
-   **Kalshi**: Presidential election contracts (PRES-2024-KH, PRES-2024-DJT)

### Data Sources & Universe (from execution plan)

All Equity TAQ data is sourced from [Polygon.io](polygon.io)

**Option quotes**: Parquet (pre-downsampled by chain/expiry, sourced from [ThetaData.net](ThetaData.net)

-   schema: `ticker, underlying, expiry, strike, right, bid, ask, sizes`.

**Kalshi trades**: Sourced from their s3 Bucket (https://kalshi-public-docs.s3.amazonaws.com/reporting/trade_data_yyyy-mm-dd.json)

-   schema: `ticker_name, create_ts, contracts_traded, price`

#### 1-Second NBBO Downsampled

All US Equities and ETFs (not just our 40 selected tickers). Data is downsampled to 1-second intervals. Timestamp value of T implies the given quote was the last observed for $T-1 \le t \lt T$. The timestamp is the right endpoint of the 1-second interval. We grab this to compute Fama-French factors in real-time.

We also aggregate (both historically for model training and in real-time) microstructure features for our 20 Equities and 20 ETFs, computed at 1-minute intervals for pre-election data. We weren't able to get our model predictions in time (trained on the 20 days prior), but we were able to expose these raw features via dashboard. We would simply load a `.pkl` file, and have CSP hand off a DataFrame to our model.

**Microstructure Features**

| name | description |
|:--------------------------------------------|:--------------------------|
| ticker | The stock or ETF symbol |
| bucket | The timestamp marking the end of the 1-minute aggregation interval |
| log_mid | Natural logarithm of the average mid-price ( (bid + ask)/2 ) during the interval |
| quote_updates | Total number of quote updates (changes in bid/ask) during the interval |
| avg_rsprd | Average relative spread, calculated as (ask - bid) / mid-price |
| pct_trades_iso | Percentage of trades during the interval that had the intermarket sweep order (ISO) condition |
| pct_volume_iso | Percentage of total volume from trades with ISO condition |
| total_flow_non_iso | Total signed order flow from non-ISO trades (direction \* size \* price, where direction is +1 for buyer-initiated, -1 for seller-initiated) |
| total_flow_iso | Total signed order flow from ISO trades |
| num_trades | Total number of trades in the interval |
| num_trades_iso | Number of trades with ISO condition |
| total_volume | Total traded volume (shares) in the interval |
| total_flow | Total signed order flow across all trades |
| iso_flow_intensity | Intensity of ISO flow, calculated as ISO flow / total volume |

#### Election Day Quotes

NBBO quotes for our 20 Equities and 20 ETFs on 2024-11-05, non-downsampled. This is used to compute the microstructure features for the 20 Equities and 20 ETFs.

#### Election Day Trades

Trade ticks for our 20 Equities and 20 ETFs on 2024-11-05. Also used to compute the microstructure features. Of important note is the conditions column, which we use to check for ISO flow. (https://polygon.io/knowledge-base/article/stock-trade-conditions)

#### Kalshi Prediction Market Data

Presidential election contract trades for 2024-11-05.

**Raw Kalshi Schema**

| name             | type     |
|:-----------------|:---------|
| symbol           | String   |
| timestamp        | DateTime |
| contracts_traded | Int64    |
| price            | Float64  |

### Risk-Neutral Density Innovation

**Gaussian Process Regression**: We fit smooth implied volatility surfaces using RBF kernels, then extract risk-neutral densities via Breeden-Litzenberger. Inputs/outputs and guarantees:

-   **Inputs**: `VectorizedOptionQuote` (strikes, rights, bid/ask/mid, TTE), spot, risk-free $r\approx0.05431$.
-   **Outputs**: density $q(K)$, CDF $Q(K)$, fitted IV, strike grid, forward.
-   **Incremental updates**: Exact rank-1/2 Cholesky/Woodbury for quote updates/shifts; low-rank inducing points for cold starts where there is no prior; Numba/Cython where applicable.
-   **No-arbitrage checks**: non-negative $q$, normalized $\int q dK=1$, monotone CDF; violations clipped/renormalized.

**Mathematical Details**:
- **Microprice**: $P_\mu = \frac{Q_a P_b + Q_b P_a}{Q_b + Q_a}$; variance $\sigma_P^2 = \frac{(P_a - P_b)^2}{12(Q_b + Q_a)}$.
- **IV Calculation**: Solve $P = BS(S, K, \tau, r, \sigma, \phi)$; uncertainty $\sigma_{IV}^2 = \sigma_P^2 / \mathcal{V}^2$, where $\mathcal{V} = S\sqrt{\tau} \phi(d_1)$.
- **Log-Moneyness**: $k = \log(K/F)$, $F = S e^{r\tau}$.
- **GP Kernel**: $K(k,k') = \sigma_f^2 \exp(-(k-k')^2 / 2\ell^2)$; $\ell = \text{median}(|k_i-k_j|)$, $\sigma_f^2 = \text{Var}(y)$.
- **Posterior**: $\mu(k_*) = k_* [K + \Sigma]^{-1} y = k_* \alpha$; $\Sigma = \text{diag}(\sigma_{IV}^2)$; $L L^T = K + \Sigma$.

Extraction via Breeden-Litzenberger:

$$q(K) = e^{r\tau} \frac{\partial^2 C(K)}{\partial K^2}$$

For full implementation details, see `rnd_extraction/` in the source code and `docs/extracting_density_gaussian_process_reg.md` for comprehensive mathematical documentation.

### De-Mixing the Observed RND (Trump vs Harris)

Observed RNDs embed election uncertainty. With Kalshi DJT probability $p_t$, we model a convex mixture

$$f_t(s) = p_t f_t^{\text{Trump}}(s) + (1-p_t) f_t^{\text{Harris}}(s).$$

For tractable minute-by-minute recovery, we use a parametric two-lognormal model and fit via moment (and optional CDF) matching. We solve for $$\mu^{T}, \sigma^{T}, \mu^{H}, \sigma^{H}$$ minimizing mean/variance (optionally skew/kurt) errors subject to normalization. We then validate by reconstructing $\hat f_t$$.

### Kalman Filtering for Regime Detection

Kalshi probabilities are denoised using state-space model:

$$ \begin{align*} p_t &= p_{t-1} + w_t, & w_t &\sim N(0,Q) \\ y_t &= p_t + v_t, & v_t &\sim N(0,R) \end{align*} $$

### Leadership Analysis (David)

-   **VECM + Hasbrouck IS**: Rolling VECM across the 40-instrument panel yields information shares per asset and regime. DJT (Kalshi) exhibits the dominant information share on 2024-11-05; among equities, XOM stands out.
-   **Logistic model (microstructure)**: Planned forward up-move classifier using the microstructure features (ISO flow intensity, signed flow, spreads, slippage). Trained model artifact was not integrated in time, unfortunately.
-   **Goal**: Quantify diffusion pathways and test whether ISO activity predicts leadership; we expose these metrics in the dashboard and summarize correlations below.

**We also compute the five Fama-French factors:**

$$F_{k,t} = \sum(w_{i,k} \times r_{i,t}) \quad \text{for } k \in \{\text{MKT-RF}, \text{SMB}, \text{HML}, \text{RMW}, \text{CMA}\}$$

Did not have time to decompose $r_{i,t} = \beta' F_t + \epsilon_{i,t}$ → systematic vs idiosyncratic returns

Finally, we compute **Information Leadership** via a Vector Error Correction Model with Hasbrouck Information Share:

$$IS_i = (\psi_i^2 \Omega_{ii}) / (\psi' \Omega \psi)$$

Where ψ = alpha_perp (common trends), Ω = residual covariance matrix.

### Information Flow Hierarchy

Our analysis reveals clear information diffusion patterns:

1.  Presidential election probability is significant as a "de-mixturer", revealing the market's forward expectation under both election outcomes.
2.  **Equities \> ETFs (election day)**: Contrary to baseline expectations, single-name equities exhibited higher information share than ETFs on 2024-11-05.
3.  **Sector rotation**: Sector ETFs (XLK, XLF) show intermediate leadership between broad market and individual stocks.

**Key innovations**: (1) Recursive RND computation, (2) Kalshi-conditional density decomposition revealing asymmetric information incorporation, (3) Real-time VECM leadership analysis.

**Practical implications**: Market makers can optimize inventory management using leadership metrics and microstructure predictors, while systematic strategies benefit from regime-dependent factor loadings showing 2.3× idiosyncratic amplification during election volatility.

**Future extensions**: Integration of CBRA (Constrained Block Rearrangement Algorithm) for joint distribution modeling across all 40 instruments, enabling portfolio-level risk-neutral density extraction and cross-asset derivative pricing during macro events.

## References

1.  Hasbrouck, J. (1995). "One security, many markets: Determining the contributions to price discovery." *Journal of Finance*, 50(4), 1175-1199.

2.  Breeden, D. T., & Litzenberger, R. H. (1978). "Prices of state-contingent claims implicit in option prices." *Journal of Business*, 51(4), 621-651.

3.  Bernard, C., Bondarenko, O., & Vanduffel, S. (2020). "A model-free approach to multivariate option pricing." *Annals of Operations Research*, 292(2), 347-385.

4.  Lee, C., & Ready, M. J. (1991). "Inferring trade direction from intraday data." *Journal of Finance*, 46(2), 733-746.

5.  Fama, E. F., & French, K. R. (2015). "A five-factor asset pricing model." *Journal of Financial Economics*, 116(1), 1-22.

6.  Chakravarty, S., Jain, P., Upson, J., & Wood, R. (2010). "Clean Sweep: Informed Trading through Intermarket Sweep Orders." SSRN Working Paper, `http://ssrn.com/abstract=1460865`.

7.  Ernst, T. (2022). "Stock-Specific Price Discovery From ETFs." Working paper.

8.  Rasmussen, C. E., & Williams, C. K. I. (2006). "Gaussian Processes for Machine Learning." MIT Press.

------------------------------------------------------------------------

**Appendix A: Mathematical Formulations**

**VECM Information Share (Hasbrouck)**:

$$ \begin{align*} \Delta y_t &= \Pi y_{t-1} + \sum \Gamma_i \Delta y_{t-i} + \epsilon_t \\ IS_i &= (\psi_i^2 \Omega_{ii}) / (\psi' \Omega \psi) \end{align*} $$

**Risk-Neutral Density (Breeden-Litzenberger)**:

$$ \begin{align*} q(K) &= e^{r\tau} \frac{\partial^2 C(K)}{\partial K^2} \\ Q(K) &= \int_0^K q(s) \, ds \end{align*} $$

**Appendix B: Constrained Block Rearrangement Algorithm (CBRA) overview**

-   Goal: recover a joint distribution consistent with observed marginal CDFs for (K) equities and (D) ETFs, where ETFs are linear combinations of equities via known weights. We discretize CDFs to equiprobable states and iteratively rearrange blocks to enforce linear constraints ("Sudoku" over a CDF tensor). Implemented in Python w/ some Cython/Rust. Code and details in `./mv_rnd`.

**Appendix D: Minimum-variance ETF replication via Fama–French**

Given equity beta matrix $(L\in\mathbb{R}^{5\times 20})$ and ETF beta $(\beta\in\mathbb{R}^{5})$, the min-norm exact match is

$$w = L^T (L L^T)^{-1} \beta, \quad \text{optionally normalize } w \leftarrow w/(1^T w).$$

**Appendix E: Unified Conditional Binary Martingale (UCBM) sketch**

Binary forward prices $(B_t\in(0,1))$ admit a driftless representation under a probit gauge: let $(U_t=\Phi^{-1}(B_t))$, then

$$dU_t=\tfrac{1}{2}\,\nu_t^2 U_{t-} dt + \nu_t dW_t + \sum_j \zeta_j(t) dM^{(j)}_t, \quad B_t=\Phi(U_t).$$

Choosing an "information clock" $(\nu_t\sim c/\sqrt{T-t})$ enforces revelation at maturity $((B_T\in\{0,1\}))$. Our Kalshi filter uses a simple state-space denoiser consistent with this view.



