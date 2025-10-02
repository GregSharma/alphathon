## 1. Mini-Paper (PDF)
[docs/MiniPaper.md](docs/MiniPaper.md)

## 2. Results Spreadsheet & Data

### Spreadsheets
Vector error correction model information leadership analysis across 20 equities and 20 ETFs:
- [plots_and_figures/vecm_leadership_data.csv](plots_and_figures/vecm_leadership_data.csv)

Runtime comparison for fitting implied risk-neutral density using Gaussian process regression with fresh starts vs incremental low-rank updates:
- [plots_and_figures/example_RND_incremental_nvda/incremental_comparison.csv](plots_and_figures/example_RND_incremental_nvda/incremental_comparison.csv)

### Plots & Figures

**Risk-neutral densities** across all 40 underlyings at 10:00 AM on November 5th, 2024:
- [plots_and_figures/example_RND_10am_1105/](plots_and_figures/example_RND_10am_1105/)

**Implied volatility surfaces** - smooth market-implied vol, unsmoothed, and double-exponential Kou jump diffusion fits:
- [plots_and_figures/iv_and_kou_jump_diffusion_fits/](plots_and_figures/iv_and_kou_jump_diffusion_fits/)

**Kalshi vs Fama factors** - comparisons of Kalshi-derived electoral probability with intraday computed Fama factor returns:
- [plots_and_figures/kalshi_and_fama/](plots_and_figures/kalshi_and_fama/)

**Kalman Filter & Logit Transformation** to denoise raw presidential prices and map to a martingale:
- [plots_and_figures/kalshi_kalman_filter.pdf](plots_and_figures/kalshi_kalman_filter.pdf)
- For context, see [docs/Greg_WkngIdeas_on_Binary_Martingales.md](docs/Greg_WkngIdeas_on_Binary_Martingales.md)
- Reference: [Election Predictions as Martingales: An Arbitrage Approach](https://arxiv.org/pdf/1703.06351) by Nassim Nicholas Taleb (NYU, October 2017)

**Regime decomposition** - moment matching to get lognormal superimposition on Tesla risk-neutral density with binary outcome of election:
- [plots_and_figures/TSLA_regime_decomp_1500.pdf](plots_and_figures/TSLA_regime_decomp_1500.pdf)

**VECM leadership analysis** - examples of leadership plot using vector error correction model across all 40 assets, lag 1 on a minute basis:
- [plots_and_figures/vecm_leadership_analysis.pdf](plots_and_figures/vecm_leadership_analysis.pdf)

## 3. Code Files

*See section 4 below for environment setup.*

**Main dashboard files** (serve live Panel dashboards):
- [graphing/Calculating_Binary_RiskNeutralDensities.py](graphing/Calculating_Binary_RiskNeutralDensities.py) - [📹 Demo Video](https://drive.google.com/file/d/1f9ivleKQddDq5SVWGY_ysZuPdMhbkZDi/view?usp=drive_link)
- [graphing/Leadership_and_FactorModel.py](graphing/Leadership_and_FactorModel.py) - [📹 Demo Video](https://drive.google.com/file/d/1LFq6dWnWWXEH2WPFlBXI4eEPgDfEbWpk/view?usp=drive_link)

**Demonstration dashboard** - calculates Fama factors intraday as well as microstructure features:
- [graphing/Demonstrate_Factors_and_Microstructure_Features.py](graphing/Demonstrate_Factors_and_Microstructure_Features.py)

**Additional test files:**
- [graphing/test_fama_and_microstructure_features.py](graphing/test_fama_and_microstructure_features.py)

**Supporting libraries:**

Risk-neutral density extraction:
- [rnd_extraction/](rnd_extraction/)

CBRA algorithm (Conditional Block Rearrangement Algorithm):
- [mv_rnd/](mv_rnd/)
- Reference: "A model-free approach to multivariate option pricing" by Carole Bernard, Oleg Bondarenko, Steven Vanduffel (Accepted: October 2020, Springer)

Fama-French factors - connecting to Wharton's WRDS to calculate Fama factors:
- [fama/](fama/)

Information leadership research (not completed due to time constraints) - historical ML pipeline to predict one-lag-ahead information leader using microstructure features up until the observed time.

## 4. Installation & Platform Access

**Requirements:**
- Python (3.12+, lower could work too, only tested at 3.12)
- Install: `pip install -e .` (uv is faster, if you have it)
- `python download_data.py` (will extract data.zip into root directory of the repo),
- then run the **Main dashboard files** above from the root directory (don't `cd` into `graphing/`)





