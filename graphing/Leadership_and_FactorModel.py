"""
Combined VECM Leadership + Fama Factors + Microstructure Dashboard

Integrates:
1. VECM Information Leadership (40 tickers)
2. Fama-French 5 Factors
3. Microstructure features (ISO flow, spread, volume)

Interactive controls to explore relationships between leadership and market microstructure.

Usage:
    python graphing/test_leadership_and_fama.py
"""

import json
import logging
import warnings
from datetime import timedelta
from typing import Dict

import numpy as np
import pandas as pd

# Suppress warnings
warnings.filterwarnings(
    "ignore",
    message="Conversion of an array with ndim > 0 to a scalar is deprecated, and will error in future.",
    category=DeprecationWarning,
)

import csp
from csp import ts
from statsmodels.tsa.vector_ar.vecm import VECM

from CSP_Options.fama import compute_fama_factors_graph
from CSP_Options.microstructure import compute_microstructure_features_basket
from CSP_Options.structs import EquityBar1m, FamaFactors, FamaReturns
from CSP_Options.utils.readers import (
    ALL_TICKERS,
    EQUITIES,
    ETFS,
    get_quotes_1s_wo_size,
    get_taq_quotes,
    get_taq_trades,
)

# Panel dashboard
try:
    import panel as pn
    import plotly.graph_objects as go

    from CSP_Options.panel_dashboard import BaseDashboard

    PANEL_AVAILABLE = True
except ImportError:
    PANEL_AVAILABLE = False
    print("❌ Panel not available. Install with: pip install panel plotly")
    exit(1)

logging.basicConfig(level=logging.INFO)

DATE_OF_INTEREST = "2024-11-05"
FACTOR_WEIGHTS_PATH = "/home/grego/Alphathon/fama/factor_weights_tickers.json"

# Load factor weights
with open(FACTOR_WEIGHTS_PATH, "r") as f:
    factor_weights = json.load(f)

ALL_FACTOR_TICKERS = set()
for factor_dict in factor_weights.values():
    ALL_FACTOR_TICKERS.update(factor_dict.keys())

start_ts = pd.Timestamp(f"{DATE_OF_INTEREST} 09:30:00", tz="America/New_York")
end_ts = pd.Timestamp(f"{DATE_OF_INTEREST} 16:00:00", tz="America/New_York")


def compute_vecm_leadership(log_mid_matrix: np.ndarray, tickers: list) -> Dict[str, float]:
    """Compute VECM leadership scores from log mid-prices."""
    try:
        df = pd.DataFrame(log_mid_matrix, columns=tickers)
        if (df.std() < 1e-10).any() or df.isna().any().any():
            return {t: 0.0 for t in tickers}

        vecm = VECM(df, k_ar_diff=1, coint_rank=1)
        res = vecm.fit()
        alpha = res.alpha
        Omega = res.sigma_u

        U, _, _ = np.linalg.svd(alpha, full_matrices=True)
        alpha_perp = U[:, -1]
        sgn = np.sign(alpha_perp[0]) if alpha_perp[0] != 0 else 1.0
        alpha_perp = sgn * alpha_perp
        denom = np.sum(np.abs(alpha_perp))
        alpha_perp = (
            alpha_perp / denom if denom > 1e-12 else np.ones(len(alpha_perp)) / len(alpha_perp)
        )

        CS = np.abs(alpha_perp)
        CS = CS / CS.sum()

        psi = alpha_perp.reshape(-1, 1)
        denom_is = float(psi.T @ Omega @ psi)
        if not np.isfinite(denom_is) or abs(denom_is) < 1e-18:
            denom_is = float(np.trace(Omega))
            denom_is = denom_is if denom_is > 0 else 1.0

        num = np.array(
            [(psi[i, 0] ** 2) * Omega[i, i] for i in range(len(alpha_perp))], dtype=float
        )
        num = np.clip(num, 0.0, None)
        if num.sum() <= 0:
            IS = np.ones_like(num) / len(num)
        else:
            IS = num / denom_is
            IS = IS / IS.sum()

        beta = IS / (CS + 1e-12)
        w = beta**2
        ILS = w / w.sum() if w.sum() > 0 else np.ones_like(w) / len(w)

        return {ticker: float(ILS[i]) for i, ticker in enumerate(tickers)}
    except Exception:
        return {t: 0.0 for t in tickers}


class LeadershipFamaDashboard(BaseDashboard):
    """Combined dashboard for leadership, Fama factors, and microstructure."""

    def __init__(self):
        super().__init__()

        # Initialize Panel
        pn.extension("plotly")

        self.data_history = {
            "times": [],
            # Leadership
            "leadership_dicts": [],
            "leaders": [],
            # Fama factors
            "HML_log": [],
            "SMB_log": [],
            "RMW_log": [],
            "CMA_log": [],
            "MKT_RF_log": [],
            "HML_ret": [],
            "SMB_ret": [],
            "RMW_ret": [],
            "CMA_ret": [],
            "MKT_RF_ret": [],
        }

        # Store microstructure for all tickers
        self.microstructure_data = {
            ticker: {
                "log_mid": [],
                "iso_flow": [],
                "total_flow": [],
                "avg_rsprd": [],
                "num_trades": [],
                "pct_iso": [],
            }
            for ticker in ALL_TICKERS
        }

        # Ticker scores for all
        self.ticker_scores = {ticker: [] for ticker in ALL_TICKERS}

        # Create interactive controls
        self.ticker_selector = pn.widgets.Select(
            name="Select Ticker",
            options=sorted(ALL_TICKERS),
            value="SPY",
            width=200,
        )

        self.factor_selector = pn.widgets.Select(
            name="Select Fama Factor",
            options=["HML", "SMB", "RMW", "CMA", "MKT_RF"],
            value="MKT_RF",
            width=200,
        )

        self.microstructure_metric = pn.widgets.Select(
            name="Microstructure Metric",
            options=["ISO Flow", "Total Flow", "Spread (bps)", "Num Trades", "% ISO"],
            value="ISO Flow",
            width=200,
        )

        # Create plot panes
        self.leadership_heatmap_pane = self._create_plot_pane(height=800)  # Taller for 40 tickers
        self.ticker_leadership_pane = self._create_plot_pane(height=400)
        self.ticker_microstructure_pane = self._create_plot_pane(height=400)
        self.factor_performance_pane = self._create_plot_pane(height=400)
        self.leadership_vs_micro_pane = self._create_plot_pane(height=400)
        self.sector_comparison_pane = self._create_plot_pane(height=400)
        self.top_leaders_pane = self._create_plot_pane(height=400)
        self.all_tickers_scatter_pane = self._create_plot_pane(height=500)

        # Stats panel
        self.stats_pane = pn.pane.Markdown("**Waiting for data...**", sizing_mode="stretch_width")

        # Bind update callbacks
        self.ticker_selector.param.watch(self._on_ticker_change, "value")
        self.factor_selector.param.watch(self._on_factor_change, "value")
        self.microstructure_metric.param.watch(self._on_metric_change, "value")

        # Layout
        self.dashboard = pn.template.FastListTemplate(
            title="📊 VECM Leadership + Fama + Microstructure Analysis (40 Tickers)",
            sidebar=[
                pn.pane.Markdown("## Interactive Controls"),
                self.ticker_selector,
                self.factor_selector,
                self.microstructure_metric,
                pn.pane.Markdown("---"),
                pn.pane.Markdown("## Statistics"),
                self.stats_pane,
                pn.pane.Markdown("### About"),
                pn.pane.Markdown(
                    "**VECM Leadership**: Information share scores\n"
                    "**Fama-French**: 5-factor model\n"
                    "**Microstructure**: ISO flow, spreads, volume\n\n"
                    "**All 40 tickers**: 20 ETFs + 20 Equities\n"
                    "Use dropdowns to explore individual tickers"
                ),
            ],
            main=[
                pn.Row(
                    pn.Column(
                        "## Leadership Heatmap (All 40 Tickers)", self.leadership_heatmap_pane
                    ),
                    sizing_mode="stretch_width",
                ),
                pn.Row(
                    pn.Column(
                        "## Selected Ticker: Leadership Over Time", self.ticker_leadership_pane
                    ),
                    pn.Column(
                        "## Selected Ticker: Microstructure", self.ticker_microstructure_pane
                    ),
                    sizing_mode="stretch_width",
                ),
                pn.Row(
                    pn.Column("## Selected Fama Factor", self.factor_performance_pane),
                    pn.Column("## Leadership vs Microstructure", self.leadership_vs_micro_pane),
                    sizing_mode="stretch_width",
                ),
                pn.Row(
                    pn.Column("## ETFs vs Equities Leadership", self.sector_comparison_pane),
                    pn.Column("## Top 5 Leaders", self.top_leaders_pane),
                    sizing_mode="stretch_width",
                ),
                pn.Row(
                    pn.Column(
                        "## All Tickers: Leadership vs Microstructure",
                        self.all_tickers_scatter_pane,
                    ),
                    sizing_mode="stretch_width",
                ),
            ],
            accent_base_color="#4169E1",
            header_background="#1a1a1a",
        )

    def update(self, t, leadership_result, fama_log, fama_ret, bars_dict):
        """Update dashboard with new data."""
        # Only update if we have leadership data (required for all plots)
        if not leadership_result:
            return

        # Debug output
        n_updates = len(self.data_history["times"])
        if n_updates % 10 == 0 or n_updates < 5:
            print(f"📊 Dashboard update #{n_updates}: {t}")
            print(f"   - Leadership: {'✓' if leadership_result else '✗'}")
            print(f"   - Fama log: {'✓' if fama_log else '✗'}")
            print(f"   - Fama ret: {'✓' if fama_ret else '✗'}")
            print(f"   - Bars: {len(bars_dict)} tickers")

        self.data_history["times"].append(t)

        # Leadership (guaranteed to exist at this point)
        leadership_scores = leadership_result["leadership"]
        leader = leadership_result["leader"]
        self.data_history["leadership_dicts"].append(leadership_scores)
        self.data_history["leaders"].append(leader)

        for ticker in ALL_TICKERS:
            self.ticker_scores[ticker].append(leadership_scores.get(ticker, 0.0))

        # Fama factors
        if fama_log:
            self.data_history["HML_log"].append(fama_log.HML)
            self.data_history["SMB_log"].append(fama_log.SMB)
            self.data_history["RMW_log"].append(fama_log.RMW)
            self.data_history["CMA_log"].append(fama_log.CMA)
            self.data_history["MKT_RF_log"].append(fama_log.MKT_RF)
        else:
            # Append None to keep arrays in sync
            self.data_history["HML_log"].append(None)
            self.data_history["SMB_log"].append(None)
            self.data_history["RMW_log"].append(None)
            self.data_history["CMA_log"].append(None)
            self.data_history["MKT_RF_log"].append(None)

        if fama_ret:
            self.data_history["HML_ret"].append(fama_ret.HML)
            self.data_history["SMB_ret"].append(fama_ret.SMB)
            self.data_history["RMW_ret"].append(fama_ret.RMW)
            self.data_history["CMA_ret"].append(fama_ret.CMA)
            self.data_history["MKT_RF_ret"].append(fama_ret.MKT_RF)
        else:
            # Append None to keep arrays in sync
            self.data_history["HML_ret"].append(None)
            self.data_history["SMB_ret"].append(None)
            self.data_history["RMW_ret"].append(None)
            self.data_history["CMA_ret"].append(None)
            self.data_history["MKT_RF_ret"].append(None)

        # Microstructure for all tickers
        for ticker in ALL_TICKERS:
            if ticker in bars_dict:
                bar = bars_dict[ticker]
                self.microstructure_data[ticker]["log_mid"].append(bar.log_mid)
                self.microstructure_data[ticker]["iso_flow"].append(bar.iso_flow_intensity)
                self.microstructure_data[ticker]["total_flow"].append(bar.total_flow)
                self.microstructure_data[ticker]["avg_rsprd"].append(bar.avg_rsprd * 10000)
                self.microstructure_data[ticker]["num_trades"].append(bar.num_trades)
                self.microstructure_data[ticker]["pct_iso"].append(bar.pct_trades_iso * 100)
            else:
                # Append None to keep arrays in sync
                self.microstructure_data[ticker]["log_mid"].append(None)
                self.microstructure_data[ticker]["iso_flow"].append(None)
                self.microstructure_data[ticker]["total_flow"].append(None)
                self.microstructure_data[ticker]["avg_rsprd"].append(None)
                self.microstructure_data[ticker]["num_trades"].append(None)
                self.microstructure_data[ticker]["pct_iso"].append(None)

        # Trim history
        self._trim_history(self.data_history, max_points=500)
        for ticker in ALL_TICKERS:
            for key in self.microstructure_data[ticker]:
                if len(self.microstructure_data[ticker][key]) > 500:
                    self.microstructure_data[ticker][key] = self.microstructure_data[ticker][key][
                        -500:
                    ]
            if len(self.ticker_scores[ticker]) > 500:
                self.ticker_scores[ticker] = self.ticker_scores[ticker][-500:]

        # Update plots
        if len(self.data_history["times"]) > 2:
            self._update_all_plots()
            self._update_stats()

    def _update_all_plots(self):
        """Update all plots with current data."""
        self.leadership_heatmap_pane.object = self._plot_leadership_heatmap()
        self.ticker_leadership_pane.object = self._plot_ticker_leadership()
        self.ticker_microstructure_pane.object = self._plot_ticker_microstructure()
        self.factor_performance_pane.object = self._plot_factor_performance()
        self.leadership_vs_micro_pane.object = self._plot_leadership_vs_microstructure()
        self.sector_comparison_pane.object = self._plot_sector_comparison()
        self.top_leaders_pane.object = self._plot_top_leaders()
        self.all_tickers_scatter_pane.object = self._plot_all_tickers_scatter()

    def _on_ticker_change(self, event):  # noqa: ARG002
        """Callback when ticker selection changes."""
        if len(self.data_history["times"]) > 2:
            self.ticker_leadership_pane.object = self._plot_ticker_leadership()
            self.ticker_microstructure_pane.object = self._plot_ticker_microstructure()
            self.leadership_vs_micro_pane.object = self._plot_leadership_vs_microstructure()

    def _on_factor_change(self, event):  # noqa: ARG002
        """Callback when factor selection changes."""
        if len(self.data_history["times"]) > 2:
            self.factor_performance_pane.object = self._plot_factor_performance()

    def _on_metric_change(self, event):  # noqa: ARG002
        """Callback when microstructure metric changes."""
        if len(self.data_history["times"]) > 2:
            self.ticker_microstructure_pane.object = self._plot_ticker_microstructure()
            self.leadership_vs_micro_pane.object = self._plot_leadership_vs_microstructure()
            self.all_tickers_scatter_pane.object = self._plot_all_tickers_scatter()

    def _plot_leadership_heatmap(self):
        """Plot leadership heatmap for all 40 tickers."""
        fig = go.Figure()

        times = pd.to_datetime(self.data_history["times"])
        n_times = len(times)

        # Build score matrix (tickers x time)
        score_matrix = np.zeros((len(ALL_TICKERS), n_times))
        for i, ticker in enumerate(ALL_TICKERS):
            for j in range(n_times):
                ld = self.data_history["leadership_dicts"][j]
                score_matrix[i, j] = ld.get(ticker, 0.0)

        # Separate ETFs and Equities for better visualization
        etf_tickers = sorted([t for t in ALL_TICKERS if t in ETFS])
        equity_tickers = sorted([t for t in ALL_TICKERS if t in EQUITIES])

        # Get indices in sorted order
        etf_indices = [ALL_TICKERS.index(t) for t in etf_tickers]
        equity_indices = [ALL_TICKERS.index(t) for t in equity_tickers]

        etf_matrix = score_matrix[etf_indices, :]
        equity_matrix = score_matrix[equity_indices, :]

        # Stack them with a separator
        combined_matrix = np.vstack([etf_matrix, equity_matrix])
        combined_names = etf_tickers + equity_tickers

        fig.add_trace(
            go.Heatmap(
                z=combined_matrix,
                x=times,
                y=combined_names,
                colorscale="YlOrRd",
                zmin=0,
                zmax=0.3,
                colorbar=dict(title="ILS Score"),
            )
        )

        # Add horizontal line separator between ETFs and Equities
        fig.add_hline(
            y=len(etf_tickers) - 0.5,
            line_dash="dash",
            line_color="black",
            line_width=2,
            opacity=0.5,
        )

        fig.update_layout(
            xaxis=dict(title="Time", gridcolor="lightgray"),
            yaxis=dict(title="Ticker", gridcolor="lightgray"),
            plot_bgcolor="white",
            height=800,  # Increased height to show all 40 tickers
            autosize=False,
            font=dict(family=self.FONT_FAMILY, size=self.FONT_SIZE),
            margin=dict(l=80, r=20, t=30, b=50),
            uirevision="leadership_heatmap",  # Preserve zoom/pan state
            annotations=[
                dict(
                    x=-0.05,
                    y=len(etf_tickers) / len(combined_names),
                    xref="paper",
                    yref="paper",
                    text=f"ETFs ({len(etf_tickers)})",
                    showarrow=False,
                    font=dict(size=12, color="blue", weight="bold"),
                    xanchor="right",
                ),
                dict(
                    x=-0.05,
                    y=(len(etf_tickers) + len(equity_tickers) / 2) / len(combined_names),
                    xref="paper",
                    yref="paper",
                    text=f"Equities ({len(equity_tickers)})",
                    showarrow=False,
                    font=dict(size=12, color="green", weight="bold"),
                    xanchor="right",
                ),
            ],
        )

        return fig

    def _plot_ticker_leadership(self):
        """Plot leadership score for selected ticker."""
        fig = go.Figure()

        ticker = self.ticker_selector.value
        times = pd.to_datetime(self.data_history["times"])

        if ticker in self.ticker_scores and len(self.ticker_scores[ticker]) > 0:
            scores = self.ticker_scores[ticker]
            color = "blue" if ticker in ETFS else "green"

            fig.add_trace(
                go.Scatter(
                    x=times,
                    y=scores,
                    mode="lines",
                    line=dict(width=2, color=color),
                    fill="tozeroy",
                    fillcolor=f"rgba({0 if color == 'blue' else 0}, {0 if color == 'blue' else 128}, {255 if color == 'blue' else 0}, 0.2)",
                    name=ticker,
                )
            )

        fig.update_layout(
            xaxis=dict(title="Time", gridcolor="lightgray"),
            yaxis=dict(title="Leadership Score (ILS)", gridcolor="lightgray", type="log"),
            plot_bgcolor="white",
            hovermode="x unified",
            height=400,
            autosize=False,
            font=dict(family=self.FONT_FAMILY, size=self.FONT_SIZE),
            margin=dict(l=50, r=20, t=30, b=50),
            uirevision="ticker_leadership",  # Preserve zoom/pan state
        )

        return fig

    def _plot_ticker_microstructure(self):
        """Plot microstructure metric for selected ticker."""
        fig = go.Figure()

        ticker = self.ticker_selector.value
        metric = self.microstructure_metric.value
        times = pd.to_datetime(self.data_history["times"])

        # Map metric to data key
        metric_map = {
            "ISO Flow": "iso_flow",
            "Total Flow": "total_flow",
            "Spread (bps)": "avg_rsprd",
            "Num Trades": "num_trades",
            "% ISO": "pct_iso",
        }

        data_key = metric_map[metric]
        if (
            ticker in self.microstructure_data
            and len(self.microstructure_data[ticker][data_key]) > 0
        ):
            values_raw = self.microstructure_data[ticker][data_key]

            # Filter out None values
            valid_idx = [i for i, v in enumerate(values_raw) if v is not None]

            if len(valid_idx) > 0:
                valid_times = times[valid_idx]
                values = [values_raw[i] for i in valid_idx]

                fig.add_trace(
                    go.Scatter(
                        x=valid_times,
                        y=values,
                        mode="lines",
                        line=dict(width=2, color="orange"),
                        fill="tozeroy",
                        fillcolor="rgba(255, 165, 0, 0.2)",
                        name=metric,
                    )
                )

        fig.update_layout(
            xaxis=dict(title="Time", gridcolor="lightgray"),
            yaxis=dict(title=metric, gridcolor="lightgray"),
            plot_bgcolor="white",
            hovermode="x unified",
            height=400,
            autosize=False,
            font=dict(family=self.FONT_FAMILY, size=self.FONT_SIZE),
            margin=dict(l=50, r=20, t=30, b=50),
            uirevision="ticker_microstructure",  # Preserve zoom/pan state
        )

        return fig

    def _plot_factor_performance(self):
        """Plot selected Fama factor log price performance."""
        fig = go.Figure()

        factor = self.factor_selector.value
        times = pd.to_datetime(self.data_history["times"])

        log_key = f"{factor}_log"

        if len(self.data_history[log_key]) > 0:
            # Filter out None values
            log_vals_raw = self.data_history[log_key]

            # Find valid indices
            valid_idx = [i for i, log_val in enumerate(log_vals_raw) if log_val is not None]

            if len(valid_idx) > 1:
                valid_times = times[valid_idx]
                log_vals = np.array([log_vals_raw[i] for i in valid_idx])

                # Normalized log price (percentage change from start)
                normalized = (log_vals - log_vals[0]) * 100

                fig.add_trace(
                    go.Scatter(
                        x=valid_times,
                        y=normalized,
                        mode="lines",
                        line=dict(width=2, color="purple"),
                        fill="tozeroy",
                        fillcolor="rgba(128, 0, 128, 0.2)",
                        name=factor,
                    )
                )

        fig.update_layout(
            xaxis=dict(title="Time", gridcolor="lightgray"),
            yaxis=dict(title="Log Price Change (%)", gridcolor="lightgray"),
            plot_bgcolor="white",
            hovermode="x unified",
            height=400,
            autosize=False,
            font=dict(family=self.FONT_FAMILY, size=self.FONT_SIZE),
            margin=dict(l=50, r=20, t=30, b=50),
            showlegend=False,
            uirevision="factor_performance",  # Preserve zoom/pan state
        )

        return fig

    def _plot_leadership_vs_microstructure(self):
        """Scatter plot: leadership score vs microstructure metric."""
        fig = go.Figure()

        ticker = self.ticker_selector.value
        metric = self.microstructure_metric.value

        # Map metric to data key
        metric_map = {
            "ISO Flow": "iso_flow",
            "Total Flow": "total_flow",
            "Spread (bps)": "avg_rsprd",
            "Num Trades": "num_trades",
            "% ISO": "pct_iso",
        }

        data_key = metric_map[metric]

        if (
            ticker in self.ticker_scores
            and len(self.ticker_scores[ticker]) > 0
            and len(self.microstructure_data[ticker][data_key]) > 0
        ):
            scores_raw = self.ticker_scores[ticker]
            micro_vals_raw = self.microstructure_data[ticker][data_key]

            # Match lengths and filter None
            min_len = min(len(scores_raw), len(micro_vals_raw))
            scores_raw = scores_raw[-min_len:]
            micro_vals_raw = micro_vals_raw[-min_len:]

            # Filter out None values
            valid_pairs = [
                (s, m)
                for s, m in zip(scores_raw, micro_vals_raw)
                if s is not None and m is not None and s > 0
            ]

            if len(valid_pairs) > 0:
                scores = np.array([p[0] for p in valid_pairs])
                micro_vals = np.array([p[1] for p in valid_pairs])

                color = "blue" if ticker in ETFS else "green"

                fig.add_trace(
                    go.Scatter(
                        x=micro_vals,
                        y=scores,
                        mode="markers",
                        marker=dict(size=6, color=color, opacity=0.6),
                        name=ticker,
                    )
                )

                # Add trendline if enough points
                if len(scores) > 10:
                    z = np.polyfit(micro_vals, scores, 1)
                    p = np.poly1d(z)
                    x_line = np.linspace(micro_vals.min(), micro_vals.max(), 100)
                    y_line = p(x_line)

                    fig.add_trace(
                        go.Scatter(
                            x=x_line,
                            y=y_line,
                            mode="lines",
                            line=dict(width=2, color="red", dash="dash"),
                            name="Trend",
                        )
                    )

        fig.update_layout(
            xaxis=dict(title=metric, gridcolor="lightgray"),
            yaxis=dict(title="Leadership Score (ILS)", gridcolor="lightgray"),
            plot_bgcolor="white",
            hovermode="closest",
            height=400,
            autosize=False,
            font=dict(family=self.FONT_FAMILY, size=self.FONT_SIZE),
            margin=dict(l=50, r=20, t=30, b=50),
            legend=dict(x=0.02, y=0.98, xanchor="left", yanchor="top"),
            uirevision="leadership_vs_micro",  # Preserve zoom/pan state
        )

        return fig

    def _plot_sector_comparison(self):
        """Plot ETF vs Equity average leadership."""
        fig = go.Figure()

        times = pd.to_datetime(self.data_history["times"])
        etf_tickers = [t for t in ALL_TICKERS if t in ETFS]
        equity_tickers = [t for t in ALL_TICKERS if t in EQUITIES]

        etf_avg = []
        equity_avg = []

        for ld in self.data_history["leadership_dicts"]:
            etf_scores = [ld.get(t, 0.0) for t in etf_tickers]
            equity_scores = [ld.get(t, 0.0) for t in equity_tickers]
            etf_avg.append(np.mean(etf_scores))
            equity_avg.append(np.mean(equity_scores))

        fig.add_trace(
            go.Scatter(
                x=times,
                y=etf_avg,
                mode="lines",
                line=dict(width=2, color="blue"),
                name="ETFs (avg)",
            )
        )

        fig.add_trace(
            go.Scatter(
                x=times,
                y=equity_avg,
                mode="lines",
                line=dict(width=2, color="green"),
                name="Equities (avg)",
            )
        )

        fig.update_layout(
            xaxis=dict(title="Time", gridcolor="lightgray"),
            yaxis=dict(title="Avg Leadership Score", gridcolor="lightgray"),
            plot_bgcolor="white",
            hovermode="x unified",
            height=400,
            autosize=False,
            font=dict(family=self.FONT_FAMILY, size=self.FONT_SIZE),
            margin=dict(l=50, r=20, t=30, b=50),
            legend=dict(x=0.02, y=0.98, xanchor="left", yanchor="top"),
            uirevision="sector_comparison",  # Preserve zoom/pan state
        )

        return fig

    def _plot_top_leaders(self):
        """Plot top 5 leadership scores over time with ticker labels."""
        fig = go.Figure()

        times = pd.to_datetime(self.data_history["times"])
        n_times = len(times)

        # Get score matrix
        score_matrix = np.zeros((n_times, len(ALL_TICKERS)))
        for i, ld in enumerate(self.data_history["leadership_dicts"]):
            for j, ticker in enumerate(ALL_TICKERS):
                score_matrix[i, j] = ld.get(ticker, 0.0)

        # Get top 5 at each timestamp
        top5_indices = np.argsort(score_matrix, axis=1)[:, -5:]
        colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

        for rank in range(5):
            y_values = []
            ticker_labels = []
            for row_idx in range(n_times):
                ticker_idx = top5_indices[row_idx, -(rank + 1)]
                y_values.append(score_matrix[row_idx, ticker_idx])
                ticker_labels.append(ALL_TICKERS[ticker_idx])

            fig.add_trace(
                go.Scatter(
                    x=times,
                    y=y_values,
                    mode="lines",
                    line=dict(width=2, color=colors[rank]),
                    name=f"Top {rank + 1}",
                    text=ticker_labels,
                    hovertemplate="<b>%{text}</b><br>Score: %{y:.6f}<br>Time: %{x}<extra></extra>",
                )
            )

        fig.update_layout(
            xaxis=dict(title="Time", gridcolor="lightgray"),
            yaxis=dict(title="Leadership Score (ILS)", gridcolor="lightgray", type="log"),
            plot_bgcolor="white",
            hovermode="closest",
            height=400,
            autosize=False,
            font=dict(family=self.FONT_FAMILY, size=self.FONT_SIZE),
            margin=dict(l=50, r=20, t=30, b=50),
            legend=dict(x=0.02, y=0.98, xanchor="left", yanchor="top"),
            uirevision="top_leaders",  # Preserve zoom/pan state
        )

        return fig

    def _plot_all_tickers_scatter(self):
        """Scatter plot of leadership vs microstructure for ALL 40 tickers (WebGL accelerated)."""
        fig = go.Figure()

        metric = self.microstructure_metric.value

        # Map metric to data key
        metric_map = {
            "ISO Flow": "iso_flow",
            "Total Flow": "total_flow",
            "Spread (bps)": "avg_rsprd",
            "Num Trades": "num_trades",
            "% ISO": "pct_iso",
        }

        data_key = metric_map[metric]

        if len(self.data_history["leadership_dicts"]) > 0:
            # Get latest values for each ticker
            latest_ld = self.data_history["leadership_dicts"][-1]

            etf_scores = []
            etf_micro = []
            etf_names = []

            equity_scores = []
            equity_micro = []
            equity_names = []

            for ticker in ALL_TICKERS:
                # Get latest microstructure value
                if len(self.microstructure_data[ticker][data_key]) > 0:
                    micro_val = self.microstructure_data[ticker][data_key][-1]
                    if micro_val is not None:
                        score = latest_ld.get(ticker, 0.0)
                        if score > 0:  # Only plot valid scores
                            if ticker in ETFS:
                                etf_scores.append(score)
                                etf_micro.append(micro_val)
                                etf_names.append(ticker)
                            else:
                                equity_scores.append(score)
                                equity_micro.append(micro_val)
                                equity_names.append(ticker)

            # Plot ETFs (WebGL for fast rendering)
            if len(etf_scores) > 0:
                fig.add_trace(
                    go.Scattergl(  # Use Scattergl for WebGL acceleration
                        x=etf_micro,
                        y=etf_scores,
                        mode="markers",
                        marker=dict(
                            size=12,
                            color="blue",
                            opacity=0.8,
                            line=dict(width=1.5, color="darkblue"),
                        ),
                        text=etf_names,
                        name="ETFs",
                        hovertemplate="<b>%{text}</b><br>"
                        + metric
                        + ": %{x:.2f}<br>Leadership: %{y:.6f}<extra></extra>",
                    )
                )

            # Plot Equities (WebGL for fast rendering)
            if len(equity_scores) > 0:
                fig.add_trace(
                    go.Scattergl(  # Use Scattergl for WebGL acceleration
                        x=equity_micro,
                        y=equity_scores,
                        mode="markers",
                        marker=dict(
                            size=12,
                            color="green",
                            opacity=0.8,
                            line=dict(width=1.5, color="darkgreen"),
                        ),
                        text=equity_names,
                        name="Equities",
                        hovertemplate="<b>%{text}</b><br>"
                        + metric
                        + ": %{x:.2f}<br>Leadership: %{y:.6f}<extra></extra>",
                    )
                )

            # Add trendline for all points combined
            if len(etf_scores) + len(equity_scores) > 10:
                all_micro = np.array(etf_micro + equity_micro)
                all_scores = np.array(etf_scores + equity_scores)

                z = np.polyfit(all_micro, all_scores, 1)
                p = np.poly1d(z)
                x_line = np.linspace(all_micro.min(), all_micro.max(), 100)
                y_line = p(x_line)

                fig.add_trace(
                    go.Scatter(
                        x=x_line,
                        y=y_line,
                        mode="lines",
                        line=dict(width=2, color="red", dash="dash"),
                        name="Trend",
                        hoverinfo="skip",
                    )
                )

        fig.update_layout(
            xaxis=dict(title=metric, gridcolor="lightgray"),
            yaxis=dict(title="Leadership Score (ILS)", gridcolor="lightgray", type="log"),
            plot_bgcolor="white",
            hovermode="closest",
            height=500,
            autosize=False,
            font=dict(family=self.FONT_FAMILY, size=self.FONT_SIZE),
            margin=dict(l=50, r=20, t=30, b=50),
            legend=dict(x=0.02, y=0.98, xanchor="left", yanchor="top"),
            uirevision="all_tickers_scatter",  # Preserve zoom/pan state
        )

        return fig

    def _update_stats(self):
        """Update statistics panel."""
        n_times = len(self.data_history["times"])
        if n_times == 0 or len(self.data_history["leaders"]) == 0:
            return

        ticker = self.ticker_selector.value
        factor = self.factor_selector.value

        # Current leader
        current_leader = self.data_history["leaders"][-1]
        current_ld = self.data_history["leadership_dicts"][-1]
        current_leader_score = current_ld[current_leader]

        # Selected ticker stats
        ticker_score = current_ld.get(ticker, 0.0)
        ticker_rank = sorted(current_ld.values(), reverse=True).index(ticker_score) + 1

        # ETF vs Equity
        etf_scores = [current_ld.get(t, 0.0) for t in ETFS]
        equity_scores = [current_ld.get(t, 0.0) for t in EQUITIES]
        etf_avg = np.mean(etf_scores)
        equity_avg = np.mean(equity_scores)

        # Factor performance (filter None values)
        factor_ret_raw = self.data_history[f"{factor}_ret"]
        factor_ret_valid = [r for r in factor_ret_raw if r is not None]
        factor_ret = np.nancumsum(factor_ret_valid)[-1] * 100 if len(factor_ret_valid) > 0 else 0.0

        stats_md = f"""
### Current Status

**Current Leader:** {current_leader} (Score: {current_leader_score:.6f})

**Selected Ticker:** {ticker}
- Leadership Score: {ticker_score:.6f}
- Rank: #{ticker_rank} of 40

---

### Sector Averages

**ETFs:** {etf_avg:.6f}  
**Equities:** {equity_avg:.6f}

---

### Selected Factor: {factor}

**Cumulative Return:** {factor_ret:+.3f}%

---

**Data Points:** {n_times}  
**Time Range:** {(self.data_history["times"][-1] - self.data_history["times"][0]).total_seconds() / 60:.0f} min

**All 40 Tickers:** 20 ETFs + 20 Equities
"""

        self.stats_pane.object = stats_md

    def show(self):
        """Show the dashboard."""
        self.dashboard.show()
        return self.dashboard


# CSP nodes and graph
@csp.node
def combined_panel_updater(
    trigger: ts[object],
    leadership: ts[object],
    fama_log: ts[FamaFactors],
    fama_ret: ts[FamaReturns],
    bars_basket: {str: ts[EquityBar1m]},
    dashboard: LeadershipFamaDashboard,
):
    """Update panel dashboard with all data."""
    with csp.state():
        s_update_count = 0

    if csp.ticked(trigger):
        s_update_count += 1

        # Debug output for first few updates
        if s_update_count <= 3:
            print(f"\n🔄 CSP Updater Tick #{s_update_count} at {csp.now()}")
            print(f"   - Leadership valid: {csp.valid(leadership)}")
            print(f"   - Fama log valid: {csp.valid(fama_log)}")
            print(f"   - Fama ret valid: {csp.valid(fama_ret)}")
            print(f"   - Bars basket items: {len(list(bars_basket.validitems()))}")

        # Update every tick (1-minute intervals)
        if s_update_count % 1 == 0:
            # Collect current bar values
            bars_dict = {}
            for ticker, bar_ts in bars_basket.validitems():
                bars_dict[ticker] = bar_ts

            dashboard.update(
                t=csp.now(),
                leadership_result=leadership if csp.valid(leadership) else None,
                fama_log=fama_log if csp.valid(fama_log) else None,
                fama_ret=fama_ret if csp.valid(fama_ret) else None,
                bars_dict=bars_dict,
            )


@csp.graph
def combined_panel_graph(dashboard: LeadershipFamaDashboard):
    """Combined graph with leadership, Fama, and microstructure."""
    print("\n🔧 Building CSP graph...")

    # Fama factors
    print(f"📈 Loading Fama factors ({len(ALL_FACTOR_TICKERS)} tickers)...")
    quotes_basket_fama = get_quotes_1s_wo_size(list(ALL_FACTOR_TICKERS))
    fama = compute_fama_factors_graph(
        quotes_basket=quotes_basket_fama,
        factor_weights_path=FACTOR_WEIGHTS_PATH,
        use_efficient=True,
    )
    print("   ✓ Fama factors loaded")

    # TAQ data for all 40 tickers
    print(f"📊 Loading TAQ data (40 tickers)...")
    quotes_basket_taq = get_taq_quotes()
    trades_basket_taq = get_taq_trades()
    print("   ✓ TAQ data loaded")

    # Microstructure features
    print("📊 Computing microstructure features...")
    bars_basket = compute_microstructure_features_basket(
        trades_basket=trades_basket_taq,
        quotes_basket=quotes_basket_taq,
        bar_interval=timedelta(minutes=1),
    )
    print("   ✓ Microstructure features configured")

    # VECM timer
    vecm_timer = csp.timer(timedelta(minutes=1))
    print("   ✓ VECM timer configured")

    @csp.node
    def vecm_leadership_streaming(
        bars_basket: {str: csp.ts[object]},
        timer: csp.ts[object],
        window_size: int = 60,
        min_bars: int = 30,
    ) -> csp.ts[object]:
        """Streaming VECM leadership computation."""
        with csp.state():
            s_bar_history: list = []
            s_tickers: list = None
            s_bar_count = 0

        if csp.ticked(bars_basket):
            s_bar_count += 1
            if s_bar_count <= 3:
                print(f"   📊 Bar update #{s_bar_count} at {csp.now()}")

            current_time = csp.now()
            bars_dict = {}
            for ticker, bar_ts in bars_basket.validitems():
                bars_dict[ticker] = bar_ts

            s_bar_history.append((current_time, bars_dict))

            if s_tickers is None:
                s_tickers = sorted(bars_dict.keys())
                print(f"   ✓ Tracking {len(s_tickers)} tickers: {s_tickers[:5]}...")

            if len(s_bar_history) > window_size:
                s_bar_history = s_bar_history[-window_size:]

        if csp.ticked(timer):
            if len(s_bar_history) < min_bars:
                if len(s_bar_history) % 10 == 0:
                    print(f"   ⏳ Building history: {len(s_bar_history)}/{min_bars} bars")
                return None

            log_mid_matrix = []
            for _, bars_dict in s_bar_history:
                row = []
                for ticker in s_tickers:
                    if ticker in bars_dict:
                        row.append(bars_dict[ticker].log_mid)
                    else:
                        row.append(np.nan)
                log_mid_matrix.append(row)

            log_mid_matrix = np.array(log_mid_matrix)

            # First VECM computation
            if len(s_bar_history) == min_bars:
                print(f"   🎯 Computing first VECM with {len(s_bar_history)} bars...")

            leadership_scores = compute_vecm_leadership(log_mid_matrix, s_tickers)

            result = {
                "timestamp": csp.now(),
                "window_bars": len(s_bar_history),
                "leadership": leadership_scores,
                "leader": max(leadership_scores, key=leadership_scores.get),
            }

            return result

    leadership = vecm_leadership_streaming(
        bars_basket=bars_basket, timer=vecm_timer, window_size=60, min_bars=30
    )

    # Use timer as trigger
    trigger = csp.timer(timedelta(seconds=5))

    combined_panel_updater(
        trigger=trigger,
        leadership=leadership,
        fama_log=fama.log_prices,
        fama_ret=fama.returns,
        bars_basket=bars_basket,
        dashboard=dashboard,
    )


def run_combined_dashboard():
    """Run the combined leadership + Fama + microstructure dashboard."""
    print("=" * 80)
    print("🚀 LAUNCHING COMBINED VECM LEADERSHIP + FAMA + MICROSTRUCTURE DASHBOARD")
    print("=" * 80)
    print(f"📅 Date: {DATE_OF_INTEREST}")
    print(f"📊 All 40 Tickers: {len(ETFS)} ETFs + {len(EQUITIES)} Equities")
    print(f"📊 Fama Factor Tickers: {len(ALL_FACTOR_TICKERS)}")
    print(f"⏰ Time range: {start_ts} → {end_ts}")
    print("\n⏳ Starting CSP engine and dashboard server...")
    print("=" * 80 + "\n")

    print("📱 Initializing dashboard...")
    dashboard = LeadershipFamaDashboard()
    print("   ✓ Dashboard initialized\n")

    import threading

    def run_csp():
        print("🚀 Starting CSP engine thread...")
        print(f"   Start: {start_ts}")
        print(f"   End: {end_ts}")
        print("   This will process historical data at accelerated speed...\n")

        csp.run(
            lambda: combined_panel_graph(dashboard=dashboard),
            starttime=start_ts,
            endtime=end_ts,
            realtime=False,
        )

        print("\n✅ CSP engine completed!")

    csp_thread = threading.Thread(target=run_csp, daemon=True)
    csp_thread.start()

    print("🌐 Opening browser dashboard...")
    print("   Dashboard will update as data streams in from CSP")
    print("   Watch for '📊 Bar update' messages in console\n")

    dashboard.show()


if __name__ == "__main__":
    run_combined_dashboard()
