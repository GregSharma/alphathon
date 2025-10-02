"""
Combined RND + Regime Decomposition Dashboard (Multi-Ticker with Batching)

Combines:
1. RND extraction from options
2. Regime decomposition with Kalshi

Features:
- Batch processing (10 tickers at a time)
- Dropdown to select ticker batch
- Start/Pause/Restart controls
- Per-ticker RND + regime decomposition visualization
"""

import warnings

warnings.filterwarnings("ignore", message="Found Below Intrinsic contracts")

import csp
import numpy as np
import pandas as pd
from csp import ts

# Regime decomposition logic
from scipy.optimize import minimize

from CSP_Options.nodes import bivariate_kalshi_filter
from CSP_Options.option_graphs import process_single_ticker
from CSP_Options.utils.readers import (
    DATE_OF_INTEREST,
    EQUITIES,
    ETFS,
    EXPIRATION,
    get_kalshi_trades,
    get_option_file,
    get_underlying_quotes,
)

# Panel imports
try:
    import panel as pn
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    from CSP_Options.panel_dashboard import BaseDashboard

    PANEL_AVAILABLE = True
except ImportError:
    PANEL_AVAILABLE = False

# =============================================================================
# Ticker Batching
# =============================================================================

# Create 4 batches of 10 tickers each
STOCKS_BATCH_1 = EQUITIES[:10]  # First 10 stocks
STOCKS_BATCH_2 = EQUITIES[10:20]  # Next 10 stocks
ETFS_BATCH_1 = ETFS[:10]  # First 10 ETFs
ETFS_BATCH_2 = ETFS[10:20]  # Next 10 ETFs

TICKER_BATCHES = {
    "Stocks Batch 1": STOCKS_BATCH_1,
    "Stocks Batch 2": STOCKS_BATCH_2,
    "ETFs Batch 1": ETFS_BATCH_1,
    "ETFs Batch 2": ETFS_BATCH_2,
}

# =============================================================================
# Regime Decomposition Logic
# =============================================================================


def lognorm_moment(mu, sig, k=1):
    """Compute k-th raw moment of lognormal distribution."""
    return np.exp(k * mu + 0.5 * k**2 * sig**2)


def decompose_rnd_online(p_trump, obs_mean, obs_std):
    """
    Decompose observed RND into Trump and Harris conditional densities.
    Returns: (mu_T, sig_T, mu_H, sig_H)
    """
    obs_var = obs_std**2

    def objective(x):
        mu_T, sig_T, mu_H, sig_H = x

        m1_T = lognorm_moment(mu_T, sig_T, 1)
        m1_H = lognorm_moment(mu_H, sig_H, 1)
        m2_T = lognorm_moment(mu_T, sig_T, 2)
        m2_H = lognorm_moment(mu_H, sig_H, 2)

        m1_mix = p_trump * m1_T + (1 - p_trump) * m1_H
        m2_mix = p_trump * m2_T + (1 - p_trump) * m2_H
        var_mix = m2_mix - m1_mix**2

        err_mean = (m1_mix - 1.0) ** 2
        err_var = (var_mix - (obs_var / obs_mean**2)) ** 2

        reg_drift = 0.5 * ((mu_T - 0.015) ** 2 + (mu_H - 0.005) ** 2)
        reg_vol = 0.5 * ((sig_T - 0.08) ** 2 + (sig_H - 0.12) ** 2)

        return err_mean + err_var + reg_drift + reg_vol

    x0 = [0.015, 0.08, 0.005, 0.12]
    bounds = [(-0.05, 0.10), (0.03, 0.30), (-0.05, 0.10), (0.03, 0.30)]

    try:
        res = minimize(objective, x0, bounds=bounds, method="L-BFGS-B")
        if not res.success:
            res = minimize(objective, x0, bounds=bounds, method="Powell")
        return res.x
    except Exception:
        return np.array([0.015, 0.08, 0.005, 0.12])


# =============================================================================
# CSP Node: Regime Decomposition
# =============================================================================


@csp.node
def regime_decomposer(
    trigger: ts[object],
    p_trump: ts[float],
    rnd_mean: ts[float],
    rnd_std: ts[float],
) -> csp.Outputs(
    mu_T=ts[float],
    sig_T=ts[float],
    mu_H=ts[float],
    sig_H=ts[float],
    expected_return_T=ts[float],
    expected_return_H=ts[float],
    vol_T=ts[float],
    vol_H=ts[float],
    return_premium=ts[float],
):
    """CSP node that decomposes observed RND into regime-conditional densities."""
    if csp.ticked(trigger) and csp.valid(p_trump, rnd_mean, rnd_std):
        mu_T, sig_T, mu_H, sig_H = decompose_rnd_online(p_trump, rnd_mean, rnd_std)

        csp.output(mu_T=mu_T)
        csp.output(sig_T=sig_T)
        csp.output(mu_H=mu_H)
        csp.output(sig_H=sig_H)
        csp.output(expected_return_T=100 * mu_T)
        csp.output(expected_return_H=100 * mu_H)
        csp.output(vol_T=100 * sig_T)
        csp.output(vol_H=100 * sig_H)
        csp.output(return_premium=100 * (mu_T - mu_H))


# =============================================================================
# CSP Graph: Combined Pipeline for Multiple Tickers
# =============================================================================


@csp.graph
def combined_multi_ticker_graph(tickers: list):
    """
    Combined RND + regime decomposition for multiple tickers.
    """
    # Get underlying quotes and Kalshi data once
    underlying_basket = get_underlying_quotes()
    kalshi_basket = get_kalshi_trades()
    djt_stream = kalshi_basket["PRES-2024-DJT"]
    kh_stream = kalshi_basket["PRES-2024-KH"]

    # Filter Kalshi probability once (shared across all tickers)
    filter_outputs = bivariate_kalshi_filter(
        djt_trades=djt_stream,
        kh_trades=kh_stream,
        obs_noise_std=0.005,
        process_var_base=1e-5,
        correlation_prior=-0.97,
        window_size=60,
    )

    expiration_ts = pd.Timestamp(
        f"{EXPIRATION[:4]}-{EXPIRATION[4:6]}-{EXPIRATION[6:]} 16:00:00", tz="America/New_York"
    )

    # Process each ticker
    for ticker in tickers:
        underlying_quote = underlying_basket[ticker]
        option_file = get_option_file(ticker)

        # RND extraction
        result = process_single_ticker(
            ticker=ticker,
            underlying_quote=underlying_quote,
            option_filename=option_file,
            expiration_ts=expiration_ts,
            max_dist=1,
            min_bid=0,
            grid_points=300,
        )

        # Regime decomposition
        regime_params = regime_decomposer(
            trigger=result.vec_quotes,
            p_trump=filter_outputs.b_djt,
            rnd_mean=result.rnd_mean,
            rnd_std=result.rnd_std,
        )

        # Output all streams with ticker prefix
        csp.add_graph_output(f"{ticker}_rnd_mean", result.rnd_mean)
        csp.add_graph_output(f"{ticker}_rnd_std", result.rnd_std)
        csp.add_graph_output(f"{ticker}_rnd_skew", result.rnd_skew)
        csp.add_graph_output(f"{ticker}_rnd_kurt", result.rnd_kurt)
        csp.add_graph_output(f"{ticker}_rnd_result", result.rnd_result)
        csp.add_graph_output(f"{ticker}_vec_quotes", result.vec_quotes)

        csp.add_graph_output(f"{ticker}_mu_T", regime_params.mu_T)
        csp.add_graph_output(f"{ticker}_sig_T", regime_params.sig_T)
        csp.add_graph_output(f"{ticker}_mu_H", regime_params.mu_H)
        csp.add_graph_output(f"{ticker}_sig_H", regime_params.sig_H)
        csp.add_graph_output(f"{ticker}_expected_return_T", regime_params.expected_return_T)
        csp.add_graph_output(f"{ticker}_expected_return_H", regime_params.expected_return_H)
        csp.add_graph_output(f"{ticker}_vol_T", regime_params.vol_T)
        csp.add_graph_output(f"{ticker}_vol_H", regime_params.vol_H)
        csp.add_graph_output(f"{ticker}_return_premium", regime_params.return_premium)

    # Add Kalshi outputs (shared)
    csp.add_graph_output("p_trump", filter_outputs.b_djt)
    csp.add_graph_output("p_harris", filter_outputs.b_kh)


# =============================================================================
# Panel Dashboard
# =============================================================================

if PANEL_AVAILABLE:

    class CombinedMultiTickerDashboard(BaseDashboard):
        """Combined RND + Regime Decomposition Dashboard for multiple tickers."""

        def __init__(self):
            super().__init__()
            self.current_batch = None
            self.current_ticker = None
            self.ticker_data = {}  # Store data per ticker
            self.latest_rnd_result = {}  # Store latest RND result per ticker
            self.latest_vec_quotes = {}  # Store latest vec_quotes per ticker

            # Control state
            self.is_running = False
            self.is_paused = False
            self.csp_thread = None
            self.should_stop = False

            # Batch selector
            self.batch_selector = pn.widgets.Select(
                name="Ticker Batch",
                options=list(TICKER_BATCHES.keys()),
                value="Stocks Batch 1",
                width=200,
            )
            self.batch_selector.param.watch(self._on_batch_change, "value")

            # Ticker selector (populated based on batch)
            self.ticker_selector = pn.widgets.Select(
                name="Ticker", options=STOCKS_BATCH_1, value=STOCKS_BATCH_1[0], width=200
            )

            # Control buttons
            self.start_button = pn.widgets.Button(name="▶ Start", button_type="success", width=150)
            self.pause_button = pn.widgets.Button(
                name="⏸ Pause", button_type="warning", width=150, disabled=True
            )
            self.restart_button = pn.widgets.Button(
                name="🔄 Restart", button_type="primary", width=150, disabled=True
            )
            self.status_pane = pn.pane.Markdown(
                "**Status:** Ready to start", sizing_mode="stretch_width"
            )

            # Wire up callbacks
            self.start_button.on_click(self._on_start)
            self.pause_button.on_click(self._on_pause)
            self.restart_button.on_click(self._on_restart)

            # Create plot panes - RND visualization
            self.iv_surface_pane = self._create_plot_pane(height=400)
            self.rnd_density_pane = self._create_plot_pane(height=400)
            self.rnd_cumulative_pane = self._create_plot_pane(height=400)
            self.rnd_moments_pane = self._create_plot_pane(height=350)

            # Regime decomposition plots
            self.prob_timeline_pane = self._create_plot_pane(height=300)
            self.prob_pie_pane = self._create_plot_pane(height=300)
            self.regime_densities_pane = self._create_plot_pane(height=400)
            self.regime_returns_pane = self._create_plot_pane(height=350)
            self.regime_vols_pane = self._create_plot_pane(height=350)
            self.return_premium_pane = self._create_plot_pane(height=350)

            # Stats panel
            self.stats_pane = pn.pane.Markdown(
                "**Waiting for data...**", sizing_mode="stretch_width"
            )

            # Layout
            self.dashboard = pn.template.FastListTemplate(
                title="🎯 Combined RND + Regime Decomposition (Multi-Ticker)",
                sidebar=[
                    pn.pane.Markdown("## Controls"),
                    self.batch_selector,
                    self.ticker_selector,
                    pn.Row(self.start_button, self.pause_button),
                    self.restart_button,
                    self.status_pane,
                    pn.pane.Markdown("---"),
                    pn.pane.Markdown("## Statistics"),
                    self.stats_pane,
                    pn.pane.Markdown("---"),
                    pn.pane.Markdown("### Legend"),
                    pn.pane.Markdown("🔴 **Trump Win Regime**"),
                    pn.pane.Markdown("🔵 **Harris Win Regime**"),
                    pn.pane.Markdown("---"),
                    pn.pane.Markdown("### About"),
                    pn.pane.Markdown(
                        "Processes batches of 10 tickers. "
                        "Select batch, click Start, then use dropdown to explore individual tickers."
                    ),
                ],
                main=[
                    pn.Row(
                        pn.Column("## Probability Timeline", self.prob_timeline_pane),
                        pn.Column("## Current Probability", self.prob_pie_pane),
                        sizing_mode="stretch_width",
                    ),
                    pn.Row(
                        pn.Column("## IV Surface (Market vs Fitted)", self.iv_surface_pane),
                        pn.Column("## RND Density", self.rnd_density_pane),
                        sizing_mode="stretch_width",
                    ),
                    pn.Row(
                        pn.Column("## Cumulative RND", self.rnd_cumulative_pane),
                        pn.Column("## RND Moments Timeline", self.rnd_moments_pane),
                        sizing_mode="stretch_width",
                    ),
                    pn.Row(
                        pn.Column("## Regime Densities", self.regime_densities_pane),
                        pn.Column("## Expected Returns", self.regime_returns_pane),
                        sizing_mode="stretch_width",
                    ),
                    pn.Row(
                        pn.Column("## Volatilities", self.regime_vols_pane),
                        pn.Column("## Return Premium", self.return_premium_pane),
                        sizing_mode="stretch_width",
                    ),
                ],
                accent_base_color="#FF6B6B",
                header_background="#1a1a1a",
            )

        def _on_batch_change(self, event):
            """Update ticker selector when batch changes."""
            batch_name = event.new
            tickers = TICKER_BATCHES[batch_name]
            self.ticker_selector.options = tickers
            self.ticker_selector.value = tickers[0]
            self._update_plots()

        def update(
            self,
            ticker,
            t,
            p_trump,
            rnd_mean,
            rnd_std,
            rnd_skew,
            ret_T,
            ret_H,
            vol_T,
            vol_H,
            premium,
            rnd_result=None,
            vec_quotes=None,
            mu_T=None,
            sig_T=None,
            mu_H=None,
            sig_H=None,
        ):
            """Update data for a specific ticker."""
            if ticker not in self.ticker_data:
                self.ticker_data[ticker] = {
                    "times": [],
                    "p_trump": [],
                    "rnd_means": [],
                    "rnd_stds": [],
                    "rnd_skews": [],
                    "ret_T": [],
                    "ret_H": [],
                    "vol_T": [],
                    "vol_H": [],
                    "premium": [],
                }

            data = self.ticker_data[ticker]
            data["times"].append(t)
            data["p_trump"].append(p_trump)
            data["rnd_means"].append(rnd_mean)
            data["rnd_stds"].append(rnd_std)
            data["rnd_skews"].append(rnd_skew)
            data["ret_T"].append(ret_T)
            data["ret_H"].append(ret_H)
            data["vol_T"].append(vol_T)
            data["vol_H"].append(vol_H)
            data["premium"].append(premium)

            # Store latest RND result and vec_quotes for visualization
            if rnd_result:
                self.latest_rnd_result[ticker] = rnd_result
            if vec_quotes:
                self.latest_vec_quotes[ticker] = vec_quotes

            # Trim history
            self._trim_history(data, max_points=500)

            # Update plots if this is the currently selected ticker
            if ticker == self.ticker_selector.value:
                self._update_plots(ticker, mu_T, sig_T, mu_H, sig_H)

        def _update_plots(self, ticker=None, mu_T=None, sig_T=None, mu_H=None, sig_H=None):
            """Update all plots for the currently selected ticker."""
            if ticker is None:
                ticker = self.ticker_selector.value
            if ticker not in self.ticker_data or len(self.ticker_data[ticker]["times"]) < 2:
                return

            data = self.ticker_data[ticker]

            # Always update timeline plots
            self.prob_timeline_pane.object = self._plot_prob_timeline(data)
            self.prob_pie_pane.object = self._plot_prob_pie(data)
            self.rnd_moments_pane.object = self._plot_rnd_moments(data)
            self.regime_returns_pane.object = self._plot_regime_returns(data)
            self.regime_vols_pane.object = self._plot_regime_vols(data)
            self.return_premium_pane.object = self._plot_return_premium(data)

            # Update RND plots if we have the latest data
            if ticker in self.latest_rnd_result and ticker in self.latest_vec_quotes:
                rnd_result = self.latest_rnd_result[ticker]
                vec_quotes = self.latest_vec_quotes[ticker]
                self.iv_surface_pane.object = self._plot_iv_surface(rnd_result, vec_quotes)
                self.rnd_density_pane.object = self._plot_rnd_density(rnd_result)
                self.rnd_cumulative_pane.object = self._plot_rnd_cumulative(rnd_result)

                # Update regime densities if we have regime params
                if (
                    mu_T is not None
                    and sig_T is not None
                    and mu_H is not None
                    and sig_H is not None
                ):
                    p_trump = data["p_trump"][-1]
                    rnd_mean = data["rnd_means"][-1]
                    self.regime_densities_pane.object = self._plot_regime_densities(
                        rnd_result, p_trump, mu_T, sig_T, mu_H, sig_H, rnd_mean
                    )

            self._update_stats(ticker, data)

        def _plot_prob_timeline(self, data):
            """Plot Kalshi probability timeline."""
            fig = go.Figure()
            times = pd.to_datetime(data["times"])
            p_trump = np.array(data["p_trump"])

            fig.add_trace(
                go.Scatter(
                    x=times,
                    y=p_trump,
                    mode="lines",
                    line=dict(color="purple", width=2.5),
                    name="p(Trump)",
                    fill="tozeroy",
                    fillcolor="rgba(128, 0, 128, 0.2)",
                )
            )

            fig.update_layout(
                xaxis=dict(title="Time", gridcolor="lightgray"),
                yaxis=dict(title="p(Trump)", range=[0, 1], gridcolor="lightgray"),
                plot_bgcolor="white",
                hovermode="x unified",
                height=300,
                autosize=False,
                font=dict(family=self.FONT_FAMILY, size=self.FONT_SIZE),
                margin=dict(l=50, r=20, t=30, b=50),
                uirevision="prob_timeline",  # Preserve zoom/pan
            )

            return fig

        def _plot_prob_pie(self, data):
            """Plot current probability as pie chart."""
            if not data["p_trump"]:
                return self._create_empty_plotly()

            p_t = data["p_trump"][-1]
            p_h = 1 - p_t

            fig = go.Figure(
                data=[
                    go.Pie(
                        labels=["Trump", "Harris"],
                        values=[p_t, p_h],
                        marker=dict(colors=["#ff6b6b", "#4ecdc4"]),
                        textinfo="label+percent",
                        textfont=dict(size=14, color="white"),
                        hole=0.4,
                    )
                ]
            )

            fig.update_layout(
                annotations=[
                    dict(
                        text=f"{p_t:.1%}<br>Trump",
                        x=0.5,
                        y=0.5,
                        font_size=12,
                        showarrow=False,
                    )
                ],
                showlegend=True,
                height=300,
                margin=dict(l=20, r=20, t=30, b=20),
                autosize=False,
                uirevision="prob_pie",  # Preserve zoom/pan
            )

            return fig

        def _plot_iv_surface(self, rnd_result, vec_quotes):
            """Plot IV surface (market vs fitted)."""
            fig = go.Figure()

            call_mask = vec_quotes.right == "c"
            put_mask = vec_quotes.right == "p"

            # Market IVs
            fig.add_trace(
                go.Scatter(
                    x=vec_quotes.strike[call_mask],
                    y=vec_quotes.iv[call_mask],
                    mode="markers",
                    marker=dict(color="purple", size=6, line=dict(width=0.5, color="black")),
                    name="Calls",
                )
            )

            fig.add_trace(
                go.Scatter(
                    x=vec_quotes.strike[put_mask],
                    y=vec_quotes.iv[put_mask],
                    mode="markers",
                    marker=dict(color="blue", size=6, line=dict(width=0.5, color="black")),
                    name="Puts",
                )
            )

            # Fitted IV
            fig.add_trace(
                go.Scatter(
                    x=rnd_result.strikes,
                    y=rnd_result.fitted_iv,
                    mode="lines",
                    line=dict(color="red", width=2.5),
                    name="GP Fit",
                )
            )

            # Confidence interval
            upper_band = rnd_result.fitted_iv + 1.96 * rnd_result.fitted_iv_std
            lower_band = rnd_result.fitted_iv - 1.96 * rnd_result.fitted_iv_std

            fig.add_trace(
                go.Scatter(
                    x=rnd_result.strikes,
                    y=upper_band,
                    mode="lines",
                    line=dict(width=0),
                    showlegend=False,
                    hoverinfo="skip",
                )
            )

            fig.add_trace(
                go.Scatter(
                    x=rnd_result.strikes,
                    y=lower_band,
                    mode="lines",
                    fill="tonexty",
                    fillcolor="rgba(255, 0, 0, 0.12)",
                    line=dict(width=0),
                    name="95% CI",
                )
            )

            fig.add_vline(
                x=rnd_result.forward_price,
                line_dash="dash",
                line_color="black",
                annotation_text=f"F=${rnd_result.forward_price:.0f}",
                annotation_position="top",
            )

            fig.update_layout(
                xaxis=dict(title="Strike ($)", gridcolor="lightgray"),
                yaxis=dict(title="Implied Volatility", gridcolor="lightgray"),
                plot_bgcolor="white",
                hovermode="x unified",
                height=400,
                autosize=False,
                font=dict(family=self.FONT_FAMILY, size=self.FONT_SIZE),
                margin=dict(l=50, r=20, t=30, b=50),
                uirevision="iv_surface",  # Preserve zoom/pan
            )

            return fig

        def _plot_rnd_density(self, rnd_result):
            """Plot RND density as bar chart."""
            fig = go.Figure()

            bar_width = rnd_result.log_moneyness[1] - rnd_result.log_moneyness[0]
            integral = np.trapz(rnd_result.rnd_density, rnd_result.log_moneyness)

            fig.add_trace(
                go.Bar(
                    x=rnd_result.log_moneyness,
                    y=rnd_result.rnd_density,
                    width=bar_width,
                    marker=dict(color="steelblue", line=dict(color="navy", width=0.5)),
                    name="RND",
                    showlegend=False,
                )
            )

            fig.add_vline(
                x=0,
                line_dash="dash",
                line_color="black",
                annotation_text="ATM",
                annotation_position="top",
            )

            fig.update_layout(
                xaxis=dict(title="Log-Moneyness", gridcolor="lightgray"),
                yaxis=dict(title="Density", gridcolor="lightgray"),
                plot_bgcolor="white",
                title=dict(
                    text=f"Integral={integral:.4f}", x=0.5, xanchor="center", font=dict(size=10)
                ),
                height=400,
                autosize=False,
                font=dict(family=self.FONT_FAMILY, size=self.FONT_SIZE),
                margin=dict(l=50, r=20, t=40, b=50),
                uirevision="rnd_density",  # Preserve zoom/pan
            )

            return fig

        def _plot_rnd_cumulative(self, rnd_result):
            """Plot cumulative RND."""
            fig = go.Figure()

            fig.add_trace(
                go.Scatter(
                    x=rnd_result.strikes,
                    y=rnd_result.rnd_cumulative,
                    mode="lines",
                    line=dict(color="green", width=2.5),
                    name="CDF",
                    showlegend=False,
                )
            )

            fig.add_vline(
                x=rnd_result.forward_price,
                line_dash="dash",
                line_color="black",
                annotation_text=f"F=${rnd_result.forward_price:.0f}",
                annotation_position="top",
            )

            fig.add_hline(y=0.5, line_dash="dot", line_color="gray", opacity=0.5)

            cdf_max = rnd_result.rnd_cumulative[-1]

            fig.update_layout(
                xaxis=dict(title="Strike ($)", gridcolor="lightgray"),
                yaxis=dict(title="Cumulative Probability", gridcolor="lightgray"),
                plot_bgcolor="white",
                title=dict(text=f"Max={cdf_max:.4f}", x=0.5, xanchor="center", font=dict(size=10)),
                hovermode="x unified",
                height=400,
                autosize=False,
                font=dict(family=self.FONT_FAMILY, size=self.FONT_SIZE),
                margin=dict(l=50, r=20, t=40, b=50),
                uirevision="rnd_cumulative",  # Preserve zoom/pan
            )

            return fig

        def _plot_regime_densities(self, rnd_result, p_trump, mu_T, sig_T, mu_H, sig_H, obs_mean):
            """Plot regime-conditional densities."""
            from scipy.stats import lognorm

            fig = go.Figure()

            # Create a wider price grid that covers the full range of both regimes
            # Use a range that extends well beyond the strike range to capture full densities
            min_strike = rnd_result.strikes.min()
            max_strike = rnd_result.strikes.max()
            strike_range = max_strike - min_strike

            # Extend the range by 50% on each side to ensure we capture the full densities
            s_min = max(0.1 * obs_mean, min_strike - 0.5 * strike_range)
            s_max = max_strike + 0.5 * strike_range
            s_grid = np.linspace(s_min, s_max, 500)

            s_norm = s_grid / obs_mean
            f_T = lognorm.pdf(s_norm, s=sig_T, scale=np.exp(mu_T)) / obs_mean
            f_H = lognorm.pdf(s_norm, s=sig_H, scale=np.exp(mu_H)) / obs_mean
            f_mix = p_trump * f_T + (1 - p_trump) * f_H

            fig.add_trace(
                go.Scatter(
                    x=s_grid,
                    y=f_T,
                    mode="lines",
                    line=dict(color="red", width=2.5),
                    name=f"Trump (p={p_trump:.2f})",
                    fill="tozeroy",
                    fillcolor="rgba(255, 107, 107, 0.15)",
                )
            )

            fig.add_trace(
                go.Scatter(
                    x=s_grid,
                    y=f_H,
                    mode="lines",
                    line=dict(color="blue", width=2.5),
                    name=f"Harris (p={1 - p_trump:.2f})",
                    fill="tozeroy",
                    fillcolor="rgba(78, 205, 196, 0.15)",
                )
            )

            fig.add_trace(
                go.Scatter(
                    x=s_grid,
                    y=f_mix,
                    mode="lines",
                    line=dict(color="black", width=2, dash="dash"),
                    name="Mixed",
                )
            )

            fig.add_vline(
                x=obs_mean,
                line_dash="dot",
                line_color="gray",
                annotation_text=f"${obs_mean:.0f}",
                annotation_position="top",
            )

            fig.update_layout(
                xaxis=dict(title=f"Price ($)", gridcolor="lightgray"),
                yaxis=dict(title="Density", gridcolor="lightgray"),
                plot_bgcolor="white",
                hovermode="x unified",
                height=400,
                autosize=False,
                font=dict(family=self.FONT_FAMILY, size=self.FONT_SIZE),
                margin=dict(l=50, r=20, t=30, b=50),
                uirevision="regime_densities",  # Preserve zoom/pan
            )

            return fig

        def _plot_rnd_moments(self, data):
            """Plot RND moments over time."""
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            times = pd.to_datetime(data["times"])

            fig.add_trace(
                go.Scatter(
                    x=times,
                    y=data["rnd_means"],
                    mode="lines",
                    line=dict(color="blue", width=2),
                    name="Mean",
                ),
                secondary_y=False,
            )

            fig.add_trace(
                go.Scatter(
                    x=times,
                    y=data["rnd_stds"],
                    mode="lines",
                    line=dict(color="red", width=2),
                    name="Std Dev",
                ),
                secondary_y=True,
            )

            fig.update_xaxes(title_text="Time", gridcolor="lightgray")
            fig.update_yaxes(title_text="Mean ($)", secondary_y=False, gridcolor="lightgray")
            fig.update_yaxes(title_text="Std Dev ($)", secondary_y=True, gridcolor="lightgray")

            fig.update_layout(
                plot_bgcolor="white",
                hovermode="x unified",
                height=350,
                autosize=False,
                font=dict(family=self.FONT_FAMILY, size=self.FONT_SIZE),
                margin=dict(l=50, r=50, t=30, b=50),
                uirevision="rnd_moments",  # Preserve zoom/pan
            )

            return fig

        def _plot_regime_returns(self, data):
            """Plot regime expected returns."""
            fig = go.Figure()
            times = pd.to_datetime(data["times"])

            fig.add_trace(
                go.Scatter(
                    x=times,
                    y=data["ret_T"],
                    mode="lines",
                    line=dict(color="red", width=2),
                    name="Trump",
                    fill="tozeroy",
                    fillcolor="rgba(255, 107, 107, 0.2)",
                )
            )

            fig.add_trace(
                go.Scatter(
                    x=times,
                    y=data["ret_H"],
                    mode="lines",
                    line=dict(color="blue", width=2),
                    name="Harris",
                    fill="tozeroy",
                    fillcolor="rgba(78, 205, 196, 0.2)",
                )
            )

            fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)

            fig.update_layout(
                xaxis=dict(title="Time", gridcolor="lightgray"),
                yaxis=dict(title="Expected Return (%)", gridcolor="lightgray"),
                plot_bgcolor="white",
                hovermode="x unified",
                height=350,
                autosize=False,
                font=dict(family=self.FONT_FAMILY, size=self.FONT_SIZE),
                margin=dict(l=50, r=20, t=30, b=50),
                uirevision="regime_returns",  # Preserve zoom/pan
            )

            return fig

        def _plot_regime_vols(self, data):
            """Plot regime volatilities."""
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            times = pd.to_datetime(data["times"])

            fig.add_trace(
                go.Scatter(
                    x=times,
                    y=data["vol_T"],
                    mode="lines",
                    line=dict(color="red", width=2),
                    name="Trump Vol",
                ),
                secondary_y=False,
            )

            fig.add_trace(
                go.Scatter(
                    x=times,
                    y=data["vol_H"],
                    mode="lines",
                    line=dict(color="blue", width=2),
                    name="Harris Vol",
                ),
                secondary_y=True,
            )

            fig.update_xaxes(title_text="Time", gridcolor="lightgray")
            fig.update_yaxes(
                title_text="Trump Vol (%)",
                title_font=dict(color="red"),
                tickfont=dict(color="red"),
                secondary_y=False,
                gridcolor="lightgray",
            )
            fig.update_yaxes(
                title_text="Harris Vol (%)",
                title_font=dict(color="blue"),
                tickfont=dict(color="blue"),
                secondary_y=True,
            )

            fig.update_layout(
                plot_bgcolor="white",
                hovermode="x unified",
                height=350,
                autosize=False,
                font=dict(family=self.FONT_FAMILY, size=self.FONT_SIZE),
                margin=dict(l=50, r=50, t=30, b=50),
                uirevision="regime_vols",  # Preserve zoom/pan
            )

            return fig

        def _plot_return_premium(self, data):
            """Plot return premium."""
            fig = go.Figure()
            times = pd.to_datetime(data["times"])
            premium = np.array(data["premium"])

            fig.add_trace(
                go.Scatter(
                    x=times,
                    y=premium,
                    mode="lines",
                    line=dict(color="purple", width=2),
                    name="Return Premium",
                    fill="tozeroy",
                    fillcolor="rgba(128, 0, 128, 0.2)",
                )
            )

            fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)

            fig.update_layout(
                xaxis=dict(title="Time", gridcolor="lightgray"),
                yaxis=dict(title="Return Differential (%)", gridcolor="lightgray"),
                plot_bgcolor="white",
                hovermode="x unified",
                height=350,
                autosize=False,
                font=dict(family=self.FONT_FAMILY, size=self.FONT_SIZE),
                margin=dict(l=50, r=20, t=30, b=50),
                uirevision="return_premium",  # Preserve zoom/pan
            )

            return fig

        def _update_stats(self, ticker, data):
            """Update statistics panel."""
            if not data["times"]:
                return

            stats_md = f"""
### Current Ticker: {ticker}

**Time:** {pd.to_datetime(data["times"][-1]).strftime("%H:%M:%S")}

**🎲 Kalshi Probability**
- p(Trump): {data["p_trump"][-1]:.1%}

**📊 RND Statistics**
- Mean: ${data["rnd_means"][-1]:.2f}
- Std Dev: ${data["rnd_stds"][-1]:.2f}

**📈 Regime Expected Returns**
- Trump: {data["ret_T"][-1]:.3f}%
- Harris: {data["ret_H"][-1]:.3f}%

**📊 Regime Volatilities**
- Trump: {data["vol_T"][-1]:.3f}%
- Harris: {data["vol_H"][-1]:.3f}%

**💰 Return Premium**
- Differential: {data["premium"][-1]:.3f}%

---

**Tickers Loaded:** {len(self.ticker_data)}
**Data Points (this ticker):** {len(data["times"])}
            """
            self.stats_pane.object = stats_md

        def _on_start(self, _event):
            """Handle start button click."""
            if not self.is_running:
                batch_name = self.batch_selector.value
                tickers = TICKER_BATCHES[batch_name]

                self.status_pane.object = f"**Status:** ▶ Running {batch_name}..."
                self.start_button.disabled = True
                self.pause_button.disabled = False
                self.restart_button.disabled = False
                self.batch_selector.disabled = True
                self.is_running = True
                self.is_paused = False
                self.should_stop = False

                # Start CSP in background thread
                import threading

                start_ts = pd.Timestamp(f"{DATE_OF_INTEREST} 09:31:00", tz="America/New_York")
                end_ts = pd.Timestamp(f"{DATE_OF_INTEREST} 16:00:00", tz="America/New_York")

                def run_csp():
                    try:
                        csp.run(
                            lambda: combined_panel_graph(tickers=tickers, dashboard=self),
                            starttime=start_ts,
                            endtime=end_ts,
                            realtime=False,
                        )
                        if not self.should_stop:
                            self.status_pane.object = "**Status:** ✅ Completed"
                            self.is_running = False
                            self.batch_selector.disabled = False
                    except Exception as e:
                        self.status_pane.object = f"**Status:** ❌ Error: {str(e)}"
                        self.is_running = False
                        self.batch_selector.disabled = False

                self.csp_thread = threading.Thread(target=run_csp, daemon=True)
                self.csp_thread.start()

        def _on_pause(self, _event):
            """Handle pause/resume button click."""
            if self.is_paused:
                self.is_paused = False
                self.pause_button.name = "⏸ Pause"
                self.status_pane.object = "**Status:** ▶ Running..."
            else:
                self.is_paused = True
                self.pause_button.name = "▶ Resume"
                self.status_pane.object = "**Status:** ⏸ Paused"

        def _on_restart(self, _event):
            """Handle restart button click."""
            self.should_stop = True
            self.is_running = False

            # Clear data
            self.ticker_data = {}

            # Reset UI
            self.prob_timeline_pane.object = self._create_empty_plotly()
            self.prob_pie_pane.object = self._create_empty_plotly()
            self.iv_surface_pane.object = self._create_empty_plotly()
            self.rnd_density_pane.object = self._create_empty_plotly()
            self.rnd_cumulative_pane.object = self._create_empty_plotly()
            self.rnd_moments_pane.object = self._create_empty_plotly()
            self.regime_densities_pane.object = self._create_empty_plotly()
            self.regime_returns_pane.object = self._create_empty_plotly()
            self.regime_vols_pane.object = self._create_empty_plotly()
            self.return_premium_pane.object = self._create_empty_plotly()
            self.stats_pane.object = "**Waiting for data...**"

            # Re-enable controls
            self.start_button.disabled = False
            self.pause_button.disabled = True
            self.pause_button.name = "⏸ Pause"
            self.restart_button.disabled = True
            self.batch_selector.disabled = False
            self.is_paused = False

            self.status_pane.object = "**Status:** 🔄 Ready to start"

        def show(self):
            """Show the dashboard."""
            self.dashboard.show()
            return self.dashboard

    # CSP node for panel updates
    @csp.node
    def combined_panel_updater(
        trigger: ts[object],
        p_trump: ts[float],
        rnd_mean: ts[float],
        rnd_std: ts[float],
        rnd_skew: ts[float],
        ret_T: ts[float],
        ret_H: ts[float],
        vol_T: ts[float],
        vol_H: ts[float],
        premium: ts[float],
        rnd_result: ts[object],
        vec_quotes: ts[object],
        mu_T: ts[float],
        sig_T: ts[float],
        mu_H: ts[float],
        sig_H: ts[float],
        ticker: str,  # Non-ts params must come after ts params
        dashboard: object,  # CombinedMultiTickerDashboard
    ):
        """Update panel dashboard with combined data (every minute)."""
        # Trigger fires every minute from the RND calculation timer
        # So we update on every tick (which is already 1-minute intervals)
        if csp.ticked(trigger) and csp.valid(
            p_trump, rnd_mean, rnd_std, rnd_skew, ret_T, ret_H, vol_T, vol_H, premium
        ):
            # Check if should stop
            if dashboard.should_stop:
                return

            # Skip updates if paused
            if dashboard.is_paused:
                return

            # Update on every tick (which is already 1-minute from the sampling timer)
            dashboard.update(
                ticker=ticker,
                t=csp.now(),
                p_trump=p_trump,
                rnd_mean=rnd_mean,
                rnd_std=rnd_std,
                rnd_skew=rnd_skew,
                ret_T=ret_T,
                ret_H=ret_H,
                vol_T=vol_T,
                vol_H=vol_H,
                premium=premium,
                rnd_result=rnd_result if csp.valid(rnd_result) else None,
                vec_quotes=vec_quotes if csp.valid(vec_quotes) else None,
                mu_T=mu_T if csp.valid(mu_T) else None,
                sig_T=sig_T if csp.valid(sig_T) else None,
                mu_H=mu_H if csp.valid(mu_H) else None,
                sig_H=sig_H if csp.valid(sig_H) else None,
            )

    @csp.graph
    def combined_panel_graph(tickers: list, dashboard: CombinedMultiTickerDashboard):
        """Combined graph with panel updates for multiple tickers."""
        # Get underlying quotes and Kalshi data
        underlying_basket = get_underlying_quotes()
        kalshi_basket = get_kalshi_trades()
        djt_stream = kalshi_basket["PRES-2024-DJT"]
        kh_stream = kalshi_basket["PRES-2024-KH"]

        # Filter Kalshi probability
        filter_outputs = bivariate_kalshi_filter(
            djt_trades=djt_stream,
            kh_trades=kh_stream,
            obs_noise_std=0.005,
            process_var_base=1e-5,
            correlation_prior=-0.97,
            window_size=60,
        )

        expiration_ts = pd.Timestamp(
            f"{EXPIRATION[:4]}-{EXPIRATION[4:6]}-{EXPIRATION[6:]} 16:00:00", tz="America/New_York"
        )

        # Process each ticker
        for ticker in tickers:
            underlying_quote = underlying_basket[ticker]
            option_file = get_option_file(ticker)

            # RND extraction
            result = process_single_ticker(
                ticker=ticker,
                underlying_quote=underlying_quote,
                option_filename=option_file,
                expiration_ts=expiration_ts,
                max_dist=1,
                min_bid=0,
                grid_points=300,
            )

            # Regime decomposition
            regime_params = regime_decomposer(
                trigger=result.vec_quotes,
                p_trump=filter_outputs.b_djt,
                rnd_mean=result.rnd_mean,
                rnd_std=result.rnd_std,
            )

            # Update dashboard for this ticker
            combined_panel_updater(
                trigger=result.vec_quotes,
                p_trump=filter_outputs.b_djt,
                rnd_mean=result.rnd_mean,
                rnd_std=result.rnd_std,
                rnd_skew=result.rnd_skew,
                ret_T=regime_params.expected_return_T,
                ret_H=regime_params.expected_return_H,
                vol_T=regime_params.vol_T,
                vol_H=regime_params.vol_H,
                premium=regime_params.return_premium,
                rnd_result=result.rnd_result,
                vec_quotes=result.vec_quotes,
                mu_T=regime_params.mu_T,
                sig_T=regime_params.sig_T,
                mu_H=regime_params.mu_H,
                sig_H=regime_params.sig_H,
                ticker=ticker,  # Non-ts params come after ts params
                dashboard=dashboard,
            )

    def run_combined_panel():
        """Run the combined multi-ticker dashboard."""
        dashboard = CombinedMultiTickerDashboard()
        dashboard.show()

else:

    def run_combined_panel():
        print("❌ Panel not available. Install with: pip install panel plotly")
        return


if __name__ == "__main__":
    run_combined_panel()
