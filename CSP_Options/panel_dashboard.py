"""
Panel + Plotly real-time dashboard framework for CSP streaming applications.

Provides base classes and utilities for building interactive dashboards that update
in real-time as CSP processes streaming data.

Key Features:
- Flicker-free updates with fixed-height plots
- Helvetica font styling
- Data history management
- Update throttling for performance
"""

import numpy as np
import pandas as pd

try:
    import panel as pn
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from scipy.stats import lognorm

    PANEL_AVAILABLE = True
except ImportError:
    PANEL_AVAILABLE = False


class BaseDashboard:
    """
    Base class for Panel + Plotly dashboards with standard styling and utilities.

    Provides common functionality for all dashboards:
    - Fixed-height plot panes (prevents flickering)
    - Helvetica font styling
    - Standard empty plot creation
    - Data history management

    Subclass this to create custom dashboards.
    """

    FONT_FAMILY = "Helvetica, Arial, sans-serif"
    FONT_SIZE = 12

    def __init__(self):
        if not PANEL_AVAILABLE:
            raise ImportError("Panel not available. Install with: pip install panel plotly scipy")

        # Initialize Panel with Plotly
        pn.extension("plotly")

    def _create_empty_plotly(self, height=400):
        """Create empty Plotly figure with standard styling."""
        fig = go.Figure()
        fig.add_annotation(
            text="Waiting for data...",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=16, color="gray", family=self.FONT_FAMILY),
        )
        fig.update_layout(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            plot_bgcolor="white",
            height=height,
            font=dict(family=self.FONT_FAMILY, size=self.FONT_SIZE),
        )
        return fig

    def _create_plot_pane(self, height=450):
        """
        Create a Plotly pane with fixed height to prevent flickering.

        Args:
            height: Fixed height in pixels

        Returns:
            Panel Plotly pane
        """
        return pn.pane.Plotly(
            self._create_empty_plotly(height),
            sizing_mode="stretch_width",
            height=height,
            config={"responsive": False},
        )

    def _apply_standard_layout(self, fig, height=450, **kwargs):
        """
        Apply standard layout settings to a Plotly figure.

        Args:
            fig: Plotly figure
            height: Height in pixels
            **kwargs: Additional layout kwargs

        Returns:
            Modified figure
        """
        layout_defaults = dict(
            plot_bgcolor="white",
            hovermode="x unified",
            height=height,
            autosize=False,
            font=dict(family=self.FONT_FAMILY, size=self.FONT_SIZE),
        )
        layout_defaults.update(kwargs)
        fig.update_layout(**layout_defaults)
        return fig

    def _trim_history(self, data_dict, max_points=500):
        """
        Trim data history to last N points for performance.

        Args:
            data_dict: Dictionary of lists to trim
            max_points: Maximum points to keep
        """
        if data_dict and len(data_dict[next(iter(data_dict))]) > max_points:
            for key in data_dict:
                data_dict[key] = data_dict[key][-max_points:]


class RegimeDashboard(BaseDashboard):
    """
    Dashboard for regime decomposition visualization.

    Displays:
    - Probability timeline and pie chart
    - Regime-conditional densities
    - Expected returns comparison
    - Volatilities comparison
    - Return premium
    """

    def __init__(self, ticker, regime_1_label="Regime 1", regime_2_label="Regime 2"):
        super().__init__()

        self.ticker = ticker
        self.regime_1_label = regime_1_label
        self.regime_2_label = regime_2_label

        self.data_history = {
            "times": [],
            "p_regime_1": [],
            "ret_1": [],
            "ret_2": [],
            "vol_1": [],
            "vol_2": [],
            "premium": [],
            "rnd_mean": [],
        }

        # Create plot panes with fixed heights to prevent flickering
        self.density_pane = self._create_plot_pane(height=450)
        self.prob_time_pane = self._create_plot_pane(height=350)
        self.prob_pie_pane = self._create_plot_pane(height=350)
        self.returns_pane = self._create_plot_pane(height=450)
        self.vols_pane = self._create_plot_pane(height=450)
        self.premium_pane = self._create_plot_pane(height=450)

        # Stats indicators
        self.stats_pane = pn.pane.Markdown("**Waiting for data...**", sizing_mode="stretch_width")

        # Layout
        self.dashboard = pn.template.FastListTemplate(
            title=f"🎯 Real-Time Regime Decomposition: {ticker}",
            sidebar=[
                pn.pane.Markdown("## Dashboard Controls"),
                self.stats_pane,
                pn.pane.Markdown("### Legend"),
                pn.pane.Markdown(f"🔴 **{regime_1_label}**"),
                pn.pane.Markdown(f"🔵 **{regime_2_label}**"),
            ],
            main=[
                pn.Row(
                    pn.Column("## Probability Timeline", self.prob_time_pane),
                    pn.Column("## Current Probability", self.prob_pie_pane),
                ),
                pn.Row(
                    pn.Column("## Regime Densities", self.density_pane),
                    pn.Column("## Expected Returns", self.returns_pane),
                ),
                pn.Row(
                    pn.Column("## Volatilities", self.vols_pane),
                    pn.Column("## Return Premium", self.premium_pane),
                ),
            ],
            accent_base_color="#FF6B6B",
            header_background="#1a1a1a",
        )

    def update(
        self,
        t,
        p_regime_1,
        ret_1,
        ret_2,
        vol_1,
        vol_2,
        premium,
        rnd_mean,
        rnd_result=None,
        mu_1=None,
        sig_1=None,
        mu_2=None,
        sig_2=None,
    ):
        """
        Update dashboard with new data.

        Args:
            t: Timestamp
            p_regime_1: Probability of regime 1
            ret_1, ret_2: Expected returns (%)
            vol_1, vol_2: Volatilities (%)
            premium: Return premium (%)
            rnd_mean: RND mean
            rnd_result: Optional RND result object (for density plot)
            mu_1, sig_1, mu_2, sig_2: Optional regime parameters
        """
        # Append to history
        self.data_history["times"].append(t)
        self.data_history["p_regime_1"].append(p_regime_1)
        self.data_history["ret_1"].append(ret_1)
        self.data_history["ret_2"].append(ret_2)
        self.data_history["vol_1"].append(vol_1)
        self.data_history["vol_2"].append(vol_2)
        self.data_history["premium"].append(premium)
        self.data_history["rnd_mean"].append(rnd_mean)

        # Trim history for performance
        self._trim_history(self.data_history, max_points=500)

        # Update plots (only if we have enough data to avoid flickering)
        if len(self.data_history["times"]) > 2:
            self.prob_time_pane.object = self._plot_probability_timeline()
            self.prob_pie_pane.object = self._plot_probability_pie()
            self.returns_pane.object = self._plot_returns()
            self.vols_pane.object = self._plot_volatilities()
            self.premium_pane.object = self._plot_premium()

            if rnd_result and mu_1 and sig_1 and mu_2 and sig_2:
                self.density_pane.object = self._plot_densities(
                    rnd_result, p_regime_1, mu_1, sig_1, mu_2, sig_2, rnd_mean
                )

        # Update stats
        self._update_stats()

    def _plot_probability_timeline(self):
        """Plot probability timeline with area fills."""
        fig = go.Figure()

        times = pd.to_datetime(self.data_history["times"])

        # Area fills
        fig.add_trace(
            go.Scatter(
                x=times,
                y=self.data_history["p_regime_1"],
                fill="tozeroy",
                fillcolor="rgba(255, 107, 107, 0.3)",
                line=dict(color="rgba(255, 107, 107, 0)", width=0),
                showlegend=False,
                hoverinfo="skip",
            )
        )

        fig.add_trace(
            go.Scatter(
                x=times,
                y=[1] * len(times),
                fill="tonexty",
                fillcolor="rgba(78, 205, 196, 0.3)",
                line=dict(color="rgba(78, 205, 196, 0)", width=0),
                showlegend=False,
                hoverinfo="skip",
            )
        )

        # Main probability line
        fig.add_trace(
            go.Scatter(
                x=times,
                y=self.data_history["p_regime_1"],
                mode="lines",
                line=dict(color="purple", width=3),
                name=f"p({self.regime_1_label})",
            )
        )

        fig.update_layout(
            yaxis=dict(title=f"p({self.regime_1_label})", range=[0, 1], gridcolor="lightgray"),
            xaxis=dict(title="Time", gridcolor="lightgray"),
            plot_bgcolor="white",
            hovermode="x unified",
            height=350,
            margin=dict(l=50, r=20, t=30, b=50),
            autosize=False,
            font=dict(family=self.FONT_FAMILY, size=self.FONT_SIZE),
        )

        return fig

    def _plot_probability_pie(self):
        """Plot current probability as pie chart."""
        if not self.data_history["p_regime_1"]:
            return self._create_empty_plotly()

        p_1 = self.data_history["p_regime_1"][-1]
        p_2 = 1 - p_1

        fig = go.Figure(
            data=[
                go.Pie(
                    labels=[self.regime_1_label, self.regime_2_label],
                    values=[p_1, p_2],
                    marker=dict(colors=["#ff6b6b", "#4ecdc4"]),
                    textinfo="label+percent",
                    textfont=dict(size=16, color="white"),
                    hole=0.4,
                )
            ]
        )

        fig.update_layout(
            annotations=[
                dict(
                    text=f"{p_1:.1%}<br>{self.regime_1_label}",
                    x=0.5,
                    y=0.5,
                    font=dict(size=14, family=self.FONT_FAMILY),
                    showarrow=False,
                )
            ],
            showlegend=True,
            height=350,
            margin=dict(l=20, r=20, t=30, b=20),
            autosize=False,
            font=dict(family=self.FONT_FAMILY, size=self.FONT_SIZE),
        )

        return fig

    def _plot_densities(self, rnd_result, p_regime_1, mu_1, sig_1, mu_2, sig_2, obs_mean):
        """Plot regime-conditional densities."""
        fig = go.Figure()

        s_grid = rnd_result.strikes
        s_norm = s_grid / obs_mean
        f_1 = lognorm.pdf(s_norm, s=sig_1, scale=np.exp(mu_1)) / obs_mean
        f_2 = lognorm.pdf(s_norm, s=sig_2, scale=np.exp(mu_2)) / obs_mean
        f_mix = p_regime_1 * f_1 + (1 - p_regime_1) * f_2

        # Regime 1 density
        fig.add_trace(
            go.Scatter(
                x=s_grid,
                y=f_1,
                mode="lines",
                line=dict(color="red", width=3),
                name=f"{self.regime_1_label} (p={p_regime_1:.3f})",
                fill="tozeroy",
                fillcolor="rgba(255, 107, 107, 0.2)",
            )
        )

        # Regime 2 density
        fig.add_trace(
            go.Scatter(
                x=s_grid,
                y=f_2,
                mode="lines",
                line=dict(color="blue", width=3),
                name=f"{self.regime_2_label} (p={1 - p_regime_1:.3f})",
                fill="tozeroy",
                fillcolor="rgba(78, 205, 196, 0.2)",
            )
        )

        # Mixed density
        fig.add_trace(
            go.Scatter(
                x=s_grid,
                y=f_mix,
                mode="lines",
                line=dict(color="black", width=2, dash="dash"),
                name="Mixed (Observed)",
            )
        )

        # Vertical line at mean
        fig.add_vline(
            x=obs_mean,
            line_dash="dot",
            line_color="gray",
            annotation_text=f"Mean: ${obs_mean:.2f}",
            annotation_position="top",
        )

        fig.update_layout(
            xaxis=dict(title=f"{self.ticker} Price ($)", gridcolor="lightgray"),
            yaxis=dict(title="Density", gridcolor="lightgray"),
            plot_bgcolor="white",
            hovermode="x unified",
            height=450,
            margin=dict(l=50, r=20, t=30, b=50),
            autosize=False,
            font=dict(family=self.FONT_FAMILY, size=self.FONT_SIZE),
        )

        return fig

    def _plot_returns(self):
        """Plot expected returns time series."""
        fig = go.Figure()

        times = pd.to_datetime(self.data_history["times"])

        # Regime 1 returns
        fig.add_trace(
            go.Scatter(
                x=times,
                y=self.data_history["ret_1"],
                mode="lines",
                line=dict(color="red", width=2.5),
                name=self.regime_1_label,
                fill="tozeroy",
                fillcolor="rgba(255, 107, 107, 0.2)",
            )
        )

        # Regime 2 returns
        fig.add_trace(
            go.Scatter(
                x=times,
                y=self.data_history["ret_2"],
                mode="lines",
                line=dict(color="blue", width=2.5),
                name=self.regime_2_label,
                fill="tozeroy",
                fillcolor="rgba(78, 205, 196, 0.2)",
            )
        )

        # Zero line
        fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)

        fig.update_layout(
            xaxis=dict(title="Time", gridcolor="lightgray"),
            yaxis=dict(title="Expected Return (%)", gridcolor="lightgray"),
            plot_bgcolor="white",
            hovermode="x unified",
            height=450,
            margin=dict(l=50, r=20, t=30, b=50),
            autosize=False,
            font=dict(family=self.FONT_FAMILY, size=self.FONT_SIZE),
        )

        return fig

    def _plot_volatilities(self):
        """Plot volatilities on same scale (users can zoom to see differences)."""
        fig = go.Figure()

        times = pd.to_datetime(self.data_history["times"])

        # Regime 1 volatility
        fig.add_trace(
            go.Scatter(
                x=times,
                y=self.data_history["vol_1"],
                mode="lines",
                line=dict(color="red", width=2.5),
                name=f"{self.regime_1_label} Vol",
            )
        )

        # Regime 2 volatility
        fig.add_trace(
            go.Scatter(
                x=times,
                y=self.data_history["vol_2"],
                mode="lines",
                line=dict(color="blue", width=2.5),
                name=f"{self.regime_2_label} Vol",
            )
        )

        fig.update_layout(
            xaxis=dict(title="Time", gridcolor="lightgray"),
            yaxis=dict(title="Implied Volatility (%)", gridcolor="lightgray"),
            plot_bgcolor="white",
            hovermode="x unified",
            height=450,
            margin=dict(l=50, r=20, t=30, b=50),
            autosize=False,
            font=dict(family=self.FONT_FAMILY, size=self.FONT_SIZE),
        )

        return fig

    def _plot_premium(self):
        """Plot return premium."""
        fig = go.Figure()

        times = pd.to_datetime(self.data_history["times"])
        premium = np.array(self.data_history["premium"])

        # Main premium line
        fig.add_trace(
            go.Scatter(
                x=times,
                y=premium,
                mode="lines",
                line=dict(color="purple", width=2.5),
                name="Return Premium",
                fill="tozeroy",
                fillcolor="rgba(128, 0, 128, 0.2)",
            )
        )

        # Conditional fills for positive (regime 1) and negative (regime 2)
        regime_1_mask = premium > 0
        regime_2_mask = premium < 0

        if np.any(regime_1_mask):
            fig.add_trace(
                go.Scatter(
                    x=times[regime_1_mask],
                    y=premium[regime_1_mask],
                    fill="tozeroy",
                    fillcolor="rgba(255, 107, 107, 0.3)",
                    line=dict(width=0),
                    showlegend=True,
                    name=f"{self.regime_1_label} Premium",
                    hoverinfo="skip",
                )
            )

        if np.any(regime_2_mask):
            fig.add_trace(
                go.Scatter(
                    x=times[regime_2_mask],
                    y=premium[regime_2_mask],
                    fill="tozeroy",
                    fillcolor="rgba(78, 205, 196, 0.3)",
                    line=dict(width=0),
                    showlegend=True,
                    name=f"{self.regime_2_label} Premium",
                    hoverinfo="skip",
                )
            )

        # Zero line
        fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)

        fig.update_layout(
            xaxis=dict(title="Time", gridcolor="lightgray"),
            yaxis=dict(title="Return Differential (%)", gridcolor="lightgray"),
            plot_bgcolor="white",
            hovermode="x unified",
            height=450,
            margin=dict(l=50, r=20, t=30, b=50),
            autosize=False,
            font=dict(family=self.FONT_FAMILY, size=self.FONT_SIZE),
        )

        return fig

    def _update_stats(self):
        """Update statistics panel."""
        if not self.data_history["times"]:
            return

        stats_md = f"""
### Current Stats
**Time:** {pd.to_datetime(self.data_history["times"][-1]).strftime("%H:%M:%S")}

**🎲 Probabilities**
- {self.regime_1_label}: {self.data_history["p_regime_1"][-1]:.1%}
- {self.regime_2_label}: {1 - self.data_history["p_regime_1"][-1]:.1%}

**📈 Expected Returns**
- {self.regime_1_label}: {self.data_history["ret_1"][-1]:.3f}%
- {self.regime_2_label}: {self.data_history["ret_2"][-1]:.3f}%

**📊 Volatilities**
- {self.regime_1_label}: {self.data_history["vol_1"][-1]:.3f}%
- {self.regime_2_label}: {self.data_history["vol_2"][-1]:.3f}%

**💰 Premium**
- Return Diff: {self.data_history["premium"][-1]:.3f}%

---

**Data Points:** {len(self.data_history["times"])}
        """
        self.stats_pane.object = stats_md

    def show(self):
        """Show the dashboard."""
        self.dashboard.show()
        return self.dashboard
