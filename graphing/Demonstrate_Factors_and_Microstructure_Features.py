"""
Real-Time Fama-French + Microstructure Panel Dashboard

Standalone Panel dashboard for visualizing Fama-French factors and microstructure features in real-time.
Separated from test_fama_full_analysis.py for cleaner code organization.

Usage:
    python graphing/test_fama_panel.py
"""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

import csp
from csp import ts
from CSP_Options.fama import compute_fama_factors_graph
from CSP_Options.microstructure import compute_microstructure_features_basket
from CSP_Options.structs import EquityBar1m, FamaFactors, FamaReturns
from CSP_Options.utils.readers import get_quotes_1s_wo_size, get_taq_quotes, get_taq_trades

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


class FamaFullAnalysisDashboard(BaseDashboard):
    """Real-time dashboard for Fama factors + microstructure + VECM leadership."""

    def __init__(self):
        super().__init__()
        self.data_history = {
            "times": [],
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
            # ETF log mids
            "SPY_log": [],
            "QQQ_log": [],
            "IWM_log": [],
            # Equity log mids
            "AAPL_log": [],
            "MSFT_log": [],
            "NVDA_log": [],
            "TSLA_log": [],
            "AMZN_log": [],
            # Microstructure (SPY)
            "SPY_iso_flow": [],
            "SPY_rsprd": [],
            "SPY_total_flow": [],
            "SPY_num_trades": [],
            "SPY_pct_iso": [],
        }

        # Create plot panes
        self.fama_log_pane = self._create_plot_pane(height=400)
        self.fama_ret_pane = self._create_plot_pane(height=400)
        self.etf_prices_pane = self._create_plot_pane(height=400)
        self.equity_prices_pane = self._create_plot_pane(height=400)
        self.spy_iso_flow_pane = self._create_plot_pane(height=350)
        self.spy_spread_pane = self._create_plot_pane(height=350)
        self.spy_flow_pane = self._create_plot_pane(height=350)
        self.spy_trades_pane = self._create_plot_pane(height=350)

        # Stats panel
        self.stats_pane = pn.pane.Markdown("**Waiting for data...**", sizing_mode="stretch_width")

        # Layout
        self.dashboard = pn.template.FastListTemplate(
            title="📊 Real-Time Fama-French + Microstructure Analysis",
            sidebar=[
                pn.pane.Markdown("## Analysis Statistics"),
                self.stats_pane,
                pn.pane.Markdown("### About"),
                pn.pane.Markdown(
                    "**Fama-French 5-Factor Model**\n"
                    "- HML: High Minus Low (value)\n"
                    "- SMB: Small Minus Big (size)\n"
                    "- RMW: Robust Minus Weak (profitability)\n"
                    "- CMA: Conservative Minus Aggressive (investment)\n"
                    "- MKT-RF: Market excess return\n\n"
                    "**Microstructure Features**\n"
                    "- ISO Flow: Intermarket sweep orders\n"
                    "- Relative Spread: Bid-ask spread/price\n"
                    "- Trade Flow: Buy/sell imbalance"
                ),
            ],
            main=[
                pn.Row(
                    pn.Column("## Fama Factor Log Prices", self.fama_log_pane),
                    pn.Column("## Fama Factor Returns (Cumulative)", self.fama_ret_pane),
                    sizing_mode="stretch_width",
                ),
                pn.Row(
                    pn.Column("## ETF Prices (% Change)", self.etf_prices_pane),
                    pn.Column("## Top Equity Prices (% Change)", self.equity_prices_pane),
                    sizing_mode="stretch_width",
                ),
                pn.Row(
                    pn.Column("## SPY: ISO Flow Intensity", self.spy_iso_flow_pane),
                    pn.Column("## SPY: Relative Spread (bps)", self.spy_spread_pane),
                    sizing_mode="stretch_width",
                ),
                pn.Row(
                    pn.Column("## SPY: Total Flow", self.spy_flow_pane),
                    pn.Column("## SPY: Number of Trades", self.spy_trades_pane),
                    sizing_mode="stretch_width",
                ),
            ],
            accent_base_color="#4169E1",
            header_background="#1a1a1a",
        )

    def update(self, t, fama_log, fama_ret, bars_dict):
        """Update dashboard with new data."""
        self.data_history["times"].append(t)

        # Fama log prices
        if fama_log:
            self.data_history["HML_log"].append(fama_log.HML)
            self.data_history["SMB_log"].append(fama_log.SMB)
            self.data_history["RMW_log"].append(fama_log.RMW)
            self.data_history["CMA_log"].append(fama_log.CMA)
            self.data_history["MKT_RF_log"].append(fama_log.MKT_RF)

        # Fama returns
        if fama_ret:
            self.data_history["HML_ret"].append(fama_ret.HML)
            self.data_history["SMB_ret"].append(fama_ret.SMB)
            self.data_history["RMW_ret"].append(fama_ret.RMW)
            self.data_history["CMA_ret"].append(fama_ret.CMA)
            self.data_history["MKT_RF_ret"].append(fama_ret.MKT_RF)

        # ETF and equity bars
        for ticker in ["SPY", "QQQ", "IWM", "AAPL", "MSFT", "NVDA", "TSLA", "AMZN"]:
            if ticker in bars_dict:
                bar = bars_dict[ticker]
                self.data_history[f"{ticker}_log"].append(bar.log_mid)

        # SPY microstructure
        if "SPY" in bars_dict:
            spy_bar = bars_dict["SPY"]
            self.data_history["SPY_iso_flow"].append(spy_bar.iso_flow_intensity)
            self.data_history["SPY_rsprd"].append(spy_bar.avg_rsprd * 10000)  # bps
            self.data_history["SPY_total_flow"].append(spy_bar.total_flow)
            self.data_history["SPY_num_trades"].append(spy_bar.num_trades)
            self.data_history["SPY_pct_iso"].append(spy_bar.pct_trades_iso * 100)  # %

        # Trim history
        self._trim_history(self.data_history, max_points=500)

        # Update plots
        if len(self.data_history["times"]) > 2:
            self.fama_log_pane.object = self._plot_fama_log_prices()
            self.fama_ret_pane.object = self._plot_fama_returns()
            self.etf_prices_pane.object = self._plot_etf_prices()
            self.equity_prices_pane.object = self._plot_equity_prices()
            self.spy_iso_flow_pane.object = self._plot_spy_iso_flow()
            self.spy_spread_pane.object = self._plot_spy_spread()
            self.spy_flow_pane.object = self._plot_spy_flow()
            self.spy_trades_pane.object = self._plot_spy_trades()

        # Update stats
        self._update_stats()

    def _plot_fama_log_prices(self):
        """Plot Fama factor log prices (normalized)."""
        fig = go.Figure()
        times = pd.to_datetime(self.data_history["times"])

        factors = ["HML", "SMB", "RMW", "CMA", "MKT_RF"]
        colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

        for i, factor in enumerate(factors):
            key = f"{factor}_log"
            if len(self.data_history[key]) > 0:
                values = np.array(self.data_history[key])
                normalized = values - values[0]  # Normalize to start
                fig.add_trace(
                    go.Scatter(
                        x=times,
                        y=normalized,
                        mode="lines",
                        line=dict(width=2, color=colors[i]),
                        name=factor,
                    )
                )

        fig.update_layout(
            xaxis=dict(title="Time", gridcolor="lightgray"),
            yaxis=dict(title="Log Price Change (Normalized)", gridcolor="lightgray"),
            plot_bgcolor="white",
            hovermode="x unified",
            height=400,
            autosize=False,
            font=dict(family=self.FONT_FAMILY, size=self.FONT_SIZE),
            margin=dict(l=50, r=20, t=30, b=50),
            legend=dict(x=0.02, y=0.98, xanchor="left", yanchor="top"),
        )

        return fig

    def _plot_fama_returns(self):
        """Plot Fama factor cumulative returns."""
        fig = go.Figure()
        times = pd.to_datetime(self.data_history["times"])

        factors = ["HML", "SMB", "RMW", "CMA", "MKT_RF"]
        colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

        for i, factor in enumerate(factors):
            key = f"{factor}_ret"
            if len(self.data_history[key]) > 0:
                values = np.array(self.data_history[key])
                cumsum = np.nancumsum(values)
                fig.add_trace(
                    go.Scatter(
                        x=times,
                        y=cumsum,
                        mode="lines",
                        line=dict(width=2, color=colors[i]),
                        name=factor,
                    )
                )

        fig.update_layout(
            xaxis=dict(title="Time", gridcolor="lightgray"),
            yaxis=dict(title="Cumulative Return", gridcolor="lightgray"),
            plot_bgcolor="white",
            hovermode="x unified",
            height=400,
            autosize=False,
            font=dict(family=self.FONT_FAMILY, size=self.FONT_SIZE),
            margin=dict(l=50, r=20, t=30, b=50),
            legend=dict(x=0.02, y=0.98, xanchor="left", yanchor="top"),
        )

        return fig

    def _plot_etf_prices(self):
        """Plot ETF prices as % change from start."""
        fig = go.Figure()
        times = pd.to_datetime(self.data_history["times"])

        etfs = ["SPY", "QQQ", "IWM"]
        colors = ["red", "blue", "green"]

        for i, ticker in enumerate(etfs):
            key = f"{ticker}_log"
            if len(self.data_history[key]) > 0:
                values = np.array(self.data_history[key])
                pct_change = (values - values[0]) * 100
                fig.add_trace(
                    go.Scatter(
                        x=times,
                        y=pct_change,
                        mode="lines",
                        line=dict(width=2, color=colors[i]),
                        name=ticker,
                    )
                )

        fig.update_layout(
            xaxis=dict(title="Time", gridcolor="lightgray"),
            yaxis=dict(title="% Change from Start", gridcolor="lightgray"),
            plot_bgcolor="white",
            hovermode="x unified",
            height=400,
            autosize=False,
            font=dict(family=self.FONT_FAMILY, size=self.FONT_SIZE),
            margin=dict(l=50, r=20, t=30, b=50),
            legend=dict(x=0.02, y=0.98, xanchor="left", yanchor="top"),
        )

        return fig

    def _plot_equity_prices(self):
        """Plot equity prices as % change from start."""
        fig = go.Figure()
        times = pd.to_datetime(self.data_history["times"])

        equities = ["AAPL", "MSFT", "NVDA", "TSLA", "AMZN"]
        colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

        for i, ticker in enumerate(equities):
            key = f"{ticker}_log"
            if len(self.data_history[key]) > 0:
                values = np.array(self.data_history[key])
                pct_change = (values - values[0]) * 100
                fig.add_trace(
                    go.Scatter(
                        x=times,
                        y=pct_change,
                        mode="lines",
                        line=dict(width=2, color=colors[i]),
                        name=ticker,
                    )
                )

        fig.update_layout(
            xaxis=dict(title="Time", gridcolor="lightgray"),
            yaxis=dict(title="% Change from Start", gridcolor="lightgray"),
            plot_bgcolor="white",
            hovermode="x unified",
            height=400,
            autosize=False,
            font=dict(family=self.FONT_FAMILY, size=self.FONT_SIZE),
            margin=dict(l=50, r=20, t=30, b=50),
            legend=dict(x=0.02, y=0.98, xanchor="left", yanchor="top"),
        )

        return fig

    def _plot_spy_iso_flow(self):
        """Plot SPY ISO flow intensity."""
        fig = go.Figure()
        times = pd.to_datetime(self.data_history["times"])

        if len(self.data_history["SPY_iso_flow"]) > 0:
            fig.add_trace(
                go.Scatter(
                    x=times,
                    y=self.data_history["SPY_iso_flow"],
                    mode="lines",
                    line=dict(width=2, color="orange"),
                    fill="tozeroy",
                    fillcolor="rgba(255, 165, 0, 0.2)",
                )
            )

        fig.update_layout(
            xaxis=dict(title="Time", gridcolor="lightgray"),
            yaxis=dict(title="ISO Flow Intensity", gridcolor="lightgray"),
            plot_bgcolor="white",
            hovermode="x unified",
            height=350,
            autosize=False,
            font=dict(family=self.FONT_FAMILY, size=self.FONT_SIZE),
            margin=dict(l=50, r=20, t=30, b=50),
            showlegend=False,
        )

        return fig

    def _plot_spy_spread(self):
        """Plot SPY relative spread in bps."""
        fig = go.Figure()
        times = pd.to_datetime(self.data_history["times"])

        if len(self.data_history["SPY_rsprd"]) > 0:
            fig.add_trace(
                go.Scatter(
                    x=times,
                    y=self.data_history["SPY_rsprd"],
                    mode="lines",
                    line=dict(width=2, color="red"),
                    fill="tozeroy",
                    fillcolor="rgba(255, 0, 0, 0.2)",
                )
            )

        fig.update_layout(
            xaxis=dict(title="Time", gridcolor="lightgray"),
            yaxis=dict(title="Spread (bps)", gridcolor="lightgray"),
            plot_bgcolor="white",
            hovermode="x unified",
            height=350,
            autosize=False,
            font=dict(family=self.FONT_FAMILY, size=self.FONT_SIZE),
            margin=dict(l=50, r=20, t=30, b=50),
            showlegend=False,
        )

        return fig

    def _plot_spy_flow(self):
        """Plot SPY total flow."""
        fig = go.Figure()
        times = pd.to_datetime(self.data_history["times"])

        if len(self.data_history["SPY_total_flow"]) > 0:
            fig.add_trace(
                go.Scatter(
                    x=times,
                    y=self.data_history["SPY_total_flow"],
                    mode="lines",
                    line=dict(width=2, color="green"),
                    fill="tozeroy",
                    fillcolor="rgba(0, 128, 0, 0.2)",
                )
            )

        fig.update_layout(
            xaxis=dict(title="Time", gridcolor="lightgray"),
            yaxis=dict(title="Total Flow", gridcolor="lightgray"),
            plot_bgcolor="white",
            hovermode="x unified",
            height=350,
            autosize=False,
            font=dict(family=self.FONT_FAMILY, size=self.FONT_SIZE),
            margin=dict(l=50, r=20, t=30, b=50),
            showlegend=False,
        )

        return fig

    def _plot_spy_trades(self):
        """Plot SPY number of trades."""
        fig = go.Figure()
        times = pd.to_datetime(self.data_history["times"])

        if len(self.data_history["SPY_num_trades"]) > 0:
            fig.add_trace(
                go.Scatter(
                    x=times,
                    y=self.data_history["SPY_num_trades"],
                    mode="lines",
                    line=dict(width=2, color="purple"),
                    fill="tozeroy",
                    fillcolor="rgba(128, 0, 128, 0.2)",
                )
            )

        fig.update_layout(
            xaxis=dict(title="Time", gridcolor="lightgray"),
            yaxis=dict(title="Number of Trades", gridcolor="lightgray"),
            plot_bgcolor="white",
            hovermode="x unified",
            height=350,
            autosize=False,
            font=dict(family=self.FONT_FAMILY, size=self.FONT_SIZE),
            margin=dict(l=50, r=20, t=30, b=50),
            showlegend=False,
        )

        return fig

    def _update_stats(self):
        """Update statistics panel."""
        n_times = len(self.data_history["times"])
        if n_times == 0:
            return

        # Calculate latest values
        latest_stats = {}
        for factor in ["HML", "SMB", "RMW", "CMA", "MKT_RF"]:
            log_key = f"{factor}_log"
            ret_key = f"{factor}_ret"
            if len(self.data_history[log_key]) > 0:
                log_vals = np.array(self.data_history[log_key])
                ret_vals = np.array(self.data_history[ret_key])
                latest_stats[factor] = {
                    "log_change": (log_vals[-1] - log_vals[0]) * 100,
                    "cum_ret": np.nancumsum(ret_vals)[-1] * 100,
                }

        stats_md = f"""
### Data Points
**Total Updates:** {n_times}  
**Time Range:** {(self.data_history["times"][-1] - self.data_history["times"][0]).total_seconds() / 60:.0f} minutes

---

### Fama Factor Performance
"""
        for factor in ["HML", "SMB", "RMW", "CMA", "MKT_RF"]:
            if factor in latest_stats:
                stats_md += f"**{factor}:** {latest_stats[factor]['cum_ret']:+.3f}% (cumulative)\n"

        # SPY microstructure stats
        if len(self.data_history["SPY_iso_flow"]) > 0:
            stats_md += f"""
---

### SPY Microstructure (Current)
**ISO Flow:** {self.data_history["SPY_iso_flow"][-1]:.4f}  
**Spread:** {self.data_history["SPY_rsprd"][-1]:.2f} bps  
**Total Flow:** {self.data_history["SPY_total_flow"][-1]:.0f}  
**Num Trades:** {self.data_history["SPY_num_trades"][-1]:.0f}  
**% ISO:** {self.data_history["SPY_pct_iso"][-1]:.1f}%
"""

        self.stats_pane.object = stats_md

    def show(self):
        """Show the dashboard."""
        self.dashboard.show()
        return self.dashboard


# CSP nodes and graph for Panel
@csp.node
def fama_panel_updater(
    trigger: ts[object],
    fama_log: ts[FamaFactors],
    fama_ret: ts[FamaReturns],
    bars_basket: {str: ts[EquityBar1m]},
    dashboard: FamaFullAnalysisDashboard,
):
    """Update panel dashboard with Fama + microstructure data."""
    with csp.state():
        s_update_count = 0

    if csp.ticked(trigger):
        s_update_count += 1
        # Update every 10 ticks to avoid overwhelming the dashboard
        if s_update_count % 10 == 0:
            # Collect current bar values
            bars_dict = {}
            for ticker, bar_ts in bars_basket.validitems():
                bars_dict[ticker] = bar_ts

            dashboard.update(
                t=csp.now(),
                fama_log=fama_log if csp.valid(fama_log) else None,
                fama_ret=fama_ret if csp.valid(fama_ret) else None,
                bars_dict=bars_dict,
            )


@csp.graph
def fama_panel_graph(dashboard: FamaFullAnalysisDashboard):
    """Fama factors + microstructure with panel dashboard updates."""
    # Fama factors
    quotes_basket_fama = get_quotes_1s_wo_size(list(ALL_FACTOR_TICKERS))
    fama = compute_fama_factors_graph(
        quotes_basket=quotes_basket_fama,
        factor_weights_path=FACTOR_WEIGHTS_PATH,
        use_efficient=True,
    )

    # TAQ data
    quotes_basket_taq = get_taq_quotes()
    trades_basket_taq = get_taq_trades()

    # Microstructure features
    bars_basket = compute_microstructure_features_basket(
        trades_basket=trades_basket_taq,
        quotes_basket=quotes_basket_taq,
        bar_interval=pd.Timedelta(minutes=1),
    )

    # Use timer as trigger (updates every 5 seconds)
    trigger = csp.timer(pd.Timedelta(seconds=5))

    fama_panel_updater(
        trigger=trigger,
        fama_log=fama.log_prices,
        fama_ret=fama.returns,
        bars_basket=bars_basket,
        dashboard=dashboard,
    )


def run_fama_panel():
    """Run real-time Fama full analysis dashboard."""
    print("=" * 80)
    print("🚀 LAUNCHING REAL-TIME FAMA FULL ANALYSIS DASHBOARD")
    print("=" * 80)
    print(f"📅 Date: {DATE_OF_INTEREST}")
    print(f"📊 Fama Factor Tickers: {len(ALL_FACTOR_TICKERS)}")
    print(f"📊 TAQ Instruments: 40")
    print("\n⏳ Starting CSP engine and dashboard server...")
    print("=" * 80 + "\n")

    dashboard = FamaFullAnalysisDashboard()

    import threading

    def run_csp():
        csp.run(
            lambda: fama_panel_graph(dashboard=dashboard),
            starttime=start_ts,
            endtime=end_ts,
            realtime=False,
        )

    csp_thread = threading.Thread(target=run_csp, daemon=True)
    csp_thread.start()

    dashboard.show()


if __name__ == "__main__":
    run_fama_panel()
