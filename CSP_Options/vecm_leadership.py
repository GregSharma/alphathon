"""VECM Leadership computation and streaming nodes"""

from datetime import timedelta
from typing import Dict

import csp
import numpy as np
import pandas as pd
from statsmodels.tsa.vector_ar.vecm import VECM

from .microstructure import compute_microstructure_features_basket
from .utils.readers import ALL_TICKERS, get_taq_quotes, get_taq_trades


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
    except (ValueError, np.linalg.LinAlgError, Exception):
        return {t: 0.0 for t in tickers}


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

    if csp.ticked(bars_basket):
        current_time = csp.now()
        bars_dict = {}
        for ticker, bar_ts in bars_basket.validitems():
            bars_dict[ticker] = bar_ts

        s_bar_history.append((current_time, bars_dict))

        if s_tickers is None:
            s_tickers = sorted(bars_dict.keys())

        if len(s_bar_history) > window_size:
            s_bar_history = s_bar_history[-window_size:]

    if csp.ticked(timer):
        if len(s_bar_history) < min_bars:
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

        log_mid_array = np.array(log_mid_matrix)
        leadership_scores = compute_vecm_leadership(log_mid_array, s_tickers)

        result = {
            "timestamp": csp.now(),
            "window_bars": len(s_bar_history),
            "leadership": leadership_scores,
            "leader": max(leadership_scores.keys(), key=lambda k: leadership_scores[k]),
        }

        return result


@csp.graph
def vecm_leadership_collector():
    """Collect VECM leadership outputs for plotting."""
    quotes_basket = get_taq_quotes()
    trades_basket = get_taq_trades()

    bars_basket = compute_microstructure_features_basket(
        trades_basket=trades_basket,
        quotes_basket=quotes_basket,
        bar_interval=timedelta(minutes=1),
    )

    vecm_timer = csp.timer(timedelta(minutes=1))
    leadership = vecm_leadership_streaming(
        bars_basket=bars_basket, timer=vecm_timer, window_size=60, min_bars=30
    )

    csp.add_graph_output("leadership", leadership)

    @csp.node
    def extract_ticker_score(leadership: csp.ts[object], ticker: str) -> csp.ts[float]:
        if csp.ticked(leadership) and leadership is not None:
            return leadership["leadership"].get(ticker, 0.0)

    @csp.node
    def extract_leader(leadership: csp.ts[object]) -> csp.ts[str]:
        if csp.ticked(leadership) and leadership is not None:
            return leadership["leader"]

    for ticker in ALL_TICKERS:
        score = extract_ticker_score(leadership, ticker)
        csp.add_graph_output(f"score_{ticker}", score)

    leader = extract_leader(leadership)
    csp.add_graph_output("leader", leader)
