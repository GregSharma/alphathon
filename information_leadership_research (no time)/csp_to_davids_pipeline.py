"""TAQ Data → Microstructure Features → VECM Format"""

from datetime import timedelta

import csp
import pandas as pd
from csp.basketlib import sample_dict

from CSP_Options.microstructure import compute_microstructure_features_basket
from CSP_Options.utils.david_helpers import prepare_for_vecm
from CSP_Options.utils.readers import ALL_TICKERS, get_taq_quotes, get_taq_trades

DATE_OF_INTEREST = "2024-11-05"
START_TIME = "09:30:00"
DURATION_HOURS = 6.5
BAR_INTERVAL_MINUTES = 1


def run_pipeline(
    start_time_str=START_TIME,
    duration_hours=DURATION_HOURS,
    bar_interval_minutes=BAR_INTERVAL_MINUTES,
    output_csv="microstructure_bars.csv",
    output_vecm="vecm_ready.csv",
):
    start_ts = pd.Timestamp(f"{DATE_OF_INTEREST} {start_time_str}", tz="America/New_York")
    end_ts = start_ts + pd.Timedelta(hours=duration_hours)
    collected_bars = []

    @csp.graph
    def pipeline_graph():
        quotes_basket = get_taq_quotes()
        trades_basket = get_taq_trades()

        bars_basket = compute_microstructure_features_basket(
            trades_basket=trades_basket,
            quotes_basket=quotes_basket,
            bar_interval=timedelta(minutes=bar_interval_minutes),
        )

        timer = csp.timer(timedelta(minutes=bar_interval_minutes))
        sampled_bars = sample_dict(timer, bars_basket)

        @csp.node
        def collect_bars(bars_dict: {str: csp.ts[object]}):
            if csp.ticked(bars_dict):
                rows = []
                for _, bar_ts in bars_dict.validitems():
                    bar = bar_ts
                    rows.append(
                        {
                            "bucket_ts": csp.now(),
                            "ticker": bar.symbol,
                            "log_mid": bar.log_mid,
                            "iso_flow_intensity": bar.iso_flow_intensity,
                            "total_flow": bar.total_flow,
                            "total_flow_non_iso": bar.total_flow_non_iso,
                            "num_trades": bar.num_trades,
                            "quote_updates": bar.quote_updates,
                            "avg_rsprd": bar.avg_rsprd,
                            "pct_trades_iso": bar.pct_trades_iso,
                        }
                    )
                collected_bars.extend(rows)

        collect_bars(sampled_bars)

    csp.run(pipeline_graph, starttime=start_ts, endtime=end_ts, realtime=False)

    if not collected_bars:
        return None, None

    df_long = pd.DataFrame(collected_bars)
    df_long.to_csv(output_csv, index=False)

    df_vecm = prepare_for_vecm(df_long, tickers=ALL_TICKERS, add_returns=True, add_tod_bins=True)
    df_vecm.to_csv(output_vecm, index=False)

    return df_long, df_vecm


def quick_test():
    return run_pipeline(
        duration_hours=5 / 60, output_csv="test_bars.csv", output_vecm="test_vecm.csv"
    )


def full_day():
    return run_pipeline(duration_hours=6.5, output_csv="full_bars.csv", output_vecm="full_vecm.csv")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "full":
        full_day()
    else:
        quick_test()
