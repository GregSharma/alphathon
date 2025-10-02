import csp
import pandas as pd

from CSP_Options.utils.readers import get_taq_quotes, get_taq_trades
from csp.basketlib import sample_dict
import logging
import pytz
from datetime import timedelta

logging.basicConfig(level=logging.INFO)

DATE_OF_INTEREST = "2024-11-05"

"""
Trades and quotes for all 40 tickers
"""

start_ts = pd.Timestamp(f"{DATE_OF_INTEREST} 09:30:00", tz="America/New_York")
end_ts = start_ts + pd.Timedelta(minutes=5)


@csp.graph
def quotes_graph():
    quotes_basket = get_taq_quotes()
    test_timer = csp.timer(timedelta(milliseconds=100))
    sampled_quotes = sample_dict(test_timer, quotes_basket)
    csp.log(
        logging.INFO,
        "quotes_basket",
        sampled_quotes,
        logger_tz=pytz.timezone("America/New_York"),
        use_thread=True,
    )


@csp.graph
def trades_graph():
    trades_basket = get_taq_trades()
    test_timer = csp.timer(timedelta(milliseconds=100))
    sampled_trades = sample_dict(test_timer, trades_basket)
    csp.log(
        logging.INFO,
        "trades_basket",
        sampled_trades,
        logger_tz=pytz.timezone("America/New_York"),
        use_thread=True,
    )


print(f"\nRunning from {start_ts} to {end_ts}...")

csp.run(
    quotes_graph,
    starttime=start_ts,
    endtime=end_ts,
    realtime=False,
)

csp.run(
    trades_graph,
    starttime=start_ts,
    endtime=end_ts,
    realtime=False,
)
