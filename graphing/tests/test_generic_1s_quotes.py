import pandas as pd

import csp
from CSP_Options.structs import EquityQuoteWoSize
from CSP_Options.utils.readers import get_quotes_1s_wo_size

DATE_OF_INTEREST = "2024-11-05"


"""
Simple test to verify we can load underlying quotes for all 40 tickers
"""
start_ts = pd.Timestamp(f"{DATE_OF_INTEREST} 09:31:00", tz="America/New_York")
end_ts = pd.Timestamp(f"{DATE_OF_INTEREST} 16:00:00", tz="America/New_York")

RANDOM_TICKERS = [
    "GME",  # GameStop - the OG meme stock
    "AMC",  # AMC Entertainment - apes together strong
    "BBBYQ",  # Bed Bath & Beyond - meme stock to bankruptcy
    "TSLA",  # Tesla - Elon memes
    "BB",  # BlackBerry - nostalgia meme
    "KOSS",  # Koss Corporation - meme headphones
    "SPCE",  # Virgin Galactic - to the moon, literally
    "NKLA",  # Nikola - rolling downhill
    "SNDL",  # Sundial Growers - weed meme
    "AAPL",  # Apple - not a meme, but always in the mix
    "PLTR",  # Palantir - meme data
    "DWAC",  # Trump SPAC - meme politics
    "CLOV",  # Clover Health - meme pump
    "BBIG",  # Vinco Ventures - meme mystery
    "SOFI",  # SoFi - meme fintech
    "FUBO",  # FuboTV - meme streaming
    "OSTK",  # Overstock - meme e-commerce
    "WISH",  # ContextLogic - meme shopping
    "RKT",  # Rocket Companies - meme mortgage
    "HCMC",  # Healthier Choices - penny meme
]


@csp.graph
def main_graph():
    quotes_basket = get_quotes_1s_wo_size(RANDOM_TICKERS)

    csp.print("quotes_basket", quotes_basket)


print(f"\nRunning from {start_ts} to {end_ts}...")

csp.run(
    main_graph,
    starttime=start_ts,
    endtime=end_ts,
    realtime=False,
)
