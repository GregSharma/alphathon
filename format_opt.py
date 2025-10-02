from clickhouse_connect import get_client
from tqdm import tqdm
import subprocess
import json

ROOTS = [
    "AAPL",
    "AMD",
    "AMZN",
    "AVGO",
    "BRKB",
    "CEG",
    "DIA",
    "DJT",
    "EEM",
    "FXI",
    "GOOG",
    "GOOGL",
    "HYG",
    "IVV",
    "IWM",
    "JPM",
    "KRE",
    "LLY",
    "LQD",
    "META",
    "MSFT",
    "MSTR",
    "NVDA",
    "PLTR",
    "QQQ",
    "SHW",
    "SMCI",
    "SMH",
    "SOXL",
    "SPY",
    "SQQQ",
    "TLT",
    "TQQQ",
    "TSLA",
    "VOO",
    "VUG",
    "XLE",
    "XLF",
    "XLU",
    "XOM",
]

from concurrent.futures import ThreadPoolExecutor

client = get_client()


def process_root(ROOT):
    client = get_client()
    query = f"""
    INSERT INTO FUNCTION s3('https://s3.us-east-005.backblazeb2.com/GregsMktData/AlphathonDatasets/options_selected_formatted/{ROOT}_20241115_20241105_20241105_1000.parquet')
    SELECT
        date,
        concat(
            leftPad(root, 6, ' '),
            formatDateTime(addHours(toDateTime(YYYYMMDDToDate(exp), 'America/New_York'), 16), '%y%m%d'),
            upper(right),
            leftPad(toString(toInt64(strike)), 8, '0')
        ) AS symbol,
        toDateTime(toUnixTimestamp(toDateTime(YYYYMMDDToDate(date), 'America/New_York')) + ms_of_day / 1000, 'America/New_York') AS timestamp,
        root AS underlying,
        --addHours(toDateTime(YYYYMMDDToDate(exp), 'America/New_York'), 16) AS expiration,
        exp AS expiration,
        CAST(strike / 1000.0 AS Float64) AS strike,
        right AS right,
        bid_size AS bid_size,
        bid_exchange AS bid_exchange,
        bid AS bid,
        bid_condition AS bid_condition,
        ask_size AS ask_size,
        ask_exchange AS ask_exchange,
        ask AS ask,
        ask_condition AS ask_condition
    FROM s3('https://s3.us-east-005.backblazeb2.com/GregsMktData/AlphathonDatasets/options_selected/{ROOT}_20241115_20241105_20241105_1000.parquet')
    ORDER BY timestamp ASC
    SETTINGS s3_truncate_on_insert=1, max_threads=16, max_insert_threads=16, max_download_threads=16
    """
    client.command(query)
    return ROOT


def process_all_roots():
    with ThreadPoolExecutor(max_workers=1) as executor:
        results = list(tqdm(executor.map(process_root, ROOTS), total=len(ROOTS)))
        print(results)


def process_all_1s_quotes():
    client = get_client()
    all_1s_quotes_query = """
    INSERT INTO FUNCTION s3('https://s3.us-east-005.backblazeb2.com/GregsMktData/AlphathonDatasets/all_1s_quotes/2024-11-05.parquet')
    SELECT *
    FROM s3('https://s3.us-east-005.backblazeb2.com/GregsMktData/all_1s_quotes/2024-11-05.parquet')
    ORDER BY timestamp ASC
    SETTINGS s3_truncate_on_insert=1, max_threads=16, max_insert_threads=16, max_download_threads=16
    """
    client.command(all_1s_quotes_query)


def get_unique_symbols(ROOT):
    client = get_client()
    query = f"""
    SELECT DISTINCT symbol FROM s3('https://s3.us-east-005.backblazeb2.com/GregsMktData/AlphathonDatasets/options_selected/{ROOT}_20241115_20241105_20241105_1000.parquet')
    """
    result = client.query(query)
    return [row[0] for row in result.result_rows]


def create_selected_instrument_options_symbols():
    instrument_options_symbols = {}

    for ROOT in tqdm(ROOTS, desc="Getting unique symbols..."):
        symbols = get_unique_symbols(ROOT)
        instrument_options_symbols[ROOT] = symbols
        print(f"{ROOT}: {len(symbols)} symbols")

    # Save to JSON file
    with open("selected_instrument_options_symbols.json", "w") as f:
        json.dump(instrument_options_symbols, f, indent=2)

    print("Saved selected_instrument_options_symbols.json")
    return instrument_options_symbols


def get_descriptions_and_sample_data(s3_path):
    full_path = (
        f"s3('https://s3.us-east-005.backblazeb2.com/GregsMktData/AlphathonDatasets/{s3_path}')"
    )
    client = get_client()
    query = f"""DESCRIBE TABLE {full_path} FORMAT Markdown"""
    subprocess.run(["chc", "-q", query])

    # Get sample data
    sample_query = f"""SELECT * FROM {full_path} LIMIT 5 FORMAT Markdown"""
    subprocess.run(["chc", "-q", sample_query])


def insert_selected_quotes():
    print("Inserting selected quotes...")
    client = get_client()
    insert_selected_quotes_query = """
    INSERT INTO FUNCTION s3
    (
    'https://s3.us-east-005.backblazeb2.com/GregsMktData/AlphathonDatasets/quotes_selected/2024-11-05.parquet'
    )
    SELECT 
        ticker AS symbol,
        ask_exchange,
        ask_price,
        ask_size,
        bid_exchange,
        bid_price,
        bid_size,
        conditions,
        indicators,
        fromUnixTimestamp64Nano(participant_timestamp) AS timestamp,
        sequence_number,
        sip_timestamp,
        tape,
        trf_timestamp
    FROM s3('https://s3.us-east-005.backblazeb2.com/GregsMktData/all_quotes/2024-11-05_filtered.parquet')
    WHERE ticker IN ('NVDA','TSLA','AAPL','MSFT','AMZN','META','AMD','GOOGL','LLY','DJT','PLTR','BRK.B','MSTR','GOOG','AVGO','XOM','CEG','JPM','SMCI','SHW','SPY','QQQ','TLT','IWM','LQD','IVV','TQQQ','HYG','FXI','VUG','VOO','XLF','SOXL','XLE','DIA','EEM','SMH','SQQQ','XLU','KRE')
    ORDER BY timestamp ASC
    SETTINGS max_threads=16, max_insert_threads=16, max_download_threads=16, s3_truncate_on_insert=1
    """
    subprocess.run(["chc", "-q", insert_selected_quotes_query])
    # client.command(insert_selected_quotes_query)


def insert_selected_trades():
    print("Inserting selected trades...")
    client = get_client()
    insert_selected_trades_query = """
    INSERT INTO FUNCTION s3('https://s3.us-east-005.backblazeb2.com/GregsMktData/AlphathonDatasets/trades_selected/2024-11-05.parquet')
    SELECT 
        ticker AS symbol,
        conditions,
        correction,
        exchange,
        id,
        fromUnixTimestamp64Nano(participant_timestamp) AS timestamp,
        price,
        sequence_number,
        sip_timestamp,
        size,
        tape,
        trf_id,
        trf_timestamp
    FROM s3('https://s3.us-east-005.backblazeb2.com/GregsMktData/all_trades/2024-11-05_filtered.parquet')
    WHERE ticker IN ('NVDA','TSLA','AAPL','MSFT','AMZN','META','AMD','GOOGL','LLY','DJT','PLTR','BRK.B','MSTR','GOOG','AVGO','XOM','CEG','JPM','SMCI','SHW','SPY','QQQ','TLT','IWM','LQD','IVV','TQQQ','HYG','FXI','VUG','VOO','XLF','SOXL','XLE','DIA','EEM','SMH','SQQQ','XLU','KRE')
    ORDER BY timestamp ASC
    SETTINGS max_threads=16, max_insert_threads=16, max_download_threads=16, s3_truncate_on_insert=1
    """
    subprocess.run(["chc", "-q", insert_selected_trades_query])
    # client.command(insert_selected_trades_query)


def download_all_data():
    subprocess.run(
        [
            "rclone",
            "copy",
            "-P",
            "bb:GregsMktData/AlphathonDatasets",
            "data/alphathon_2025",
        ]
    )


if __name__ == "__main__":
    # process_all_roots()
    # process_all_1s_quotes()
    # create_selected_instrument_options_symbols()
    # insert_selected_quotes()
    # insert_selected_trades()
    print("Getting descriptions and sample data...")
    print("Sample Options Data for AAPL on 2024-11-05, Expiring on 2024-11-15")
    print(
        "Note -- ALL of the option chains on our 20 Equities and 20 ETFs are expiring on 2024-11-15"
    )
    get_descriptions_and_sample_data(
        "options_selected_formatted/AAPL_20241115_20241105_20241105_1000.parquet"
    )
    print(
        "Sample 1s Quotes for 2024-11-05 --- ALL US Equities and ETFs -- not just our 20 Equities and ETFs"
    )
    print(
        "Note - this data is downsampled to 1 second intervals. The timestamp is the right endpoint of the 1 second interval."
    )
    print(
        "This is consistent with the timestmap at which this data could be computed in real time."
    )
    get_descriptions_and_sample_data("all_1s_quotes/2024-11-05.parquet")
    print("Feature Set for Pre-Election Data")
    print("Sample Pre-Election Data for 2024-11-04 --- just our 20 Equities and ETFs")
    get_descriptions_and_sample_data("pre-elec/pre-elec-2024-11-04.parquet")
    print("Sample Quotes for 2024-11-05 --- ONLY our 20 Equities and ETFs")
    get_descriptions_and_sample_data("quotes_selected/2024-11-05.parquet")
    print("Sample Trades for 2024-11-05 --- ONLY our 20 Equities and ETFs")
    get_descriptions_and_sample_data("trades_selected/2024-11-05.parquet")
    print("Sample Kalshi Data for 2024-11-05 --- ONLY our 20 Equities and ETFs")
    get_descriptions_and_sample_data("kalshi_election/2024-11-05.parquet")
    print("Sample Selected Quotes Output from insert_selected_quotes")
    get_descriptions_and_sample_data("quotes_selected/2024-11-05.parquet")
    print("Sample Selected Trades Output from insert_selected_trades")
    get_descriptions_and_sample_data("trades_selected/2024-11-05.parquet")
    download_all_data()
