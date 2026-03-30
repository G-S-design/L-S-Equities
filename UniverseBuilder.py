import yfinance as yf
from yfinance import EquityQuery
import csv
import pandas as pd 

SIZE = 50 # Change this to decide how many stocks from each sector you want
OUTPUT_PATH = "universe.csv"
SECTOR_LIST = [
    "Technology",
    "Healthcare",
    "Financial Services",
    "Consumer Cyclical",
    "Communication Services",
    "Industrials",
    "Consumer Defensive",
    "Energy",
    "Utilities",
    "Real Estate",
    "Materials",  # map this to "Basic Materials"
]

SECTOR_MAP = {
    "Materials": "Basic Materials",
}
SECTOR_RULES = {
    "Technology": {"region": "us", "min_mcap": 100_000_000},
    "Healthcare": {"region": "us", "min_mcap": 50_000_000},
    "Financial Services": {"region": "us", "min_mcap": 75_000_000},
    "Energy": {"region": "us", "min_mcap": 30_000_000},
}
DEFAULT_RULE = {"region": "us", "min_mcap": 20_000_000}

def build_criteria(region: str, sector: str, min_mcap: int) -> EquityQuery:
    return EquityQuery(
        "and",
        [
            EquityQuery("is-in", ["region", region]),
            EquityQuery("eq", ["sector", sector]),
            EquityQuery("gt", ["intradaymarketcap", min_mcap]),
        ],
    )

def run_sector_screen(my_sector_name: str, size: int = 50):
    yahoo_sector = SECTOR_MAP.get(my_sector_name, my_sector_name)
    rule = SECTOR_RULES.get(my_sector_name, DEFAULT_RULE)

    criteria = build_criteria(
        region=rule["region"],
        sector=yahoo_sector,
        min_mcap=rule["min_mcap"],
    )

    return yf.screen(
        query=criteria,
        size=size,
        sortField="intradaymarketcap",
        sortAsc=False,
    )


def screen_all_sectors(size: int, outputpath):

    all_rows = []
    ticker_list = []

    for sector in SECTOR_LIST:
        resp = run_sector_screen(sector, size)

        for stock in resp.get("quotes", []):
            row = {
                "sector": sector,
                "ticker": stock["symbol"],
                "name": stock.get("longName", "N/A"),
                "market_cap": stock.get("intradaymarketcap", None),
            }

            all_rows.append(row)
            ticker_list.append(stock["symbol"])

    # --- De-dupe by company name (not ticker), then write clean CSV ---
    df = pd.DataFrame(all_rows)

    # Clean names so tiny differences don't prevent matching
    df["name_clean"] = (
        df["name"]
        .astype(str)
        .str.lower()
        .str.replace(r"\s+", " ", regex=True)
        .str.replace(",", "", regex=False)
        .str.strip()
    )

    # Keep the biggest market cap listing for each company name
    df["market_cap"] = pd.to_numeric(df["market_cap"], errors="coerce")
    df = df.sort_values("market_cap", ascending=False)
    df = df.drop_duplicates(subset="name_clean", keep="first")

    # Drop helper column and save
    df = df.drop(columns=["name_clean"])
    df.to_csv(outputpath, index=False)

    # Return tickers from cleaned universe
    return df["ticker"].tolist()

if __name__ == "__main__":
    tickers = screen_all_sectors(SIZE, OUTPUT_PATH)
    print("Universe size:", len(tickers))   
    
    
# for sector in SECTOR_LIST:
#     print(f"\n=== {sector} ===")
#     resp = run_sector_screen(sector, size=30)
#     for stock in resp.get("quotes", []):
#         print(f"{stock['symbol']}: {stock.get('longName', 'N/A')}")