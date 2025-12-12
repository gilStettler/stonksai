"""
Loads historical daily data (TIME_SERIES_DAILY) from Alpha Vantage
for a user-selected ticker and saves it as a CSV file.
"""

import os
import sys
import csv
import requests
from pathlib import Path
from typing import Dict, Tuple, Any
from dotenv import load_dotenv

# Load API key from .env file
load_dotenv()
API_KEY = os.getenv("ALPHAVANTAGE_API_KEY")

BASE_URL = "https://www.alphavantage.co/query"
DATA_DIR = Path("alphavantage_data")

# Available Tickers
TICKER_SELECTION: Dict[str, Tuple[str, str, str]] = {
    "1": ("0QKI.LON", "SWISSCOM ORD SHS", "London Stock Exchange"),
    "2": ("0QLR.LON", "NOVARTIS ORD SHS", "London Stock Exchange"),
    "3": ("NSRGY", "Nestlé SA ADR", "US Stock Exchange"),
    "4": ("RHO6.FRK", "Roche Holding Ltd AD", "Frankfurt Stock Exchange"),
    "5": ("ABBNY", "ABB Ltd", "US Stock Exchange"),
    "6": ("UBS", "UBS Group AG Registered Ordinary Shares (UBS)", "US Stock Exchange"),
    "7": ("0QP2.LON", "ZURICH INSURANCE GROUP ORD SHS", "London Stock Exchange"),
    "8": ("0QKY.LON", "HOLCIM LTD ORD SHS", "London Stock Exchange"),
    "9": ("0QNO.LON", "LONZA GROUP ORD SHS", "London Stock Exchange"),
    "10": ("0QPS.LON", "GIVAUDAN ORD SHS", "London Stock Exchange"),
    "11": ("0A0D.LON", "ALCON ORD SHS", "London Stock Exchange"),
    "12": ("0Z4C.LON", "SIKA ORD SHS", "London Stock Exchange"),
    "13": ("0QOQ.LON", "PARTNERS GROUP HOLDING ORD SHS", "London Stock Exchange"),
    "14": ("0QMG.LON", "SWISS LIFE HOLDING ORD SHS", "London Stock Exchange"),
    "15": ("AMRZ", "Amrize Ltd Ordinary Shares", "US Stock Exchange"),
    "16": ("0QQ2.LON", "GEBERIT ORD SHS	", "London Stock Exchange"),
    "17": ("0QMW.LON", "KUEHNE & NAGEL ORD SHS", "London Stock Exchange"),
    "18": ("0QK6.LON", "LOGITECH INTERNATIONAL ORD SHS", "London Stock Exchange"),
}


def choose_ticker() -> str:
    """
    User can choose a ticker from the available options.
    """
    print("Choose a ticker from the following options:\n")
    for key, (symbol, name, exchange) in TICKER_SELECTION.items():
        print(f"{key}: {symbol} - {name} ({exchange})")
    print()

    while True:
        choice = input("Enter the number corresponding to your choice: ").strip()
        if choice in TICKER_SELECTION:
            symbol = TICKER_SELECTION[choice][0]
            print(f"\nYou selected: {symbol}\n")
            return symbol
        else:
            print("Invalid choice. Please try again.\n")


def fetch_historical_data(ticker: str) -> Dict[str, Any]:
    """
    Fetch historical daily data for the given ticker from Alpha Vantage.
    Returns the 'Time Series (Daily)' dictionary.
    """
    if not API_KEY or API_KEY == "your api key":
        print("Error: Please set your Alpha Vantage API key in the environment variable 'ALPHAVANTAGE_API_KEY'.")
        sys.exit(1)

    params = {
        "function": "TIME_SERIES_DAILY",
        "symbol": ticker,
        "outputsize": "full",  # or 'compact' for the last 100 data points
        "apikey": API_KEY,
    }

    try:
        response = requests.get(BASE_URL, params=params, timeout=15)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data: {e}")
        sys.exit(1)

    data = response.json()

    if "Error Message" in data:
        print(f"Error from API: {data['Error Message']}")
        sys.exit(1)

    if "Note" in data:
        # This usually indicates that the API call frequency limit has been reached
        print("Note from Alpha Vantage:")
        print(data["Note"])
        print("You may need to wait a bit and try again.")
        sys.exit(1)

    if "Time Series (Daily)" not in data:
        print("Unexpected response format from API.")
        sys.exit(1)

    return data["Time Series (Daily)"]


def save_data_to_csv(ticker: str, time_series_daily: Dict[str, Dict[str, str]]) -> Path:
    """
    Save the historical data to a CSV file.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    file_path = DATA_DIR / f"{ticker}_daily_data.csv"

    # CSV header
    header = ["date", "open", "high", "low", "close", "volume"]

    # sort by date
    dates = sorted(time_series_daily.keys())

    with file_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for date in dates:
            values = time_series_daily[date]
            row = [
                date,
                values.get("1. open"),
                values.get("2. high"),
                values.get("3. low"),
                values.get("4. close"),
                values.get("5. volume"),
            ]
            writer.writerow(row)

    return file_path


# ----------------------------------------
# Main
# ----------------------------------------
def main() -> None:
    symbol = choose_ticker()
    print(f"Fetching historical data for {symbol}...")
    time_series_daily = fetch_historical_data(symbol)
    csv_file_path = save_data_to_csv(symbol, time_series_daily)
    print(f"Data saved to {csv_file_path.resolve()}")


if __name__ == "__main__":
    main()
