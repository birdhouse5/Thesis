import yfinance as yf
import pandas as pd

# List of 30 curated tickers
tickers_info = [
    # Likely in 1987
    ("AAPL", ""), ("MSFT", ""), ("JNJ", ""), ("XOM", ""), ("PG", ""),
    ("KO", ""), ("GE", "2018-06-26"), ("JPM", ""), ("HD", ""), ("DIS", ""),
    ("BAC", ""), ("PEP", ""), ("IBM", ""), ("WMT", ""), ("CVX", ""),
    
    # Currently in S&P 500 (2025)
    ("GOOGL", ""), ("META", ""), ("AMZN", ""), ("TSLA", ""), ("MA", ""),
    ("V", ""), ("UNH", ""), ("NVDA", ""), ("BRK-B", ""), ("ADBE", ""),
    ("NFLX", ""), ("CRM", ""), ("AVGO", ""), ("COST", ""), ("LLY", ""),
]

def get_first_available_date(ticker):
    try:
        data = yf.download(ticker, start="1980-01-01", end="1987-01-05", progress=False)
        if data.empty:
            data = yf.download(ticker, start="1980-01-01", progress=False)
        if data.empty:
            print(f"No data for {ticker}")
            return None
        return data.index[0].strftime("%Y-%m-%d")
    except Exception as e:
        print(f"Error with {ticker}: {e}")
        return None

# Collect records
records = []
for ticker, end_date in tickers_info:
    print(f"Processing {ticker}...")
    start_date = get_first_available_date(ticker)
    if start_date:
        records.append([ticker, start_date, end_date])

# Save to CSV
df = pd.DataFrame(records, columns=["ticker", "start_date", "end_date"])
df.to_csv("data/sp500_constituents.csv", index=False)
print("\nSaved to constituents.csv")
