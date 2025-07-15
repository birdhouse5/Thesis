import yfinance as yf
import pandas as pd
import re
from datetime import datetime
import time

def extract_tickers_from_file(filename):
    """
    Extract ticker symbols from a file containing tickers in format ("A"), ("B"), etc.
    """
    tickers = []
    
    try:
        with open(filename, 'r') as file:
            content = file.read()
            
        # Use regex to find all ticker symbols in parentheses and quotes
        pattern = r'\("([A-Z]+)"\)'
        matches = re.findall(pattern, content)
        
        # Remove duplicates and sort
        tickers = sorted(list(set(matches)))
        
        print(f"Found {len(tickers)} unique tickers")
        return tickers
        
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
        return []
    except Exception as e:
        print(f"Error reading file: {e}")
        return []

def get_ticker_date_range(ticker, max_retries=3):
    """
    Get the start and end date for a ticker from Yahoo Finance
    """
    for attempt in range(max_retries):
        try:
            # Create ticker object
            stock = yf.Ticker(ticker)
            
            # Get historical data for maximum period
            hist = stock.history(period="max")
            
            if hist.empty:
                print(f"Warning: No data found for {ticker}")
                return None, None
            
            # Get start and end dates
            start_date = hist.index[0].strftime('%Y-%m-%d')
            end_date = hist.index[-1].strftime('%Y-%m-%d')
            
            print(f"✓ {ticker}: {start_date} to {end_date}")
            return start_date, end_date
            
        except Exception as e:
            print(f"Attempt {attempt + 1} failed for {ticker}: {e}")
            if attempt < max_retries - 1:
                time.sleep(1)  # Wait 1 second before retry
            else:
                print(f"✗ Failed to get data for {ticker} after {max_retries} attempts")
                return None, None
    
    return None, None

def create_constituents_csv(input_file, output_file='data/sp500_constituents.csv'):
    """
    Main function to create the constituents CSV file
    """
    print("Starting S&P 500 ticker data extraction...")
    print("=" * 50)
    
    # Extract tickers from file
    tickers = extract_tickers_from_file(input_file)
    
    if not tickers:
        print("No tickers found. Exiting.")
        return
    
    print(f"\nProcessing {len(tickers)} tickers...")
    print("=" * 50)
    
    # Store results
    results = []
    failed_tickers = []
    
    # Process each ticker
    for i, ticker in enumerate(tickers, 1):
        print(f"[{i}/{len(tickers)}] Processing {ticker}...")
        
        start_date, end_date = get_ticker_date_range(ticker)
        
        if start_date and end_date:
            results.append({
                'ticker': ticker,
                'start_date': start_date,
                'end_date': end_date
            })
        else:
            failed_tickers.append(ticker)
        
        # Add a small delay to avoid overwhelming the API
        time.sleep(0.1)
    
    # Create DataFrame and save to CSV
    if results:
        df = pd.DataFrame(results)
        df.to_csv(output_file, index=False)
        
        print("\n" + "=" * 50)
        print(f"SUCCESS: Created {output_file} with {len(results)} tickers")
        print(f"Failed tickers: {len(failed_tickers)}")
        
        if failed_tickers:
            print(f"Failed tickers: {', '.join(failed_tickers)}")
        
        # Display sample of results
        print("\nSample of results:")
        print(df.head(10).to_string(index=False))
        
    else:
        print("No data was successfully retrieved.")

# Example usage
if __name__ == "__main__":
    # Run the script
    create_constituents_csv('data/ticker_symbols.txt')
    
    # Optional: If you want to specify a different output filename
    # create_constituents_csv('ticker_symbols.txt', 'my_constituents.csv')