import pandas as pd
df = pd.read_parquet("data/sp500_ohlcv_dataset.parquet")
df = df.reset_index()

# Select only the price columns you're interested in
price_columns = ['open', 'high', 'low', 'close', 'adj_close']

# Group by date and check if all rows have complete data for all price columns
def check_complete_data(group):
    return group[price_columns].notna().all(axis=1).all()

complete_dates = df.groupby('date').apply(check_complete_data)

# Find the first date where all tickers have complete data
first_complete_date = complete_dates[complete_dates].index[0]
print(f"First complete date: {first_complete_date}")


# Now drop all rows before that date
df_filtered = df[df['date'] >= first_complete_date]
df_filtered.to_parquet('data/cleaned_sp500_dataset.parquet')