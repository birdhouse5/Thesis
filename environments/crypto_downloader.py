from data import PortfolioDataset

# This will trigger the new chunked download
dataset = PortfolioDataset(
    asset_class="crypto",
    force_recreate=True  # Force download with new method
)

# Verify the results
print(dataset.full_data.shape)
print(dataset.full_data['date'].min(), "to", dataset.full_data['date'].max())
print(f"{dataset.full_data['ticker'].nunique()} tickers")