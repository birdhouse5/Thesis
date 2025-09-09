from data_preparation import create_dataset, create_crypto_dataset
from dataset import create_split_datasets

def test_sp500():
    print("\n=== Testing S&P500 pipeline ===")
    sp500_path = create_dataset(force_recreate=False)  # uses Yahoo Finance parquet
    datasets_sp = create_split_datasets(sp500_path,
                                        train_end="2015-12-31",
                                        val_end="2020-12-31",
                                        proportional=False)

    for split, ds in datasets_sp.items():
        info = ds.get_split_info()
        print(f"S&P500 {split}: {info['num_days']} days, "
              f"{info['num_assets']} assets, {info['num_features']} features, "
              f"shape={ds.data.shape}")

def test_crypto():
    print("\n=== Testing Crypto pipeline ===")
    crypto_path = create_crypto_dataset(force_recreate=True)  # Binance REST sampler
    datasets_crypto = create_split_datasets(crypto_path,
                                            proportional=True,
                                            proportions=(0.7, 0.2, 0.1))

    for split, ds in datasets_crypto.items():
        info = ds.get_split_info()
        print(f"Crypto {split}: {info['num_days']} days, "
              f"{info['num_assets']} assets, {info['num_features']} features, "
              f"shape={ds.data.shape}")

if __name__ == "__main__":
    test_sp500()
    test_crypto()
