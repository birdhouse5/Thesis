import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler

def calculate_technical_indicators_robust(df):
    """
    Calculate robust technical indicators for RL portfolio optimization
    Handles edge cases and prevents extreme values
    
    Parameters:
    df (pd.DataFrame): DataFrame with columns [date, ticker, open, high, low, close, adj_close, volume, is_active]
    
    Returns:
    pd.DataFrame: Original DataFrame with additional technical indicator columns
    """
    result_df = df.copy().sort_values(['ticker', 'date'])
    
    def calculate_indicators_for_ticker(group):
        group = group.reset_index(drop=True)
        
        # Extract OHLCV data
        open_price, high, low, close = group['open'], group['high'], group['low'], group['close']
        adj_close, volume = group['adj_close'], group['volume']
        price = adj_close  # Use adjusted close for calculations
        
        # 1. MACD (Moving Average Convergence Divergence)
        def calculate_macd(prices, fast=12, slow=26, signal=9):
            ema_fast = prices.ewm(span=fast).mean()
            ema_slow = prices.ewm(span=slow).mean()
            macd_line = ema_fast - ema_slow
            signal_line = macd_line.ewm(span=signal).mean()
            histogram = macd_line - signal_line
            return macd_line, signal_line, histogram
        
        macd, macd_signal, macd_hist = calculate_macd(price)
        group['MACD'] = macd
        group['MACD_Signal'] = macd_signal
        group['MACD_Histogram'] = macd_hist
        
        # 2. RSI (Relative Strength Index)
        def calculate_rsi(prices, period=14):
            delta = prices.diff()
            gain = delta.where(delta > 0, 0).rolling(window=period, min_periods=1).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period, min_periods=1).mean()
            rs = gain / loss.replace(0, np.nan)
            return 100 - (100 / (1 + rs))
        
        group['RSI'] = calculate_rsi(price)
        
        # 3. KDJ (Stochastic with J line) - ROBUST VERSION
        def calculate_kdj_robust(high, low, close, k_period=14, d_period=3):
            lowest_low = low.rolling(window=k_period, min_periods=1).min()
            highest_high = high.rolling(window=k_period, min_periods=1).max()
            
            # Prevent division by zero
            price_range = highest_high - lowest_low
            price_range = price_range.replace(0, np.nan)
            
            rsv = 100 * ((close - lowest_low) / price_range)
            rsv = rsv.clip(0, 100).fillna(50)  # Fill NaN with neutral value
            
            # Use simple moving average for stability
            k_percent = rsv.rolling(window=d_period, min_periods=1).mean().clip(0, 100)
            d_percent = k_percent.rolling(window=d_period, min_periods=1).mean().clip(0, 100)
            j_percent = (3 * k_percent - 2 * d_percent).clip(-50, 150)  # Allow moderate overshoot
            
            return k_percent, d_percent, j_percent
        
        k_percent, d_percent, j_percent = calculate_kdj_robust(high, low, price)
        group['KDJ_K'] = k_percent
        group['KDJ_D'] = d_percent
        group['KDJ_J'] = j_percent
        
        # 4. Williams %R - ROBUST VERSION
        def calculate_williams_r_robust(high, low, close, period=14):
            highest_high = high.rolling(window=period, min_periods=1).max()
            lowest_low = low.rolling(window=period, min_periods=1).min()
            
            price_range = highest_high - lowest_low
            price_range = price_range.replace(0, np.nan)
            
            williams_r = -100 * ((highest_high - close) / price_range)
            return williams_r.clip(-100, 0).fillna(-50)  # Fill NaN with neutral value
        
        group['WILLIAMS_R'] = calculate_williams_r_robust(high, low, price)
        
        # 5. Stochastic Fast - ROBUST VERSION
        def calculate_stochf_robust(high, low, close, k_period=5, d_period=3):
            lowest_low = low.rolling(window=k_period, min_periods=1).min()
            highest_high = high.rolling(window=k_period, min_periods=1).max()
            
            price_range = highest_high - lowest_low
            price_range = price_range.replace(0, np.nan)
            
            stochf_k = 100 * ((close - lowest_low) / price_range)
            stochf_k = stochf_k.clip(0, 100).fillna(50)
            
            stochf_d = stochf_k.rolling(window=d_period, min_periods=1).mean().clip(0, 100)
            
            return stochf_k, stochf_d
        
        stochf_k, stochf_d = calculate_stochf_robust(high, low, price)
        group['STOCHF_K'] = stochf_k
        group['STOCHF_D'] = stochf_d
        
        # 6. ATR (Average True Range)
        def calculate_atr(high, low, close, period=14):
            tr1 = high - low
            tr2 = abs(high - close.shift(1))
            tr3 = abs(low - close.shift(1))
            true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            return true_range.rolling(window=period, min_periods=1).mean()
        
        group['ATR'] = calculate_atr(high, low, price)
        
        # 7. NATR (Normalized ATR)
        atr = group['ATR']
        group['NATR'] = (100 * atr / price).clip(0, 200)  # Cap at reasonable level
        
        # 8. AROONOSC (Aroon Oscillator)
        def calculate_aroon_osc(high, low, period=14):
            def aroon_calc(series, period, is_high=True):
                result = []
                for i in range(len(series)):
                    start = max(0, i - period + 1)
                    window = series.iloc[start:i+1]
                    if len(window) == 0:
                        result.append(np.nan)
                    else:
                        if is_high:
                            periods_since = len(window) - 1 - window.argmax()
                        else:
                            periods_since = len(window) - 1 - window.argmin()
                        aroon_val = ((len(window) - periods_since) / len(window)) * 100
                        result.append(aroon_val)
                return pd.Series(result, index=series.index)
            
            aroon_up = aroon_calc(high, period, True)
            aroon_down = aroon_calc(low, period, False)
            return aroon_up - aroon_down
        
        group['AROONOSC'] = calculate_aroon_osc(high, low)
        
        # 9. CCI (Commodity Channel Index)
        def calculate_cci(high, low, close, period=20):
            typical_price = (high + low + close) / 3
            sma_tp = typical_price.rolling(window=period, min_periods=1).mean()
            mad = typical_price.rolling(window=period, min_periods=1).apply(
                lambda x: np.mean(np.abs(x - np.mean(x))) if len(x) > 0 else 0
            )
            mad = mad.replace(0, np.nan)
            cci = (typical_price - sma_tp) / (0.015 * mad)
            return cci.clip(-300, 300)  # Reasonable bounds for RL
        
        group['CCI'] = calculate_cci(high, low, price)
        
        # 10. CMO (Chande Momentum Oscillator)
        def calculate_cmo(prices, period=14):
            momentum = prices.diff()
            positive_sum = momentum.where(momentum > 0, 0).rolling(window=period, min_periods=1).sum()
            negative_sum = abs(momentum.where(momentum < 0, 0)).rolling(window=period, min_periods=1).sum()
            total_sum = positive_sum + negative_sum
            total_sum = total_sum.replace(0, np.nan)
            return (100 * (positive_sum - negative_sum) / total_sum).fillna(0)
        
        group['CMO'] = calculate_cmo(price)
        
        # 11. MFI (Money Flow Index)
        def calculate_mfi(high, low, close, volume, period=14):
            typical_price = (high + low + close) / 3
            raw_money_flow = typical_price * volume
            
            positive_flow = raw_money_flow.where(typical_price > typical_price.shift(1), 0)
            negative_flow = raw_money_flow.where(typical_price < typical_price.shift(1), 0)
            
            positive_mf = positive_flow.rolling(window=period, min_periods=1).sum()
            negative_mf = negative_flow.rolling(window=period, min_periods=1).sum()
            
            money_ratio = positive_mf / negative_mf.replace(0, np.nan)
            return 100 - (100 / (1 + money_ratio)).fillna(50)
        
        group['MFI'] = calculate_mfi(high, low, price, volume)
        
        # 12. Bollinger Bands
        def calculate_bollinger_bands(prices, period=20, std_dev=2):
            sma = prices.rolling(window=period, min_periods=1).mean()
            std = prices.rolling(window=period, min_periods=1).std()
            upper_band = sma + (std * std_dev)
            lower_band = sma - (std * std_dev)
            bb_width = ((upper_band - lower_band) / sma).fillna(0)
            bb_position = ((prices - lower_band) / (upper_band - lower_band)).clip(0, 1).fillna(0.5)
            return upper_band, lower_band, bb_width, bb_position
        
        bb_upper, bb_lower, bb_width, bb_pos = calculate_bollinger_bands(price)
        group['BB_Upper'] = bb_upper
        group['BB_Lower'] = bb_lower
        group['BB_Width'] = bb_width
        group['BB_Position'] = bb_pos
        
        # 13. VWAP (Volume Weighted Average Price)
        typical_price = (high + low + price) / 3
        group['VWAP'] = (typical_price * volume).cumsum() / volume.cumsum()
        
        # 14. ROC (Rate of Change)
        group['ROC'] = ((price / price.shift(10)) - 1) * 100
        group['ROC'] = group['ROC'].clip(-50, 100)  # Cap extreme moves
        
        return group
    
    # Apply calculations to each ticker group
    result_df = result_df.groupby('ticker', group_keys=False).apply(calculate_indicators_for_ticker)
    return result_df.reset_index(drop=True)


def normalize_for_rl_portfolio_optimization(df, normalization_method='minmax'):
    """
    Comprehensive normalization for RL portfolio optimization
    
    Parameters:
    df: DataFrame with technical indicators
    normalization_method: 'minmax', 'standard', or 'mixed'
    
    Returns:
    df_normalized: DataFrame with normalized features
    scalers: Dictionary of fitted scalers for inverse transformation
    """
    result_df = df.copy()
    scalers = {}
    
    # Define feature categories for different normalization approaches
    price_features = ['open', 'high', 'low', 'close', 'adj_close', 'BB_Upper', 'BB_Lower', 'VWAP']
    bounded_indicators = ['RSI', 'KDJ_K', 'KDJ_D', 'MFI', 'STOCHF_K', 'STOCHF_D', 'BB_Position']  # Already 0-100 or 0-1
    oscillators = ['MACD', 'MACD_Signal', 'MACD_Histogram', 'KDJ_J', 'WILLIAMS_R', 'AROONOSC', 'CCI', 'CMO', 'ROC']
    volatility_features = ['ATR', 'NATR', 'BB_Width', 'volume']
    
    def normalize_ticker_group(group):
        group = group.copy()
        
        # 1. Price features: MinMax normalization per ticker (captures relative price movements)
        for feature in price_features:
            if feature in group.columns:
                scaler = MinMaxScaler()
                values = group[[feature]].values
                group[f'{feature}_norm'] = scaler.fit_transform(values).flatten()
                scalers[f'{feature}_{group.iloc[0]["ticker"]}'] = scaler
        
        # 2. Bounded indicators: Simple division (already in known ranges)
        for feature in bounded_indicators:
            if feature in group.columns:
                if feature in ['RSI', 'KDJ_K', 'KDJ_D', 'MFI', 'STOCHF_K', 'STOCHF_D']:
                    group[f'{feature}_norm'] = group[feature] / 100.0  # Scale to [0,1]
                elif feature == 'BB_Position':
                    group[f'{feature}_norm'] = group[feature]  # Already [0,1]
                elif feature == 'WILLIAMS_R':
                    group[f'{feature}_norm'] = (group[feature] + 100) / 100.0  # Scale [-100,0] to [0,1]
        
        # 3. Oscillators: Robust standardization (handles extreme values better)
        for feature in oscillators:
            if feature in group.columns:
                # Use robust standardization (median and IQR)
                median_val = group[feature].median()
                q75, q25 = group[feature].quantile(0.75), group[feature].quantile(0.25)
                iqr = q75 - q25
                if iqr > 0:
                    group[f'{feature}_norm'] = (group[feature] - median_val) / iqr
                    group[f'{feature}_norm'] = group[f'{feature}_norm'].clip(-3, 3)  # Cap outliers
                else:
                    group[f'{feature}_norm'] = 0
        
        # 4. Volatility features: Log-transform then standardize (handles skewness)
        for feature in volatility_features:
            if feature in group.columns:
                # Log transform to handle skewness, then standardize
                log_values = np.log1p(group[feature])  # log1p handles zeros
                scaler = StandardScaler()
                group[f'{feature}_norm'] = scaler.fit_transform(log_values.values.reshape(-1, 1)).flatten()
                scalers[f'{feature}_{group.iloc[0]["ticker"]}'] = scaler
        
        return group
    
    # Apply normalization per ticker
    print("Applying ticker-specific normalization for RL...")
    result_df = result_df.groupby('ticker', group_keys=False).apply(normalize_ticker_group)
    
    # 5. Create returns for RL (essential for portfolio optimization)
    result_df['returns'] = result_df.groupby('ticker')['adj_close'].pct_change()
    result_df['log_returns'] = np.log(result_df['adj_close'] / result_df.groupby('ticker')['adj_close'].shift(1))
    
    # 6. Add market-relative features (important for portfolio optimization)
    # Calculate market return (equal-weighted average)
    market_return = result_df.groupby('date')['returns'].mean().reset_index()
    market_return.columns = ['date', 'market_return']
    result_df = result_df.merge(market_return, on='date', how='left')
    
    # Calculate excess returns (return relative to market)
    result_df['excess_returns'] = result_df['returns'] - result_df['market_return']
    
    # 7. Add rolling volatility (important for risk management in RL)
    result_df['volatility_5d'] = result_df.groupby('ticker')['returns'].rolling(window=5, min_periods=1).std().reset_index(0, drop=True)
    result_df['volatility_20d'] = result_df.groupby('ticker')['returns'].rolling(window=20, min_periods=1).std().reset_index(0, drop=True)
    
    return result_df.reset_index(drop=True), scalers


def prepare_rl_features(df_normalized):
    """
    Prepare final feature set for RL portfolio optimization
    """
    # Select normalized features for RL
    feature_columns = [col for col in df_normalized.columns if col.endswith('_norm')]
    rl_features = ['returns', 'log_returns', 'excess_returns', 'market_return', 'volatility_5d', 'volatility_20d']
    
    # Combine all features
    all_features = feature_columns + rl_features
    
    print(f"Selected {len(feature_columns)} normalized technical indicators")
    print(f"Added {len(rl_features)} RL-specific features")
    print(f"Total features for RL: {len(all_features)}")
    
    return df_normalized[['date', 'ticker'] + all_features], all_features


# Main execution pipeline
def create_rl_dataset():
    """
    Complete pipeline: Load data → Calculate indicators → Normalize → Prepare for RL
    """
    print("=== CREATING RL PORTFOLIO OPTIMIZATION DATASET ===")
    
    # 1. Load data
    print("\n1. Loading cleaned S&P 500 dataset...")
    df = pd.read_parquet("../data/cleaned_sp500_dataset.parquet")
    print(f"   Loaded {df.shape[0]:,} rows, {df.shape[1]} columns")
    print(f"   Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"   Tickers: {df['ticker'].nunique()} ({', '.join(sorted(df['ticker'].unique())[:10])}...)")
    
    # 2. Calculate technical indicators
    print("\n2. Calculating robust technical indicators...")
    df_with_indicators = calculate_technical_indicators_robust(df)
    
    # Show new indicators
    original_cols = set(df.columns)
    new_indicator_cols = [col for col in df_with_indicators.columns if col not in original_cols]
    print(f"   Added {len(new_indicator_cols)} technical indicators")
    print(f"   Indicators: {', '.join(new_indicator_cols[:10])}...")
    
    # 3. Normalize for RL
    print("\n3. Applying RL-optimized normalization...")
    df_normalized, scalers = normalize_for_rl_portfolio_optimization(df_with_indicators)
    
    # 4. Prepare final RL features
    print("\n4. Preparing final RL feature set...")
    df_rl_ready, feature_list = prepare_rl_features(df_normalized)
    
    # 5. Quality checks
    print("\n5. Quality assessment...")
    print(f"   Final dataset shape: {df_rl_ready.shape}")
    print(f"   Missing values: {df_rl_ready.isnull().sum().sum():,}")
    print(f"   Date range preserved: {df_rl_ready['date'].min()} to {df_rl_ready['date'].max()}")
    
    # Check feature distributions
    norm_features = [col for col in df_rl_ready.columns if col.endswith('_norm')]
    if norm_features:
        feature_stats = df_rl_ready[norm_features[:5]].describe()
        print(f"\n   Sample normalized feature statistics:")
        print(feature_stats.round(3))
    
    # 6. Save datasets
    print("\n6. Saving datasets...")
    df_with_indicators.to_parquet('../data/sp500_with_technical_indicators_robust.parquet')
    df_normalized.to_parquet('../data/sp500_normalized_for_rl.parquet')
    df_rl_ready.to_parquet('../data/sp500_rl_ready.parquet')
    
    # Save feature list and scalers
    import pickle
    with open('../data/rl_feature_list.pkl', 'wb') as f:
        pickle.dump(feature_list, f)
    with open('../data/rl_scalers.pkl', 'wb') as f:
        pickle.dump(scalers, f)
    
    print("   ✅ sp500_with_technical_indicators_robust.parquet")
    print("   ✅ sp500_normalized_for_rl.parquet")
    print("   ✅ sp500_rl_ready.parquet")
    print("   ✅ rl_feature_list.pkl")
    print("   ✅ rl_scalers.pkl")
    
    print("\n🎉 RL dataset creation complete!")
    print(f"\n📊 Ready for RL training with {len(feature_list)} features across {df_rl_ready['ticker'].nunique()} stocks")
    
    return df_rl_ready, feature_list, scalers


# Execute the pipeline
if __name__ == "__main__":
    rl_dataset, features, scalers = create_rl_dataset()