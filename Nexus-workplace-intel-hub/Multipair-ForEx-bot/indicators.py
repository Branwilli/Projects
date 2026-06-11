import numpy as np
import pandas as pd


#df = pd.read_csv('GBPUSD1.csv', sep='\t', header=None)

def standardize_df(df):
    """
    Standardizes raw financial data (e.g., from MetaTrader 5) into a uniform, 
    clean 15-minute OHLCV candle format optimized for model training and live prediction.
    """
    df = df.copy() # Avoid modifying original DataFrame

    # Normalize column names if 'time' is present (MetaTrader 5 format)
    if "time" in df.columns:
        df = df.rename(columns={
            'time': 'datetime',         # Standardize to 'datetime'
            'tick_volume': 'volume'     # Standardize to 'volume'
        })

    # Dynamic column selection to ensure we only keep relevant columns and avoid KeyErrors
    required_cols = ["datetime", "open", "high", "low", "close", "volume"]
    df = df[[col for col in required_cols if col in df.columns]]

    # Dimension handling: If 'datetime' is a DataFrame (e.g., from multi-index), flatten it
    if isinstance(df["datetime"], pd.DataFrame):
        df["datetime"] = df["datetime"].iloc[:, 0]

    # Convert 'datetime' to pandas datetime format, handling various formats and errors
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True, unit='s', errors='coerce')
    
    df = df.dropna(subset=["datetime"])

    df = df.set_index("datetime").sort_index()

    df = df.loc[df.index >= df.index.max() - pd.Timedelta(days=365)]

    # Resample to 15-minute intervals, ensuring we have a consistent OHLCV format
    df = df.resample('15min').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()

    return df

def compute_rsi(series, period=14):
    """
    Calculates the Relative Strength Index (RSI) for a given pandas Series 
    using J. Welles Wilder's Exponential Moving Average (EMA) smoothing method.
    """
    delta = series.diff()   # Calculate price changes
    gain = delta.clip(lower=0).ewm(alpha=1/period, adjust=False).mean() 
    loss = -delta.clip(upper=0).ewm(alpha=1/period, adjust=False).mean() 
    rs = gain / (loss + 1e-9)   # Calculate Relative Strength, adding small epsilon to avoid division by zero
    
    return 100 - (100 / (1 + rs))   # Convert to RSI scale (0-100)

def add_features(df):
    """
    Advanced feature engineering pipeline for a 15-minute Forex trading dataframe.
    Generates technical, momentum, volatility, multi-timeframe, and interaction features.
    """

    # Structrual Sanity Checks
    if not isinstance(df, pd.DataFrame):
        raise ValueError(f"Expected DataFrame, got {type(df)}")

    if 'close' not in df.columns:
        raise ValueError("Missing 'close' column")

    #Returns 
    df['ret_1'] = df['close'].pct_change(1)
    df['ret_5'] = df['close'].pct_change(5)

    #Volatility
    df['volatility'] = df['ret_1'].ewm(span=20, adjust=False).std()
    df['volatility_spike'] = df['volatility'] / df['volatility'].ewm(span=50, adjust=False).mean()

    #Volume
    df['volume_spike'] = df['close'].pct_change().abs() / df['close'].pct_change().abs().ewm(span=20).mean()

    #Range
    df['range'] = df['high'] - df['low']
    df['range_expansion'] = df['range'] / df['range'].ewm(span=20, adjust=False).mean()

    #Breakouts
    df['rolling_high'] = df['high'].cummax()
    df['rolling_low'] = df['low'].cummin()

    df['high_break'] = (df['close'] > df['rolling_high'].shift(1)).astype(int)
    df['low_break'] = (df['close'] < df['rolling_low'].shift(1)).astype(int)

    #Moving Averages
    df['ma_10'] = df['close'].ewm(span=10, adjust=False).mean()
    df['ma_20'] = df['close'].ewm(span=20, adjust=False).mean()
    df['trend_strength'] = df['ma_10'] - df['ma_20']

    #Momentum 
    df['momentum_5'] = df['close'].pct_change(5)

    #RSI
    df['RSI'] = compute_rsi(df['close'])

    #Multi-timeframe 
    df_1h = df.resample('1h').last()
    df['hourly_trend'] = df_1h['close'].ewm(span=20, adjust=False).mean().reindex(df.index, method='ffill')
    
    df_daily = df.resample('1D').last()
    df['daily_trend'] = df_daily['close'].ewm(span=50).mean().reindex(df.index, method='ffill')

    df['htf_bias'] = (df['close'] > df['daily_trend']).astype(int)
    df['htf_bias_hourly'] = (df['close'] > df['hourly_trend']).astype(int)
    
    #Opportunity Interactions 
    df['vol_trend'] = df['volatility'] * df['trend_strength']
    df['vol_spike_strength'] = df['volatility_spike'] * df['range_expansion']
    df['volume_vol'] = df['volume_spike'] * df['volatility']
    df['expansion_strength'] = df['range_expansion'] * df['trend_strength']

    #Direction Interactions 
    df['rsi_mom'] = df['RSI'] * df['momentum_5']
    df['trend_bias'] = df['trend_strength'] * df['htf_bias']
    df['multi_tf_trend'] = df['hourly_trend'] * df['daily_trend']
    df['break_direction'] = df['high_break'] - df['low_break']

    # Bullish/Bearish Interactions
    df['bullish_signal'] = ((df['trend_strength'] > 0).astype(int) * (df['momentum_5'] > 0).astype(int) * (df['RSI'] > 50).astype(int))

    df['bearish_signal'] = ((df['trend_strength'] < 0).astype(int) * (df['momentum_5'] < 0).astype(int) * (df['RSI'] < 50).astype(int))

    df['bullish_rebound'] = ((df['hourly_trend'] > 0) & (df['RSI'] < 30) & (df['momentum_5'] > 0)).astype(int)
    df['bearish_rebound'] = ((df['hourly_trend'] < 0) & (df['RSI'] > 70) & (df['momentum_5'] < 0)).astype(int)

    df['bullish_continuation'] = ((df['hourly_trend'] > 0) & (df['momentum_5'] > 0) & (df['high_break'] == 1)).astype(int)
    df['bearish_continuation'] = ((df['hourly_trend'] < 0) & (df['momentum_5'] < 0) & (df['low_break'] == 1)).astype(int)

    #Hybrid Features 
    df['trend_vol_mom'] = df['trend_strength'] * df['volatility'] * df['momentum_5']
    df['rsi_trend'] = df['RSI'] * df['trend_strength']
    
    df['trend_norm'] = df['trend_strength'] / (df['volatility'] + 1e-9)
    df['rsi_centered'] = df['RSI'] - 50
    df['momentum_sign'] = np.sign(df['momentum_5'])

    df['trend_rsi'] = df['trend_norm'] * df['rsi_centered']
    df['trend_mom'] = df['trend_norm'] * df['momentum_5']
    df['direction_strength'] = df['trend_strength'] * df['momentum_5']

    df = df.replace([np.inf, -np.inf], np.nan) # Replace infinities with NaN before filling
    df = df.ffill().bfill() # Forward-fill then back-fill to handle any remaining NaN values from feature calculations

    return df

def filter_data(df):
    """
    Filters out quiet or range-bound market periods, keeping only periods 
    where short-term momentum significantly exceeds recent historical volatility.
    """
    trend_std = df['trend_strength'].rolling(50).std()  # Calculate rolling standard deviation of trend strength
    mask = abs(df['trend_strength']) > trend_std    # Keep rows where absolute trend strength exceeds its recent volatility

    return df[mask]

def add_targets(df):
    """
    Generates multi-stage labels for Reinforcement Learning or Supervised models.
    Stage 1 (Opportunity): Predicts whether a volatility breakout will occur.
    Stage 2 (Direction): Predicts whether a price move will trend Up (+1) or Down (0).
    """
    df = df.copy()

    future_return = df['close'].shift(-1) /df['close'] - 1  # Extract next candle return as target variable

    threshold = abs(future_return).quantile(0.70)   # Set threshold for significant moves (e.g., top 30% of returns)
    
    #Stage 1: Opportunity
    cond = (
        (df['trend_strength'] > df['trend_strength'].ewm(span=50).mean()).astype(int) +
        (df['volatility'] > df['volatility'].ewm(span=50).mean()).astype(int) +
        (df['vol_trend'] > df['vol_trend'].ewm(span=50).mean()).astype(int) +
        (df['momentum_5'].abs() > df['momentum_5'].ewm(span=50).mean()).astype(int) 
    )

    df['target_opportunity'] = (
        (abs(future_return) > threshold) &
        (cond >= 2)
    ).astype(int)

    #Stage 2: Direction
    direction_cond = (
        (df['volatility_spike'] > df['volatility_spike'].ewm(span=50).mean()).astype(int) +
        (df['volume_spike'] > df['volume_spike'].ewm(span=50).mean()).astype(int) +
        (df['trend_strength'].abs() > df['trend_strength'].ewm(span=50).mean()).astype(int)
    )
    
    df['target_direction'] = np.where(
        (future_return > 0) & (direction_cond >= 1), 1,
        np.where(
            (future_return < 0) & (direction_cond >= 1), 0,
            np.nan
        )
    )
    return df

def get_strong_feature(df):
    targets = ['target_opportunity', 'target_direction', 'open', 'high', 'low', 'close']
    features = [col for col in df.columns if col not in targets]

    corr_matrix = df.corr()
    threshold = 0.01  

    features_to_keep = set()

    for target in targets:
        target_corr = corr_matrix[target].abs()
        
        relevant = target_corr[features][target_corr > threshold].index.tolist()
        features_to_keep.update(relevant)

    df = df[list(features_to_keep) + targets]
    return df

def smooth_labels(y, smoothing=0.1):
    """
    Applies Label Smoothing regularization to binary targets (0 or 1).
    Prevents the neural network from becoming overconfident in its predictions
    """
    return y * (1-smoothing) + 0.5 * smoothing 

#------------------------- For live data -------------------------------

def update_features(df):
    """
    Performs real-time, iterative feature engineering on the latest single 
    streaming candle inside a live trading loop. Mimics batch operations without lookahead bias.
    """
    i = -1
    idx = df.index[i]

    # RETURNS
    close = df['close']
    df.loc[idx, 'ret_1'] = close.iloc[i] / close.iloc[i-1] - 1
    df.loc[idx, 'ret_5'] = close.iloc[i] / close.iloc[i-5] - 1
    df.loc[idx, 'momentum_5'] = df.loc[idx, 'ret_5']

    # EMA HELPER
    def ema(prev, new, span):
        alpha = 2 / (span + 1)
        return alpha * new + (1 - alpha) * prev

    # MOVING AVERAGES
    prev_ma10 = df['ma_10'].iloc[i-1]
    prev_ma20 = df['ma_20'].iloc[i-1]

    price = close.iloc[i]
    prev_hourly_trend = df['hourly_trend'].iloc[i-1]
    prev_daily_trend = df['daily_trend'].iloc[i-1]

    df.loc[idx, 'ma_10'] = ema(prev_ma10, price, 10)
    df.loc[idx, 'ma_20'] = ema(prev_ma20, price, 20)

    df.loc[idx, 'trend_strength'] = (df.loc[idx, 'ma_10'] - df.loc[idx, 'ma_20'])

    # VOLATILITY (EWM STD APPROX)
    prev_vol = df['volatility'].iloc[i-1]
    ret = df.loc[idx, 'ret_1']

    alpha_vol = 2 / (20 + 1)

    # EWM variance update
    prev_var = prev_vol ** 2
    new_var = (1 - alpha_vol) * prev_var + alpha_vol * (ret ** 2)
    df.loc[idx, 'volatility'] = np.sqrt(new_var)

    # Volatility spike
    prev_vol_mean = df['volatility'].iloc[i-1]
    df.loc[idx, 'volatility_spike'] = df.loc[idx, 'volatility'] / (prev_vol_mean + 1e-9)

    # VOLUME
    prev_vol_ema = df['volume'].iloc[i-1]
    df.loc[idx, 'volume_spike'] = df['volume'].iloc[i] / (prev_vol_ema + 1e-9)

    # RANGE
    high = df['high'].iloc[i]
    low = df['low'].iloc[i]

    df.loc[idx, 'range'] = high - low

    prev_range_ema = df['range'].iloc[i-1]
    df.loc[idx, 'range_expansion'] = df.loc[idx, 'range'] / (prev_range_ema + 1e-9)

    # ROLLING HIGH/LOW
    prev_high = df['rolling_high'].iloc[i-1]
    prev_low = df['rolling_low'].iloc[i-1]

    df.loc[idx, 'rolling_high'] = max(prev_high, high)
    df.loc[idx, 'rolling_low'] = min(prev_low, low)

    df.loc[idx, 'high_break'] = int(price > prev_high)
    df.loc[idx, 'low_break'] = int(price < prev_low)

    # RSI (WINDOWED)
    window = 14
    closes = close.iloc[-window-1:]

    delta = closes.diff()
    gain = delta.clip(lower=0).mean()
    loss = -delta.clip(upper=0).mean()

    rs = gain / (loss + 1e-9)
    df.loc[idx, 'RSI'] = 100 - (100 / (1 + rs))

    # HTF (CARRY FORWARD)

    df.loc[idx, 'hourly_trend'] = ema(prev_hourly_trend, price, span=20)
    df.loc[idx, 'daily_trend'] = ema(prev_daily_trend, price, span=50)

    df.loc[idx, 'htf_bias'] = int(price > df.loc[idx, 'daily_trend'])
    df.loc[idx, 'htf_bias_hourly'] = int(price > df.loc[idx, 'hourly_trend'])

    # INTERACTIONS (OPPORTUNITY)
    df.loc[idx, 'vol_trend'] = (df.loc[idx, 'volatility'] * df.loc[idx, 'trend_strength'])
    df.loc[idx, 'vol_spike_strength'] = (df.loc[idx, 'volatility_spike'] * df.loc[idx, 'range_expansion'])
    
    df.loc[idx, 'volume_vol'] = (df.loc[idx, 'volume_spike'] * df.loc[idx, 'volatility'])
    df.loc[idx, 'expansion_strength'] = (df.loc[idx, 'range_expansion'] * df.loc[idx, 'trend_strength'])

    # INTERACTIONS (DIRECTION)
    df.loc[idx, 'rsi_mom'] = (df.loc[idx, 'RSI'] * df.loc[idx, 'momentum_5'])
    df.loc[idx, 'trend_bias'] = (df.loc[idx, 'trend_strength'] * df.loc[idx, 'htf_bias'])

    df.loc[idx, 'multi_tf_trend'] = (df.loc[idx, 'hourly_trend'] * df.loc[idx, 'daily_trend'])
    df.loc[idx, 'break_direction'] = (df.loc[idx, 'high_break'] - df.loc[idx, 'low_break'])

    # Bullish/Bearish Interactions
    df.loc[idx, 'bullish_signal'] = ((df.loc[idx, 'trend_strength'] > 0).astype(int) * (df.loc[idx, 'momentum_5'] > 0).astype(int) * (df.loc[idx, 'RSI'] > 50).astype(int))

    df.loc[idx, 'bearish_signal'] = ((df.loc[idx, 'trend_strength'] < 0).astype(int) * (df.loc[idx, 'momentum_5'] < 0).astype(int) * (df.loc[idx, 'RSI'] < 50).astype(int))

    df.loc[idx, 'bullish_rebound'] = ((df.loc[idx, 'hourly_trend'] > 0) & (df.loc[idx, 'RSI'] < 30) & (df.loc[idx, 'momentum_5'] > 0)).astype(int)
    df.loc[idx, 'bearish_rebound'] = ((df.loc[idx, 'hourly_trend'] < 0) & (df.loc[idx, 'RSI'] > 70) & (df.loc[idx, 'momentum_5'] < 0)).astype(int)

    df.loc[idx, 'bullish_continuation'] = ((df.loc[idx, 'hourly_trend'] > 0) & (df.loc[idx, 'momentum_5'] > 0) & (df.loc[idx, 'high_break'] == 1)).astype(int)
    df.loc[idx, 'bearish_continuation'] = ((df.loc[idx, 'hourly_trend'] < 0) & (df.loc[idx, 'momentum_5'] < 0) & (df.loc[idx, 'low_break'] == 1)).astype(int)

    # HYBRID FEATURES
    df.loc[idx, 'trend_vol_mom'] = (
        df.loc[idx, 'trend_strength'] *
        df.loc[idx, 'volatility'] *
        df.loc[idx, 'momentum_5']
    )

    df.loc[idx, 'rsi_trend'] = (df.loc[idx, 'RSI'] * df.loc[idx, 'trend_strength'])
    df.loc[idx, 'trend_norm'] = (df.loc[idx, 'trend_strength'] /(df.loc[idx, 'volatility'] + 1e-9))

    df.loc[idx, 'rsi_centered'] = df.loc[idx, 'RSI'] - 50
    df.loc[idx, 'momentum_sign'] = np.sign(df.loc[idx, 'momentum_5'])

    df.loc[idx, 'trend_rsi'] = (df.loc[idx, 'trend_norm'] * df.loc[idx, 'rsi_centered'])
    df.loc[idx, 'trend_mom'] = (df.loc[idx, 'trend_norm'] * df.loc[idx, 'momentum_5'])

    df.loc[idx, 'direction_strength'] = (df.loc[idx, 'trend_strength'] * df.loc[idx, 'momentum_5'])

    df.loc[idx, 'sentiment_score'] = df['sentiment_score'].iloc[i-1]
    # FINAL CLEANUP
    df.loc[idx] = df.loc[idx].replace([np.inf, -np.inf], 0)
    df.loc[idx] = df.loc[idx].fillna(0)

    return df

def append_new_candle(df, new_row):
    """
    Concatenates a newly closed streaming candle into the primary historical DataFrame
    and triggers an iterative feature update exclusively for that new row.
    """
    df = pd.concat([df, new_row]).sort_index()  # Append new row and ensure index is sorted

    # Only perform feature update if we have enough data to calculate features (e.g., at least 50 rows for rolling calculations)
    if len(df) > 50:
        df = update_features(df)

    return df