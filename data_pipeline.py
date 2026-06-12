import yfinance as yf
import pandas as pd
import numpy as np
import datetime
import os
from config import TICKERS, MACRO_TICKERS, FRED_MACRO_SERIES, TRAIN_YEARS

def calculate_rsi(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate the Relative Strength Index (RSI)."""
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0.0)).copy()
    loss = (-delta.where(delta < 0, 0.0)).copy()
    
    # Use exponential moving average for smoothing
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
    
    # Avoid division by zero
    rs = avg_gain / avg_loss.replace(0.0, 1e-9)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate the Average True Range (ATR)."""
    high_low = df['High'] - df['Low']
    high_close = (df['High'] - df['Close'].shift()).abs()
    low_close = (df['Low'] - df['Close'].shift()).abs()
    
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    
    # Simple moving average of True Range
    atr = true_range.rolling(window=period, min_periods=1).mean()
    return atr

def fetch_macro_series(start_date: datetime.date, end_date: datetime.date) -> pd.DataFrame:
    """
    Fetch macro economic series (Crude Oil) from yfinance futures CL=F,
    falling back to FRED (DCOILWTICO) if yfinance is unavailable.
    """
    macro_df = pd.DataFrame()
    
    # 1. Try yfinance CL=F (WTI Crude Futures)
    try:
        oil_ticker = MACRO_TICKERS["Crude_Oil"]
        oil_data = yf.download(oil_ticker, start=start_date, end=end_date, progress=False)
        if not oil_data.empty:
            # Handle MultiIndex columns if any
            if isinstance(oil_data.columns, pd.MultiIndex):
                oil_close = oil_data['Close'][oil_ticker]
            else:
                oil_close = oil_data['Close']
            
            macro_df['Crude_Oil'] = oil_close
            print("Successfully fetched WTI Crude Futures from yfinance.")
            return macro_df
    except Exception as e:
        print(f"yfinance macro fetch failed: {e}. Falling back to FRED...")

    # 2. Try FRED (DCOILWTICO)
    try:
        fred_code = FRED_MACRO_SERIES["Crude_Oil_FRED"]
        fred_data = web.DataReader(fred_code, "fred", start_date, end_date)
        if not fred_data.empty:
            macro_df['Crude_Oil'] = fred_data[fred_code]
            print("Successfully fetched WTI Crude Spot from FRED.")
            return macro_df
    except Exception as e:
        print(f"FRED macro fetch failed: {e}. Creating fallback mock macro series...")
        
    # 3. Fallback mock macro series (just 75.0 constant)
    dates = pd.date_range(start=start_date, end=end_date)
    macro_df = pd.DataFrame(index=dates)
    macro_df['Crude_Oil'] = 75.0
    print("Mock macro data generated as safety fallback.")
    return macro_df

def build_ticker_features(ticker: str, start_date: datetime.date, end_date: datetime.date, macro_df: pd.DataFrame) -> pd.DataFrame:
    """
    Download ticker data, compute technical indicators, and merge macro indicators
    to construct the full feature matrix.
    """
    print(f"Processing features for {ticker}...")
    df = yf.download(ticker, start=start_date, end=end_date, progress=False)
    if df.empty:
        raise ValueError(f"No price data returned for ticker {ticker}")
        
    # Handle MultiIndex columns in yfinance (usually happens in multi-ticker downloads or newer versions)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Patch today's live session data if history download returns NaNs
    try:
        if not df.empty:
            last_idx = df.index[-1]
            if df.loc[last_idx].isna().any():
                today_df = yf.download(ticker, period="1d", progress=False)
                if isinstance(today_df.columns, pd.MultiIndex):
                    today_df.columns = today_df.columns.get_level_values(0)
                if not today_df.empty:
                    today_idx = today_df.index[-1]
                    if last_idx == today_idx:
                        for col in ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']:
                            if col in df.columns and col in today_df.columns:
                                val = today_df.loc[today_idx, col]
                                if pd.notna(val):
                                    df.loc[last_idx, col] = float(val)
    except Exception as e:
        print(f"Error patching live data for pipeline on {ticker}: {e}")

    # Core Technical indicators
    df['RSI_14'] = calculate_rsi(df, period=14)
    df['ATR_14'] = calculate_atr(df, period=14)
    df['ATR_14_Pct'] = df['ATR_14'] / df['Close']
    
    # SMAs
    df['SMA_50'] = df['Close'].rolling(window=50, min_periods=1).mean()
    df['SMA_200'] = df['Close'].rolling(window=200, min_periods=1).mean()
    
    # Ratios (Structural Features)
    df['Close_to_SMA_50'] = df['Close'] / df['SMA_50']
    df['SMA_50_to_200'] = df['SMA_50'] / df['SMA_200']
    
    # Volatility
    df['Return_Daily'] = df['Close'].pct_change()
    df['Volatility_30'] = df['Return_Daily'].rolling(window=30, min_periods=1).std()

    # Join Macro Data
    # Convert indexes to datetime to ensure alignment
    df.index = pd.to_datetime(df.index)
    macro_df.index = pd.to_datetime(macro_df.index)
    
    df = df.join(macro_df, how='left')
    # Forward-fill macro data on market holidays/weekends
    df['Crude_Oil'] = df['Crude_Oil'].ffill().bfill()
    
    # Rolling Z-score normalization for Crude Oil (using a 60-day short-term or 252-day long-term window)
    # Using 120-day window to balance history requirements vs speed
    window = 120
    rolling_mean = df['Crude_Oil'].rolling(window=window, min_periods=30).mean()
    rolling_std = df['Crude_Oil'].rolling(window=window, min_periods=30).std().replace(0.0, 1e-9)
    df['Crude_Oil_Z'] = (df['Crude_Oil'] - rolling_mean) / rolling_std
    df['Crude_Oil_Z'] = df['Crude_Oil_Z'].ffill().fillna(0.0)

    # Drop intermediate column to keep feature set clean
    df.drop(columns=['Return_Daily'], inplace=True)
    
    # Fill remaining NaNs from moving averages gracefully
    df.ffill().bfill(inplace=True)
    
    return df

def get_complete_dataset(tickers=TICKERS) -> dict:
    """
    Downloads and compiles technical and macro features for all tickers in parallel.
    Returns a dictionary of DataFrames keyed by stock ticker.
    """
    end_date = datetime.date.today()
    start_date = end_date - datetime.timedelta(days=int(365 * TRAIN_YEARS))
    
    print(f"Ingesting data from {start_date} to {end_date}...")
    macro_df = fetch_macro_series(start_date, end_date)
    
    datasets = {}
    from concurrent.futures import ThreadPoolExecutor, as_completed
    
    def process_ticker(ticker):
        try:
            df = build_ticker_features(ticker, start_date, end_date, macro_df)
            return ticker, df
        except Exception as e:
            print(f"Error building features for {ticker}: {e}")
            return ticker, None

    max_workers = min(15, len(tickers))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_ticker = {executor.submit(process_ticker, ticker): ticker for ticker in tickers}
        for future in as_completed(future_to_ticker):
            ticker, df = future.result()
            if df is not None:
                datasets[ticker] = df
                
    return datasets

if __name__ == "__main__":
    # Test data pipeline
    print("Testing Technical & Macro Data Pipeline...")
    data = get_complete_dataset(tickers=["TATAPOWER.NS"])
    if "TATAPOWER.NS" in data:
        df = data["TATAPOWER.NS"]
        print("Success! Dataset shape:", df.shape)
        print("Columns:")
        print(df.columns.tolist())
        print("\nLatest data point:")
        print(df.tail(1)[['Close', 'RSI_14', 'ATR_14', 'Close_to_SMA_50', 'Crude_Oil', 'Crude_Oil_Z']])
    else:
        print("Failed to build dataset.")
