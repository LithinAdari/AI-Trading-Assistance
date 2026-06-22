import os
import json
import time
import requests
import io
import pandas as pd
import yfinance as yf
from datetime import datetime
from config import BREAKOUT_SCAN_FILE

def download_nse_symbols():
    """
    Downloads the official list of equity symbols from the NSE archives.
    Returns a list of symbols with '.NS' suffix.
    """
    url = "https://nsearchives.nseindia.com/content/equities/EQUITY_L.csv"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/100.0.0.0 Safari/537.36"
    }
    try:
        response = requests.get(url, headers=headers, timeout=15)
        if response.status_code == 200:
            df = pd.read_csv(io.StringIO(response.text))
            df.columns = [c.strip() for c in df.columns]
            
            # Filter to keep only regular equity shares ('EQ' series)
            if 'SERIES' in df.columns:
                df = df[df['SERIES'].str.strip() == 'EQ']
                
            symbols = [f"{s.strip()}.NS" for s in df['SYMBOL'].tolist() if str(s).strip()]
            return sorted(list(set(symbols)))
    except Exception as e:
        print(f"Error downloading NSE symbols: {e}")
    return []

def run_breakout_scan(progress_callback=None, symbols_list=None):
    """
    Batch-downloads 2d daily Close prices for equities and filters for daily gain >= +10%.
    progress_callback: optional function with signature (current_chunk, total_chunks, status_msg)
    """
    if symbols_list is not None:
        symbols = symbols_list
    else:
        symbols = download_nse_symbols()
        
    if not symbols:
        if progress_callback:
            progress_callback(1, 1, "Failed to download NSE symbols list. Aborting.")
        return []
        
    chunk_size = 400
    chunks = [symbols[i:i + chunk_size] for i in range(0, len(symbols), chunk_size)]
    total_chunks = len(chunks)
    
    breakout_stocks = []
    
    for idx, chunk in enumerate(chunks):
        if progress_callback:
            progress_callback(
                idx, 
                total_chunks, 
                f"Scanning chunk {idx+1}/{total_chunks} ({len(chunk)} tickers)..."
            )
            
        try:
            # yf.download handles batch querying in parallel
            data = yf.download(chunk, period="5d", group_by='ticker', progress=False)
            
            for ticker in chunk:
                try:
                    if isinstance(data.columns, pd.MultiIndex):
                        if ticker in data.columns.get_level_values(0):
                            ticker_df = data[ticker]
                        else:
                            continue
                    else:
                        ticker_df = data
                        
                    if 'Close' in ticker_df.columns and 'Volume' in ticker_df.columns:
                        close_series = ticker_df['Close'].dropna()
                        vol_series = ticker_df['Volume'].dropna()
                        
                        if len(close_series) >= 2 and len(vol_series) >= 2:
                            prev_close = float(close_series.iloc[-2])
                            curr_close = float(close_series.iloc[-1])
                            
                            if prev_close > 0:
                                pct_change = (curr_close - prev_close) / prev_close * 100
                                
                                # Pre-breakout momentum filter: +1.5% to +7.0%
                                if 1.5 <= pct_change <= 7.0:
                                    curr_vol = int(vol_series.iloc[-1])
                                    # Calculate average volume of the previous days (up to 4 days)
                                    prev_vol_avg = vol_series.iloc[:-1].mean()
                                    
                                    # Volume surge confirmation: >= 1.5x average
                                    if prev_vol_avg > 0 and curr_vol >= 1.5 * prev_vol_avg:
                                        breakout_stocks.append({
                                            "Ticker": ticker,
                                            "Prev Close": round(prev_close, 2),
                                            "Live Price": round(curr_close, 2),
                                            "Change %": round(pct_change, 2),
                                            "Volume": curr_vol
                                        })
                except Exception:
                    pass
        except Exception as chunk_err:
            print(f"Error scanning chunk {idx+1}: {chunk_err}")
            
    if progress_callback:
        progress_callback(total_chunks, total_chunks, f"Scan finished! Found {len(breakout_stocks)} breakout stocks.")
        
    return breakout_stocks

def save_scan_results(results):
    """Saves the scan results to breakout_scan_results.json."""
    scan_data = {
        "last_scan_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "stocks": results
    }
    try:
        with open(BREAKOUT_SCAN_FILE, "w") as f:
            json.dump(scan_data, f, indent=4)
        print(f"Saved {len(results)} breakout scan results to {BREAKOUT_SCAN_FILE}")
        return True
    except Exception as e:
        print(f"Error saving breakout scan results: {e}")
    return False

def load_scan_results():
    """Loads the breakout scan results from breakout_scan_results.json."""
    if os.path.exists(BREAKOUT_SCAN_FILE):
        try:
            with open(BREAKOUT_SCAN_FILE, "r") as f:
                return json.load(f)
        except Exception as e:
            print(f"Error reading breakout scan results: {e}")
    return None
