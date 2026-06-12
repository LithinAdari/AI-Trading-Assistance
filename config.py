import os

import json
import urllib.request
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
NIFTY500_CACHE_FILE = os.path.join(BASE_DIR, "nifty500_config.json")
CUSTOM_TICKERS_FILE = os.path.join(BASE_DIR, "custom_tickers.json")

def load_nifty500_config():
    """
    Downloads Nifty 500 from NSE, groups by Industry (Sectors), and caches locally.
    Returns: (tickers_list, sectors_dict)
    """
    url = "https://archives.nseindia.com/content/indices/ind_nifty500list.csv"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    try:
        # Try fetching from NSE India
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=10) as response:
            df = pd.read_csv(response)
        
        tickers = []
        sectors = {}
        for _, row in df.iterrows():
            sym = str(row['Symbol']).strip()
            ind = str(row['Industry']).strip()
            if not sym or not ind:
                continue
            t = f"{sym}.NS"
            tickers.append(t)
            sectors.setdefault(ind, []).append(t)
            
        tickers = sorted(list(set(tickers)))
        
        # Save to cache file
        config_data = {
            "tickers": tickers,
            "sectors": sectors
        }
        with open(NIFTY500_CACHE_FILE, "w") as f:
            json.dump(config_data, f, indent=4)
            
        print("Successfully updated Nifty 500 configurations from NSE India.")
        return tickers, sectors
    except Exception as e:
        print(f"Warning: Failed to fetch online Nifty 500 list ({e}). Loading cached fallback...")
        if os.path.exists(NIFTY500_CACHE_FILE):
            try:
                with open(NIFTY500_CACHE_FILE, "r") as f:
                    config_data = json.load(f)
                return config_data["tickers"], config_data["sectors"]
            except Exception as e2:
                print(f"Error reading Nifty 500 cache file: {e2}")
        
        # Absolute backup default tickers if cache is missing/corrupted
        backup_tickers = [
            "HAL.NS", "BEL.NS", "ASTRAMICRO.NS", "HDFCBANK.NS", "ICICIBANK.NS", 
            "AXISBANK.NS", "TATAPOWER.NS", "RELIANCE.NS", "BOSCHLTD.NS", "OLAELEC.NS", 
            "TCS.NS", "INFY.NS", "CYIENT.NS", "ITC.NS", "HINDUNILVR.NS", "NESTLEIND.NS"
        ]
        backup_sectors = {
            "Industrial Manufacturing": ["HAL.NS", "BEL.NS", "ASTRAMICRO.NS", "BOSCHLTD.NS"],
            "Financial Services": ["HDFCBANK.NS", "ICICIBANK.NS", "AXISBANK.NS"],
            "Energy & Power": ["TATAPOWER.NS", "RELIANCE.NS", "OLAELEC.NS"],
            "Information Technology": ["TCS.NS", "INFY.NS", "CYIENT.NS"],
            "Consumer Goods": ["ITC.NS", "HINDUNILVR.NS", "NESTLEIND.NS"]
        }
        return backup_tickers, backup_sectors

ALL_TICKERS = []
SECTORS = {}
TICKERS = ALL_TICKERS

def load_all_tickers_and_sectors():
    """
    Loads both online/cached Nifty 500 configuration and local custom tickers database.
    Returns: (tickers_list, sectors_dict)
    """
    tickers, sectors = load_nifty500_config()
    
    # Load custom tickers and merge
    if os.path.exists(CUSTOM_TICKERS_FILE):
        try:
            with open(CUSTOM_TICKERS_FILE, "r") as f:
                custom_data = json.load(f)
            
            c_tickers = custom_data.get("tickers", [])
            c_sectors = custom_data.get("sectors", {})
            
            # Merge tickers
            tickers = sorted(list(set(tickers + c_tickers)))
            
            # Merge sectors
            for sector_name, tickers_in_sector in c_sectors.items():
                existing_list = sectors.setdefault(sector_name, [])
                sectors[sector_name] = sorted(list(set(existing_list + tickers_in_sector)))
        except Exception as e:
            print(f"Error loading custom tickers from database: {e}")
            
    return tickers, sectors

def refresh_config():
    """
    Reloads configurations from NSE/cache and custom database,
    and updates global variables in-place.
    """
    global ALL_TICKERS, SECTORS
    new_tickers, new_sectors = load_all_tickers_and_sectors()
    
    # Mutate in-place
    ALL_TICKERS.clear()
    ALL_TICKERS.extend(new_tickers)
    
    SECTORS.clear()
    SECTORS.update(new_sectors)
    
    print(f"Configuration refreshed. Total tickers: {len(ALL_TICKERS)}, Total sectors: {len(SECTORS)}")

# Run initial load
refresh_config()



# Macro indicators (yfinance & FRED proxies)
MACRO_TICKERS = {
    "Crude_Oil": "CL=F"  # WTI Crude Oil Futures (yfinance)
}
FRED_MACRO_SERIES = {
    "Crude_Oil_FRED": "DCOILWTICO"  # WTI Crude Oil Spot (FRED)
}

# Machine learning parameters
TRAIN_HORIZON_DAYS = 5
TRAIN_YEARS = 4  # Pull 4 years of historical data

MODEL_PARAMS = {
    "classifier": {
        "n_estimators": 100,
        "max_depth": 3,
        "learning_rate": 0.05,
        "random_state": 42,
        "eval_metric": "logloss"
    },
    "regressor": {
        "n_estimators": 80,
        "max_depth": 3,
        "learning_rate": 0.03,
        "random_state": 42,
        "reg_alpha": 2.0,       # L1 regularization
        "reg_lambda": 10.0,     # L2 regularization
        "subsample": 0.8,
        "colsample_bytree": 0.8
    }
}

# Path configurations
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
PORTFOLIO_FILE = os.path.join(BASE_DIR, "portfolio_holdings.json")
RECOMMENDATIONS_FILE = os.path.join(BASE_DIR, "recommendations.json")
VALIDATION_ERRORS_FILE = os.path.join(BASE_DIR, "validation_errors.json")
BREAKOUT_SCAN_FILE = os.path.join(BASE_DIR, "breakout_scan_results.json")

# Ensure models directory exists
os.makedirs(MODELS_DIR, exist_ok=True)

# Brokerage Rate Parameters for Indian Equity Delivery
BROKER_CONFIGS = {
    "Zerodha": {
        "brokerage_percent": 0.0,
        "brokerage_flat": 0.0,
        "brokerage_max": 0.0,
        "stt_percent": 0.1,  # 0.1% on buy & sell
        "exchange_txn_percent": 0.00322,  # NSE equity delivery: 0.00322%
        "sebi_turnover_percent": 0.0001,  # Rs 10 per crore (0.0001%)
        "stamp_duty_percent": 0.015,  # 0.015% on buy only
        "gst_percent": 18.0  # 18% on (brokerage + exchange transaction charge + sebi turnover fee)
    },
    "Groww": {
        "brokerage_percent": 0.05,  # 0.05% per order
        "brokerage_flat": 0.0,
        "brokerage_max": 20.0,  # Max Rs 20 per order
        "stt_percent": 0.1,
        "exchange_txn_percent": 0.00322,
        "sebi_turnover_percent": 0.0001,
        "stamp_duty_percent": 0.015,
        "gst_percent": 18.0
    },
    "Angel One": {
        "brokerage_percent": 0.0,
        "brokerage_flat": 0.0,
        "brokerage_max": 0.0,
        "stt_percent": 0.1,
        "exchange_txn_percent": 0.00322,
        "sebi_turnover_percent": 0.0001,
        "stamp_duty_percent": 0.015,
        "gst_percent": 18.0
    },
    "Custom (Flat 0.25%)": {
        "brokerage_percent": 0.25,
        "brokerage_flat": 0.0,
        "brokerage_max": 999999.0,
        "stt_percent": 0.0,
        "exchange_txn_percent": 0.0,
        "sebi_turnover_percent": 0.0,
        "stamp_duty_percent": 0.0,
        "gst_percent": 0.0
    }
}
# Trigger file reload for Streamlit watcher
