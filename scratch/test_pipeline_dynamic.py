import os
import json
import yfinance as yf
import sys

# Ensure working directory is added to sys.path
sys.path.append(os.path.abspath(os.path.dirname(__file__) + "/.."))

import config

def test_dynamic_reload():
    print("Initial list of tickers in config:", len(config.ALL_TICKERS))
    print("Initial sectors count:", len(config.SECTORS))
    
    # Save a temporary test ticker
    test_ticker = "ZOMATO.NS"
    test_sector = "Consumer Services" # fallback/known sector
    
    # Check if nifty500_config.json contains it or check yfinance info
    # We will write directly to custom_tickers.json for testing reload
    custom_file = config.CUSTOM_TICKERS_FILE
    
    # Read existing custom tickers
    custom_data = {"tickers": [], "sectors": {}}
    if os.path.exists(custom_file):
        try:
            with open(custom_file, "r") as f:
                custom_data = json.load(f)
        except Exception:
            pass
            
    tickers_list = custom_data.setdefault("tickers", [])
    sectors_dict = custom_data.setdefault("sectors", {})
    
    if test_ticker not in tickers_list:
        tickers_list.append(test_ticker)
        sectors_dict.setdefault(test_sector, []).append(test_ticker)
        
        with open(custom_file, "w") as f:
            json.dump(custom_data, f, indent=4)
            
        print(f"\nAdded {test_ticker} to custom_tickers.json on disk.")
    else:
        print(f"\n{test_ticker} already exists in custom_tickers.json.")
        
    # Now call config refresh
    print("Refreshing config...")
    config.refresh_config()
    
    # Verify that config.ALL_TICKERS contains ZOMATO.NS
    assert test_ticker in config.ALL_TICKERS, "Failed: Ticker not in config.ALL_TICKERS after refresh!"
    assert test_ticker in config.SECTORS[test_sector], f"Failed: Ticker not in config.SECTORS[{test_sector}] after refresh!"
    
    print("\nSUCCESS! In-place reload updated the in-memory configuration variables successfully!")
    print("Updated TICKERS length:", len(config.ALL_TICKERS))
    print(f"Sector '{test_sector}' now contains:", config.SECTORS[test_sector])

if __name__ == "__main__":
    test_dynamic_reload()
