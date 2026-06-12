import argparse
import sys
from data_pipeline import get_complete_dataset
from sentiment_analyzer import get_ticker_news_sentiment
from model_engine import train_and_predict_all
from config import TICKERS, ALL_TICKERS
from breakout_scanner import run_breakout_scan, save_scan_results

def main():
    parser = argparse.ArgumentParser(description="Orchestrator for Trading Assistant and Portfolio Monitor")
    parser.add_argument(
        "--tickers", 
        type=str, 
        help="Comma-separated list of tickers to override default watchlist"
    )
    parser.add_argument(
        "--all-sectors",
        action="store_true",
        help="Run model pipeline and prediction for ALL sector tickers and refresh breakout scans"
    )
    parser.add_argument(
        "--scan-breakouts",
        action="store_true",
        help="Scan all Indian stocks for +10%% daily breakouts and save results"
    )
    
    args = parser.parse_args()
    
    # 1. Check if we should only run the breakout scan
    if args.scan_breakouts:
        print("--- Initiating Full Market Breakout Scan (+10% Changers) ---")
        try:
            results = run_breakout_scan(progress_callback=lambda curr, tot, msg: print(f"[{curr}/{tot}] {msg}"))
            save_scan_results(results)
            print("--- Breakout Scan Completed Successfully ---")
            sys.exit(0)
        except Exception as e:
            print(f"Error running breakout scan: {e}")
            sys.exit(1)
            
    # Determine target watchlist
    if args.all_sectors:
        watchlist = ALL_TICKERS
    elif args.tickers:
        watchlist = [t.strip() for t in args.tickers.split(",") if t.strip()]
    else:
        watchlist = TICKERS
        
    print(f"--- Trading Assistant Job Initiated ---")
    print(f"Watchlist: {watchlist}")
    
    try:
        # 1. Scraping headlines and calculating sentiment in parallel
        print("\n[Step 1/3] Scraping news headlines and scoring sentiment in parallel...")
        sentiments = {}
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        def fetch_sentiment_worker(ticker):
            try:
                res = get_ticker_news_sentiment(ticker)
                return ticker, res["average_sentiment"]
            except Exception:
                return ticker, 0.0
                
        max_workers = min(20, len(watchlist))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_ticker = {executor.submit(fetch_sentiment_worker, ticker): ticker for ticker in watchlist}
            for future in as_completed(future_to_ticker):
                ticker, score = future.result()
                sentiments[ticker] = score
                print(f"-> {ticker} average sentiment: {score:+.3f}")
                
        # 2. Fetching technical historical data and macro variables
        print("\n[Step 2/3] Fetching historical price bins and macro factors...")
        datasets = get_complete_dataset(tickers=watchlist)
        
        if not datasets:
            print("Error: No datasets downloaded. Aborting pipeline.")
            sys.exit(1)
            
        # 3. Model training & prediction output
        print("\n[Step 3/3] Training sector models and exporting execution brackets...")
        recs = train_and_predict_all(datasets, sentiments)
        
        # 4. If running all sectors, also run the daily breakout scan for full automated updates
        if args.all_sectors:
            print("\n[Step 4/4] Refreshing daily breakout scan list (+10% stocks)...")
            try:
                results = run_breakout_scan(progress_callback=lambda curr, tot, msg: print(f"[{curr}/{tot}] {msg}"))
                save_scan_results(results)
            except Exception as e:
                print(f"Error during automated breakout scan: {e}")
                
        print("\n--- Pipeline Completed Successfully ---")
        print(f"Saved forecasts for {len(recs)} symbols.")
        sys.exit(0)
        
    except Exception as e:
        print(f"\nPipeline Job Failed with Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
