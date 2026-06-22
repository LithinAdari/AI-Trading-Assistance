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
        print("--- Initiating Predictive Breakout Scan (Pre-Breakout Filter -> ML Prediction) ---")
        try:
            target_symbols = None
            if args.tickers:
                target_symbols = [t.strip() for t in args.tickers.split(",") if t.strip()]
                print(f"Targeting custom list of {len(target_symbols)} tickers.")
                
            # Phase 1: Filter candidates
            candidates = run_breakout_scan(progress_callback=lambda curr, tot, msg: print(f"[{curr}/{tot}] {msg}"), symbols_list=target_symbols)
            
            if not candidates:
                print("No pre-breakout candidates found. Exiting.")
                save_scan_results([])
                sys.exit(0)
                
            candidate_tickers = [c["Ticker"] for c in candidates]
            print(f"--- Phase 2: Running ML Prediction Pipeline on {len(candidate_tickers)} candidates ---")
            
            # Get sentiments
            sentiments = {t: 0.0 for t in candidate_tickers}
            from concurrent.futures import ThreadPoolExecutor, as_completed
            def fetch_sentiment_worker(ticker):
                try:
                    res = get_ticker_news_sentiment(ticker)
                    return ticker, res["average_sentiment"]
                except Exception:
                    return ticker, 0.0
                    
            max_workers = min(20, len(candidate_tickers))
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_ticker = {executor.submit(fetch_sentiment_worker, ticker): ticker for ticker in candidate_tickers}
                for future in as_completed(future_to_ticker):
                    ticker, score = future.result()
                    sentiments[ticker] = score
                    
            # Run ML Pipeline
            datasets = get_complete_dataset(tickers=candidate_tickers)
            recs_dict = train_and_predict_all(datasets, sentiments)
            
            # Map recs back to dictionary by ticker for easy lookup
            recs_by_ticker = {r["ticker"]: r for r in recs_dict} if isinstance(recs_dict, list) else recs_dict
            
            # Filter for >= 10% predicted return
            predicted_breakouts = []
            for c in candidates:
                t = c["Ticker"]
                if t in recs_by_ticker:
                    # predictions save return as fraction (e.g. 0.12)
                    pred_ret = recs_by_ticker[t].get("predicted_return", 0.0) * 100.0
                    if pred_ret >= 10.0:
                        c["Predicted Return %"] = round(pred_ret, 2)
                        predicted_breakouts.append(c)
                        
                        # Automatically add to custom_tickers.json so it appears in the watchlist / Neural Analyst sections
                        try:
                            from config import BASE_DIR
                            CUSTOM_TICKERS_FILE = os.path.join(BASE_DIR, "custom_tickers.json")
                            custom_data = {"tickers": [], "sectors": {}}
                            if os.path.exists(CUSTOM_TICKERS_FILE):
                                with open(CUSTOM_TICKERS_FILE, "r") as f:
                                    custom_data = json.load(f)
                            
                            tickers_list = custom_data.setdefault("tickers", [])
                            sectors_dict = custom_data.setdefault("sectors", {})
                            
                            if t not in tickers_list:
                                # Fetch sector
                                industry = "Breakout Scans"
                                try:
                                    ticker_info = yf.Ticker(t).info
                                    fetched_ind = ticker_info.get("industry")
                                    if fetched_ind:
                                        industry = fetched_ind
                                except Exception:
                                    pass
                                
                                tickers_list.append(t)
                                sectors_dict.setdefault(industry, []).append(t)
                                
                                custom_data["tickers"] = sorted(list(set(tickers_list)))
                                for s_name in sectors_dict:
                                    sectors_dict[s_name] = sorted(list(set(sectors_dict[s_name])))
                                custom_data["sectors"] = sectors_dict
                                
                                with open(CUSTOM_TICKERS_FILE, "w") as f:
                                    json.dump(custom_data, f, indent=4)
                                print(f"Successfully auto-added {t} to watchlist under sector '{industry}'")
                        except Exception as add_err:
                            print(f"Failed to auto-add {t} to watchlist: {add_err}")
                        
            save_scan_results(predicted_breakouts)
            print(f"--- Predictive Breakout Scan Completed Successfully. Found {len(predicted_breakouts)} future breakouts. ---")
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
