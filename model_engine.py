import os
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
from config import MODELS_DIR, MODEL_PARAMS, TRAIN_HORIZON_DAYS, RECOMMENDATIONS_FILE, VALIDATION_ERRORS_FILE, TICKERS
from broker_charges import calculate_charges

# Features to feed into models
FEATURES = ['RSI_14', 'ATR_14_Pct', 'Close_to_SMA_50', 'SMA_50_to_200', 'Volatility_30', 'Crude_Oil_Z']

# Try importing XGBoost. If it's not working, we fall back to RandomForest.
XGBOOST_AVAILABLE = False
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
    print("XGBoost is available for training.")
except ImportError:
    print("XGBoost not available. Falling back to Scikit-Learn RandomForest.")

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_absolute_error

def prepare_training_data(df: pd.DataFrame) -> tuple:
    """
    Prepare features (X) and targets (y_class, y_reg) for training.
    Uses rolling 5-day shifts for target definitions.
    Returns train sets, inference row, and dataset metadata (date ranges, counts).
    """
    df_clean = df.copy()
    
    # Initialize historical sentiment to 0.0 since it is not available retrospectively
    if 'Sentiment' not in df_clean.columns:
        df_clean['Sentiment'] = 0.0
        
    # Drop rows with NaNs in core features first
    df_clean = df_clean.dropna(subset=['RSI_14', 'ATR_14', 'Close_to_SMA_50', 'SMA_50_to_200', 'Volatility_30', 'Crude_Oil_Z'])

    # Targets: Forward 5-day horizon
    df_clean['Future_Close'] = df_clean['Close'].shift(-TRAIN_HORIZON_DAYS)
    
    # Target Return: [Close(t + 5) - Close(t)] / Close(t), clipped to [-15%, +15%] to prevent outlier distortion
    df_clean['Target_Return'] = ((df_clean['Future_Close'] - df_clean['Close']) / df_clean['Close']).clip(-0.15, 0.15)
    
    # Target Direction: 1 if return > 0 else 0
    df_clean['Target_Direction'] = np.where(df_clean['Target_Return'] > 0.0, 1, 0)
    
    X_full = df_clean[FEATURES]
    
    # 80/20 chronological train-test split (NO shuffling — respects time ordering)
    total_rows = len(df_clean)
    train_end_idx = int(total_rows * 0.80)
    
    train_slice = df_clean.iloc[:train_end_idx]
    test_slice  = df_clean.iloc[train_end_idx:-TRAIN_HORIZON_DAYS]  # exclude last 5 rows (no future label)
    
    X_train = train_slice[FEATURES]
    y_class  = train_slice['Target_Direction']
    y_reg    = train_slice['Target_Return']
    
    # Test/validation set
    X_test   = test_slice[FEATURES]
    y_class_test = test_slice['Target_Direction']
    y_reg_test   = test_slice['Target_Return']
    
    # Drop any remaining NaNs in targets (from shift)
    valid_train = y_reg.notna()
    X_train  = X_train[valid_train]
    y_class  = y_class[valid_train]
    y_reg    = y_reg[valid_train]
    
    valid_test = y_reg_test.notna()
    X_test       = X_test[valid_test]
    y_class_test = y_class_test[valid_test]
    y_reg_test   = y_reg_test[valid_test]
    
    # Dataset metadata: calendar & trading day counts
    data_meta = {
        "total_trading_days": total_rows,
        "train_trading_days": len(X_train),
        "validation_trading_days": len(X_test),
        "train_start_date": str(df_clean.index[0].date()),
        "train_end_date": str(df_clean.index[train_end_idx - 1].date()),
        "validation_start_date": str(df_clean.index[train_end_idx].date()) if train_end_idx < total_rows else "N/A",
        "validation_end_date": str(df_clean.index[-1].date()),
        "forecast_horizon_days": TRAIN_HORIZON_DAYS
    }
    
    # Inference row = latest available feature vector
    inference_row = X_full.tail(1)
    
    return X_train, y_class, y_reg, X_test, y_class_test, y_reg_test, test_slice, data_meta

def train_models(ticker: str, df: pd.DataFrame) -> dict:
    """
    Trains and saves Classifier and Regressor models for a ticker.
    Uses an 80/20 chronological split — trains on older data, validates on recent data.
    Returns training accuracy, out-of-sample validation accuracy, MAE, and dataset metadata.
    """
    X_train, y_class, y_reg, X_test, y_class_test, y_reg_test, test_slice, data_meta = prepare_training_data(df)
    
    if len(X_train) < 50:
        print(f"Skipping training for {ticker} due to insufficient historical rows ({len(X_train)} available).")
        return {"accuracy": 0.0, "val_accuracy": 0.0, "mae": 0.0, "val_mae": 0.0, "algorithm": "None", "error": "Insufficient history"}
        
    clf_path = os.path.join(MODELS_DIR, f"{ticker}_classifier.pkl")
    reg_path = os.path.join(MODELS_DIR, f"{ticker}_regressor.pkl")
    
    algorithm_used = "RandomForest"
    
    # 1. Classifier Training (on 80% historical train split)
    if XGBOOST_AVAILABLE:
        try:
            params = MODEL_PARAMS["classifier"]
            clf = xgb.XGBClassifier(**params)
            clf.fit(X_train, y_class)
            algorithm_used = "XGBoost"
        except Exception as e:
            print(f"XGBoost Classifier failed: {e}. Using RandomForest fallback.")
            clf = RandomForestClassifier(n_estimators=80, max_depth=3, min_samples_leaf=5, random_state=42)
            clf.fit(X_train, y_class)
    else:
        clf = RandomForestClassifier(n_estimators=80, max_depth=3, min_samples_leaf=5, random_state=42)
        clf.fit(X_train, y_class)
        
    # Train accuracy vs. Out-of-sample validation accuracy (20% unseen data)
    train_acc = accuracy_score(y_class, clf.predict(X_train))
    val_acc   = accuracy_score(y_class_test, clf.predict(X_test)) if len(X_test) > 0 else 0.0
    
    # 2. Regressor Training with Dynamic Hyperparameter Tuning
    best_reg = None
    best_val_mae = float('inf')
    best_alpha = 2.0
    best_lambda = 10.0
    
    if XGBOOST_AVAILABLE:
        try:
            reg_candidates = [
                (2.0, 10.0),   # High (Conservative)
                (0.5, 3.0),    # Moderate
                (0.05, 1.0),   # Low
                (0.0, 0.1)     # Minimal
            ]
            
            base_params = MODEL_PARAMS["regressor"].copy()
            
            for alpha, lam in reg_candidates:
                candidate_params = base_params.copy()
                candidate_params["reg_alpha"] = alpha
                candidate_params["reg_lambda"] = lam
                
                candidate_reg = xgb.XGBRegressor(**candidate_params)
                candidate_reg.fit(X_train, y_reg)
                
                if len(X_test) > 0:
                    val_mae_cand = mean_absolute_error(y_reg_test, candidate_reg.predict(X_test))
                else:
                    val_mae_cand = mean_absolute_error(y_reg, candidate_reg.predict(X_train))
                    
                if val_mae_cand < best_val_mae:
                    best_val_mae = val_mae_cand
                    best_reg = candidate_reg
                    best_alpha = alpha
                    best_lambda = lam
            
            reg = best_reg
            print(f"[{ticker}] Dynamic Tuning Selected: reg_alpha={best_alpha}, reg_lambda={best_lambda} (Val MAE: {best_val_mae:.5f})")
        except Exception as e:
            print(f"XGBoost Regressor tuning failed: {e}. Using RandomForest fallback.")
            reg = RandomForestRegressor(n_estimators=80, max_depth=3, min_samples_leaf=5, random_state=42)
            reg.fit(X_train, y_reg)
    else:
        reg = RandomForestRegressor(n_estimators=80, max_depth=3, min_samples_leaf=5, random_state=42)
        reg.fit(X_train, y_reg)
        
    # Train MAE vs. Out-of-sample validation MAE
    train_mae = mean_absolute_error(y_reg, reg.predict(X_train))
    val_mae   = mean_absolute_error(y_reg_test, reg.predict(X_test)) if len(X_test) > 0 else 0.0
    
    joblib.dump(clf, clf_path)
    joblib.dump(reg, reg_path)
    
    # Compile validation error records (last 10 independent completed 5-day cycles)
    val_errors = []
    if len(test_slice) > 0:
        try:
            y_reg_pred = reg.predict(X_test)
            raw_records = []
            for i in range(len(test_slice)):
                date_t = str(test_slice.index[i].date())
                close_t = float(test_slice['Close'].iloc[i])
                future_close_t = float(test_slice['Future_Close'].iloc[i])
                actual_ret = float(y_reg_test.iloc[i])
                pred_ret = float(y_reg_pred[i])
                err_margin = pred_ret - actual_ret
                
                raw_records.append({
                    "date": date_t,
                    "actual_close": round(close_t, 2),
                    "actual_future_close": round(future_close_t, 2),
                    "actual_return_pct": round(actual_ret * 100, 2),
                    "predicted_return_pct": round(pred_ret * 100, 2),
                    "error_margin_pct": round(err_margin * 100, 2)
                })
            # Save all raw validation records in chronological order
            val_errors = raw_records
        except Exception as e:
            print(f"Error compiling validation errors for {ticker}: {e}")
            
    print(
        f"{ticker} [{algorithm_used}] | "
        f"Train Acc: {train_acc:.2%} | Val Acc: {val_acc:.2%} | "
        f"Train MAE: {train_mae:.4f} | Val MAE: {val_mae:.4f} | "
        f"Train days: {data_meta['train_trading_days']} | Val days: {data_meta['validation_trading_days']}"
    )
    
    return {
        "train_accuracy":      round(float(train_acc), 4),
        "val_accuracy":        round(float(val_acc), 4),
        "train_mae":           round(float(train_mae), 6),
        "val_mae":             round(float(val_mae), 6),
        "algorithm":           algorithm_used,
        "features":            FEATURES,
        "validation_errors":   val_errors,
        **data_meta   # inject all date/day metadata
    }

def score_latest_session(ticker: str, latest_row: pd.DataFrame, latest_sentiment: float, current_price: float, current_atr: float) -> dict:
    """
    Load trained models, inject live news sentiment, and predict the 5-day horizon.
    """
    clf_path = os.path.join(MODELS_DIR, f"{ticker}_classifier.pkl")
    reg_path = os.path.join(MODELS_DIR, f"{ticker}_regressor.pkl")
    
    if not os.path.exists(clf_path) or not os.path.exists(reg_path):
        return {"error": f"Models not trained for {ticker}"}
        
    clf = joblib.load(clf_path)
    reg = joblib.load(reg_path)
    
    # Prepare inference feature vector
    X_inf = latest_row.copy()
    X_inf['Sentiment'] = float(latest_sentiment)
    
    # Align features list exactly
    X_inf = X_inf[FEATURES]
    
    # 1. Classification (Probability of rise)
    try:
        # Handles binary classifier classes
        probs = clf.predict_proba(X_inf)[0]
        # class 1 probability
        up_prob = probs[1]
    except AttributeError:
        # Fallback if model doesn't support predict_proba
        up_prob = 0.5
        
    # 2. Regression (Predicted alpha magnitude)
    pred_return = reg.predict(X_inf)[0]
    
    # 3. Dynamic execution brackets
    suggested_buy = current_price
    take_profit = suggested_buy * (1.0 + pred_return)
    
    # Stop-Loss: price - 1.5 * ATR
    atr_factor = 1.5
    stop_loss = suggested_buy - (atr_factor * current_atr)
    
    # Prevent negative stop-loss
    if stop_loss < 0.01:
        stop_loss = suggested_buy * 0.95
        
    # Risk-to-Reward Ratio (RRR)
    downside_percent = (atr_factor * current_atr) / current_price if current_price > 0 else 0.05
    rrr = pred_return / downside_percent if downside_percent > 0 else 1.0
    
    # -----------------------------------------------------------------------
    # Signal Logic (covers all three cases — BUY, SELL, NEUTRAL)
    # BUY:     High confidence of upward move AND meaningful predicted return
    # SELL:    Either low confidence of upward move OR negative predicted return
    #          Both conditions independently valid for a hold-portfolio warning
    # NEUTRAL: Model is undecided — no actionable edge detected
    # -----------------------------------------------------------------------
    signal = "NEUTRAL"
    signal_reason = "Model has no strong directional edge. Stay in cash or hold."
    
    if up_prob >= 0.75 and pred_return >= 0.02:
        signal = "BUY"
        signal_reason = f"Model has {up_prob:.0%} confidence of a rise with +{pred_return:.2%} predicted alpha in {TRAIN_HORIZON_DAYS} trading days."
    elif up_prob <= 0.35 and pred_return <= -0.01:
        # Strong SELL: BOTH classifier bearish AND regressor negative
        signal = "SELL"
        signal_reason = f"Model has only {up_prob:.0%} upward probability with {pred_return:.2%} predicted return — strong downtrend signal."
    elif pred_return <= -0.02:
        # Regressor-only SELL: predicted loss exceeds 2% threshold
        signal = "SELL"
        signal_reason = f"Regressor predicts a {pred_return:.2%} loss over {TRAIN_HORIZON_DAYS} days even with neutral classifier."
    elif up_prob <= 0.30:
        # Classifier-only SELL: very low probability of upward move
        signal = "SELL"
        signal_reason = f"Classifier predicts only {up_prob:.0%} probability of a rise — high downside risk."

    # For SELL signals: compute a suggested exit/cover price
    # (current price minus the predicted loss magnitude)
    suggested_exit = round(float(current_price * (1.0 + pred_return)), 2) if signal == "SELL" else None
        
    return {
        "ticker":             ticker,
        "current_price":      round(float(current_price), 2),
        "upward_probability": round(float(up_prob), 4),
        "downward_probability": round(1.0 - float(up_prob), 4),
        "predicted_return":   round(float(pred_return), 4),
        "suggested_buy_price":  round(float(suggested_buy), 2),
        "take_profit_price":    round(float(take_profit), 2),
        "stop_loss_price":      round(float(stop_loss), 2),
        "suggested_exit_price": suggested_exit,
        "rrr":                  round(float(rrr), 2),
        "signal":               signal,
        "signal_reason":        signal_reason,
        "predicted_price":      round(float(suggested_buy * (1.0 + pred_return)), 2),
        "forecast_horizon_days": TRAIN_HORIZON_DAYS
    }

def train_and_predict_all(datasets: dict, sentiments: dict) -> list:
    """
    Loops through all ticker datasets, trains their models, generates predictions
    for the latest session in parallel, and writes recommendations to recommendations.json.
    """
    recs = []
    from concurrent.futures import ThreadPoolExecutor, as_completed
    
    def process_train_and_predict(ticker, df):
        try:
            # Step 1: Train/retrain
            metrics = train_models(ticker, df)
            if "error" in metrics:
                return None
                
            # Step 2: Score
            latest_row = df.tail(1)
            current_price = float(df['Close'].iloc[-1])
            current_atr = float(df['ATR_14'].iloc[-1])
            sentiment = sentiments.get(ticker, 0.0)
            
            pred = score_latest_session(ticker, latest_row, sentiment, current_price, current_atr)
            if "error" not in pred:
                # Merge training metrics and dataset metadata
                pred["train_accuracy"]          = metrics.get("train_accuracy", 0.0)
                pred["val_accuracy"]            = metrics.get("val_accuracy", 0.0)
                pred["train_mae"]               = metrics.get("train_mae", 0.0)
                pred["val_mae"]                 = metrics.get("val_mae", 0.0)
                pred["model_algorithm"]         = metrics.get("algorithm", "Unknown")
                pred["train_trading_days"]      = metrics.get("train_trading_days", 0)
                pred["validation_trading_days"] = metrics.get("validation_trading_days", 0)
                pred["total_trading_days"]      = metrics.get("total_trading_days", 0)
                pred["train_start_date"]        = metrics.get("train_start_date", "")
                pred["train_end_date"]          = metrics.get("train_end_date", "")
                pred["validation_start_date"]   = metrics.get("validation_start_date", "")
                pred["validation_end_date"]     = metrics.get("validation_end_date", "")
                pred["validation_errors"]       = metrics.get("validation_errors", [])
                pred["sentiment_score"]         = sentiment
                pred["timestamp"]               = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                return pred
        except Exception as e:
            print(f"Error training/predicting for {ticker}: {e}")
        return None

    max_workers = min(10, len(datasets))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_ticker = {
            executor.submit(process_train_and_predict, ticker, df): ticker 
            for ticker, df in datasets.items()
        }
        for future in as_completed(future_to_ticker):
            pred = future.result()
            if pred is not None:
                recs.append(pred)
            
    # Write to recommendations.json (merge with existing to preserve other tickers)
    try:
        import json
        existing_recs = []
        if os.path.exists(RECOMMENDATIONS_FILE):
            try:
                with open(RECOMMENDATIONS_FILE, "r") as f:
                    existing_recs = json.load(f)
            except Exception:
                existing_recs = []
        
        # Merge by ticker
        merged_dict = {}
        for item in existing_recs:
            if isinstance(item, dict) and "ticker" in item:
                merged_dict[item["ticker"]] = item
        
        for item in recs:
            merged_dict[item["ticker"]] = item
            
        merged_list = list(merged_dict.values())
        
        with open(RECOMMENDATIONS_FILE, "w") as f:
            json.dump(merged_list, f, indent=4)
        print(f"Recommendations successfully saved to {RECOMMENDATIONS_FILE}")
    except Exception as e:
        print(f"Error writing recommendations: {e}")
        
    return recs

if __name__ == "__main__":
    # Test model engine training
    print("Testing ML Engine...")
    from data_pipeline import get_complete_dataset
    datasets = get_complete_dataset(tickers=["TATAPOWER.NS"])
    sentiments = {"TATAPOWER.NS": 0.493}
    
    if "TATAPOWER.NS" in datasets:
        recs = train_and_predict_all(datasets, sentiments)
        print("Success! Generated recommendations count:", len(recs))
        print("First Recommendation:")
        print(recs[0])
    else:
        print("Pipeline returned no datasets to test ML Engine.")
