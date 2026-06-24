import os
import sys
import time
import subprocess

def test_pipeline_caching():
    print("==================================================")
    print("Testing ML Pipeline Caching and Speed Performance")
    print("==================================================")
    
    # 1. Run with --force-retrain (Fresh fit)
    print("\n[Run 1/2] Executing with --force-retrain (training from scratch)...")
    cmd_retrain = [sys.executable, "scheduler.py", "--scan-breakouts", "--tickers", "TATAPOWER.NS", "--force-retrain"]
    start_time = time.time()
    res_retrain = subprocess.run(cmd_retrain, capture_output=True, text=True)
    duration_retrain = time.time() - start_time
    
    print(f"Status Code: {res_retrain.returncode}")
    print(f"Duration (Fresh Fit): {duration_retrain:.2f} seconds")
    print("STDOUT:")
    print(res_retrain.stdout)
    print("STDERR:")
    print(res_retrain.stderr)
        
    # 2. Run without --force-retrain (Cached fit)
    print("\n[Run 2/2] Executing in Cached Mode (Fast Mode)...")
    cmd_cached = [sys.executable, "scheduler.py", "--scan-breakouts", "--tickers", "TATAPOWER.NS"]
    start_time = time.time()
    res_cached = subprocess.run(cmd_cached, capture_output=True, text=True)
    duration_cached = time.time() - start_time
    
    print(f"Status Code: {res_cached.returncode}")
    print(f"Duration (Cached/Fast Mode): {duration_cached:.2f} seconds")
    print("STDOUT:")
    print(res_cached.stdout)
    print("STDERR:")
    print(res_cached.stderr)

if __name__ == "__main__":
    test_pipeline_caching()
