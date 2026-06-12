import re

with open("app.py", "r", encoding="utf-8") as f:
    lines = f.readlines()

vars_to_find = ["TICKERS", "SECTORS", "PORTFOLIO_FILE", "RECOMMENDATIONS_FILE", "BROKER_CONFIGS", "ALL_TICKERS", "BASE_DIR"]

for i, line in enumerate(lines, 1):
    for var in vars_to_find:
        # Match word boundaries to avoid false positives (e.g. TICKERS in ALL_TICKERS)
        if re.search(r'\b' + var + r'\b', line):
            print(f"Line {i}: {line.strip()}")
