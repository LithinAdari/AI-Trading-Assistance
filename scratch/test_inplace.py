# Test in-place modification of lists/dicts across modules

# Mock config module
class MockConfig:
    def __init__(self):
        self.TICKERS = ["A", "B"]
        self.SECTORS = {"Sector1": ["A"], "Sector2": ["B"]}
    
    def refresh(self):
        new_tickers = ["A", "B", "C"]
        new_sectors = {"Sector1": ["A"], "Sector2": ["B"], "Sector3": ["C"]}
        
        self.TICKERS.clear()
        self.TICKERS.extend(new_tickers)
        
        self.SECTORS.clear()
        self.SECTORS.update(new_sectors)

# Mock app module imports
c = MockConfig()
TICKERS = c.TICKERS
SECTORS = c.SECTORS

print("Before refresh:")
print("TICKERS reference identical:", TICKERS is c.TICKERS)
print("SECTORS reference identical:", SECTORS is c.SECTORS)
print("TICKERS:", TICKERS)
print("SECTORS:", SECTORS)

c.refresh()

print("\nAfter refresh:")
print("TICKERS:", TICKERS)
print("SECTORS:", SECTORS)
