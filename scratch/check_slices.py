import re

with open("app.py", "r", encoding="utf-8") as f:
    content = f.read()

# Find any occurrences of slicing on tickers or options
slices = re.findall(r'\[\s*:\s*\d+\s*\]', content)
print("Slices found in app.py:", slices)

# Let's inspect where they are
for match in re.finditer(r'\[\s*:\s*\d+\s*\]', content):
    start = max(0, match.start() - 50)
    end = min(len(content), match.end() + 50)
    print(f"Context: {content[start:end].strip()}")
