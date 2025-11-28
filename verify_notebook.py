import json
import pandas as pd
import numpy as np
import scipy.stats as si
import matplotlib.pyplot as plt
import io
import sys

# Mock yfinance
class MockYF:
    def download(self, tickers, start=None, end=None, progress=False, **kwargs):
        print(f"Mocking download for {tickers}")
        # Create a dummy dataframe for 2 years
        dates = pd.date_range(start='2023-01-01', periods=500, freq='B')
        # Simulate a random walk
        np.random.seed(42)
        returns = np.random.normal(0, 0.02, 500)
        price = 100 * np.exp(np.cumsum(returns))
        df = pd.DataFrame({'Close': price, 'Adj Close': price}, index=dates)
        return df

sys.modules['yfinance'] = MockYF()
import yfinance as yf

# Load notebook
file_path = r"c:\Users\amrit\Desktop\FINAL_DRM\DRM-GA\DRM_Project.ipynb"
with open(file_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Extract code
code = ""
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        print(f"Processing code cell (length {len(source)}): {source[:50]}...")
        # Skip pip install or magic commands
        lines = source.split('\n')
        clean_lines = []
        for line in lines:
            if line.strip().startswith('!') or line.strip().startswith('%'):
                continue
            clean_lines.append(line)
        code += "\n".join(clean_lines) + "\n"

# Dump extracted code for debugging
with open('extracted_code.py', 'w', encoding='utf-8') as f:
    f.write(code)
print("Extracted code dumped to extracted_code.py")

# Execute code
# We need to capture stdout to check for P&L output

from contextlib import redirect_stdout
f = io.StringIO()

print("Starting verification execution...")
try:
    # We need to handle plotting to not block
    plt.show = lambda: print("Plot generated.")
    
    exec(code, globals())
    print("Execution successful.")
except Exception as e:
    print(f"Execution failed: {e}")
    import traceback
    traceback.print_exc()
    
    # Print code with line numbers
    print("\nCode around error:")
    lines = code.split('\n')
    # Try to extract line number from traceback
    import sys
    exc_type, exc_value, exc_traceback = sys.exc_info()
    if exc_traceback:
        # Go to the last frame
        tb = exc_traceback
        while tb.tb_next:
            tb = tb.tb_next
        lineno = tb.tb_lineno
        start = max(0, lineno - 10)
        end = min(len(lines), lineno + 10)
        for i in range(start, end):
            print(f"{i+1}: {lines[i]}")


output = f.getvalue()
with open('verify_log.txt', 'w', encoding='utf-8') as log_file:
    log_file.write(output)
    
    # Check for specific outputs
    if "P&L:" in output:
        log_file.write("\nVerification: P&L calculated.")
    else:
        log_file.write("\nVerification: P&L NOT found.")

    if "Final Hedged P&L:" in output:
        log_file.write("\nVerification: Hedged P&L calculated.")
    else:
        log_file.write("\nVerification: Hedged P&L NOT found.")

    if "ATM Call Greeks" in output:
        log_file.write("\nVerification: Greeks calculated.")
    else:
        log_file.write("\nVerification: Greeks NOT found.")

