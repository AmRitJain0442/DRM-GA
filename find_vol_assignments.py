import json
import re

file_path = r"c:\Users\amrit\Desktop\FINAL_DRM\DRM-GA\DRM_Project.ipynb"

with open(file_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

print("Searching for assignments to 'vol' or 'sigma'...")

for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        source = cell['source']
        for j, line in enumerate(source):
            # Check for assignments to vol
            if re.search(r'\bvol\s*=', line) or re.search(r'\bsigma\s*=', line):
                print(f"Cell {i}, Line {j}: {line.strip()}")
