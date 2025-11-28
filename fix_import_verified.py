import json
import os

file_path = r"c:\Users\amrit\Desktop\FINAL_DRM\DRM-GA\DRM_Project.ipynb"

print(f"Reading {file_path}...")
with open(file_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

found = False
target_line = "    from datetime import datetime\n"
replacement_line = "    from datetime import datetime, timedelta\n"

for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        source = cell['source']
        for j, line in enumerate(source):
            if target_line in line:
                print(f"Found target line in cell {i}, line {j}")
                source[j] = replacement_line
                found = True

if found:
    print(f"Writing to {file_path}...")
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)
    
    # Verify
    print("Verifying write...")
    with open(file_path, 'r', encoding='utf-8') as f:
        nb_verify = json.load(f)
    
    verified = False
    for cell in nb_verify['cells']:
        if cell['cell_type'] == 'code':
            for line in cell['source']:
                if replacement_line in line:
                    verified = True
                    break
    
    if verified:
        print("SUCCESS: File updated and verified.")
    else:
        print("FAILURE: File write appeared successful but content not updated.")
else:
    print("Target line not found.")
