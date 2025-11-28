import json

file_path = r"c:\Users\amrit\Desktop\FINAL_DRM\DRM-GA\DRM_Project.ipynb"

with open(file_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Find the cell with the import issue
found = False
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        if "from datetime import datetime" in source and "timedelta" in source:
            print("Found the target cell.")
            
            new_source = []
            source_lines = cell['source']
            
            for line in source_lines:
                if "from datetime import datetime" in line:
                    new_source.append("    from datetime import datetime, timedelta\n")
                else:
                    new_source.append(line)
            
            cell['source'] = new_source
            found = True
            break

if found:
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)
    print("Notebook updated successfully.")
else:
    print("Target cell not found.")
