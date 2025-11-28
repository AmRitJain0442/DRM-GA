import json

file_path = r"c:\Users\amrit\Desktop\FINAL_DRM\DRM-GA\DRM_Project.ipynb"

with open(file_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

found = False
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        if "from datetime import datetime" in source and "timedelta" in source:
            print(f"Found target cell at index {i}")
            
            new_source = []
            for line in cell['source']:
                if "from datetime import datetime" in line:
                    # Preserve indentation
                    indent = line[:line.find("from")]
                    new_line = f"{indent}from datetime import datetime, timedelta\n"
                    print(f"Replacing:\n{repr(line)}\nWith:\n{repr(new_line)}")
                    new_source.append(new_line)
                else:
                    new_source.append(line)
            
            cell['source'] = new_source
            found = True
            break

if found:
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)
    print("Notebook updated.")
else:
    print("Target cell not found.")
