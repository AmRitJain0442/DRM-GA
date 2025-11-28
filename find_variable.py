import json

file_path = r"c:\Users\amrit\Desktop\FINAL_DRM\DRM-GA\DRM_Project.ipynb"

with open(file_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

found = False
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        if "def bsm_greeks" in source:
            with open('cell_content.txt', 'a', encoding='utf-8') as out:
                out.write(f"Cell {i}:\n")
                out.write(source)
                out.write("\n" + "="*20 + "\n")
            print(f"Found 'def bsm_greeks' in cell {i}.")
            found = True

if not found:
    print("'cost_call' not found in any code cell.")
