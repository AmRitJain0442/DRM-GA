import json

file_path = r"c:\Users\amrit\Desktop\FINAL_DRM\DRM-GA\DRM_Project.ipynb"

with open(file_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Iterate through cells to fix indentation
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = cell['source']
        new_source = []
        for i, line in enumerate(source):
            # Fix T_init indentation
            if "T_init = len(data) / 252.0" in line:
                new_source.append("T_init = len(data) / 252.0  # Dynamic duration based on data\n")
            # Fix call_liability indentation if needed (it was inside a loop, so indentation is expected there)
            # But let's check if I messed up other lines.
            # T_t line was inside loop, so it should be indented.
            # "    T_t = max(T_init - (i / 252.0), 1e-5)\n" -> This looks correct if loop started before.
            
            else:
                new_source.append(line)
        cell['source'] = new_source

with open(file_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Notebook indentation fixed.")
