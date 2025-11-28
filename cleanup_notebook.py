import json

file_path = r"c:\Users\amrit\Desktop\FINAL_DRM\DRM-GA\DRM_Project.ipynb"

with open(file_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Iterate through cells to clean up
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = cell['source']
        new_source = []
        skip_next = False
        for i, line in enumerate(source):
            if skip_next:
                skip_next = False
                continue
            
            # Remove duplicate comment
            if "# Time to maturity (decaying)" in line:
                # Check if next line is the same
                if i + 1 < len(source) and "# Time to maturity (decaying)" in source[i+1]:
                    new_source.append(line)
                    skip_next = True # Skip the duplicate
                    continue
            
            new_source.append(line)
        cell['source'] = new_source

with open(file_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Notebook cleaned up.")
