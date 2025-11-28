import json

file_path = r"c:\Users\amrit\Desktop\FINAL_DRM\DRM-GA\DRM_Project.ipynb"

with open(file_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Find the cell with the error
found = False
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        if "# PHASE 2A: PROCESS REAL NSE OPTION DATA" in source:
            print("Found the target cell.")
            
            new_source = []
            source_lines = cell['source']
            
            for i, line in enumerate(source_lines):
                # Look for the problematic line
                if "one_month_atm = one_month_options[one_month_options['Strike'] == atm_strike].iloc[0]" in line:
                    # Replace with robust logic
                    indent = line[:line.find("one_month_atm")]
                    
                    new_source.append(f"{indent}# Try to find exact match first\n")
                    new_source.append(f"{indent}exact_match = one_month_options[one_month_options['Strike'] == atm_strike]\n")
                    new_source.append(f"{indent}if not exact_match.empty:\n")
                    new_source.append(f"{indent}    one_month_atm = exact_match.iloc[0]\n")
                    new_source.append(f"{indent}else:\n")
                    new_source.append(f"{indent}    # Find closest strike\n")
                    new_source.append(f"{indent}    # Create a copy to avoid SettingWithCopyWarning\n")
                    new_source.append(f"{indent}    temp_options = one_month_options.copy()\n")
                    new_source.append(f"{indent}    temp_options['diff'] = abs(temp_options['Strike'] - atm_strike)\n")
                    new_source.append(f"{indent}    one_month_atm = temp_options.sort_values('diff').iloc[0]\n")
                    new_source.append(f"{indent}    print(f\"⚠ Exact ATM strike {{atm_strike}} not found. Using closest strike: {{one_month_atm['Strike']}}\")\n")
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
