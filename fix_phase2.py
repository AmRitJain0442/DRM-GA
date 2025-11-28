import json
import numpy as np

file_path = r"c:\Users\amrit\Desktop\FINAL_DRM\DRM-GA\DRM_Project.ipynb"

with open(file_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Find the Phase 2 cell
phase_2_cell_index = -1
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        if "PHASE 2: SYNTHETIC PORTFOLIO" in source:
            phase_2_cell_index = i
            break

if phase_2_cell_index != -1:
    print(f"Found Phase 2 cell at index {phase_2_cell_index}")
    cell = nb['cells'][phase_2_cell_index]
    source_lines = cell['source']
    
    new_source = []
    
    # We will reconstruct the cell content with fixes
    # 1. Update Parameters to be dynamic
    # 2. Update Simulation Loop
    
    # Helper to check if a line is part of the loop or parameters
    
    # We'll rewrite the parameters section
    # And the loop section
    
    # Let's just replace the entire content of the cell with the corrected version
    # preserving the imports or function definitions if they are there.
    # Actually, bsm_price and bsm_delta are defined there. I should keep them.
    # And bsm_greeks was added there (or appended to it).
    
    # Wait, I appended bsm_greeks to the cell where bsm_delta was. 
    # If that was the Phase 2 cell, then bsm_greeks is in it.
    
    # I will read the current source and modify specific parts.
    
    modified_source = []
    skip = False
    
    for line in source_lines:
        # Fix T_init
        if "T_init = 2.0" in line:
            modified_source.append("    T_init = len(data) / 252.0  # Dynamic duration based on data\n")
            continue
        
        # Fix T_t calculation in the loop
        if "T_t = max(T_init - (i / 252.0), 1e-5)" in line:
            modified_source.append("    # Time to maturity (decaying)\n")
            modified_source.append("    T_t = max(T_init - (i / 252.0), 1e-5)\n") 
            # Actually, if T_init is exactly len(data)/252, then at i=len(data)-1, 
            # T_t = len/252 - (len-1)/252 = 1/252.
            # At i=len(data), it would be 0. But loop goes up to len(data)-1.
            # So T_t will be 1/252 at the last step. This is correct for "just before expiry".
            # If we want the payoff AT expiry, we might want to simulate one more step or assume the last step is close enough.
            # I'll stick with the existing formula but with corrected T_init.
            continue
            
        # Fix Call Liability in Dynamic Hedging
        if "call_liability = call_price_t" in line:
             modified_source.append("    call_liability = call_price_t\n")
             continue

        modified_source.append(line)
    
    # Replace the source
    nb['cells'][phase_2_cell_index]['source'] = modified_source
    print("Updated Phase 2 cell.")
    
else:
    print("Phase 2 cell not found.")

# Save the notebook
with open(file_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Notebook updated successfully.")
