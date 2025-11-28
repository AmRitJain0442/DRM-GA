import json

file_path = r"c:\Users\amrit\Desktop\FINAL_DRM\DRM-GA\DRM_Project.ipynb"

with open(file_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Find Phase 4 cell (Cell 15 based on previous output, or search)
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = cell['source']
        # Check if this is the Phase 4 cell
        if "CONSOLIDATED EXCEL EXPORT" in "".join(source):
            new_source = []
            inserted = False
            for line in source:
                # Insert definitions before df_task_a
                if "df_task_a = pd.DataFrame({" in line and not inserted:
                    new_source.append("# Define missing variables for Task A Comparison\n")
                    new_source.append("cost_call = call_init\n")
                    new_source.append("payoff_call = max(stock_prices[-1] - K, 0)\n")
                    new_source.append("pnl_call = payoff_call - cost_call\n")
                    new_source.append("roc_call = (pnl_call / cost_call) * 100 if cost_call != 0 else 0\n")
                    new_source.append("\n")
                    new_source.append("cost_synth = initial_synth_cost\n")
                    new_source.append("payoff_synth = final_synth_value\n")
                    new_source.append("roc_synth = (pnl_synth / cost_synth) * 100 if cost_synth != 0 else 0\n")
                    new_source.append("\n")
                    inserted = True
                
                new_source.append(line)
            cell['source'] = new_source
            print("Fixed undefined variables in Phase 4.")
            break

with open(file_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Notebook updated.")
