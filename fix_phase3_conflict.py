import json

file_path = r"c:\Users\amrit\Desktop\FINAL_DRM\DRM-GA\DRM_Project.ipynb"

with open(file_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Find Cell 11 (Phase 3 Task B)
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = cell['source']
        source_text = "".join(source)
        if "PHASE 3 - TASK B: BSM MODEL & GREEKS" in source_text:
            new_source = []
            skip_def = False
            
            for line in source:
                # Remove definition of bsm_greeks
                if "def bsm_greeks(" in line:
                    skip_def = True
                    continue
                if skip_def:
                    if "return delta_call, delta_put, vega" in line:
                        skip_def = False
                    continue
                
                # Update calls
                if "delta_c, delta_p, vega_atm = bsm_greeks(S, K, T, r, sigma)" in line:
                    new_source.append("g_c = bsm_greeks(S, K, T, r, sigma, 'call')\n")
                    new_source.append("g_p = bsm_greeks(S, K, T, r, sigma, 'put')\n")
                    new_source.append("delta_c = g_c['delta']\n")
                    new_source.append("delta_p = g_p['delta']\n")
                    new_source.append("vega_atm = g_c['vega']\n")
                    continue
                
                if "dc, dp, _ = bsm_greeks(s, K, T, r, sigma)" in line:
                    new_source.append("    gc = bsm_greeks(s, K, T, r, sigma, 'call')\n")
                    new_source.append("    gp = bsm_greeks(s, K, T, r, sigma, 'put')\n")
                    new_source.append("    dc = gc['delta']\n")
                    new_source.append("    dp = gp['delta']\n")
                    continue
                
                if "_, _, vega = bsm_greeks(S, K, T, r, v)" in line:
                    new_source.append("    g = bsm_greeks(S, K, T, r, v, 'call')\n")
                    new_source.append("    vega = g['vega']\n")
                    continue
                
                new_source.append(line)
            
            cell['source'] = new_source
            print("Fixed Phase 3 Task B cell.")
            break

with open(file_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Notebook updated.")
