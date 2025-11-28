import json
import os
import numpy as np # Not needed for script but good practice? No, script just manipulates strings.

file_path = r"c:\Users\amrit\Desktop\FINAL_DRM\DRM-GA\DRM_Project.ipynb"

with open(file_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Define bsm_greeks function code
bsm_greeks_code = [
    "\n",
    "# BSM Greeks Function\n",
    "def bsm_greeks(S, K, T, r, sigma, option_type='call'):\n",
    "    \"\"\"\n",
    "    Calculate all BSM Greeks: Delta, Gamma, Theta, Vega, Rho\n",
    "    \"\"\"\n",
    "    if T <= 1e-5:\n",
    "        return {'delta': 0, 'gamma': 0, 'theta': 0, 'vega': 0, 'rho': 0}\n",
    "    \n",
    "    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))\n",
    "    d2 = d1 - sigma * np.sqrt(T)\n",
    "    \n",
    "    # Delta\n",
    "    if option_type == 'call':\n",
    "        delta = si.norm.cdf(d1)\n",
    "    else:\n",
    "        delta = si.norm.cdf(d1) - 1\n",
    "        \n",
    "    # Gamma (same for call and put)\n",
    "    gamma = si.norm.pdf(d1) / (S * sigma * np.sqrt(T))\n",
    "    \n",
    "    # Vega (same for call and put)\n",
    "    # Vega is typically expressed as change in price for 1% change in volatility\n",
    "    vega = S * si.norm.pdf(d1) * np.sqrt(T) / 100 \n",
    "    \n",
    "    # Theta\n",
    "    # Theta is typically expressed as change in price for 1 day passage of time\n",
    "    if option_type == 'call':\n",
    "        theta = (- (S * si.norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * si.norm.cdf(d2)) / 365\n",
    "    else:\n",
    "        theta = (- (S * si.norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * si.norm.cdf(-d2)) / 365\n",
    "        \n",
    "    # Rho\n",
    "    # Rho is typically expressed as change in price for 1% change in interest rate\n",
    "    if option_type == 'call':\n",
    "        rho = (K * T * np.exp(-r * T) * si.norm.cdf(d2)) / 100\n",
    "    else:\n",
    "        rho = (-K * T * np.exp(-r * T) * si.norm.cdf(-d2)) / 100\n",
    "        \n",
    "    return {\n",
    "        'delta': delta,\n",
    "        'gamma': gamma,\n",
    "        'theta': theta,\n",
    "        'vega': vega,\n",
    "        'rho': rho\n",
    "    }\n"
]

# Find the cell with bsm_delta and insert bsm_greeks
found = False
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = cell['source']
        # Check if bsm_delta is defined in this cell
        if any('def bsm_delta' in line for line in source):
            # Find the end of bsm_delta function
            # It ends before "# Initial Prices" or similar
            insert_idx = -1
            for i, line in enumerate(source):
                if '# Initial Prices' in line:
                    insert_idx = i
                    break
            
            if insert_idx != -1:
                # Insert before Initial Prices
                cell['source'][insert_idx:insert_idx] = bsm_greeks_code
                found = True
                print("Inserted bsm_greeks function.")
            else:
                # Append to end if not found (fallback)
                cell['source'].extend(bsm_greeks_code)
                found = True
                print("Appended bsm_greeks function.")
            break

if not found:
    print("Could not find cell with bsm_delta.")

# Add Phase 3 Cell
phase_3_code = [
    "# PHASE 3: OPTION GREEKS ANALYSIS\n",
    "\n",
    "print(f\"\\n{'=' * 60}\")\n",
    "print(f\"PHASE 3: OPTION GREEKS ANALYSIS\")\n",
    "print(f\"{'=' * 60}\")\n",
    "\n",
    "# 1. Calculate Greeks for ATM Option\n",
    "greeks_atm = bsm_greeks(S0, K, T_init, r, sigma, 'call')\n",
    "print(f\"ATM Call Greeks (S={S0:.2f}, K={K:.2f}, T={T_init}y, r={r:.2%}, sigma={sigma:.2%}):\")\n",
    "for g, v in greeks_atm.items():\n",
    "    print(f\"  - {g.capitalize()}: {v:.4f}\")\n",
    "\n",
    "# 2. Plot Greeks vs Stock Price\n",
    "S_range = np.linspace(S0 * 0.8, S0 * 1.2, 100)\n",
    "greeks_S = {'delta': [], 'gamma': [], 'theta': [], 'vega': [], 'rho': []}\n",
    "\n",
    "for s in S_range:\n",
    "    g = bsm_greeks(s, K, T_init, r, sigma, 'call')\n",
    "    for k in greeks_S:\n",
    "        greeks_S[k].append(g[k])\n",
    "\n",
    "fig, axes = plt.subplots(3, 2, figsize=(15, 15))\n",
    "fig.suptitle('Option Greeks Sensitivity to Stock Price', fontsize=16)\n",
    "\n",
    "axes[0, 0].plot(S_range, greeks_S['delta'], color='blue')\n",
    "axes[0, 0].set_title('Delta')\n",
    "axes[0, 0].grid(True)\n",
    "\n",
    "axes[0, 1].plot(S_range, greeks_S['gamma'], color='green')\n",
    "axes[0, 1].set_title('Gamma')\n",
    "axes[0, 1].grid(True)\n",
    "\n",
    "axes[1, 0].plot(S_range, greeks_S['theta'], color='red')\n",
    "axes[1, 0].set_title('Theta (Daily)')\n",
    "axes[1, 0].grid(True)\n",
    "\n",
    "axes[1, 1].plot(S_range, greeks_S['vega'], color='purple')\n",
    "axes[1, 1].set_title('Vega (1% Vol Change)')\n",
    "axes[1, 1].grid(True)\n",
    "\n",
    "axes[2, 0].plot(S_range, greeks_S['rho'], color='brown')\n",
    "axes[2, 0].set_title('Rho (1% Rate Change)')\n",
    "axes[2, 0].grid(True)\n",
    "\n",
    "# Hide empty subplot\n",
    "axes[2, 1].axis('off')\n",
    "\n",
    "plt.tight_layout()\n",
    "plt.show()\n"
]

# Create new cell
new_cell = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": phase_3_code
}

# Append new cell to the end of the notebook
nb['cells'].append(new_cell)
print("Added Phase 3 cell.")

with open(file_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Notebook updated successfully.")
