import numpy as np
from scipy.stats import norm

S_values = [3598.40, 3700, 3800, 3900, 4000, 4069.60]
K = 3598.40
r = 0.06
sigma = 0.2557
T_init = 0.25

def bsm_delta(S, K, T, r, sig):
    d1 = (np.log(S/K) + (r + 0.5*sig**2)*T) / (sig*np.sqrt(T))
    return norm.cdf(d1)

def bsm_price(S, K, T, r, sig):
    d1 = (np.log(S/K) + (r + 0.5*sig**2)*T) / (sig*np.sqrt(T))
    d2 = d1 - sig*np.sqrt(T)
    return S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)

# Initial setup
init_call = bsm_price(S_values[0], K, T_init, r, sigma)
init_delta = bsm_delta(S_values[0], K, T_init, r, sigma)
cash = init_call - init_delta * S_values[0]  # Receive call premium, pay for stock
shares = init_delta

print(f'Initial Setup:')
print(f'  Call Premium Received: {init_call:.2f}')
print(f'  Stock Investment: {init_delta * S_values[0]:.2f}')
print(f'  Initial Cash: {cash:.2f}')
print(f'  Initial Delta: {shares:.4f}')
print(f'  Initial Portfolio Value = {shares*S_values[0] + cash - init_call:.2f}\n')

for i, S in enumerate(S_values[1:]):
    T = T_init - (i+1)*10/252  # Assume 10 days between observations
    
    # Calculate new delta
    new_delta = bsm_delta(S, K, T, r, sigma)
    delta_chg = new_delta - shares
    
    # Update cash (grow at risk-free rate, then adjust for rebalancing)
    cash = cash * np.exp(r*10/252) - delta_chg * S
    
    # Update shares
    shares = new_delta
    
    # Calculate call liability
    call_lib = bsm_price(S, K, T, r, sigma)
    
    # Portfolio value
    pv = shares * S + cash - call_lib
    
    print(f'Step {i+1}: S={S:.2f}, T={T:.4f}')
    print(f'  Delta={shares:.4f}, Cash={cash:.2f}')
    print(f'  Call Liability={call_lib:.2f}')
    print(f'  Portfolio Value (Hedging P&L)={pv:.2f}\n')
