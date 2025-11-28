import numpy as np
import scipy.stats as si

def bsm_greeks(S, K, T, r, sigma, option_type='call'):
    """
    Calculate all BSM Greeks: Delta, Gamma, Theta, Vega, Rho
    """
    if T <= 1e-5:
        return {'delta': 0, 'gamma': 0, 'theta': 0, 'vega': 0, 'rho': 0}
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    # Delta
    if option_type == 'call':
        delta = si.norm.cdf(d1)
    else:
        delta = si.norm.cdf(d1) - 1
        
    # Gamma (same for call and put)
    gamma = si.norm.pdf(d1) / (S * sigma * np.sqrt(T))
    
    # Vega (same for call and put)
    # Vega is typically expressed as change in price for 1% change in volatility
    vega = S * si.norm.pdf(d1) * np.sqrt(T) / 100 
    
    # Theta
    # Theta is typically expressed as change in price for 1 day passage of time
    if option_type == 'call':
        theta = (- (S * si.norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * si.norm.cdf(d2)) / 365
    else:
        theta = (- (S * si.norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * si.norm.cdf(-d2)) / 365
        
    # Rho
    # Rho is typically expressed as change in price for 1% change in interest rate
    if option_type == 'call':
        rho = (K * T * np.exp(-r * T) * si.norm.cdf(d2)) / 100
    else:
        rho = (-K * T * np.exp(-r * T) * si.norm.cdf(-d2)) / 100
        
    return {
        'delta': delta,
        'gamma': gamma,
        'theta': theta,
        'vega': vega,
        'rho': rho
    }

# Test call
S0 = 100
K = 100
T_init = 1.0
r = 0.05
sigma = 0.2
print(bsm_greeks(S0, K, T_init, r, sigma, 'call'))
