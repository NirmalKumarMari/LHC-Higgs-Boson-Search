"""Median expected significance (Cowan et al., Z_A) versus total yield at fixed S/B."""

import numpy as np
import matplotlib.pyplot as plt

# Example function for median significance approximation (Z_A, Eq. 97 of arXiv:1007.1727)
def median_significance(s, b):
    """Calculate median significance using the formula for approximated Z_A."""
    from numpy import sqrt, log
    if s == 0:
        return 0
    return np.sqrt(2 * ((s + b) * np.log(1 + s/b) - s))

# Example fixed signal to background ratio based on your event selection efficiencies:
signal_to_background_ratio = 0.1  # for example, 1 signal per 10 background

# Vary total events (signal + background)
total_yields = np.linspace(10, 5000, 500)

# Calculate corresponding signal and background yields keeping ratio fixed
signal_yields = total_yields * (signal_to_background_ratio / (1 + signal_to_background_ratio))
background_yields = total_yields - signal_yields

# Calculate median significances
median_significances = np.array([median_significance(s, b) for s, b in zip(signal_yields, background_yields)])

# Plot median expected significance vs total yield
plt.figure(figsize=(8,5))
plt.plot(total_yields, median_significances, label='Median Expected Significance')
plt.axhline(5, color='red', linestyle='--', label='5 Sigma Threshold')
plt.xlabel('Total Yield (Signal + Background)')
plt.ylabel('Median Expected Significance (σ)')
plt.title('Median Expected Significance vs Total Yield')
plt.legend()
plt.grid(True)
plt.show()

# 2D plot of total yield vs signal yield with color indication of median significance
plt.figure(figsize=(8,6))
sc = plt.scatter(total_yields, signal_yields, c=median_significances, cmap='viridis', marker='o')
plt.colorbar(sc, label='Median Expected Significance (σ)')
plt.xlabel('Total Yield (Signal + Background)')
plt.ylabel('Signal Yield')
plt.title('Median Expected Significance by Signal and Total Yield')
plt.grid(True)
plt.show()
