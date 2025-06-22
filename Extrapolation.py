"""
The code below extrapolates from known values of $\beta$ and the corresponding $a$ values new $a$ values for lower values of $\beta$.

"""

import numpy as np
import scipy.optimize as scp
import matplotlib.pyplot as plt

plt.style.use('seaborn-v0_8-darkgrid')

# We use the given values of beta and the corresponding values of a
beta = np.array([2.20, 2.30, 2.40, 2.50, 2.60, 2.70, 2.85])
a = np.array([0.211, 0.166, 0.120, 0.0857, 0.0612, 0.0457, 0.02845])


def model(beta, A, B):
    return A * np.exp(-B * beta)

# Using curve fitting from scipy with an initial guess of (1,1), giving us the constants A and B
parameters, covaraiance = scp.curve_fit(model, beta, a, p0 = (1.0,1.0))
A = parameters[0]
B = parameters[1]

beta_guess = np.arange(1.9, 2.2, 0.05)
a_guess = model(beta_guess, A, B)  

for i in range(len(beta_guess)):
    print(f"Beta = {beta_guess[i]}, a = {a_guess[i]:.6f}")

plt.scatter(beta, a, label='Known data',color='royalblue')
plt.plot(beta_guess, a_guess,'--', label='Extrapolated', color='crimson')
plt.title(r'Extrapolation of $a$ from $\beta$')
plt.xlabel(r'$\beta$')
plt.ylabel(r'$a$ (fm)')
plt.grid(True)
plt.legend()
plt.show()
