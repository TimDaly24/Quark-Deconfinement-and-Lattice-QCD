import numpy as np
import matplotlib.pyplot as plt

plt.style.use('seaborn-v0_8-darkgrid')
poly = np.array(np.loadtxt('L_avg_1.9_2.3_0.025.txt'))
suscep = np.array(np.loadtxt('L2_avg_1.9_2.3_0.025.txt'))

poly2 = np.array(np.loadtxt('L_avg_2.3_2.4_0.025.txt'))
suscep2 = np.array(np.loadtxt('L2_avg_2.3_2.4_0.025.txt'))

# Combine arrays
loop = np.concatenate((poly, poly2))
susceptibility = np.concatenate((suscep, suscep2))

# Remove last two elements
loop = loop[:-2]
susceptibility = susceptibility[:-2]

# Define corresponding beta values (also remove last two values)
betas = np.arange(1.9, 2.35, 0.025)

# Plot Polyakov loop
plt.figure(figsize=(8, 6))
plt.scatter(betas, loop, color='royalblue', label=r'$\langle L \rangle$', s=50,edgecolors='black', alpha=0.7)
#plt.plot(betas,loop)
plt.title(r'Expectation Value of Polyakov Loop vs $\beta$', fontsize=16)
plt.xlabel(r'$\beta$', fontsize=14)
plt.ylabel(r'$\langle L \rangle$', fontsize=14)
plt.grid(True, linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.show()

# Plot Susceptibility
plt.figure(figsize=(8, 6))
plt.scatter(betas, susceptibility, color='seagreen', label='Susceptibility', s=50,edgecolors='black', alpha=0.7)
#plt.plot(betas, susceptibility)
plt.title(r'Polyakov Loop Susceptibility $\chi_L$', fontsize=16)
plt.xlabel(r'$\beta$', fontsize=14)
plt.ylabel('Susceptibility', fontsize=14)
plt.grid(True, linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.show()
