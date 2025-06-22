import numpy as np
import matplotlib.pyplot as plt
import statistics

# Load in precalculated values
plaqs = (np.loadtxt('plaq_betas.txt'))
N = len(plaqs)
volume = 8 * 8 * 8 * 4 

def jackknife(plaq):
    plaq_mean = np.mean(plaq)
    std_vals_plaq = 0
    bias = 0
    
    for n in range(N):
        theta_plaq = np.delete(plaq,n)
        theta_plaq_mean = np.mean(theta_plaq)
        std_vals_plaq += (theta_plaq_mean -plaq_mean)**2
        bias += theta_plaq_mean
        
    theta_tilde =  1/N * bias
    unbiased = plaq_mean - (N-1)*(theta_tilde - plaq_mean)
    
    return np.sqrt((N-1)/N * std_vals_plaq) , plaq_mean, unbiased

plaq_error, plaq_mean, unbiased = jackknife(plaqs)

print(f" plaquette : {plaq_mean} + {plaq_error}")
print(f"Unbiased : {unbiased}")


        
    