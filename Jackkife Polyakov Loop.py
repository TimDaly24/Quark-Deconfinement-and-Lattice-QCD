"""
This performs a Jackknife analysis of the Polyakoov loop expectatio nvalue and suscpetibility. 
It makes use of values obtained from the main algorithm and saved for analysis. 
I have split up the values between the confined and deconfined phase to give an accurate representation of mean values and errors as the phase transition
 introduces a sharp increase that can skew the average values if not dealt with correctly. It also calculates the unbiased operator which gives a mean value of the observable accounting for the error.
"""

import numpy as np
import matplotlib.pyplot as plt
import statistics

# Take in saved values for the Polyakov loop
poly = np.array(np.loadtxt('L_avg_1.9_2.3_0.025.txt'))
suscep = np.array(np.loadtxt('L2_avg_1.9_2.3_0.025.txt'))

poly2 = np.array(np.loadtxt('L_avg_2.3_2.4_0.025.txt'))
suscep2 = np.array(np.loadtxt('L2_avg_2.3_2.4_0.025.txt'))


loop = np.concatenate((poly, poly2))
susceptibility = np.concatenate((suscep, suscep2))

betas = np.arange(1.9, 2.35, 0.025)
index = np.where(np.isclose(betas, 2.20))[0] # get index of phase transiton 
volume = 8 * 8 * 8 * 4

def jackknife(L):
    
    # Split into confined and deconfined phases
    L1 = L[:12]
    L2 = L[12:]
    
    # Calculate overall mean
    L_mean_con = np.mean(L1)
    L_mean_decon = np.mean(L2)
    
    L2_mean_con = np.mean(L1**2) 
    L2_mean_decon = np.mean(L2**2) 
    
    chi_con = volume * (L2_mean_con -L_mean_con**2)
    chi_decon = volume * (L2_mean_decon - L_mean_decon**2)
    
    
    std_vals_chi_con = 0
    std_vals_chi_decon = 0
    
    std_vals_L_con = 0
    std_vals_L_decon = 0
    N1 = len(L1)
    N2 = len(L2)
    
    bias_chi_con = 0
    bias_L_con = 0
    
    bias_chi_decon = 0
    bias_L_decon = 0
    
    # Errors for confined phase
    for n in range(N1):
        theta_L = np.delete(L1,n)
        theta_L_mean = np.mean(theta_L)
        theta_L2_mean = np.mean(theta_L**2)
        chi_theta = (volume * (theta_L2_mean - theta_L_mean**2))
        std_vals_chi_con += (chi_theta-chi_con)**2
        std_vals_L_con += (theta_L_mean - L_mean_con)**2
        
        bias_chi_con += chi_theta
        bias_L_con += theta_L_mean
        
    # Errors for deconfined phase    
    for n in range(N2):
        theta_L = np.delete(L2,n)
        theta_L_mean = np.mean(theta_L)
        theta_L2_mean = np.mean(theta_L**2)
        chi_theta = (volume * (theta_L2_mean - theta_L_mean**2))
        std_vals_chi_decon += (chi_theta-chi_decon)**2
        std_vals_L_decon += (theta_L_mean - L_mean_decon)**2
        
        bias_chi_decon += chi_theta
        bias_L_decon += theta_L_mean
        
    chi_thetacon = np.sqrt((N1-1)/N1 * std_vals_chi_con)
    chi_thetadecon = np.sqrt((N2-1)/N2 * std_vals_chi_decon)
    
    L_con = np.sqrt((N1-1)/N1 * std_vals_L_con)
    L_decon = np.sqrt((N2-1)/N2 * std_vals_L_decon)
    
    # Unbiased operator calculations
    chi_con_tilde =  1/N1 * bias_chi_con
    L_con_tilde =  1/N1 * bias_L_con
    
    chi_decon_tilde =  1/N2 * bias_chi_decon
    L_decon_tilde =  1/N2 * bias_L_decon
    
    unbiased_chi_con = chi_con - (N1-1) * (chi_con_tilde- chi_con)
    unbiased_L_con = L_mean_con - (N1-1) * (L_con_tilde- L_mean_con)
    
    unbiased_chi_decon = chi_decon - (N2-1) * (chi_decon_tilde- chi_decon)
    unbiased_L_decon = L_mean_decon - (N2-1) * (L_decon_tilde- L_mean_decon)
    
    return chi_thetacon, L_con, chi_thetadecon, L_decon, chi_con, chi_decon, L_mean_con,L_mean_decon, unbiased_chi_con,unbiased_chi_decon,unbiased_L_con,unbiased_L_decon


chi_con_err, L_con_err, chi_decon_err, L_decon_err, chi_con, chi_decon, L_mean_con, L_mean_decon, unbiased_chi_con,unbiased_chi_decon,unbiased_L_con,unbiased_L_decon = jackknife(loop)

print("Confined phase:")
print(f"L = {L_mean_con} + {L_con_err}, chi= {chi_con} + {chi_con_err}")


print("Deconfined phase:")
print(f"L = {L_mean_decon} + {L_decon_err}, chi = {chi_decon} + {chi_decon_err}")

print("Confined phase unbiased:")
print(f"  L = {unbiased_L_con}   chi = {unbiased_chi_con}")
print("Deconfined phase unbiased:")
print(f"  L = {unbiased_L_decon}  chi = {unbiased_chi_decon}")


        
    
        
        
    