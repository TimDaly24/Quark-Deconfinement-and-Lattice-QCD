"""
Here is the primary algorithm upon which all other code and functions depends. 
Here we perform the heatbath algorithm and overrelaxation method for a range of $\beta$ and sweeps. 
We calculate and store the average plaquette, Polyakov loop expectation and susceptibility and plot these values.
"""

import numpy as np
from numpy import log,pi,arccos,sin, cos
import matplotlib.pyplot as plt
from datetime import datetime

start_time = datetime.now()

print('Start Time:', start_time.strftime('%H:%M:%S'))
plt.style.use('seaborn-v0_8-darkgrid')

# Define lattice dimensions
Nx = 8
Ny = 8
Nz = 8
Nt = 4
betas = np.arange(2.0, 2.7, 0.05)
lattice = np.array([Nx,Ny,Nz,Nt])
nwarm = 20 # Number of sweeps for thermalisation
runs = 100 
discard = 1 # How many configurations we discard between measurements


directions = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])


# Here the initial kick is a hot start that gives each link variable a random configuration

def random_vars():
    u = np.random.randn(4)
    norm = np.linalg.norm(u)
    return u/norm

def initial_kick():
    U = np.zeros((Nx, Ny, Nz, Nt, 4, 4))
    for i in range(Nx):
        for j in range(Ny):
            for k in range(Nz):
                for l in range(Nt):
                    for mu in range(4): 
                        U[i, j, k, l, mu] = random_vars() # Returns a random normalised vector representing the four real numbers in the matrix representation
    return U


# Takes the four real parameters and returns the matrix form
def makematrix(u):
    return np.array([[u[0] + u[3]*1j, u[1]*1j+u[2]],[u[1]*1j-u[2],u[0]-u[3]*1j]])

def trace(u):
    return 2*u[0]

# Here we define how to multiply any two matrices in our chosen representation, n represents a normal matrix while d is the daggered version of the matrix.
def nn(u,v):
    w0 = u[0]*v[0]-u[3]*v[3]-u[1]*v[1]-u[2]*v[2]
    w1 = u[0]*v[1]+u[3]*v[2]+u[1]*v[0]-u[2]*v[3]
    w2 = u[0]*v[2]-u[3]*v[1]+u[1]*v[3]+u[2]*v[0]
    w3 = u[0]*v[3]+u[3]*v[0]-u[1]*v[2]+u[2]*v[1]
    return np.array([w0,w1,w2,w3])

def nd(u,v):
    w0 = u[0]*v[0]+u[3]*v[3]+u[1]*v[1]+u[2]*v[2]
    w1 = -u[0]*v[1]-u[3]*v[2]+u[1]*v[0]+u[2]*v[3]
    w2 = -u[0]*v[2]+u[3]*v[1]-u[1]*v[3]+u[2]*v[0]
    w3 = -u[0]*v[3]+u[3]*v[0]+u[1]*v[2]-u[2]*v[1]
    return np.array([w0,w1,w2,w3])

def dn(u,v):
    w0 = u[0]*v[0]+u[3]*v[3]+u[1]*v[1]+u[2]*v[2]
    w1 = u[0]*v[1]-u[3]*v[2]-u[1]*v[0]+u[2]*v[3]
    w2 = u[0]*v[2]+u[3]*v[1]-u[1]*v[3]-u[2]*v[0]
    w3 = u[0]*v[3]-u[3]*v[0]+u[1]*v[2]-u[2]*v[1]
    return np.array([w0,w1,w2,w3])

def dd(u,v):
    w0 = u[0]*v[0]-u[3]*v[3]-u[1]*v[1]-u[2]*v[2]
    w1 = -u[0]*v[1]+u[3]*v[2]-u[1]*v[0]-u[2]*v[3]
    w2 = -u[0]*v[2]-u[3]*v[1]+u[1]*v[3]-u[2]*v[0]
    w3 = -u[0]*v[3]-u[3]*v[0]-u[1]*v[2]+u[2]*v[1]
    return np.array([w0,w1,w2,w3])



# Returns the staple sum $S_\mu(x)$ for a given lattice site $x$ and direction $\mu$ while taking boundary conditions into account
def staple(x,mu,U):
    staple_sum = np.array([0,0,0,0])
    for v in range(4):
        if v != mu:
            xmu = np.mod((x + directions[mu]),lattice)
            x_plus_v  = np.mod((x + directions[v]), lattice)
            xmuv = np.mod((x + directions[mu] - directions[v]),lattice)
            x_minus_v = np.mod((x - directions[v]),lattice)
            
            first = (nn(U[tuple(xmu)+ (v,)],dd(U[tuple(x_plus_v)+ (mu,)],U[tuple(x)+ (v,)]))) 
            second = dn(U[tuple(xmuv)+ (v,)],dn(U[tuple(x_minus_v)+ (mu,)],U[tuple(x_minus_v)+ (v,)]))
            staple_sum = staple_sum + first + second 
    return staple_sum

def action(x,mu,U):
    stap = staple(x,mu,U)
    return stap[0] - betas/2 * trace(nn(U[tuple(x)+ (mu,)],stap))

def plaquette(x,mu,v,U):
    xmu = np.mod((x + directions[mu]),lattice)
    xv = np.mod((x + directions[v]), lattice)
    
    first = dd(U[tuple(xv)+ (mu,)],U[tuple(x)+ (v,)])
    second = nn(U[tuple(xmu)+ (v,)],first)
    third = nn(U[tuple(x)+ (mu,)],second)
    return 1/2 * trace(third)

def average_plaquette(U):
    plaq_sum = 0
    for i in range(Nx):
        for j in range(Ny):
            for k in range(Nz):
                for l in range(Nt):
                    for mu in range(4):
                        for v in range(mu+1,4):
                            plaq_sum += plaquette([i,j,k,l], mu, v,U)
                            
    return 1/(6*(Nx**3*Nt)) * plaq_sum 
                            

def norm(m):
    return np.sqrt(m[0]**2+m[1]**2+m[2]**2+m[3]**2)

# Returns the value $a_4$ for use in the heatbath algorithm when obtaining a new link configuration
def randfn(a):
    R = 2
    d = 0
    count = 0
    while R**2 > 1 - d/2:
        X1 = -np.log(np.random.rand())/a
        X2 = -np.log(np.random.rand())/a
        C = np.cos(2 * np.pi * np.random.rand())**2
        d = X1 * C + X2
        R = np.random.rand()
        count += 1
        a_4 = 1-d
    return a_4,count


def heatbath(U,beta):
    count = []
    for i in range(Nx):
        for j in range(Ny):
            for k in range(Nz):
                for l in range(Nt):
                    for mu in range(4):  
                        x = np.array([i, j, k, l])
                        stap = staple(x,mu,U)
                        xi = norm(stap)
                        s = stap/xi
                        a4, c= randfn(beta*xi) 
                        count.append(c)
                        r = np.sqrt(1-a4**2)
                        
                        # Generate random angles
                        th = np.pi * np.random.rand()
                        phi = 2 * np.pi * np.random.rand()
                        
                        cth = np.cos(th)
                        sth = np.sqrt(1 - cth * cth)  
                        
                        a = [a4,r * sth * np.cos(phi),r * sth * np.sin(phi), r * cth]
                        g_new = nd(a, s)
                        U[i,j,k,l,mu] = g_new
                        
    return U
    
# Overrelaxation algorithm to be run 5 times after each heatbath update to help explore the sample space more efficiently 
def overrelaxation(U):
    for i in range(Nx):
        for j in range(Ny):
            for k in range(Nz):
                for l in range(Nt):
                    for mu in range(4):  
                        x = np.array([i, j, k, l])
                        stap = staple(x,mu, U)
                        s = stap/norm(stap)
                        s = [s[0],-s[1],-s[2],-s[3]]
                        w = dd(s,U[x[0],x[1],x[2],x[3],mu]) 
                        g_new = nd(w,s)
                        U[i, j, k, l, mu] = g_new
                    
    return U

# Function to test that direction around plaquette is irrelevant, useful for checking the plaquette function is working correctly 
def test_plaquette(x,mu,v):       
    ting = initial_kick()     
    clockwise = plaquette(x, mu, v,ting)
    counter = plaquette(x, v, mu, ting)
    print(clockwise)
    print(counter)
    

# Calculation of Polyakov loop, we compute $L$ and $L^2$ for the susceptibility
def polyakov_loop(U):
    total_p = 0
    total_p2 = 0
    spatial = Nx * Ny * Nz

    for i in range(Nx):
        for j in range(Ny):
            for k in range(Nz):
                pl = U[i, j, k, 0, 3]

                for t in np.arange(1, Nt):
                    pl = nn(pl, U[i, j, k, t, 3])
                
                p = trace(pl)
                total_p += p
                total_p2 += (p**2)

    avg_p = total_p/spatial
    avg_p2 = total_p2/spatial

    return avg_p, avg_p2

# Tadpole improvement to be implemented after thermalisation to improve approximation to continuous model
def tadpole(U):
    u_0 = (average_plaquette(U))**0.25
    U_new = U/u_0
    return U_new


L_avg = []
L2_avg = []
plaq_betas = []

# Here we apply the heatbath and overrelaxation methods looping over all values of beta 

for b in range(len(betas)):
    L_vals = []
    L_2vals =[]
    U = initial_kick() # Give all the link variables an intial radnom configurations
    plaquette_vals = []

    # Thermalisation
    for _ in range(nwarm):
        U= heatbath(U,betas[b])
        
    U = tadpole(U) # Apply tadpole improvement after thermalisation
        
    for a in range(runs):
        U = heatbath(U, betas[b])

        for _ in range(5):
           U = overrelaxation(U)
           
        L,L2 = polyakov_loop(U)
        L_vals.append(L)
        L_2vals.append(L2)
        plaquette_vals.append(average_plaquette(U))
    

    plaq_betas.append(np.mean(plaquette_vals))
    L_avg.append(np.mean(L_vals))
    L2_avg.append(np.mean(L_2vals))
    print(f"Beta:{betas[b]}")
    
        
volume = Nx * Ny * Nz * Nt
L_avg = np.array(L_avg)
L2_avg = np.array(L2_avg)
var = volume*(L2_avg - L_avg**2)


# Expectation value of Polyakov loop
plt.scatter(betas, L_avg, color='crimson', label=r'$\langle L \rangle$', s=50)
plt.title(r'Expectation Value of Polyakov Loop vs $\beta$')
plt.xlabel(r'$\beta$')
plt.ylabel(r'$\langle L \rangle$')
plt.grid(True)
plt.show()

# Susceptibility
plt.scatter(betas, var, color='green', label='Susceptibility', s=50)
plt.title(r'Polyakov Loop Susceptibility $\chi_L$')
plt.xlabel(r'$\beta$')
plt.ylabel('Susceptibility')
plt.grid(True)
plt.show()

end_time = datetime.now()
print('Duration: {}'.format(end_time - start_time))