"""
This code analyses the noise (standard deviation) of the average plaquette for lattice sizes from 4^3 x 4 to 12^3 x 4. 
It calculates the standard deviation and how much it oscillates about the constant value of the average plaquette that is independent of lattice size. 
We also compose a function that calculates how the noise decreases as lattice size increases for any size lattice.
"""
import numpy as np
from numpy import log,pi,arccos,sin, cos
import matplotlib.pyplot as plt
from datetime import datetime
import statistics

plt.style.use('seaborn-v0_8-darkgrid')


start_time = datetime.now()

print('Start Time:', start_time.strftime('%H:%M:%S'))

beta = 1.9
nwarm = 20
runs = 100


directions = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])



def initial_kick(Nx,Ny,Nz,Nt):
    U = np.zeros((Nx, Ny, Nz, Nt, 4, 4))
    for i in range(Nx):
        for j in range(Ny):
            for k in range(Nz):
                for l in range(Nt):
                    for mu in range(4): 
                        U[i, j, k, l, mu] = random_vars()
    return U



def makematrix(u):
    return np.array([[u[0] + u[3]*1j, u[1]*1j+u[2]],[u[1]*1j-u[2],u[0]-u[3]*1j]])

def trace(u):
    return 2*u[0]
 


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

def random_vars():
    u = np.random.randn(4)
    norm = np.linalg.norm(u)
    return u/norm


def staple(x,mu,U,Nx,Ny,Nz,Nt):
    staple_sum = np.array([0,0,0,0])
    lattice = np.array([Nx,Ny,Nz,Nt])
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
    return stap[0] - beta/2 * trace(nn(U[tuple(x)+ (mu,)],stap))

def plaquette(x,mu,v,U,Nx,Ny,Nz,Nt):
    lattice = np.array([Nx,Ny,Nz,Nt])
    xmu = np.mod((x + directions[mu]),lattice)
    xv = np.mod((x + directions[v]), lattice)
    
    first = dd(U[tuple(xv)+ (mu,)],U[tuple(x)+ (v,)])
    second = nn(U[tuple(xmu)+ (v,)],first)
    third = nn(U[tuple(x)+ (mu,)],second)
    return 1/2 * trace(third)

def average_plaquette(U,Nx,Ny,Nz,Nt):
    plaq_sum = 0
    for i in range(Nx):
        for j in range(Ny):
            for k in range(Nz):
                for l in range(Nt):
                    for mu in range(4):
                        for v in range(mu+1,4):
                            plaq_sum += plaquette([i,j,k,l], mu, v,U,Nx,Ny,Nz,Nt)
                            
    return 1/(6*(Nx**3*Nt)) * plaq_sum 
                            

def norm(m):
    return np.sqrt(m[0]**2+m[1]**2+m[2]**2+m[3]**2)


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
    return 1 - d,count

def heatbath(U,beta,Nx,Ny,Nz,Nt):
    count = []
    for i in range(Nx):
        for j in range(Ny):
            for k in range(Nz):
                for l in range(Nt):
                    for mu in range(4):  
                        x = np.array([i, j, k, l])
                        stap = staple(x,mu,U,Nx,Ny,Nz,Nt)
                        xi = norm(stap)
                        s = stap/xi
                        a4 ,c= randfn(beta*xi) 
                        count.append(c)
                        r = np.sqrt(1-a4**2)
                        
                        th = np.pi * np.random.rand()
                        phi = 2 * np.pi * np.random.rand()
                        
                        cth = np.cos(th)
                        sth = np.sqrt(1 - cth * cth)  
                        
                        a = [a4,r * sth * np.cos(phi), r * sth * np.sin(phi), r * cth]
                        gnew = nd(a, s)
                        U[i,j,k,l,mu] = gnew
                        
    return U



constant = []
deviation = []

lattice = np.arange(4,13)

# Runs the algorithm for various lattice sizes and calulates the mean value of the plaquette aswell as the standard deviation

for i in range(4,13):
    U = initial_kick(i,i,i,4)
    plaquette_vals = []

    for _ in range(nwarm):
        U= heatbath(U,beta,i,i,i,4)
    
    for a in np.arange(1,runs):
        U = heatbath(U,beta,i,i,i,4)
        plaquette_vals.append(average_plaquette(U,i,i,i,4))
        
    constant.append(np.mean(plaquette_vals))
    deviation.append(np.std(plaquette_vals))


# Then we want to calculate the constant alpha for our fitted function allowing us to formulate a function for the noise of a given lattice

alpha_vals = []
for i in range(len(deviation)):
    N = i+4
    alpha_vals.append(deviation[i] * np.sqrt(N**3 * 4))
    
print("Alpha: ", np.mean(alpha_vals))
expected = []

for i in lattice:
    expected.append(np.mean(alpha_vals)/np.sqrt(i**3*4))
    
plt.figure()
plt.plot(lattice,deviation, label='Actual Values',color='royalblue')
plt.plot(lattice,expected,label='Calculated Function ',color='crimson')
plt.xlabel('Lattice Spatial extent')
plt.ylabel(r'$\sigma$')
plt.legend()
plt.grid(True)
plt.title('Plaquette Fluctuations vs Lattice Size')
plt.show()
    


end_time = datetime.now()
print('Duration: {}'.format(end_time - start_time))