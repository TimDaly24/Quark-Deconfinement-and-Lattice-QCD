"""
This code analyses the burn-in period and convergence of the average plaquettte to a roughly constant value for various lattice sizes

"""
import numpy as np
from numpy import log,pi,arccos,sin, cos
import matplotlib.pyplot as plt
from datetime import datetime

plt.style.use('seaborn-v0_8-darkgrid')
start_time = datetime.now()

print('Start Time:', start_time.strftime('%H:%M:%S'))

beta = 1.9
runs = 200


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



plaquette_vals4 = []
plaquette_vals8 = []
action_vals = []

        
U4 = initial_kick(4,4,4,4)

# For the $8^3\times 4$ matrix we load in a precalculated array
loaded_flat_matrix = np.loadtxt("matrix.txt") 
U8 = loaded_flat_matrix.reshape(8, 8, 8, 4, 4, 4)  

# Calculating average plaquette values and updating link variables using the heatbath algorithm
for a in np.arange(1,runs):
    
    U4 = heatbath(U4,beta,4,4,4,4)
    U8 = heatbath(U8,beta,8, 8, 8, 4)
    
        
    plaquette_vals4.append(average_plaquette(U4,4,4,4,4))
    plaquette_vals8.append(average_plaquette(U8,8,8,8,4))


plt.plot(plaquette_vals4, label=r'$4^3\times 4$ lattice', color='royalblue')
plt.plot(plaquette_vals8, label=r'$8^3\times 4$ lattice', color='crimson')
plt.title('Average Plaquette Values vs Configurations')
plt.xlabel('Configurations')
plt.ylabel('Average Plaquette')
plt.grid(True)
plt.legend()
plt.show()


end_time = datetime.now()
print('Duration: {}'.format(end_time - start_time))

