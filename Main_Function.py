import numpy as np
from numpy import log,pi,arccos,sin, cos
import matplotlib.pyplot as plt
from datetime import datetime







def unitary(M):
    mat = np.array(makematrix(M))  # Ensure it's a NumPy array
    dag = mat.conj().T  # Compute conjugate transpose
    identity = np.eye(2)  # 2x2 Identity matrix
    is_unitary = np.allclose(np.matmul(mat, dag), identity)
    is_su2 = np.isclose(np.linalg.det(mat), 1)
    
    return is_unitary and is_su2

def initial_kick1():
    # Performance note here might be more efficient to have the loop in a different order
    U = np.zeros((Nx, Ny, Nz, Nt, 4, 4))
    for i in range(Nx):
        for j in range(Ny):
            for k in range(Nz):
                for l in range(Nt):
                    for mu in range(4): # Pretty sure this should be 4
                        U[i, j, k, l, mu] = random_vars()
    return U

def initial_kick():
    U = 2 * np.random.rand(Nx, Ny, Nz, Nt, 4, 4) - 1
    
    # Normalize the gauge field
    for x in range(Nx):
        for y in range(Ny):
            for z in range(Nz):
                for t in range(Nt):
                    for mu in range(4):
                        u = U[x, y, z, t, mu]# has a colon here maybe 
                        U[x, y, z, t, mu] = u / norm(u)
    
    return U




# Takes the parameters and returns the matrix in terms of the pauli basis
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
    w0 = u[0]*v[0]-u[3]*v[3]-u[1]*v[1]-u[2]*v[2] # changed this line 
    w1 = -u[0]*v[1]+u[3]*v[2]-u[1]*v[0]-u[2]*v[3]
    w2 = -u[0]*v[2]-u[3]*v[1]+u[1]*v[3]-u[2]*v[0]
    w3 = -u[0]*v[3]-u[3]*v[0]-u[1]*v[2]+u[2]*v[1]
    return np.array([w0,w1,w2,w3])

def random_vars():
    u = np.random.randn(4)
    norm = np.linalg.norm(u)
    return u/norm


def staple(x,mu,U):
    staple_sum = np.array([0,0,0,0])
    for v in range(4):
        if v != mu:
            xmu = np.mod((x + directions[mu]),lattice)
            x_plus_v  = np.mod((x + directions[v]), lattice)
            xmuv = np.mod((x + directions[mu] - directions[v]),lattice)
            x_minus_v = np.mod((x - directions[v]),lattice)
            
            first = (nn(U[tuple(xmu)+ (v,)],dd(U[tuple(x_plus_v)+ (mu,)],U[tuple(x)+ (v,)]))) # changed to a v here
            second = dn(U[tuple(xmuv)+ (v,)],dn(U[tuple(x_minus_v)+ (mu,)],U[tuple(x_minus_v)+ (v,)]))
            staple_sum = staple_sum + first + second # I'm pretty sure its ok here adding the 4 numbers instead of matrices
    return staple_sum

def action(x,mu,U):
    stap = staple(x,mu,U)
    return stap[0] - beta/2 * trace(nn(U[tuple(x)+ (mu,)],stap))

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


def randfn(a):
    R = 2
    d = 0
    count = 0
    while R**2 > 1 - d / 2:
        X1 = -np.log(np.random.rand()) / a
        X2 = -np.log(np.random.rand()) / a
        C = np.cos(2 * np.pi * np.random.rand())**2
        d = X1 * C + X2
        R = np.random.rand()
        count += 1
    return 1 - d,count

def heatbath(U,beta):
    count = []
    for i in range(Nx):
        for j in range(Ny):
            for k in range(Nz):
                for l in range(Nt):
                    for mu in range(4):  # Pretty sure this should be 4
                        x = np.array([i, j, k, l])
                        stap = staple(x,mu,U)
                        xi = norm(stap)
                        s = stap/xi
                        a4 ,c= randfn(beta*xi) #Changed this to beta*xi
                        count.append(c)
                        r = np.sqrt(1-a4**2)
                        
                        
                        # Generate random angles
                        th = np.pi * np.random.rand()
                        phi = 2 * np.pi * np.random.rand()
                        
                        # Compute trigonometric values
                        cth = np.cos(th)
                        sth = np.sqrt(1 - cth * cth)  
                        
                        a = [a4,r * sth * np.cos(phi), r * sth * np.sin(phi), r * cth]
                        #a = [0.4,0.1,0.2,0.3]
                        gnew = nd(a, s)
                        U[i,j,k,l,mu] = gnew
                        
    return U
    
    
def overrelaxation(U):
    for i in range(Nx):
        for j in range(Ny):
            for k in range(Nz):
                for l in range(Nt):
                    for mu in range(4):  
                        x = np.array([i, j, k, l])
                        stap = staple(x,mu,U)
                        s = stap/norm(stap)
                        # They have a step here where they take the inverse of 
                        s = [s[0],-s[1],-s[2],-s[3]]# Not sure about this here 
                        w = dd(s,U[x[0],x[1],x[2],x[3],mu]) #changed this from nd
                        gnew = nd(w,s)# changed this from nn
                        U[i, j, k, l, mu] = gnew
                    
    return U

def test_plaquette(x,mu,v):       
    ting = initial_kick()     
    
    clockwise = plaquette(x, mu, v,ting)
    counter = plaquette(x, v, mu, ting)
    print(clockwise)
    print(counter)
    
    
def polyakov_loops(U):
    total = 0
    spatial_extent = Nx * Ny * Nz

    for i in range(Nx):
        for j in range(Ny):
            for k in range(Nz):
                pl = U[i,j,k,0,3]

                for t in np.arange(1,Nt):  
                    pl = nn(pl,U[i, j, k, t, 3])  # Not quite sure of matrix multiplication order here
                
                total += trace(pl) #maybe a 1/3 here 

    return total/ spatial_extent# So this returns 



def polyakov_loop(U):
    total_P = 0
    total_P2 = 0
    spatial_extent = Nx * Ny * Nz

    for i in range(Nx):
        for j in range(Ny):
            for k in range(Nz):
                pl = U[i, j, k, 0, 3]

                for t in np.arange(1, Nt):
                    pl = nn(pl, U[i, j, k, t, 3])
                
                P = trace(pl)
                total_P += P
                total_P2 += (P * P)

    avg_P = total_P / spatial_extent
    avg_P2 = total_P2 / spatial_extent

    return avg_P, avg_P2

def tadpole(U):
    u_0 = (average_plaquette(U))**0.25
    U_new = U/u_0
    return U_new
    


#loaded_flat_matrix = np.loadtxt("matrix.txt")  # Load the saved data
#U = loaded_flat_matrix.reshape(Nx, Ny, Nz, Nt, 4, 4)  #Reshape
#U = initial_kick()


def main():
    L_matrix = np.zeros((len(betas), runs))
    L_avg = []
    L2_avg = []
    plaq_betas = []

    for b in range(len(betas)):
        L_vals = []
        L_2vals =[]
        U = initial_kick()
        plaquette_vals = []
        
        for _ in range(nwarm):
            U= heatbath(U,betas[b])
            
        #U = tadpole(U)
            
        for a in range(runs):#changed from arange
            U = heatbath(U,betas[b])
            
            
            
            L,L2 = polyakov_loop(U)
            L_vals.append(L)
            L_2vals.append(L2)
            #for _ in range(5):
               # U = overrelaxation(U)
            
            plaquette_vals.append(average_plaquette(U))
            #poly_vals.append(polyakov_loop(U))

        plaq_betas.append(np.mean(plaquette_vals))
        L_avg.append(np.mean(L_vals))
        L2_avg.append(np.mean(L_2vals))
        print(f"Beta:{betas[b]}")
        
            
    lattice_volume = Nx * Ny * Nz * Nt
    L_avg = np.array(L_avg)
    L2_avg = np.array(L2_avg)
    var = lattice_volume*(L2_avg - L_avg**2)


    # Plot 2: Expectation value of Polyakov loop
    plt.figure(figsize=(8, 6))
    plt.scatter(betas, L_avg, color='crimson', label=r'$\langle L \rangle$', s=50, edgecolors='black', alpha=0.7)
    plt.title(r'Expectation Value of Polyakov Loop vs $\beta$', fontsize=16)
    plt.xlabel(r'$\beta$', fontsize=14)
    plt.ylabel(r'$\langle L \rangle$', fontsize=14)
    plt.grid(True, linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.show()

    # Plot 3: Susceptibility
    plt.figure(figsize=(8, 6))
    plt.scatter(betas, var, color='seagreen', label='Susceptibility', s=50, edgecolors='black', alpha=0.7)
    plt.title(r'Polyakov Loop Susceptibility $\chi_L$', fontsize=16)
    plt.xlabel(r'$\beta$', fontsize=14)
    plt.ylabel('Susceptibility', fontsize=14)
    plt.grid(True, linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.show()

   


start_time = datetime.now()

print('Start Time:', start_time.strftime('%H:%M:%S'))
plt.style.use('seaborn-v0_8-darkgrid')

#Define lattice dimensions
Nx = 8
Ny = 8
Nz = 8
Nt = 4
betas = np.arange(2.0,2.5,0.05)# Changed to betas
lattice = np.array([Nx,Ny,Nz,Nt])
nwarm = 20
runs = 150 # Maybe not so many iterations
discard = 1


directions = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])

main()

end_time = datetime.now()
print('Duration: {}'.format(end_time - start_time))