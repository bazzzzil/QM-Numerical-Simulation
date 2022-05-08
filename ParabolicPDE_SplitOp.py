"""
Date: Tuesday, November 25, 2019
Author: Layale Bazzi

Solving PDEs (Schrodinger equation) using the split operator methods. 
Both split operator techniques are used to deal with
non-commuting operators in the exponential argument, generally due to 
time-evolution of a wavefunction. This is treated first by a Taylor expansion
and clever algebraic manipulation to turn an expression exp(A+B) into 
exp(A/2)exp(B)exp(A/2). The Baker-Campbell-Hausdorff lemma can also be used to
split the exponential operator.

The next step depends on the type of technique to be implemented. The general 
split step operation requires that an explicit basis change is performed for one
of the two operators in the sequence. As solutions are frequently displayed in 
position space, the momentum operator is chosen to undergo a basis change.
The momentum matrix is diagonalized and applied in sequence with its unitary
transformation matrices to act on the initial wavefunction. This is done 
iteratively over time to compute the wavefunction at every time step.

The second method involves using the Fast Fourier Transform (FFT) algorithm to
change the space we are working in to properly apply the momentum operator. 
This is identical to a basis change but is much faster than a diagonalization
procedure. The first potential operator acts on the initial wavefunction, then
it is FFT'd into momentum (k) space, where the momentum operator in its own
basis is not a derivative, but a multiplication. Then, after the operation, the
resultant is IFFT'd back into position space, where the second potential term
acts on the rest of the operation. This is done iteratively over time to 
compute the wavefunction at every time step.

This code can solve time dependent and time-independent QHO potentials by 
editing the value of the angular frequency. If a time-independent potential is to
be used, please edit the w variable so that it is an integer and not an array.
If a time-dependent potential is to be used, please edit the w variable so that
it is an array and not an integer.

In this code two cases are solved: arbitrary and Gaussian wavefunctions.

Method used: Split Operator Method, FFT Split Operator Method
"""
#%% Importing necessary packages
# numpy for calculations, matplotlib.pyplot for plotting, scipy for fft & linalg
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from scipy.fftpack import fft, ifft, fftshift, ifftshift
from scipy import linalg
import NumMethods as nm

###############################################################################
## Functions ##
###############################################################################

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# fft_op: Performs the shifted fast fourier transform of any input function
#          (real array). This is so that the result is displayed with the 
#          centre frequency in the middle index.
def fft_op(func_x):
    return fftshift(fft(fftshift(func_x)))
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ifft_op: Performs the inverse shifted fast fourier transform of any input
#           function (real array). This is so that the result is displayed with 
#           the centre spatial/temporal variable in the middle index.
def ifft_op(func_k):
    return ifftshift(ifft(ifftshift(func_k)))
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Function that streamlines plotting the wavefunction inside the QHO potential
# avg_H is the exectation value of energy used to raise the wavefunction up such
# that it lies on the energy level that keeps it bounded in the potential (used
# mostly for aesthetic reasons). 
def wfxnPlot(V,wfxn,x,L,avg_H):
    plt.plot(x,V)
    plt.ylim([0,2+avg_H])
    plt.xlim([-L,L])
    plt.plot(x,np.abs(wfxn)+avg_H)
    plt.xlabel('Position')
    plt.ylabel('Amplitude')
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Function that plots TDSE solutions to eigenfunction solutions for Problem 1
# The function plots the wavefunction at the final point in time
def eigenQHO(x,t,wfxn,wfxn_true): 
    # Overlap plot
    plt.figure()
    plt.plot(x,np.abs(wfxn[:,-1]),label='Computed')
    plt.plot(x,np.abs(wfxn_true[:,-1]),label='True')
    plt.xlabel('Position')
    plt.ylabel('Amplitude')
    plt.title('Comparison of Computed and True Wavefunctions (QHO)')
    plt.legend()
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Function that plots the the 3D representation of the wavefunction in time
# and in space. The second part plots the initial wavefunction with the 
# wavefunction at a later time in the same position. For a time-dependent 
# potential, the packet should be "thinner" as the potential narrows over time.
def miscPlots(x,t,wfxn):
    # 3D plot
    X,Y=np.meshgrid(x,t)
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_surface(X, Y, np.abs(wfxn[:,:].transpose()))
    plt.xlabel('Position')
    plt.ylabel('Time')
    plt.title('Full representation of time-evolved wavefunction in QHO')
    
    # Overlap plot
    # If plotting time-independent wavefunction, use positions 0 and 62
    # If plotting time-dependent wavefunction ,use positions 0 and 87
    # In reference to the two lines below plt.figure()
    plt.figure()
    plt.plot(x,np.abs(wfxn[:,0]),label='Initial')
    plt.plot(x,np.abs(wfxn[:,87]),label='Final')
    plt.xlabel('Position')
    plt.ylabel('Amplitude')
    plt.title('Comparison of Initial and Final Wavefunctions (QHO)')
    plt.legend()
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# DiagForm: Takes some non-diagonal matrix T and returns it diagonalized and in
#           exponential form with its unitary matrix transformation U. The 
#           eigenvalues g are the arguments in the exponential form of T.
def DiagForm(T,tau):
    # Finding eigenvectors/values of T
    # g is array of eigenvalues, v is matrix of eigenvectors
    g,v=linalg.eigh(T)

    # Converting eigenvectors to unitary matrix U
    U=np.matrix(v)

    # Creating exponential matrix for T
    # Eigenvalues g become arguments in the exponential T operator
    exp_T=np.diag(np.exp(-1j*tau*g))
    return U, exp_T
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# SplitOp: SplitOp is the general split operator method where the exponential
#           argument containing two non-commuting operators is split into three
#           parts for calculations. This function is for one potential and one
#           kinetic term, where the kinetic term is inserted in its diagonalized
#           exponential form. V is the potential term in matrix form, exp_T is
#           the diagonalized kinetic term in exponential matrix form, U is the
#           unitary basis change for the kinetic term, wfxn is the wavefunction
#           at a point in time, t is the time array and dt is the time-step.
#           N and nt are the number of space and time points, respectively.
def SplitOp(V,exp_T,U,wfxn,dt):
    exp_V=np.diag(np.exp(-0.5j*dt*V))
    # The line below is the operation: 
    # exp(-i*V*tau/2)*U*exp(-i(d^2/dx^2)*tau)*U^(dagger)*exp(-i*V*tau/2)*wfxn
    return (exp_V @ U @ exp_T @ U.getH() @ exp_V).dot(wfxn)
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# SplitFFT: SplitFFT is the FFT split operator method where the exponential
#           argument containing two non-commuting operators is split into three
#           parts for calculations. This function is for one potential and one
#           kinetic term, where the kinetic term is inserted in its momentum
#           space form. V is the potential term in matrix form, exp_k is
#           the k-space kinetic term in exponential matrix form, wfxn is the
#           wavefunction at a point in time and dt is the time-step. N and nt
#           are the number of space and time points, respectively.
def SplitFFT(V,exp_k,wfxn,dt):
    exp_V=np.exp(-0.5j*dt*V)
    # The line below is the operation: 
    # exp(-i*V*tau/2)*ifft[exp(-(i/2)k^2*tau)*fft[*exp^(-i*V*tau/2)*wfxn)]]
    return exp_V*ifft_op(exp_k*fft_op(exp_V*wfxn))
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Calculates expectation value of energy of QHO
def expect_H(wfxn,L,w0,x,N):
    ## Finding <H> by doing <V>+<T> ##
    
    ## Expectation value of position squared ##
    # First doing <V>, which is just <x^2>*0.5*w^2 in atomic units
    avg_V=np.real(nm.simpson(N,x,np.conj(wfxn)*(x**2)*wfxn,-L,L)*0.5*w0**2)
    print("Average over expectation values of V, <V>:",'{:,.5}'.format(np.average(avg_V)))
    
    ## Expectation value of momentum squared ##
    # Now doing <T>, which is just <p^2>*0.5 = -1*<d^2/dx^2>*0.5 in atomic units
    d2wfxn=np.zeros(N,dtype=complex) # allocating memory
    d2wfxn=nm.three_point_two(N,x,wfxn,-L,L) # calculating 2nd derivative of wfxn
    
    avg_T=np.real(nm.simpson(N-2,x[1:N-1],np.conj(wfxn[1:N-1])*d2wfxn[1:N-1],-L,L)*-0.5)
    print("Expectation value of T, <T>:",'{:,.5}'.format(avg_T))
    
    ## Expectation value of Hamiltonian ##
    # <H>=<V>+<T>
    return avg_V+avg_T

#%%
###############################################################################
## Using Split Operator Methods to Evolve Wavefunction in QHO
###############################################################################
# Evolution of arbitrary wavefunction composed of linear combination of energy 
# eigenfunctions in Quantum Harmonic Oscillator (QHO)

print("Solving time-dependent QHO for arbitrary wavefunction"
      " composed of energy eigenfunctions.")

"""#### Defining Parameters ####"""
N=750 # Number of space points
L=10 # Width being solved over
x=np.linspace(-L,L,N) # Grid of space points
dx=x[1]-x[0] # Spatial step
nt=100 # Number of time points
t=np.linspace(0,10,nt) # Grid of time points
dt=t[1]-t[0] # Time step
dk=2*np.pi/(N*dx) # spectral step
k=np.arange(-(N-1)*dk/2,N*dk/2,dk) # Grid of k-space

"""#### Wavefunction and Potential Parameters ####"""
# Frequency of harmonic oscillator potential
w0=1 # Time-independent potential

# Eigenfunctions of QHO (Hermite polynomials)
wfxn0=np.exp(-w0*0.5*x**2)
wfxn1=np.sqrt(2*w0)*x*wfxn0
wfxn2=(1/np.sqrt(2))*(2*w0*x**2-1)*wfxn0
wfxn3=(1/(np.sqrt(3))*(2*np.sqrt(w0)**3*x**3-3*np.sqrt(w0)*x))*wfxn0
# Initial wavefunction
wfxn=(1/np.sqrt(3))*(wfxn1+wfxn2+wfxn3)

print("Normalizing wavefunction...")
S=nm.simpson(N,x,wfxn*np.conj(wfxn),-L,L)
A=1/np.sqrt(S)
wfxn=A*wfxn

# Harmonic Oscillator potential 
# np.outer used to generate matrix (outer product) if time-dependent
V=0.5*w0**2*x**2

# Energy expectation value
avg_H=expect_H(wfxn,L,w0,x,N)
print("Average over expectation values of H, <H>:",'{:,.5}'.format(np.average(avg_H)))

print("Number of points in space, N:", N)
print("Number of points in time, nt:", nt)
print("Spatial step size, dx:",'{:,.5}'.format(dx))
print("Temporal step size, dt:", '{:,.5}'.format(dt))
print("Spectral step size, dk:",'{:,.5}'.format(dk))

"""#### True Solution ####"""

n=np.array([0,1,2,3])
En=w0*(n+0.5)

wfxn_true=np.zeros([len(x),len(t)],dtype=complex)
wfxn_true[:,0]=wfxn

for i in range(len(t)-1):
    wfxn_true[:,i+1]=(1/np.sqrt(4))*(np.exp(-1j*En[0]*t[i])*wfxn0
                    +np.exp(-1j*En[1]*t[i])*wfxn1
                    +np.exp(-1j*En[2]*t[i])*wfxn2+np.exp(-1j*En[3]*t[i])*wfxn3)

# Normalization
print("Normalizing true solution...")
S=nm.simpson(N,x,wfxn_true[:,-1]*np.conj(wfxn_true[:,-1]),-L,L)
A=1/np.sqrt(S)
wfxn_true=A*wfxn_true

#%%
"""#### General Split Operator Method ####"""

# Applying general Split Operator method to solve TDSE

# Setting up kinetic energy matrix
# 3-pt formula for second derivative
lower_diag = np.ones(N-1)
main_diag = -2*np.ones(N) 
upper_diag = np.ones(N-1)
Tfac=1/(dx**2)

T=-0.5*Tfac*nm.tridiag(lower_diag,main_diag,upper_diag)
U,exp_T=DiagForm(T,dt)

# Allocating and initializing wavefunction
wfxn_sol=np.zeros([N,nt],dtype=complex)
wfxn_sol[:,0]=wfxn

# Solving TDSE using split operator
for j in range(nt-1):
    wfxn_sol[:,j+1]=SplitOp(V,exp_T,U,wfxn_sol[:,j],dt)

# Displaying results #

# Plotting initial wavefunction inside initial potential
plt.figure()
wfxnPlot(V,wfxn_sol[:,0],x,L,avg_H)
plt.title('Wavefunction of QHO at time t=0')
plt.show()

# Plotting all timesteps at once on the same plot
plt.figure()
for j in range(nt-1):
    wfxnPlot(V,wfxn_sol[:,j],x,L,avg_H)
    plt.title('Time evolution of QHO')
    plt.show()

# Checking if time-evolution retained normalization of function at the end
# This allows us to validate the model and ensure it is not broken
print(r"Checking if wfxn is still normalized...")
S=nm.simpson(N,x,wfxn_sol[:,-1]*np.conj(wfxn_sol[:,-1]),-L,L)
print("Total probability:",'{:,.5}'.format(np.real(S)))

# Plotting true solutions with computed solutions
eigenQHO(x,t,wfxn_sol,wfxn_true)

#%%
"""#### FFT Split Operator Method ####"""

# Applying FFT Split Operator method to solve TDSE

# Defining kinetic energy in k-space
exp_k=np.exp(-0.5j*dt*k**2)

# Allocating and initializing wavefunction
wfxn_sol=np.zeros([N,nt],dtype=complex)
wfxn_sol[:,0]=wfxn

# Solving TDSE using fft split operator
for j in range(nt-1):
    wfxn_sol[:,j+1]=SplitFFT(V,exp_k,wfxn_sol[:,j],dt)

# Displaying results #

# Plotting initial wavefunction inside initial potential
plt.figure()
wfxnPlot(V,wfxn_sol[:,0],x,L,avg_H)
plt.title('Wavefunction of QHO at time t=0')
plt.show()

# Plotting all timesteps at once on the same plot
plt.figure()
for j in range(nt-1):
    wfxnPlot(V,wfxn_sol[:,j],x,L,avg_H)
    plt.title('Time evolution of QHO (FFT method)')
    plt.show()

# Checking if time-evolution retained normalization of function at the end
# This allows us to validate the model and ensure it is not broken
print(r"Checking if wfxn is still normalized...")
S=nm.simpson(N,x,wfxn_sol[:,-1]*np.conj(wfxn_sol[:,-1]),-L,L)
print("Total probability:",'{:,.5}'.format(np.real(S)))

# Plotting true solutions with computed solutions
eigenQHO(x,t,wfxn_sol,wfxn_true)

#%%
###############################################################################
## Using Split Operator Methods to Evolve Wavefunction in QHO
###############################################################################
# Evolution of Gaussian wavepacket in Quantum Harmonic Oscillator (QHO)

print("Solving time-dependent QHO for Gaussian wavefunction.")

"""#### Defining Parameters ####"""
N=257 # Number of space points
L=10 # Width being solved over
x=np.linspace(-L,L,N) # Grid of space points
dx=x[1]-x[0] # Spatial step
nt=100 # Number of time points
t=np.linspace(0,10,nt) # Grid of time points
dt=t[1]-t[0] # Time step
dk=2*np.pi/(N*dx) # spectral step
k=np.arange(-(N-1)*dk/2,N*dk/2,dk) # Grid of k-space

"""#### Wavefunction and Potential Parameters ####"""
p0=2 # Initial momentum of wavepacket
sigma=0.4 # Initial width of wavepacket
div=1/((2*np.pi*sigma**2)**0.25) # Constants bunched together 

# Initial wavefunction
wfxn=np.exp(1j*p0*x)*np.exp((-x**2)/((2*sigma)**2))*div

print("Normalizing wavefunction...")
S=nm.simpson(N,x,wfxn*np.conj(wfxn),-L,L)
A=1/np.sqrt(S)
wfxn=A*wfxn

# Frequency of harmonic oscillator potential
### Please edit these two lines below (comment/uncomment) ###
w=1+0.1*t # Time-dependent potential
# w=1 # Time-independent potential
### End user edits ###

# Harmonic Oscillator potential 
# np.outer used to generate matrix (outer product) if time-dependent
V=0.5*np.outer(w**2,x**2)

avg_H=expect_H(wfxn,L,w,x,N)
print("Average over expectation values of H, <H>:",'{:,.5}'.format(np.average(avg_H)))

print("Number of points in space, N:", N)
print("Number of points in time, nt:", nt)
print("Spatial step size, dx:",'{:,.5}'.format(dx))
print("Temporal step size, dt:", '{:,.5}'.format(dt))
print("Spectral step size, dk:",'{:,.5}'.format(dk))

#%%
"""#### General Split Operator Method ####"""

# Applying general Split Operator method to solve TDSE

# Setting up kinetic energy matrix
# 3-pt formula for second derivative
lower_diag = np.ones(N-1)
main_diag = -2*np.ones(N) 
upper_diag = np.ones(N-1)
Tfac=1/(dx**2)

T=-0.5*Tfac*nm.tridiag(lower_diag,main_diag,upper_diag)
U,exp_T=DiagForm(T,dt)

# Allocating and initializing wavefunction
wfxn_sol=np.zeros([N,nt],dtype=complex)
wfxn_sol[:,0]=wfxn

### For time-dependent potentials ###
if type(w)==np.ndarray:
    # Plotting time evolution of potential
    plt.figure()
    for j in range(nt-1):
        wfxnPlot(V[j,:],np.ones([N,1])*-1*avg_H[j],x,L,avg_H[j])
        plt.title('Time evolution of potential')
    plt.show()
    
    # Solving TDSE using split operator
    for j in range(nt-1):
        wfxn_sol[:,j+1]=SplitOp(V[j,:],exp_T,U,wfxn_sol[:,j],dt)

    # Displaying results #
    
    # Plotting one wavefunction inside one potential
    plt.figure() 
    d=2 # Grid point number
    d_time=(10/nt)*(d+1) # Calculation of the time with respect to boundaries
    wfxnPlot(V[d,:],wfxn_sol[:,d],x,L,avg_H[d])
    plt.title('Time evolution of QHO at time t=%i'%d_time)
    plt.show()
    
    # Plotting all timesteps at once on the same plot
    plt.figure()
    for j in range(nt-1):
        wfxnPlot(V[j,:],wfxn_sol[:,j],x,L,avg_H[j])
        plt.title('Time evolution of QHO')
        plt.show()

### For time-independent potentials ###
elif type(w)==int:
    V=np.squeeze(np.asarray(V)) # Converting outer product into scalar array
    
    # Solving TDSE using split operator
    for j in range(nt-1):
        wfxn_sol[:,j+1]=SplitOp(V,exp_T,U,wfxn_sol[:,j],dt)
    
    # Displaying results #
    
    # Plotting initial wavefunction inside initial potential
    plt.figure()
    wfxnPlot(V,wfxn_sol[:,0],x,L,avg_H)
    plt.title('Wavefunction of QHO at time t=0')
    plt.show()
    
    # Plotting all timesteps at once on the same plot
    plt.figure()
    for j in range(nt-1):
        wfxnPlot(V,wfxn_sol[:,j],x,L,avg_H)
        plt.title('Time evolution of QHO')
        plt.show()

# Checking if time-evolution retained normalization of function at the end
# This allows us to validate the model and ensure it is not broken
print(r"Checking if wfxn is still normalized...")
S=nm.simpson(N,x,wfxn_sol[:,-1]*np.conj(wfxn_sol[:,-1]),-L,L)
print("Total probability:",'{:,.5}'.format(np.real(S)))

# Other useful plots
miscPlots(x,t,wfxn_sol)

#%%
"""#### FFT Split Operator Method ####"""

# Applying FFT Split Operator method to solve TDSE

# Defining kinetic energy in k-space
exp_k=np.exp(-0.5j*dt*k**2)

# Allocating and initializing wavefunction
wfxn_sol=np.zeros([N,nt],dtype=complex)
wfxn_sol[:,0]=wfxn

### For time-dependent potentials ###
if type(w)==np.ndarray:
    # Plotting time evolution of potential
    plt.figure()
    for j in range(nt-1):
        wfxnPlot(V[j,:],np.ones([N,1])*-1*avg_H[j],x,L,avg_H[j])
        plt.title('Time evolution of potential')
    plt.show()
    
    # Solving TDSE using fft split operator
    for j in range(nt-1):
        wfxn_sol[:,j+1]=SplitFFT(V[j,:],exp_k,wfxn_sol[:,j],dt)

    # Displaying results #
    
    # Plotting one wavefunction inside one potential
    plt.figure() 
    d=79 # Grid point number
    d_time=(10/nt)*(d+1) # Calculation of the time with respect to boundaries
    wfxnPlot(V[d,:],wfxn_sol[:,d],x,L,avg_H[d])
    plt.title('Time evolution of QHO (FFT) at time t=%i'%d_time)
    plt.show()
    
    # Plotting all timesteps at once on the same plot
    plt.figure()
    for j in range(nt-1):
        wfxnPlot(V[j,:],wfxn_sol[:,j],x,L,avg_H[j])
        plt.title('Time evolution of QHO (FFT method)')
        plt.show()

### For time-independent potentials ###
elif type(w)==int:
    V=np.squeeze(np.asarray(V)) # Converting outer product into scalar array
    
    # Solving TDSE using fft split operator
    for j in range(nt-1):
        wfxn_sol[:,j+1]=SplitFFT(V,exp_k,wfxn_sol[:,j],dt)
    
    # Displaying results #
    
    # Plotting initial wavefunction inside initial potential
    plt.figure()
    wfxnPlot(V,wfxn_sol[:,0],x,L,avg_H)
    plt.title('Wavefunction of QHO at time t=0')
    plt.show()
    
    # Plotting all timesteps at once on the same plot
    plt.figure()
    for j in range(nt-1):
        wfxnPlot(V,wfxn_sol[:,j],x,L,avg_H)
        plt.title('Time evolution of QHO (FFT method)')
        plt.show()

# Checking if time-evolution retained normalization of function at the end
# This allows us to validate the model and ensure it is not broken
print(r"Checking if wfxn is still normalized...")
S=nm.simpson(N,x,wfxn_sol[:,-1]*np.conj(wfxn_sol[:,-1]),-L,L)
print("Total probability:",'{:,.5}'.format(np.real(S)))

# Other useful plots
miscPlots(x,t,wfxn_sol)

