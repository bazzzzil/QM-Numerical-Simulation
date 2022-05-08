"""
Date: Friday, November 1, 2019
Author: Layale Bazzi

This code solves the time-dependent Schrodinger equation for a free particle
using Crank-Nicolson method. In the free particle model, the potential is 0.
Equation to solve: psi_i^n+1 = inv(2-P)*P*psi_i^n
where psi is the wavefunction and P is the tridiagonal matrix containing the
Hamiltonian H=p^2/2m


"""
#%% Importing necessary packages
# numpy for calculations, matplotlib.pyplot for plotting
import numpy as np
import matplotlib.pyplot as plt
import NumMethods as nm

# https://stackoverflow.com/a/16657002
def tridiag(a, b, c, k1=-1, k2=0, k3=1):
    return np.diag(a, k1) + np.diag(b, k2) + np.diag(c, k3)

#%%

L=10 # Width being solved over
N=1000 # Number of space points
x=np.linspace(-L/2,L/2,N) # Grid of space points
sigma=0.1 # Width of wavefunction
p0=2 # Initial momentum

# Initial wavefunction
div=1/((2*np.pi)**0.25*(sigma**1/2))
wfxn=np.exp(1j*p0*x)*np.exp((-x**2)/((2*sigma)**2))*div

# Finding Normalization Constant
S=nm.simpson(N,x,wfxn*np.conj(wfxn),-L/2,L/2)
A=1/np.sqrt(abs(S))
print("Normalization constant:",np.real(A))

# New wavefunction:
wfxn=A*wfxn
print("Testing normalization...")

print(r"Checking if wfxn is normalized...")
S=nm.simpson(N,x,wfxn*np.conj(wfxn),-L/2,L/2)
print("Probability:",np.real(S))

## Finding <H> by doing <T> ##

## Expectation value of momentum squared ##
# Now doing <T>, which is just <p^2>*0.5 = -1*<d^2/dx^2>*0.5 in atomic units

d2wfxn=np.zeros(N,dtype=complex) # allocating memory
d2wfxn=nm.three_point_two(N,x,wfxn,-L/2,L/2) # calculating 2nd derivative of wfxn

exp_T=np.real(nm.simpson(N-2,x[1:N-1],wfxn[1:N-1]*d2wfxn[1:N-1],-L/2,L/2)*-0.5)
print("Expectation value of T, <T>:",exp_T)


print("Expectation value of H, <H>:",exp_T)
#%% Finding smallest "wavelength" for spatial grid spacing in CN solution
plt.plot(x,np.abs(wfxn))
plt.title('Wavefunction of QHO to propagate in time')
plt.show()
nt=100
t=np.linspace(0,50,nt)

h=sigma # h < (smallest wavelength)/100

# Finding smallest time period for temporal grid spacing in CN solution
# Characteristic time period T=2*pi/exp_H, because E=hbar*w

T=2*np.pi/exp_T # 

tau = (t[-1]/nt)/10 # tau < T/10

#%% Solving TDSE using Crank-Nicholson
# h=(L-(-L))/(N-1) # Grid spacing -> rearrange for L
# (N-1)*h/2=L

g=(tau/h**2)*0.5j # Constants bunched together

V=np.zeros(N) # Free particle

B=-2*g-1.0j*tau*V+1

lower_diag = np.ones(N-1)*g; 
main_diag = B; 
upper_diag = np.ones(N-1)*g

P = tridiag(lower_diag,main_diag,upper_diag)
Q = np.linalg.inv(2*np.identity(N)-P)

R=np.matmul(Q,P)

wfxn_sol=np.zeros([len(x),len(t)],dtype=complex)
wfxn_sol[:,0]=wfxn

for k in range(len(t)-1):
    wfxn_sol[:,k+1]=R.dot(wfxn_sol[:,k])

for j in range(len(t)-1):
    plt.plot(x,np.abs(wfxn_sol[:,j]))
    # plt.draw()
    plt.title('Time-evolution of free particle')
    # plt.show()
# plt.show()

print(r"Checking if wfxn is normalized...") # should be 1
S=nm.simpson(N,x,wfxn_sol[:,-1]*np.conj(wfxn_sol[:,-1]),-L/2,L/2)
print("Probability:",np.real(S))




