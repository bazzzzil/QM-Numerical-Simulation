"""
Date: Friday, November 1, 2019
Author: Layale Bazzi

Solving coupled ODEs using Runge-Kutta (RK4) method. In-house and SciPy.
Examples: simple harmonic oscillator, TDSE, 
Quantum Liouville equation without and with Lindblad terms.

The in-house RK4 was compared with the SciPy RK4 solution to evaluate the error, 
which increases with the detuning. This is because we are approaching the regime 
where the Rotating Wave Approximation (RWA) fails. RWA is used when the detuning 
is very small, i.e.: the transition frequency ω0 is close to the driving 
frequency ω. This implies the system is being excited on resonance, or very 
close to resonance, thus we can approximate the solution by killing off fast 
rotating terms. As we deviate from this condition, RWA no longer holds and thus 
the solution becomes more unstable.

"""
#%% Importing necessary packages
# numpy for calculations, matplotlib.pyplot for plotting, scipy for integrating
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate

# Solving ODEs using Runge-Kutta 
# The Runge-Kutta algorithm employs the Taylor series expansion of the function
# we wish to find about some point. Then, in a separate operation we expand the 
# original function as a power series and compare the two expressions. Going up
# to the fourth order term in the expansions is what leads to the name 'RK4', 
# as it requires the calculation of 4 coefficients to find the function:
# For some dy/dt=g(y,t);
# y(t+T) = y(t) + (1/6)*(c1 + 2*c2 + 2*c3 + c4)
#%%
###############################################################################
## Solving Simple Harmonic Oscillator using RK4 (in-house vs SciPy)
###############################################################################
# Solving coupled first order ODEs in the SHO problem
# dx/dt = v
# dv/dt = -4pi^2 x

print("\nSolving Simple Harmonic Oscillator using RK4 (in-house vs SciPy)")
      
### In-house RK4 ###

# Defining functions to integrate
def dvdt(t,x): # dv/dt
    k=-4*np.pi**2
    return k*x

def dxdt(t,v): # dx/dt
    return v

N=10000 # Number of grid points
t=np.linspace(0,5,N) # Time array
x=np.zeros(N,dtype=complex) # Allocating space for x solution
v=np.zeros(N,dtype=complex) # Allocating space for v solution
x[0]=0 # Initial condition x(t=0)=0
v[0]=1 # Initial condition v(t=0)=1
w=1/6; # Weighting factor used in RK4

# Exact analytical solutions for comparison with numerical solutions
xexact=(1/(2*np.pi))*np.sin(2*np.pi*t)
vexact=np.cos(2*np.pi*t)

# Begin coupled RK4
for i in range(N-1): # Range ends at N-1 due to Python array syntax 0...N-1
    T=t[i+1]-t[i] # Time step
    
    # Solving dv/dt
    c1v=T*dvdt(t[i],x[i])
    c2v=T*(dvdt(t[i]+T*0.5,x[i])+c1v*0.5)
    c3v=T*(dvdt(t[i]+T*0.5,x[i])+c2v*0.5)
    c4v=T*(dvdt(t[i]+T,x[i])+c3v)
    v[i+1]=v[i]+w*(c1v + 2*c2v + 2*c3v + c4v) # Updating the next point in v
    
    # Solving dx/dt
    c1x=T*dxdt(t[i],v[i])
    c2x=T*(dxdt(t[i]+T*0.5,v[i])+c1x*0.5)
    c3x=T*(dxdt(t[i]+T*0.5,v[i])+c2x*0.5)
    c4x=T*(dxdt(t[i]+T,v[i])+c3x) 
    x[i+1]=x[i]+w*(c1x + 2*c2x + 2*c3x + c4x) # Updating the next point in x

### SciPy ###

# Defining functions to integrate in an array to use in SciPy
# First element of F is dx/dt, second is dv/dt
# F[[dx/dt=v],[dv/dt=-4(pi^2)x]]
def dFdt(t,F):
    k=-4*np.pi**2
    return np.array([F[1], k*F[0]])

# Initial conditions
F0=np.array([0,1]) # x(t=0)=0, v(t=0)=1
t0=0 # t=0

# Setting integrand in scipy integrator and calling RK4 method
r=integrate.complex_ode(dFdt).set_integrator('dopri5') 
r.set_initial_value(F0,t0) # Setting initial values
xv=np.zeros((2,N),dtype=complex) # Allocating space for solutions in 2xN matrix

# Begin coupled SciPy RK4
for i in range(1,t.size):
    xv[:,i]=r.integrate(t[i])
    if not r.successful():
        raise RuntimeError("Could Not Integrate")

# Plotting solutions for position
plt.plot(t,x,label='In-House RK4')
plt.plot(t,np.real(xv[0,:]),label='Scipy RK4')
plt.plot(t,xexact,label='Analytical')
plt.title('Comparison of SHO position solutions')
plt.xlabel('Time')
plt.ylabel('Position')
plt.ticklabel_format(axis='both', style='sci', scilimits=(1,3))
plt.legend(loc=3)
plt.show()

# Plotting solutions for velocity
plt.plot(t,v,label='In-House RK4')
plt.plot(t[1:],np.real(xv[1,1:]),label='Scipy RK4')
plt.plot(t,vexact,label='Analytical')
plt.title('Comparison of SHO velocity solutions')
plt.xlabel('Time')
plt.ylabel('Velocity')
plt.ticklabel_format(axis='both', style='sci', scilimits=(1,3))
plt.legend(loc=3)
plt.show()

# Plotting errors in position
plt.plot(t,abs(xexact-x),label='In-House RK4 Error (x)')
plt.plot(t[1:],abs(xexact[1:]-np.real(xv[0,1:])),label='Scipy RK4 Error (x)')
plt.title('Errors in SciPy and in-house RK4 for Position')
plt.xlabel('Time')
plt.ylabel('Position')
plt.ticklabel_format(axis='both', style='sci', scilimits=(1,3))
plt.legend()
plt.show()

# Plotting errors in velocity
plt.plot(t,abs(vexact-v),label='In-House RK4 Error (v)')
plt.plot(t[1:],abs(vexact[1:]-np.real(xv[1,1:])),label='Scipy RK4 Error (v)')
plt.legend()
plt.title('Errors in SciPy and in-house RK4 for Velocity')
plt.xlabel('Time')
plt.ylabel('Velocity')
plt.ticklabel_format(axis='both', style='sci', scilimits=(1,3))
plt.show()

#%%
###############################################################################
## Solving TDSE in RWA for 2L atom driven by external harmonic field
###############################################################################
# Solving the Time-Dependent Schrodinger equation (TDSE) in the Rotating
# Wave Approximation for a two-level atom driven by an external harmonic field
# da/dt = i(WW/2)exp(idt)b(t)
# db/dt = i(WW/2)exp(-idt)a(t)

print("\nSolving TDSE in RWA for 2L atom driven by external harmonic field")
      
### In-house RK4 ###
      
N=10000 # Number of grid points
t=np.linspace(0,100,N) # Time array
a=np.zeros(N,dtype=complex) # Allocating space for solution to coefficient a
b=np.zeros(N,dtype=complex) # Allocating space for solution to coefficient b
# Initial conditions
a[0]=1/(np.sqrt(2))
b[0]=-1/(np.sqrt(2))
w=1/6; # Weighting factor used in RK4
WW=2 # Energy in units of hbar
# Detuning (w-w0)
# d=-0.5
# d=-0.05
d=0.05
# d=0.5

# Defining functions to integrate
def dadt(t,b): # da/dt
    k=1.0j*(WW/2)
    return k*np.exp(1.0j*d*t)*b

def dbdt(t,a): # db/dt
    k=1.0j*(WW/2)
    return k*np.exp(-1.0j*d*t)*a

# Begin coupled RK4
for i in range(N-1): # Range ends at N-1 due to Python array syntax 0...N-1
    T=t[i+1]-t[i] # Time step
    
    # Solving da/dt
    c1a=T*dadt(t[i],b[i])
    c2a=T*(dadt(t[i]+T*0.5,b[i])+c1a*0.5)
    c3a=T*(dadt(t[i]+T*0.5,b[i])+c2a*0.5)
    c4a=T*(dadt(t[i]+T,b[i])+c3a)
    a[i+1]=a[i]+w*(c1a + 2*c2a + 2*c3a + c4a) # Updating the next point in a
    
    # Solving db/dt
    c1b=T*dbdt(t[i],a[i])
    c2b=T*(dbdt(t[i]+T*0.5,a[i])+c1b*0.5)
    c3b=T*(dbdt(t[i]+T*0.5,a[i])+c2b*0.5)
    c4b=T*(dbdt(t[i]+T,a[i])+c3b) 
    b[i+1]=b[i]+w*(c1b + 2*c2b + 2*c3b + c4b) # Updating the next point in b
    
### SciPy ###

# Defining functions to integrate in an array to use in SciPy
# First element of F is da/dt, second is db/dt
# F[[da/dt = i(WW/2)exp(idt)b(t)],[db/dt = i(WW/2)exp(-idt)a(t)]]
def dFdt(t,F):
    k=1.0j*(WW/2)
    return np.array([k*np.exp(1.0j*d*t)*F[1], k*np.exp(-1.0j*d*t)*F[0]])

# Initial conditions
F0=np.array([a[0],b[0]]) # a(t=0)=1/sqrt(2), b(t=0)=-1/sqrt(2)
t0=0 # t=0

# Setting integrand in scipy integrator and calling RK4 method
r=integrate.complex_ode(dFdt).set_integrator('dopri5') 
r.set_initial_value(F0,t0) # Setting initial values
ab=np.zeros((2,N),dtype=complex) # Allocating space for solutions in 2xN matrix

# Begin coupled SciPy RK4
for i in range(1,t.size):
    ab[:,i]=r.integrate(t[i])
    if not r.successful():
        raise RuntimeError("Could Not Integrate")

# Plotting solutions for in-house RK4
plt.plot(t,np.abs(a)**2,label='a(t)')
plt.plot(t,np.abs(b)**2,label='b(t)')
plt.title('Time Evolution of Energy Level Probabilities using in-house RK4')
plt.xlabel('Time')
plt.ylabel('Probability')
plt.legend(loc=3)
plt.show()

# Plotting solutions for SciPy
plt.plot(t[1:],np.abs(ab[0,1:])**2,label='a(t)')
plt.plot(t[1:],np.abs(ab[1,1:])**2,label='b(t)')
# plt.title('Time Evolution of Energy Level Probabilities using SciPy RK4')
plt.xlabel('Time')
plt.ylabel('Probability')
plt.legend(loc=3)
plt.show()

#%%
###############################################################################
## Solving QLE using RWA Hamiltonian
###############################################################################
# Solving the Quantum Liouville Equation using the Rotating Wave Approximation
# (RWA) Hamiltonian for a spin-half particle in a static magnetic field driven
# by an oscillating magnetic field. This is done using the density matrix
# formalism where the decoherence (Lindblad) terms are zero.
# The equations were simplified from a system of 4 eqns to 2
# Solving:  dp/dt=-i[H,p]

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def SpinOps(Operator=0):
    
        # Sz
        if Operator==0:
            return 0.5*np.array([[1,0],[0,-1]])
        # Sy
        elif Operator==1:
            return 0.5*np.array([[0,-1j],[1j,0]])

        # Sx
        elif Operator==2:
            return 0.5*np.array([[0,1],[1,0]])

print("\nSolving QLE using RWA Hamiltonian")
      
### In-house RK4 ###

N=10000
t=np.linspace(0,20*np.pi,N)
p=np.zeros([4,N],dtype=complex)
w=1/6;
p[0,0]=1/2 # pbb
p[1,0]=1/2 # pba
p[2,0]=1/2 # pab
p[3,0]=1/2 # paa
w1=2 # Energy of RF pulse in units of hbar, w1<<w0
d=0.1 # Detuning
# d=0

# Defining functions to integrate
def pbb_dt(t,p): # dpbb/dt
    Vba=(1j*w1/2)*np.exp(1.0j*d*t)
    Vab=(1j*w1/2)*np.exp(-1.0j*d*t)
    return Vba*p[2]-Vab*np.conj(p[2])

def pab_dt(t,p): # dpab/dt
    Vab=(1j*w1/2)*np.exp(-1.0j*d*t)
    return (2*p[0]-1)*Vab

for i in range(N-1): # Range ends at N-1 due to Python array syntax 0...N-1
    T=(t[i+1]-t[i])/5 # Time step
        
    # Solving dpbb/dt
    c1bb=T*pbb_dt(t[i],p[:,i])
    c2bb=T*(pbb_dt(t[i]+T*0.5,p[:,i])+c1bb*0.5)
    c3bb=T*(pbb_dt(t[i]+T*0.5,p[:,i])+c2bb*0.5)
    c4bb=T*(pbb_dt(t[i]+T,p[:,i])+c3bb) 
    p[0,i+1]=p[0,i]+w*(c1bb + 2*c2bb + 2*c3bb + c4bb) # Updating the next point in b
    
    # Solving dpab/dt
    c1ab=T*pab_dt(t[i],p[:,i])
    c2ab=T*(pab_dt(t[i]+T*0.5,p[:,i])+c1ab*0.5)
    c3ab=T*(pab_dt(t[i]+T*0.5,p[:,i])+c2ab*0.5)
    c4ab=T*(pab_dt(t[i]+T,p[:,i])+c3ab) 
    p[2,i+1]=p[2,i]+w*(c1ab + 2*c2ab + 2*c3ab + c4ab)

p[3,:]=1-p[0,:]
p[1,:]=np.conj(p[2,:])
    
### SciPy ###

# Defining functions to integrate in an array to use in SciPy
# First element of F is dpbb/dt, second is dpab/dt
def dFdt(t,F):
    Vba=(1j*w1/2)*np.exp(1.0j*d*t)
    Vab=(1j*w1/2)*np.exp(-1.0j*d*t)
    return np.array([Vba*F[1]-Vab*np.conj(F[1]),(2*F[0]-1)*Vab])

# Initial conditions
F0=np.array([p[0,0],p[1,0]]) # pbb=1/2, pab=1/2
t0=0 # t=0

# Setting integrand in scipy integrator and calling RK4 method
r=integrate.complex_ode(dFdt).set_integrator('dopri5') 
r.set_initial_value(F0,t0) # Setting initial values
p2=np.zeros((2,N),dtype=complex) # Allocating space for solutions in 2xN matrix
pba=np.zeros((1,N),dtype=complex) 
paa=np.zeros((1,N),dtype=complex) 

# Begin coupled SciPy RK4
for i in range(1,t.size):
    p2[:,i]=r.integrate(t[i])
    if not r.successful():
        raise RuntimeError("Could Not Integrate")
        
paa=1-p2[0,:]
pba=np.conj(p2[1,:])
plt.figure()
# Plotting solutions for in-house RK4
plt.plot(t,np.abs(p[0,:]),label=r'$\rho_{bb}$')
plt.plot(t,np.abs(p[3,:]),label=r'$\rho_{aa}$')
plt.title('Time Evolution of Energy Level Population')
plt.xlabel('Time')
plt.ylabel('Population')
plt.legend(loc=4)
plt.show()

plt.figure()
plt.plot(t,np.real(p[1,:]),label=r'$\rho_{ba}$ (real)')
plt.plot(t,np.imag(p[1,:]),label=r'$\rho_{ba}$ (imag)')
plt.title(r'Time Evolution of $\rho_{ba}$ Coherence')
plt.xlabel('Time')
plt.ylabel('Coherence Amplitude')
plt.legend(loc=4)
plt.show()

plt.figure()
plt.plot(t,np.real(p[2,:]),label=r'$\rho_{ab}$ (real)')
plt.plot(t,np.imag(p[2,:]),label=r'$\rho_{ab}$ (imag)')
plt.title(r'Time Evolution of $\rho_{ab}$ Coherence')
plt.xlabel('Time')
plt.ylabel('Coherence Amplitude')
plt.legend()
plt.show()

# # Plotting solutions for SciPy
# plt.plot(t[1:],np.abs(p2[0,1:])**2,label=r'$\rho_{bb}$')
# plt.plot(t[1:],np.abs(paa[1:])**2,label=r'$\rho_{aa}$')
# plt.title('Time Evolution of Energy Level Populations using SciPy RK4')
# plt.xlabel('Time')
# plt.ylabel('Population')
# plt.legend(loc=2)
# plt.show()

# # Plotting errors in paa
# plt.plot(t[1:],abs(paa[1:]-p[3,1:]))
# plt.title(r'Comparing in-house RK4 to SciPy for $\rho_{aa}$')
# plt.xlabel('Time')
# plt.ylabel('Difference')
# plt.ticklabel_format(axis='both', style='sci', scilimits=(1,3))
# plt.show()

# # Plotting errors in pbb
# plt.plot(t[1:],abs(p2[0,1:]-p[0,1:]))
# plt.title(r'Comparing in-house RK4 to SciPy for $\rho_{bb}$')
# plt.xlabel('Time')
# plt.ylabel('Difference')
# plt.ticklabel_format(axis='both', style='sci', scilimits=(1,3))
# plt.show()

rho=np.zeros([N,2,2],dtype=complex)
rho[:,0,0]=p[0,:]
rho[:,0,1]=p[1,:]
rho[:,1,0]=p[2,:]
rho[:,1,1]=p[3,:]
avg_Sz=np.zeros(N,dtype=complex) # tr(Sz rho)
avg_Sy=np.zeros(N,dtype=complex) # tr(Sz rho)
avg_Sx=np.zeros(N,dtype=complex) # tr(Sz rho)
rhoTrace=np.zeros(N,dtype=complex)

for i in range(N):
    avg_Sz[i]=np.trace( SpinOps(0) @ rho[i,:,:])
    avg_Sy[i]=np.trace( SpinOps(1) @ rho[i,:,:])
    avg_Sx[i]=np.trace( SpinOps(2) @ rho[i,:,:])
    rhoTrace[i]=np.trace(rho[i,:,:])

#Plotting results
plt.figure
plt.plot(t,np.abs(avg_Sz),label=r'$<S1z>$')
plt.plot(t,np.abs(avg_Sy),label=r'$<S1y>$')
plt.plot(t,np.abs(avg_Sx),label=r'$<S1x>$')
plt.legend()

plt.figure()
plt.plot(t,np.abs(rhoTrace))
plt.ylim([0,2])

#%%
###############################################################################
## Solving QLE for 2LS system with spontaneous emission
###############################################################################
# Solving the Quantum Liouville Equation for a two-level system with spontaneous
# emission (Lindblad term) using the density matrix formalism. The system of 
# equations was reduced to 3 equations and 3 unknowns instead of 4.
# See hardcopy for more details.

print("\nSolving QLE for 2LS system with spontaneous emission")

### In-house RK4 ###

N=10000
t=np.linspace(0,10,N)
p=np.zeros([4,N],dtype=complex)
w=1/6;
p[0,0]=1/2 # pbb
p[1,0]=1/2 # pba
p[2,0]=1/2 # pab
p[3,0]=1/2 # paa
W=2 #
d=0.1 # Detuning
# d=0
# gamma=0.1 # Decoherence rate
# gamma=0.01
gamma=0

# Defining functions to integrate
def pbb_dt(t,p): # dpbb/dt
    Vba=-1.0j*(W/2)*np.exp(1.0j*d*t)
    Vab=-1.0j*(W/2)*np.exp(-1.0j*d*t)
    return Vba*p[2]-Vab*p[1] -gamma*p[0]

def pba_dt(t,p): # dpba/dt
    Vba=-1.0j*(W/2)*np.exp(1.0j*d*t)
    return (1-2*p[0])*Vba-0.5*gamma*p[1]

def pab_dt(t,p): # dpab/dt
    Vab=-1.0j*(W/2)*np.exp(-1.0j*d*t)
    return (2*p[0]-1)*Vab-0.5*gamma*p[2]

for i in range(N-1): # Range ends at N-1 due to Python array syntax 0...N-1
    T=t[i+1]-t[i] # Time step

    # Solving dpbb/dt
    c1bb=T*pbb_dt(t[i],p[:,i])
    c2bb=T*(pbb_dt(t[i]+T*0.5,p[:,i])+c1bb*0.5)
    c3bb=T*(pbb_dt(t[i]+T*0.5,p[:,i])+c2bb*0.5)
    c4bb=T*(pbb_dt(t[i]+T,p[:,i])+c3bb) 
    p[0,i+1]=p[0,i]+w*(c1bb + 2*c2bb + 2*c3bb + c4bb)
     
    # Solving dpba/dt
    c1ba=T*pba_dt(t[i],p[:,i])
    c2ba=T*(pba_dt(t[i]+T*0.5,p[:,i])+c1ba*0.5)
    c3ba=T*(pba_dt(t[i]+T*0.5,p[:,i])+c2ba*0.5)
    c4ba=T*(pba_dt(t[i]+T,p[:,i])+c3ba)
    p[1,i+1]=p[1,i]+w*(c1ba + 2*c2ba + 2*c3ba + c4ba)

    # Solving dpab/dt
    c1ab=T*pab_dt(t[i],p[:,i])
    c2ab=T*(pab_dt(t[i]+T*0.5,p[:,i])+c1ab*0.5)
    c3ab=T*(pab_dt(t[i]+T*0.5,p[:,i])+c2ab*0.5)
    c4ab=T*(pab_dt(t[i]+T,p[:,i])+c3ab) 
    p[2,i+1]=p[2,i]+w*(c1ab + 2*c2ab + 2*c3ab + c4ab)

p[3,:]=1-p[0,:]

### SciPy ###

# Defining functions to integrate in an array to use in SciPy
# First element of F is dpbb/dt, second is dpab/dt, third is dpaa/dt
def dFdt(t,F):
    Vba=-1.0j*(W/2)*np.exp(1.0j*d*t)
    Vab=-1.0j*(W/2)*np.exp(-1.0j*d*t)
    return np.array([Vba*F[2]-Vab*F[1] -gamma*F[0],
                     (1-2*F[0])*Vba-0.5*gamma*F[1],
                     (2*F[0]-1)*Vab-0.5*gamma*F[2]])

# Initial conditions
F0=np.array([p[0,0],p[1,0],p[2,0]]) #  pbb=1/2, pba=1/2, pab=1/2
t0=0 # t=0

# Setting integrand in scipy integrator and calling RK4 method
r=integrate.complex_ode(dFdt).set_integrator('dopri5') 
r.set_initial_value(F0,t0) # Setting initial values
p2=np.zeros((3,N),dtype=complex) # Allocating space for solutions in 2xN matrix
paa=np.zeros((1,N),dtype=complex) 

# Begin coupled SciPy RK4
for i in range(1,t.size):
    p2[:,i]=r.integrate(t[i])
    if not r.successful():
        raise RuntimeError("Could Not Integrate")
        
pba=1-p2[0,:]

# Plotting solutions for in-house RK4
plt.plot(t,np.abs(p[0,:]),label=r'$\rho_{bb}$')
plt.plot(t,np.abs(p[3,:]),label=r'$\rho_{aa}$')
plt.title('Time Evolution of Energy Level Populations')
plt.xlabel('Time')
plt.ylabel('Population')
plt.legend(loc=4)
plt.show()

plt.plot(t,np.real(p[1,:]),label=r'$\rho_{ba}$ (real)')
plt.plot(t,np.imag(p[1,:]),label=r'$\rho_{ba}$ (imag)')
plt.title(r'Time Evolution of $\rho_{ba}$ Coherence')
plt.xlabel('Time')
plt.ylabel('Coherence Amplitude')
plt.legend(loc=4)
plt.show()

plt.plot(t,np.real(p[2,:]),label=r'$\rho_{ab}$ (real)')
plt.plot(t,np.imag(p[2,:]),label=r'$\rho_{ab}$ (imag)')
plt.title(r'Time Evolution of $\rho_{ab}$ Coherence')
plt.xlabel('Time')
plt.ylabel('Coherence Amplitude')
plt.legend(loc=4)
plt.show()

# Plotting solutions for SciPy
# First point emitted due to problems with integration
plt.plot(t[1:],np.abs(p2[0,1:]),label=r'$\rho_{bb}$')
plt.plot(t[1:],np.abs(p2[2,1:]),label=r'$\rho_{aa}$')
plt.title('Time Evolution of Energy Level Populations SciPy')
plt.xlabel('Time')
plt.ylabel('Population')
plt.legend(loc=4)
plt.show()

plt.plot(t[1:],np.real(pba[1:]),label=r'$\rho_{ba}$ (real)')
plt.plot(t[1:],np.imag(pba[1:]),label=r'$\rho_{ba}$ (imag)')
plt.title(r'Time Evolution of $\rho_{ba}$ Coherence SciPy')
plt.xlabel('Time')
plt.ylabel('Coherence Amplitude')
plt.legend(loc=4)
plt.show()

plt.plot(t[1:],np.real(p2[1,1:]),label=r'$\rho_{ab}$ (real)')
plt.plot(t[1:],np.imag(p2[1,1:]),label=r'$\rho_{ab}$ (imag)')
plt.title(r'Time Evolution of $\rho_{ab}$ Coherence SciPy')
plt.xlabel('Time')
plt.ylabel('Coherence Amplitude')
plt.legend(loc=4)
plt.show()

























