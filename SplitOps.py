import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft, ifft, fftshift, ifftshift
from scipy import linalg
import NumMethods as nm


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
    ax = fig.add_subplot(projection='3d')
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