# -*- coding: utf-8 -*-
"""
Functions to use in Numerical Methods (update after every class)
"""
import numpy as np
from scipy.fftpack import fft, ifft, fftshift,ifftshift
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# two_point: Outputs numerical derivative (real float) of any input function
#            (real array y) based on number of sampled points (integer N),
#            domain (real array x) using two-point method: f'=(f(i+1)-f(i))/h
def two_point(N,x,y,a,b):
    d=np.zeros(N)
    h=(b-a)/(N-1) # Spacing between each point in range ((end-start)/(N-1))
    for i in range(0,N-1): # Range ends at N-1 due to Python array syntax 0...N-1
        d[i]=(y[i+1]-y[i])/h # Numerical derivative
    return d;

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# three_point: Outputs numerical derivative (real float) of any input function
#              (real array y) based on number of sampled points (integer N),
#              domain (real array x) using three-point method:
#              f'=(f(i+1)-f(i-1))/2h
def three_point(N,x,y,a,b):
    d=np.zeros(N)
    h=(b-a)/(N-1) # Spacing between each point in range ((end-start)/(N-1))
    for i in range(1,N-1): # Range starts at 1 in order to sample point behind
                           # and ends at N-1 for Python syntax
        d[i]=(y[i+1]-y[i-1])/(2*h) # Numerical derivative
    return d;
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# five_point:  Outputs numerical derivative (real float) of any input function
#              (real array y) based on number of sampled points (integer N),
#              domain (real array x) using five-point method:
#              f'=f(i-2)-8f(i)+8f(i+1)-f(i+2)/12h
def five_point(N,x,y,a,b):
    d=np.zeros(N) # Array for numerical derivative
    h=(b-a)/(N-1) # Spacing between each point in range ((end-start)/(N-1))
    for i in range(2,N-2): # Range starts at 2 in order to sample points behind
                           # and ends at N-2 to sample points in front
        d[i]=(y[i-2]-8*y[i-1]+8*y[i+1]-y[i+2])/(12*h) # Numerical derivative
    return d;
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# three_point_two: Outputs numerical second derivative (real float) of any input
#                  function (real array y) based on number of sampled points
#                  (integer N), domain (real array x) using three-point method:
#                  f''=(f(i+1)-f(i)/h - f(i)-f(i-1)/h)/h
def three_point_two(N,x,y,a,b):
    d=np.zeros(N,dtype=complex) # Array for numerical derivative
    h=(b-a)/(N-1) # Spacing between each point in range ((end-start)/(N-1))
    for i in range(1,N-1): # Range starts at 2 in order to sample points behind
                           # and ends at N-2 to sample points in front
        d[i]=(((y[i+1]-y[i])/h)-((y[i]-y[i-1])/h))/h # Numerical second derivative
    return d;
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# invr: Inverts elements in real array by raising elements to the power of -1.
#       Returns changed real array. Array can be of type intger or float.
def invr(arr):
    b=np.zeros(len(arr))
    for i in range(0,len(arr)-1):
        b[i]=arr[i]**(-1)
    return b;
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# two_point3D: Outputs numerical partial derivative of any input 3D function 
#              (real array f) based on number of sampled points (integer N),
#              and independent real variables x, y using 3D two-point method:
#              f'=(f(i+1,j)-f(i,j))/h ; f'=(f(i,j+1)-f(i,j))/h
#              Last argument determines which dimension to differentiate along;
#              Input t==0 differentiates along y, anything else: along x.
def two_point3D(N,x,y,f,t,a,b):
    d=np.zeros([N,N])
    h=(b-a)/(N-1)
    if t==0: # If t==0, differentiate along y
        for j in range(0,N-1): # Outer loop to shift columns
            for i in range(0,N-1): # Nested loop to differentiate along a row
                d[i,j]=(f[i+1,j]-f[i,j])/h # Numerical derivative
    else: # Else, differentiate along x
        for i in range(0,N-1): # Outer loop to shift rows
            for j in range(0,N-1): # Nested loop to differentiate along a column
                d[i,j]=(f[i,j+1]-f[i,j])/h # Numerical derivative
    return d;
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# three_point3D: Outputs numerical partial derivative of any input 3D function 
#                (real array f) based on number of sampled points (integer N),
#                and independent real variables x, y using 3D three-point method: 
#                f'=(f(i+1,j)-f(i-1,j))/2h, f'=(f(i,j+1)-f(i,j-1))/2h
#                Last argument determines which dimension to differentiate along;
#                Input t==0 differentiates along y, anything else: along x.
def three_point3D(N,x,y,f,t,a,b):
    d=np.zeros([N,N]) # Array for numerical derivative
    h=(b-a)/(N-1) # Spacing between each point in range ((end-start)/(N-1))
    if t==0: # Differentiate along y
        for j in range(1,N-1):
            for i in range(1,N-1): 
                d[i,j]=(f[i+1,j]-f[i-1,j])/(2*h) 
    else: # Differentiate along x
        for i in range(1,N-1):
            for j in range(1,N-1):
                d[i,j]=(f[i,j+1]-f[i,j-1])/(2*h)
    return d;
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# five_point3D: Outputs numerical partial derivative of any input 3D function 
#               (real array f) based on number of sampled points (integer N),
#               and independent real variables x, y using 3D three-point method: 
#               f'=f(i-2,j)-8f(i,j)+8f(i+1,j)-f(i+2,j)/12h,
#               f'=f(i,j-2)-8f(i,j)+8f(i,j+1)-f(i,j+2)/12h
#               Last argument determines which dimension to differentiate along;
#               Input t==0 differentiates along y, anything else: along x.
def five_point3D(N,x,y,f,t,a,b):
    d=np.zeros([N,N]) # Array for numerical derivative
    h=(b-a)/(N-1) # Spacing between each point in range ((end-start)/(N-1))
    if t==0: # Differentiate along y
        for j in range(2,N-2):
            for i in range(2,N-2): 
                d[i,j]=(f[i-2,j]-8*f[i-1,j]+8*f[i+1,j]-f[i+2,j])/(12*h)
    else: # Differentiate along x
        for i in range(2,N-2):
            for j in range(2,N-2):
                d[i,j]=(f[i,j-2]-8*f[i,j-1]+8*f[i,j+1]-f[i,j+2])/(12*h)
    return d;
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# trapezoid: Outputs numerical integral (quadrature) of any input function
#            (real array f) based on number of sampled points (integer N), and
#            independent real variable x using trapezoid rule:
#            Taking average between two (consecutive) points and 
#            multiplying by spacing h
def trapezoid(N,x,y,a,b):
    h=(b-a)/(N-1) # Grid spacing 
    f=np.zeros(N,dtype=complex) # Allocating memory for calculations between two points
    for i in range(0,N-1):
        f[i]=y[i+1]+y[i] # Calculating sum between adjacent points
    return np.sum(f)*(h/2); # Summing all values and multiplying by h/2
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# simpson: Outputs numerical integral (quadrature) of any input function
#          (real array f) based on number of sampled points (integer N), and
#          independent real variables x using Simpson rule:
#          Using only even N, formula is to sum over these elements up to N/2:
#          F[i]=(f[2*i]+4*f[2*i+1]+f[2*i+2])
def simpson(N,x,y,a,b):
    h=(b-a)/(N-1) # Grid spacing
    f=np.zeros(N,dtype=complex) # Allocating memory for calculations between two points
    for i in range(0,int(N/2)-1): # Summing to half the sample points N
        f[i]=(y[2*i]+4*y[2*i+1]+y[2*i+2]) # Applying Simpson Rule and 
                                          # assigning result to array
    return np.sum(f)*(h/3); # Summing over all values and multiplying by h/3
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~    
# prod: Outputs product of two input functions or scalars.
def prod(h,g):
    return h*g
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# bisect: Outputs average between two points.
def bisect(r,t):
    return (r+t)/2
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# bisector: Finds root of any input function (defined as a function in the main
#           code) based on (real) range [a,b] used to generate a guess x0 and
#           tolerance (tol). The method essentially attempts to find the point
#           where the absolute value of the function is less than the tolerance,
#           which should be close to zero.  Returns root x0 & iter for comparison
#           Uses: prod, bisect functions
def bisector(func,a,b,tol):
    y=prod(func(a),func(b))
    iter=0
    if y>0: # Checking to see if bisector valid (product must be <0)
        print("Impossible to use bisector method.")
        return 0
    elif y==0: # Checking if root is already guessed
        print("One of the values is a root.")
        return 0
    else:
        x0=bisect(a,b) # Uses average between range as guess
        y1=prod(func(a),func(x0)) # Calculating RHS of potential root
        y2=prod(func(x0),func(b)) # Calculating LHS of potential root
        while abs(func(x0))>tol: # while tol is not satisfied, stay in loop
            iter=iter+1
            x0=bisect(a,b) # updating the guess
            if y1<0: # if RHS < 0, then we bisect that region
                b=x0 # updating b as x0 (x0 is now the left range)
                x0=bisect(a,x0) # updating guess
            if y2<0: # if LHS < 0, then we bisect that region
                a=x0 # updating a as x0 (x0 is now the right range)
                x0=bisect(b,x0) # updating guess
            # recalculating RHS,LHS of potential root
            y1=prod(func(a),func(x0))
            y2=prod(func(x0),func(b))
            
        return x0,iter
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# newtRaph: Finds root of any input function (defined as a function in the main
#           code) based on input (real) range [a,b], first derivative of f and 
#           tolerance (tol). The method essentially attempts to find the point 
#           where the absolute value of the function is less than the tolerance, 
#           which should be close to zero. iter is returned for comparison
#           Applies Newton-Raphson method: x[k+1]=x[k]-f(x[k])/f'(x[k])
def newtRaph(func,funcP,a,b,tol):
    x0=(a+b)/2 # Initial guess
    iter=0
    fract=func(x0)/funcP(x0) # Calculating update increment

    while abs(func(x0))>tol: # while tol is not satisfied, stay in loop
        iter=iter+1
        x0=x0-fract # Updating guess
        fract=func(x0)/funcP(x0) # updating increment
        
    return x0,iter
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# secRoot: Finds root of any input function (defined as a function in the main
#          code) based on input (real) range [a,b], int number of grid points N
#          and tolerance (tol). The method essentially attempts to find the 
#          point where the absolute value of the function is less than the 
#          tolerance, which should be close to zero. Applies secant method: 
#          x[k+1]=x[k]-((x[k]-x[k-1])/(f(x[k])-f([k-1])))*f(x[k])
#          This is used when the derivative of a function is not known
#          Applies two-point rule for numerical derivative
#          x1 is returned as the root because it seems to approach it faster
#          than x0 in the algorithm; iter is also returned for comparison
def secRoot(func,a,b,N,tol):
    iter=0;
    h=(b-a)/(N-1) # step size between x1 and x0 for derivative
    x0=(a+b)/2 # initial guess
    x1=x0+h # second point 
    fract=((x1-x0)/(func(x1)-func(x0)))*func(x1) # update increment for x0
    
    while abs(func(x0))>tol:  # while tol is not satisfied, stay in loop
        iter=iter+1
        if abs(func(x1))<tol: # break when tolerance is satisfied with x1
            break
        else:
            x0=x0-fract # Updating guess
            x1=x0+h # Using updated guess to get second point
            fract=((x1-x0)/(func(x1)-func(x0)))*func(x1) # updating increment
            
    return x1, iter
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# secRootP2: Same as secRoot, but modified for Problem 2 in HW3 to allow initial
#            guess x0 and step size input h. Also returns x[k] and f(x[k])
def secRootP2(func,tol,x0,h):
    x=[]
    f=[]
    iter=0;
    x1=x0+h # second point
    x.append(x1)
    f.append(func(x1))
    fract=((x1-x0)/(func(x1)-func(x0)))*func(x1) # update increment for x0
    while abs(func(x0))>tol:  # while tol is not satisfied, stay in loop
        iter=iter+1
        if abs(func(x1))<tol: # break when tolerance is satisfied with x1
            break
        else:
            x.append(x1)
            f.append(func(x1))
            x0=x0-fract # Updating guess
            x1=x0+h # Using updated guess to get second point
            fract=((x1-x0)/(func(x1)-func(x0)))*func(x1) # updating increment
            
    return x1,iter,x,f
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# newtRaphP3: Same as newtRaph, but modified for Problem 3 in HW3 to allow initial
#             guess x0.
def newtRaphP3(func,funcP,x0,tol):
    iter=0
    fract=func(x0)/funcP(x0) # Calculating update increment

    while abs(func(x0))>tol: # while tol is not satisfied, stay in loop
        iter=iter+1
        x0=x0-fract # Updating guess
        fract=func(x0)/funcP(x0) # updating increment
        
    return x0,iter
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# secMin: Modified secant method for finding minimum in an input function (func)
#         by minimizing its input (analytical) derivative funcP. Parameters
#         passed include tolerance (tol), initial guess x0, step size dx 
#         and maximum iterations (max). Returns x1 as the root and number of
#         iterations. This differs from the standard secant in that it checks
#         to see if the function value increases or decreases with every
#         iteration. It also calculates the second derivative using finite
#         element method (two-point rule). The method essentially attempts to 
#         find the point where the absolute value of the derivative function
#         is less than the tolerance, which should be close to zero.
#         fract is used to update the guess x0, which in turn is used to update
#         the next point x1 in conjunction with the passed step size dx.
#          x1 is returned as the root because it seems to approach it faster
#          than x0 in the algorithm
def secMin(func,funcP,tol,x0,dx,max):
    iter=0 # initializing iterations
    x1=x0+dx # creating x1 for derivative calculation
    fract=((x1-x0)/(funcP(x1)-funcP(x0)))*funcP(x1) # update increment for x0
    while abs(funcP(x1))>tol and iter<max: # while tol is not satisfied,
                                           # and max not reached, stay in loop
        iter=iter+1
        if func(x1)<func(x0): # if function is decreasing from x0 to x1, go
            x0=x0-fract # update x0
            x1=x0+dx # update x1 in direction of x0
            fract=((x1-x0)/(funcP(x1)-funcP(x0)))*funcP(x1) # update fract
        elif func(x1)>func(x0): # if function is increasing from x0 to x1, go
            x1=x0 # update x1 to x0 (go backwards)
            x0=x0-fract # update x0
            fract=(-(x1-x0)/(funcP(x1)-funcP(x0)))*funcP(x1) # update fract
            
    return x1,iter
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# gradr: Function that calculates the gradient for any input function and is
#        generalized to n dimensions. It takes as input the real array/vector
#        r, the input function func (passed as a function), the integer step
#        size, step and the number of dimensions dimN.
def gradr(r,func,step,dimN):
    d=np.zeros(dimN) # initializing array to hold grad values
    for i in range(dimN): # looping over all dimensions
        mask=np.zeros_like(r) # mask that updates 1 dimension per iteration
        mask[i]=step # assigning step size to mask array element
        d[i]=((func(r+mask)-func(r)))/step # calculating derivative in 1 dim
    return d
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# length: Function that calculates the length of any input array and is
#         generalized to n-dimensions. Takes in an array and returns a scalar.
def length(dr):
    dl=np.zeros(len(dr)) # array
    for i in range(len(dr)):
        dl[i]=dr[i]**2 # squares every element
    return np.sqrt(np.sum(dl)) # Sums every element in the array, takes sqrt
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# steep: Function that minimizes any input function based on principle of 
#        steepest descent. This is essentially the modified secant method
#        generalized to n-dimensions. Takes in function to be minimized as an
#        input function, the tolerance tol, the independent n-D real array r,
#        the step size step, the number of dimensions dimN, the adjustable 
#        parameter a and the maximum number of iterations max.
def steep(func,tol,r,step,dimN,a,max):
    iter=0 # initializing iterations
    dr=gradr(r,func,step,dimN) # calling gradr for initial grad calculations
    r1=r+step # initializing r1 for checking
    while abs(length(dr))>tol and iter<max: # If the length is smaller than
                                            # tol, and max iter reached, break
        iter=iter+1
        l=1/length(dr) # to save on divisions in algorithm, calculate it here
        if func(r1)<func(r): # if function is decreasing from r to r1, go
            for i in range(len(r)):
                r[i]=r[i]-a*dr[i]*l # adjust r
                r1[i]=r[i]+step # go forward with r1
            dr=gradr(r,func,step,dimN) # re-calculate grad
        elif func(r1)>func(r): # if function is increasing from r to r1, go
            for i in range(len(r)):
                r1[i]=r[i] # update r1 to r (go backwards)
                r[i]=r[i]-a*dr[i]*l # adjust r
            dr=gradr(-r,func,step,dimN) # re-calculate grad
    return r1,iter
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# fft_d: Calculates the first derivative of any function over some space N using
#        the Fast Fourier Transform (fft) algorithm. This function FTs the input
#        f(x) then multiplies it by some constant in k-space depending on the
#        value of k
def fft_d(func,x,L,N):
    k=np.linspace(0,N-1,N)
    # k=2*np.pi/L*np.linspace(0,N/2-1,N)
    fac=2j*np.pi/L
    func_fft=fft(func(x))
    # func_fft=ifftshift(fft(ifftshift(func(x))))
    
    der_fft=np.zeros(len(func_fft),dtype=complex)
    for i in range(len(k)):
        if k[i]<N/2:
            der_fft[i]=func_fft[i]*fac*k[i]
        elif k[i]>N/2:
            der_fft[i]=func_fft[i]*fac*(k[i]-N)
        elif k[i]==N/2:
            der_fft[i]=0
    return der_fft
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~    
# # Source: https://stackoverflow.com/a/16657002
# # tridiag: Outputs a tridiagonal matrix based on the np.diag function. Requires
# #          3 input arrays and 3 input locations for np.diag. The diagonal 
# #          elements of the tridiag matrix is denoted as b and is associated with
# #          the k2 variable. The top offdiagonal elements are contained in c with
# #          the k3 variable. The bottom offdiagonal elements are contained in a
# #          with the k1 variable. The np.diag function creates diagonal matrices,
# #         which are then added to form the tridiagonal matrix.
def tridiag(a, b, c, k1=-1, k2=0, k3=1):
    return np.diag(a, k1) + np.diag(b, k2) + np.diag(c, k3)    
    
    
    
    
    
    