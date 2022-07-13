import array
import vegas
import math
import random as r
import numpy as np
#initializing the variables
N = 20
eps = 1.4
a= 0.5
N_cor= 20
N_cf =1000
w0 = 1
c = 2

##for anharmonic oscillator assuming eq.69 and comparing with eq.67 gives chi2 = -1/24 and chi3 = -1/6
tilde = True #set action that removes ghost states using xtilde
##creating arrays using numpy
x = np.zeros(N)
G = np.zeros((N_cf,N))

###defining the functions

def update(x):
    for j in range(0,N):
        old_x = x[j]
        old_Sj = S_anharmonic(j,x_tilde(0.5,-1/24,-1/6,x)) 
        x[j]=x[j]+r.uniform(-eps,eps)
        dS = S_anharmonic(j,x_tilde(0.5,-1/24,-1/6,x))-old_Sj
        if dS>0 and math.exp(-dS)<r.uniform(0,1):
            x[j] = old_x


def S(j,x):  #original local action, improved potential, do not remove the ghost
    jp = (j+1)%N
    jm = (j-1)%N
    return a*((x[j]**2)/2) + x[j]*(x[j]-x[jp]-x[jm])/a


def S_imp(j,x): ###improved action in the point x[j] of the lattice, first term potential the second kinetic term
    jpp= (j+2)%N
    jp= (j+1)%N
    jm= (j-1)%N
    jmm= (j-2)%N
    return a*(x[j]**2)/2 -(1/2*a)*x[j]*( (x[j]*(-2+ (1/3)*(x[jp]+x[jm]-x[j]))+(x[jp] +x[jm])*(1-(1/12)*(x[jp]+x[jm]))) +(x[jp]*(1+(1/3)*x[jp]-(1/12)*(2*x[jpp]+x[j]))) +(x[jm]*(1+(1/3)*x[jm]-(1/12)*(2*x[jmm]+x[j]))))
def compute_G(x,n):
    g =0
    for j in range(0,N):
        g = g + x[j]*x[(j+n)%N]
    return g/N

def MCaverage(x,G):
    for j in range(0,N):
        x[j] =0
    for j in range(0, 5*N_cor):
        update(x)
    for alpha in range(0,N_cf):
        for j in range(0, N_cor):
            update(x)
        for n in range(1,N):
            G[alpha][n-1] = compute_G(x,n)

def bootstrap(G):
    N_cf = len(G)
    G_bootstrap = []
    for i in range(0,N_cf):
        alpha = int(r.uniform(0,N_cf))
        G_bootstrap.append(G[alpha])
    return G_bootstrap

def bin(G,binsize):
    G_binned = []
    for i in range(0, len(G), binsize):
        G_avg =0
        for j in range(0, binsize):
            G_avg = G_avg + G[i+j]
        G_binned.append(G_avg/binsize)
    return G_binned
def avg(G):
    avg = np.sum(G,axis=0)/len(G)
    return avg
def sdev(G):
    g = np.asarray(G)
    return np.absolute(avg(g**2)-avg(g)**2)**0.5

def deltaE(G):
    avgG = avg(G)
    adE = np.log(np.absolute(avgG[:-1]/avgG[1:]))
    return adE/a
def bootstrap_deltaE(G,nbstrap=1000):
    avgE = np.absolute(deltaE(G))
    bsE = []
    for i in range(nbstrap):
        g = bootstrap(G)
        bsE.append(deltaE(g))
    bsE = np.array(bsE)
    sdevE = sdev(bsE)
    for i in range(int(len(avgE)/2)):
        print("n, deltaEn, error: %f, %f, %f" % (i+1, avgE[i],sdevE[i]) )
def delta(x,j):
    delta = (x[(j+1)%(N-1)] -2*x[j]+x[(j-1)%(N-1)])/a**2
    return delta

def x_tilde(chi1, chi2,chi3,x):  ###for the harmonic oscillator a simple computation gives chi2 = -1 and chi1 =0
    x_tilde = np.zeros(N)
    for j in range(0,N):
        x_tilde[j] = x[j]-  chi1*(a**2)*delta(x,j)-chi2*(a**2)*(w0**2)*x[j]
    return x_tilde

def S_anharmonic(j,x):#potential of eq. 69, first 4 terms
    jpp= (j+2)%N
    jp= (j+1)%N
    jm= (j-1)%N
    jmm= (j-2)%N
    return (1+2*w0*(x[j]**2))*a*(x[j]**2)/2 +((a**3)/24)*((x[j]+4*(x[j]**3))**2) -(a**2)*(x[j]**2)/2+((a**4)/2)*(x[j]**4)/4 -(1/2*a)*x[j]*( (x[j]*(-2+ (1/3)*(x[jp]+x[jm]-x[j]))+(x[jp] +x[jm])*(1-(1/12)*(x[jp]+x[jm]))) +(x[jp]*(1+(1/3)*x[jp]-(1/12)*(2*x[jpp]+x[j]))) +(x[jm]*(1+(1/3)*x[jm]-(1/12)*(2*x[jmm]+x[j]))))


MCaverage(x,G)
G = bin(G,200)
bootstrap_deltaE(G)
    


    
    
