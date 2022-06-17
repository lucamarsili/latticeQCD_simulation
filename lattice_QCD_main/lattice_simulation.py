#########################################################################################
#simulation of the lattice, S_new is the action of the eq. 114 while S_new_improved is based on eq. 103
#S is the previous action I have used which still gives good results.
#In  S_new I have used plaquette_new as method for computing loops, here I am using the convention of 0506036. 
#





########################################################################################

import numpy as np
from numba import njit 
import math
import random as r
import random_matrices_generator as g
import numba

######define properties of lattice
beta = 1.719
u0 = 0.797
N = 4

N_cor = 20
N_cf = 100
 
#link = [[[[[0]*N]*N]*N]*N]*4
#@njit
def initialize_links():
    link = [[[[[0]*4]*N]*N]*N]*N
    for i in range(0,N):
        for j in range(0,N):
            for k in range(0,N):
                for l in range(0,N):
                    for d in range(0,4):
                        link[i][j][k][l][d] = np.identity(3)
    return link



def moveup(x,d):
    x[d] += 1
    if (x[d]>N-1):
        x[d] = x[d]-N+1
    return x

def movedown(x,d):
    x[d] -= 1
    if (x[d]<0):
        x[d] = x[d]+ N-1
    return x

def plaquette(mu,nu, x,link):
    staple = link[x[0]][x[1]][x[2]][x[3]][mu]
    #g.printing_matrix(staple)
    xpmu = moveup(x,mu)
    xmmu = movedown(x,mu)
    xpnu = moveup(x,nu)
    xmnu = movedown(x,nu)
    xpmunu = moveup(xpnu,mu)
    staple = np.dot(staple,  link[xpmu[0]][xpmu[1]][xpmu[2]][xpmu[3]][nu])
    staple = np.dot(staple,  np.conjugate(link[xpmunu[0]][xpmunu[1]][xpmunu[2]][xpmunu[3]][mu]))
    #g.printing_matrix(staple)
    staple = np.dot(staple, np.conjugate(link[xpnu[0]][xpnu[1]][xpnu[2]][xpnu[3]][nu]))
    return np.absolute((1/3)*np.real(np.trace(staple)))
    #g.printing_matrix(np.dot(staple,np.transpose(np.conjugate(staple))))


def plaquette_new(mu,nu, x,link):
    staple = link[x[0]][x[1]][x[2]][x[3]][mu]
    #g.printing_matrix(staple)
    xpmu = moveup(x,mu)
    xmmu = movedown(x,mu)
    xpnu = moveup(x,nu)
    xmnu = movedown(x,nu)
    xpmunu = moveup(xpnu,mu)
    staple = np.dot(staple,  link[xpmu[0]][xpmu[1]][xpmu[2]][xpmu[3]][nu])
    staple = np.dot(staple,  np.conjugate(link[xpnu[0]][xpnu[1]][xpnu[2]][xpnu[3]][mu]))
    #g.printing_matrix(staple)
    staple = np.dot(staple, np.conjugate(link[x[0]][x[1]][x[2]][x[3]][nu]))
    return np.absolute((1/3)*np.real(np.trace(staple)))
    #g.printing_matrix(np.dot(staple,np.transpose(np.conjugate(staple))))

def S_new(mu,x,link):  ###works better 0.48 and acceptance rate 0.45/0.50 don't touch
    a = 0
    for deltamu in range(1,mu+1):
        a = a +  plaquette(mu,mu-deltamu,x,link) + plaquette(mu,mu-deltamu,movedown(x,mu-deltamu),link)
    for deltamu in range(1,3-mu+1):
        a = a + plaquette(mu+deltamu,mu,movedown(x,mu+deltamu),link)+ plaquette(mu+deltamu,mu,x,link)
    return -beta*a
def S_new_improved(mu,x,link):
    a = 0
    for deltamu in range(1,mu+1):
        a = a+ wilson_loop_2aa_new(mu,mu-deltamu,x,link)+wilson_loop_a2a_new(mu,mu-deltamu,x,link)
    for deltamu in range(1,3-mu+1):
        a = a + wilson_loop_a2a_new(mu+deltamu,mu,movedown(x,mu+deltamu),link)+ wilson_loop_2aa_new(mu+deltamu,mu,movedown(movedown(x,mu+deltamu),mu+deltamu),link)
    return S_new(mu,x,link)*(1/u0**4) + beta*a*(1/u0**6)*(1/12)
      
    
def wilson_loop_a2a_new(mu,nu, x,link):
    staple = link[x[0]][x[1]][x[2]][x[3]][mu]
    #g.printing_matrix(staple)
    xpmu = moveup(x,mu)
    xmmu = movedown(x,mu)
    xpnu = moveup(x,nu)
    xpmunu = moveup(xpnu,mu)
    xppnu = moveup(moveup(x,nu),nu)
    xmnu = movedown(x,nu)
    xppmunu = moveup(moveup(moveup(xpnu,mu),mu),nu)
    staple = np.dot(staple,  link[xpmu[0]][xpmu[1]][xpmu[2]][xpmu[3]][nu])
    staple = np.dot(staple,  link[xpmunu[0]][xpmunu[1]][xpmunu[2]][xpmunu[3]][nu])
    staple = np.dot(staple,  np.conjugate(link[xppnu[0]][xppnu[1]][xppnu[2]][xppnu[3]][mu]))
    staple = np.dot(staple,  np.conjugate(link[xpnu[0]][xpnu[1]][xpnu[2]][xpnu[3]][nu]))
    #g.printing_matrix(staple)
    staple = np.dot(staple, np.conjugate(link[x[0]][x[1]][x[2]][x[3]][nu]))
    return np.absolute((1/3)*np.real(np.trace(staple)))
    #g.printing_matrix(np.dot(staple,np.transpose(np.conjugate(staple))))

def wilson_loop_2aa_new(mu,nu,x,link):
    staple = link[x[0]][x[1]][x[2]][x[3]][mu]
    #g.printing_matrix(staple)
    xpmu = moveup(x,mu)
    xmmu = movedown(x,mu)
    xpnu = moveup(x,nu)
    xpmunu = moveup(xpnu,mu)
    xppmu = moveup(moveup(x,mu),mu)
    xmnu = movedown(x,nu)
    xppmunu = moveup(moveup(moveup(xpnu,mu),mu),nu)
    staple = np.dot(staple,  link[xpmu[0]][xpmu[1]][xpmu[2]][xpmu[3]][mu])
    staple = np.dot(staple,  link[xppmu[0]][xppmu[1]][xppmu[2]][xppmu[3]][nu])
    staple = np.dot(staple,  np.conjugate(link[xpmunu[0]][xpmunu[1]][xpmunu[2]][xpmunu[3]][mu]))
    staple = np.dot(staple,  np.conjugate(link[xpnu[0]][xpnu[1]][xpnu[2]][xpnu[3]][mu]))
    #g.printing_matrix(staple)
    staple = np.dot(staple, np.conjugate(link[x[0]][x[1]][x[2]][x[3]][nu]))
    return np.absolute((1/3)*np.real(np.trace(staple)))
    

def S(nu,x,link):  ###works better 0.48 and acceptance rate 0.45/0.50 don't touch
    a = 0
    for deltanu in range(0,3-nu):
        deltanu = deltanu + 1
        a = a +  plaquette(nu+deltanu,nu, movedown(x,nu+deltanu),link)
    for deltanu in range(0,nu):
        deltanu = deltanu+1
        a = a + plaquette(nu,nu-deltanu,x,link)
    return -beta*a





    
R = g.random_matrices()
#bubblesort_autojit = numba.jit(bubblesort)

def update(link):     #### change rate 0.4 with eps = 0.24
    oldlink = [[[[[0]*4]*N]*N]*N]*N
    c =0
    tot =0
    for i in range(0,N):
        for j in range(0,N):
            for k in range(0,N):
                for l in range(0,N):
                    for d in range(0,4):
                        oldlink[i][j][k][l][d] = link[i][j][k][l][d]
                        x = [i,j,k,l]
                        old_S = S_new_improved(d,x,link)
                        I = (int(r.uniform(0,200)))
                       # R = g.random_matrices()[I]
                        link[i][j][k][l][d] = np.dot(R[I],link[i][j][k][l][d])
                        dS = S_new_improved(d,x,link)-old_S
                        tot = tot + 1
                        if dS>0 and math.exp(-dS)<r.uniform(0,1):
                            link[i][j][k][l][d] = oldlink[i][j][k][l][d]
                            c = c+ 1
    return 1-c/tot
def compute(link):
    x = [0]*4
    loop = 0
    for i in range(0,N):
        for j in range(0,N):
            for k in range(0,N):
                for l in range(0,N):
                    x = [i,j,k,l]
                    for mu  in range(0,4):
                        for dperp in range(0,3):
                            if (dperp != mu):
                                loop =  loop + wilson_loop_a2a_new(mu,dperp,x,link)
    return loop/(N*N*N*N*12)

def MCaverage(link):
    plaquette_avg = [0]*N_cf
    for i in range(0,50): ###thermalization
        print(("%f" % i))
        print("%f" % update(link))
        
    for alpha in range(0,N_cf):
        print("%f" % alpha)
        for j in range(0,N_cor):
            update(link)
           
        plaquette_avg[alpha] = compute(link)
    return plaquette_avg


















#link[2][2][2][2][2]=g.unitary_matrix_generator()

#x = movedown(moveup(x,1),3)
link = initialize_links()
loop_average_array = MCaverage(link)
f =0
for i in range(0,N_cf):
    f += loop_average_array[i]
    #print("%f" % loop_average_array[i])
f = f/N_cf


print("%f" % f)
#print("%f" % a)
#g.printing_matrix(np.dot(link[3][4][5][2][2],np.transpose(np.conjugate(link[3][4][5][2][2]))))
#g.printing_matrix(link[3][4][6][2][2])
#link[3][4][6][2][2] = link[1][2][7][3][0]
#for i in range(0,50):
 #   print("%f" %i)
#for i in range(0,25):
 #   update(link)
#print("%f" % update(link))
#g.printing_matrix(link[3][4][6][2][2])

#######
