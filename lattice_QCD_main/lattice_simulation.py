#########################################################################################





########################################################################################

import numpy as np
from numba import njit 
import math
import random as r
import random_matrices_generator as g
import numba

######define properties of lattice
beta = 5.5
u0 = 0.797
N = 4

N_cor = 20
N_cf = 100
 ###question here
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
        a = a +  plaquette(mu,mu-deltamu,x,link) 
    for deltamu in range(1,3-mu+1):
        a = a + plaquette(mu+deltamu,mu,movedown(x,mu+deltamu),link) 
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
                        old_S = S_new(d,x,link)
                        I = (int(r.uniform(0,200)))
                       # R = g.random_matrices()[I]
                        link[i][j][k][l][d] = np.dot(R[I],link[i][j][k][l][d])
                        dS = S_new(d,x,link)-old_S
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
                                loop =  loop + plaquette_new(mu,dperp,x,link)
    return loop/(N*N*N*N*12)
def compute_V(link,m):
    x_zero = [0,0,0,0]
    return aV(x_zero,link,m)

def smeared(d,x,link, epsilon,n):
    oldlink = np.zeros((3,3))
    for i in range(0,n):
        oldlink = oldlink+ link[x[0]][x[1]][x[2]][x[3]][d] + delta_2(d,x,link)
    return oldlink

def delta_2_rho(rho,mu,x,link):
    xpmu = moveup(x,mu)
    xprho = moveup(x,rho)
    xmmu = movedown(x,mu)
    xmrho = movedown(x,rho)
    xmrhopmu = moveup(xmrho,mu)
    staple1 = np.dot(link[x[0]][x[1]][x[2]][x[3]][rho], link[xprho[0]][xprho[1]][xprho[2]][xprho[3]][mu])
    staple1 = np.dot(staple1, np.conjugate( link[xpmu[0]][xpmu[1]][xpmu[2]][xpmu[3]][rho]))
    staple2 = np.dot(np.conjugate(link[xmrho[0]][xmrho[1]][xmrho[2]][xmrho[3]][rho]),link[xmrho[0]][xmrho[1]][xmrho[2]][xmrho[3]][mu])
    staple2 = np.dot(staple2, link[xmrhopmu[0]][xmrhopmu[1]][xmrhopmu[2]][xmrhopmu[3]][rho])
    return (1/u0**2)*(staple1 -2*(u0**2)*link[x[0]][x[1]][x[2]][x[3]][mu]+ staple2)
def delta_2(mu,x,link):
    f = np.zeros((3,3))
    for rho in range(0,4):
        f = f +  delta_2_rho(rho,mu,x,link)
    return 0.08*f

        
    

x0=[0,0,0,0]
def MCaverage_potential(link):
    output = []
    avg_V0 = [0]*N_cf
    avg_V1 = [0]*N_cf
    avg_V2 = [0]*N_cf
    avg_V3 = [0]*N_cf
    avg_plaquette = [0]*N_cf
    avg_loopa2a = [0]*N_cf
    for i in range(0,50): ###thermalization
        print(("%f" % i))
        print("%f" % update(link))
        
    for alpha in range(0,N_cf):
        print("%f" % alpha)
        for j in range(0,N_cor):
            update(link)
        avg_V0[alpha]= compute_V(link,1)
        #print("%f" % W_smeared(x0,link,2,3))
        avg_V1[alpha] = compute_V(link,2)
        #print("%f" % aV(x0,link,1))
        avg_V2[alpha] = compute_V(link,3)
        avg_V3[alpha] = compute_V(link,4)
        avg_plaquette[alpha] = compute(link)
    
    output.append(avg(avg_V0))    
    output.append(avg(avg_V1))
    #print("%f %f %f" % (potential[0], potential[1],potential[2]) )
    output.append(avg(avg_V2))
    output.append(avg(avg_V3))
    output.append(avg(avg_plaquette))
    output.append(sdev_abs(avg_plaquette))
    #print("%f %f %f" % (avg(avg_G1), potential[1],potential[2]) )    
    return np.asarray(output)



def W(x,link,n,m):   #W(x,t) = W(ma,na)
    staple = link[x[0]][x[1]][x[2]][x[3]][0]
    x = moveup(x,0)
    for i in range(0,n-1):
        staple = np.dot(staple, link[x[0]][x[1]][x[2]][x[3]][0])
        x = moveup(x,0)
    for j in range(0,m):
        x = moveup(x,1)
        staple = np.dot(staple, link[x[0]][x[1]][x[2]][x[3]][1])
    for k in range(0,n):
        x = movedown(x,0)
        staple = np.dot(staple, np.conjugate(link[x[0]][x[1]][x[2]][x[3]][0]))
    for l in range(0,m):
        x = movedown(x,1)
        staple = np.dot(staple, np.conjugate(link[x[0]][x[1]][x[2]][x[3]][1]))
    return (1/3)*np.absolute(np.trace(staple))


def aV(x,link,m):
    return W(x,link,0,m)/W(x,link,1,m)

def W_smeared(x,link,n,m):   #W(x,t) = W(ma,na)
    staple = smeared(0,x,link,0.08,4)
    for i in range(0,n):
        x = moveup(x,0)
        staple = np.dot(staple, smeared(0,x,link,0.08,4))
    for j in range(0,m):
        x = moveup(x,1)
        staple = np.dot(staple, smeared(1,x,link,0.08,4))
    for k in range(0,n):
        x = movedown(x,0)
        staple = np.dot(staple, np.conjugate(smeared(0,x,link,0.08,4)))
    for l in range(0,m):
        staple = np.dot(staple, np.conjugate(smeared(1,x,link,0.08,4)))
    return (1/3)*np.absolute(np.trace(staple))

def aV_smeared(x,link,m):
    return W_smeared(x,link,5,m)/W_smeared(x,link,6,m)


def avg(f):
    c =0
    tot = 0
    for i in range(0, len(f)):
        c =c+ f[i]
        tot = tot+1
    return c/tot


def sdev(G):
    g = np.asarray(G)
    return np.absolute(avg(g**2)-avg(g)**2)**0.5


def bootstrap(G):
    G_bootstrap = []
    for i in range(0,len(G)):
        alpha = int(r.uniform(0,N_cf))
        G_bootstrap.append(G[alpha])
    return G_bootstrap

def sdev_abs(G,nbstrap = 100):
    bsE = []
    for i in range(nbstrap):
        g = bootstrap(G)
        bsE.append(avg(g))
    bsE = np.array(bsE)
    return sdev(bsE)
        


#link[2][2][2][2][2]=g.unitary_matrix_generator()

#x = movedown(moveup(x,1),3)
link = initialize_links()
V = MCaverage_potential(link)
np.savetxt('V_wilsonfull_notsmeared.txt',V,)
#for i in range(0,N_cf):
#    f += loop_average_array[i]
#    #print("%f" % loop_average_array[i])




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

########idea save after 50 update and start from it, themralization only one time
