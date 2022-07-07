#########################################################################################
#Structure of the program:
#We initialize the link as identity, then we use alternatively Wilson action or Improved action to evolve the lattice using the metropolis algorithm. update() evolves the lattice and MC_average_potential run the simulation. The first 50 steps thermalize the system and then we update 20 times each measure.
#The action is computed using formula 94 and S_new keep account of all the plaquette in which contributes the link that we want to update.





########################################################################################

import numpy as np
from numba import njit 
import math
import random as r
import random_matrices_generator as g
import numba
#debug variables
test = False
test_rate = False
test_potential = False
######define properties of lattice
beta = 5.5
u0 = 0.797
N = 4

N_cor = 20
N_cf = 100



def initialize_links():
    link = [[[[[0]*4]*N]*N]*N]*N
    for i in range(0,N):
        for j in range(0,N):
            for k in range(0,N):
                for l in range(0,N):
                    for d in range(0,4):
                        link[i][j][k][l][d] = np.identity(3)
    return link

###############################################################################################################################################################
#Here we compute the S_new
#x is an array and represent a site in the lattice
#link[x[0]][x[1]][x[2]][x[3]][d] means U_d(n)


def moveup(x,d):#move the site in the lattice in the positive direction along d
    x[d] += 1
    if (x[d]>N-1):
        x[d] = x[d]-N+1
    return x

def movedown(x,d):#move the site in the lattice in the negative direction along d
    x[d] -= 1
    if (x[d]<0):
        x[d] = x[d]+ N-1
    return x

def plaquette_new(mu,nu, x,link):#compute plaquette following eq. 88
    staple = link[x[0]][x[1]][x[2]][x[3]][mu]
    xpmu = moveup(x,mu)
    xmmu = movedown(x,mu)
    xpnu = moveup(x,nu)
    xmnu = movedown(x,nu)
    xpmunu = moveup(xpnu,mu)
    staple = np.dot(staple,  link[xpmu[0]][xpmu[1]][xpmu[2]][xpmu[3]][nu])
    staple = np.dot(staple,  np.conjugate(link[xpnu[0]][xpnu[1]][xpnu[2]][xpnu[3]][mu]))
    if test:
        print("Test plaquette:")
        g.printing_matrix(staple)
        print("Test Unitarity:")
        g.printing_matrix(np.dot(staple,np.transpose(np.conjugate(staple))))
    staple = np.dot(staple, np.conjugate(link[x[0]][x[1]][x[2]][x[3]][nu]))
    return np.absolute((1/3)*np.real(np.trace(staple)))



#the idea here is consider all the contributes in which is present U_mu(x) in which mu > nu


def S_new(mu,x,link):  ###works better 0.49+-0.006 and acceptance rate 0.45/0.50 #write the plaquette contribution as: U_mu(x)U_{mu-deltamu}(x+mu)U^dagger_{mu}(x+(mu-deltamu))U^dagger_{mu-deltamu}(x)
    a = 0
    for deltamu in range(1,mu+1):#contribution of the first factor
        a = a +  plaquette_new(mu,mu-deltamu,x,link) 
#write the plaquette contribution as U_mu+deltamu(x)U_muU^dagger_mu+deltamuU^dagger_mu
    for deltamu in range(1,3-mu+1):#contribution of second factor
        a = a + plaquette_new(mu+deltamu,mu,movedown(x,mu+deltamu),link) 
    return -beta*a


def S_new_improved(mu,x,link): #same procedure here, we consider following the same reasoning also the wilson loop 2axa
    a = 0
    for deltamu in range(1,mu+1):
        a = a+ wilson_loop_2aa_new(mu,mu-deltamu,x,link)+wilson_loop_a2a_new(mu,mu-deltamu,x,link)
    for deltamu in range(1,3-mu+1):
        a = a + wilson_loop_a2a_new(mu+deltamu,mu,movedown(x,mu+deltamu),link)+ wilson_loop_2aa_new(mu+deltamu,mu,movedown(movedown(x,mu+deltamu),mu+deltamu),link)
    return S_new(mu,x,link)*(1/u0**4) + beta*a*(1/u0**6)*(1/12)
      
###############################################################################################################################################################    
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
    if test:
        print("test_2axa:")
        g.printing_matrix(staple)
        g.printing_matrix(np.dot(staple,np.transpose(np.conjugate(staple))))
    staple = np.dot(staple, np.conjugate(link[x[0]][x[1]][x[2]][x[3]][nu]))
    return np.absolute((1/3)*np.real(np.trace(staple)))
    

def wilson_loop_2aa_new(mu,nu,x,link):
    staple = link[x[0]][x[1]][x[2]][x[3]][mu]
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
    if test:
        print("test_ax2a:")
        g.printing_matrix(staple)
    staple = np.dot(staple, np.conjugate(link[x[0]][x[1]][x[2]][x[3]][nu]))
    return np.absolute((1/3)*np.real(np.trace(staple)))
    






    
R = g.random_matrices()#array which contains all the matrices R used for updating the links


def update(link):     #### change rate 0.4/0.5 with eps = 0.24
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
                        link[i][j][k][l][d] = np.dot(R[I],link[i][j][k][l][d])
                        dS = S_new(d,x,link)-old_S
                        tot = tot + 1
                        if dS>0 and math.exp(-dS)<r.uniform(0,1):
                            link[i][j][k][l][d] = oldlink[i][j][k][l][d]
                            c = c+ 1
    return 1-c/tot

def compute(link): #sum all the plaquettes in the lattice and then averaging
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

def compute_V(link,m): #compute the potential between two static quarks at distance m*a along the x axis
    x_zero = [0,0,0,0]
    return aV(x_zero,link,m)

################################################################################
#methods for computing smeared potential


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

##################################################################################################################################        
    

x0=[0,0,0,0]#starting point for evaluating W, used later for the potential
########################################################################################################
#method that run the simulation

def MCaverage_potential(link):
#initialize lists
    output = []
    avg_V0 = [0]*N_cf
    avg_V1 = [0]*N_cf
    avg_V2 = [0]*N_cf
    avg_V3 = [0]*N_cf
    avg_plaquette = [0]*N_cf
    avg_loopa2a = [0]*N_cf
    for i in range(0,50): ###thermalization
        if test_rate:
            print(("%f" % i))
            print("%f" % update(link))
        
    for alpha in range(0,N_cf):
        print("%f" % alpha)
        for j in range(0,N_cor):
            update(link)
        #save the measure of the potential at each r
        avg_V0[alpha]= compute_V(link,1)
        avg_V1[alpha] = compute_V(link,2)
        avg_V2[alpha] = compute_V(link,3)
        avg_V3[alpha] = compute_V(link,4)
        if test_potential:
            print("%f" % W_smeared(x0,link,2,3))
            print("%f" % aV(x0,link,1))
        #save measure of the Wilson loop
        avg_plaquette[alpha] = compute(link)
    #avg compute the average of each array
    #append the average to the output array, th results are written in a txt file
    output.append(avg(avg_V0))    
    output.append(avg(avg_V1))
    output.append(avg(avg_V2))
    output.append(avg(avg_V3))
    output.append(avg(avg_plaquette))
    #error of the Wilson loop average computed with bootstrap method
    output.append(sdev_abs(avg_plaquette))   
    return np.asarray(output)

########################################################################################################

#methods for computing potential

def W(x,link,n,m):   #W(x,t) = W(ma,na)
    staple = np.identity(3)
    for i in range(0,n):
        staple = np.dot(staple, link[x[0]][x[1]][x[2]][x[3]][0])
        x = moveup(x,0)    
    for j in range(0,m):
        staple = np.dot(staple, link[x[0]][x[1]][x[2]][x[3]][1])
        x = moveup(x,1)
    for k in range(0,n):
        x = movedown(x,0)
        staple = np.dot(staple, np.conjugate(link[x[0]][x[1]][x[2]][x[3]][0]))
    
    for l in range(0,m):
        x = movedown(x,1)
        staple = np.dot(staple, np.conjugate(link[x[0]][x[1]][x[2]][x[3]][1]))
    return (1/3)*(np.trace(staple))


def aV(x,link,m):
    return W(x,link,2,m)/W(x,link,3,m)

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

##################################################################################################
 #tools for computing average and standard deviation given an array. Bootstrap method is used
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
        
###########################################################################################################################################################


link = initialize_links()
V = MCaverage_potential(link)
np.savetxt('V_wilsonfull_notsmeared.txt',V,)






