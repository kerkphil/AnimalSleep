# -*- coding: utf-8 -*-
"""
This program is used to calibrate the Economics of Sleep model where the 
fundamental period is less than a day.  The model is one where the activity 
for the current period is optimally chosen to be one of two possible values:
work/play or sleep. The stock of sleep determines productivity in work, and 
shifts the itersection of the sleep cutoff level (d) with respect to the 
circadian cycle

This version of the program solves for daily steady state pattern by using a 
value function iteration on discretized grids to solve for discrete 
approximations to the policy and jump functions.  The value function includes
the value of the circadian cycle as an exogenous state variable.  Simulation 
uses these function, but the outcomes of that simulation are restricted to the 
discrete point on the grids for the variables.  Given starting values we
simulate for several days so that the model settles into a steady state pattern
that repeats over a 24-hour period.

This version treats A as a jump variable with only two values, sleep or awake.
It uses a power function for consumption and a power function for leisure.

The code chooses values for muS and sig to match the data for mean and standard
deviation of hour slept for capitve zoo elephants

Code written by Kerk l. Phillips
Oct. 11, 2018
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl
from numba import njit

name = 'GSleepElephantSimpleCe'

@njit
def moddefs(H, y, Am, A, z, mparams):
    '''
    H is the homeostatic process now
    y is the value of the circadian cycle now
    A is an indicator for working or sleeping now
    Am is an indicator for working or sleeping last period
    
    UC is utility from consumption >= 0
    Uy is utility from following the circadian cycle

    U is total utility
    '''
    
    # unpack mparams
    (q, nu_S, nu_W, kappa, muW, muS, lambd, chiS, eta, xi, mu, \
           gamma, yvect, rho, sig) = mparams
    # effective labor as a function of sleep stock
    b = np.exp(z)* (mu + xi*(muW-H)**eta)
    # if working (A=0)
    if (A == 0):
        # from trunk-to-mouth budget constraint
        C = b
        # consumption greater than savings is not allowed
        if np.isinf(C):
            UC = -1.E+50
        else:
            UC = C**gamma
        Hp =  muW + (H - muW)*np.exp(-1/nu_W)
    # if sleeping
    else:
        C = 0.
        UC = 0.
        Hp = muS + (H - muS)*np.exp(-1/nu_S)
    # switching cost
    Usw = lambd*np.abs(Am-A)
    # total utility
    U = UC - chiS*np.abs(Hp - y)**kappa - Usw
    
    return U, C, Hp

@njit
def runsim(T, epshist, mparams):
    Hhist = np.zeros(T+1)
    yhist = np.zeros(T+1)
    Ahist = np.zeros(T+1)
    zhist = np.zeros(T+1)
    Chist = np.zeros(T)
    bhist = np.zeros(T)
    Uhist = np.zeros(T)
    
    Hhist[0] = 0.
    Ahist[0] = 0
    iy = 0
    yhist[0] = yvect[0]
    
    for t in range(0, T):
        zhist[t+1] = rho*zhist[t] + epshist[t]
        if iy == q-1:
            iy = 0
        else:
            iy = iy + 1
        yhist[t+1] = yvect[iy]
            
        Usleep, Csleep, Hpsleep = \
            moddefs(Hhist[t], yhist[t], Ahist[t], 1, zhist[t], mparams)
        Uwake, Cwake, Hpwake = \
            moddefs(Hhist[t], yhist[t], Ahist[t], 0, zhist[t], mparams)
    
        if Usleep > Uwake:
            Ahist[t+1] = 1
            Hhist[t+1] = Hpsleep
            Chist[t] = Csleep
        else:
            Ahist[t+1] = 0
            Hhist[t+1] = Hpwake
            Chist[t] = Cwake
    
    return Hhist, yhist, Ahist, Chist, bhist, Uhist, zhist

@njit
def SMM(inparams, T, epshist, extraparams):
    # upack inparams
    muS = inparams[0] 
    if inparams[1] > 0.:
        sig = inparams[1]
    else:
        sig = 0.  
    
    (q, nu_S, nu_W, kappa, muW, lambd, chiS, eta, xi, mu, \
           gamma, yvect, rho) = extraparams
     
    mparams2 = (q, nu_S, nu_W, kappa, muW, muS, lambd, chiS, eta, xi, mu, \
           gamma, yvect, rho, sig)
     
    Hhist, yhist, Ahist, Chist, bhist, Uhist, zhist = runsim(T, epshist, mparams2)

    HrsSlept = np.zeros(ndays)
    for d in range(0,ndays):
        HrsSlept[d] = np.sum(Ahist[d:d+q-1])/pph

    HrsMean = np.mean(HrsSlept)
    HrsStd  = np.std(HrsSlept)
    
    outarray = np.array([HrsMean, HrsStd])
    
    sse = np.sum(np.abs(outarray - np.array([6.28, 0.21])))
    
    print(muS, sig, outarray, sse)
    
    return sse

     
def plothist(H, C, A, y, b, U, t):
    
    plt.figure()
    plt.subplot(2,2,1)
    plt.plot(t, H, label='H')
    plt.plot(t, y, label='y')
    plt.title('homeostatic and circadian')
    plt.xticks([])
    
    plt.subplot(2,2,2)
    plt.plot(t, A, label='A')
    plt.title('sleep / waking')
    plt.xticks([])
    
    plt.subplot(2,2,3)
    plt.plot(t, C, label='C')
    plt.title('consumption')
    #plt.xlabel('time')
    
    plt.subplot(2,2,4)
    plt.plot(t, U, label='U')
    plt.title('utility')
    #plt.xlabel('time')


# main program

# define parameters
pph = 6            # period per hour
q = 24*pph         # number of periods per day
nu_W = 8.*q/24.    # decay rate while awake
nu_S = 8.*q/24.    # decay rate while sleeping
kappa = 2.         # curvature of circadian penalty
muW = 1.0          # waking asymptote for homeostatic process
muS =   -2.242612940812111     # sleeping asymptote for homeostatic process
lambd = .2         # wake/sleep switching cost
chiS =  20.        # utility weight on circadian cycle
eta = 1.           # curvature of b(H) functuion
mu = 0.            # additive term for b(H) function
xi = 1.0           # scaling factor for b(H) function
gamma = 0.8         # curvature of consumption utility
sig = 0.01405219236974973*.22/0.17830366653549215  # standard deviation of z innovations
rho = .9**(24/q)   # autocorrelation of z innovations

# set up sine way for circadian cycle
yvect = np.linspace(0., 2*np.pi, num = q+1)
yvect = -np.cos(yvect)
#plt.plot(yvect)
#plt.show()

# Simulate with shocks
ndays = 10000
T = ndays*q    # number of periods to simulate
epshist = sig*pkl.load(open('epshist.pkl', 'rb'))

inparams =np.array([muS, sig])
extraparams = (q, nu_S, nu_W, kappa, muW, lambd, chiS, eta, xi, mu, \
           gamma, yvect, rho)

f = lambda inparams: SMM(inparams, T, epshist, extraparams)

from scipy.optimize import fmin

soln, junk = fmin(f, inparams, xtol=0.001, ftol=0.001, retall=1)

muS = soln[0]
sig = soln[1]

mparams1 = (q, nu_S, nu_W, kappa, muW, muS, lambd, chiS, eta, xi, mu, \
       gamma, yvect, rho, sig)
 
Hhist, yhist, Ahist, Chist, bhist, Uhist, zhist = runsim(T, epshist, mparams1)

HrsSlept = np.zeros(ndays)
for d in range(0,ndays):
    HrsSlept[d] = np.sum(Ahist[d:d+q-1])/pph

HrsMean = np.mean(HrsSlept)
HrsStd  = np.std(HrsSlept)
HrsAuto = np.corrcoef(HrsSlept[0:ndays-1],HrsSlept[1:ndays])
HrsAuto = HrsAuto[0,1]

print('muS:   ', muS)
print('sigma: ', sig)
print('average hours of sleep per day:   ', HrsMean)
print('st dev of hours of sleep per day: ', HrsStd)
print('autocorreation of sleep per day: ', HrsAuto)
print(' ')
data = (HrsMean, HrsStd)
pkl.dump(data, open( name + '.pkl', 'wb' ) )


# Simulate with no shocks to find SS
sig = 0.
ndays = 5
T = ndays*q    # number of periods to simulate
epshist = sig*np.random.normal(0., 1., T+1)

Hhist, yhist, Ahist, Chist, bhist, Uhist, zhist = runsim(T, epshist, mparams1)

# plot typical SS cycle over 2 days
start = T - 2*q 
# plot data
H = Hhist[start:T]
C = Chist[start:T]
A = Ahist[start:T]
y = yhist[start:T]
b = bhist[start:T]
U = Uhist[start:T]
t = range(start, T)

plothist(H, C, A, y, b, U, t)

plt.savefig(name + '_SS.pdf', format='pdf', dpi=2000)
plt.show()

# plot typical SS cycle over 1 day
start = T - q - 24*pph
end = T - 24*pph + 1

A = Ahist[start:end]
y = yhist[start:end]

time = (0, 4, 8, 12, 16, 20, 24)
plt.figure()
plt.subplot(2,1,1)
plt.plot(A)
plt.ylabel('sleep/awake')
plt.xticks(np.arange(0, q+1, step=pph*4),time)
plt.subplot(2,1,2)
plt.plot(y)
plt.xlabel('time of day')
plt.ylabel('circadian')
plt.xticks(np.arange(0, q+1, step=pph*4),time)
plt.savefig(name + '_SSsleep.pdf', format='pdf', dpi=2000)
plt.show()

unique, counts = np.unique(Ahist[T-q:T], return_counts=True)
print('fraction waking:   ', float(counts[0])/q, 'hours:', 24*float(counts[0])/q)
print('fraction sleeping :', float(counts[1])/q, 'hours:', 24*float(counts[1])/q)