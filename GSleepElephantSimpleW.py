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

Code written by Kerk l. Phillips
Mar. 11, 2017
Adapted to the general model
Nov. 18, 2017
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl

name = 'GSleepElephantSimpleW'

def moddefs(H, y, Am, A, z, *mparams):
    '''
    H is the homeostatic process now
    y is the value of the circadian cycle now
    A is an indicator for working or sleeping now
    Am is an indicator for working or sleeping last period
    
    UC is utility from consumption >= 0
    Uy is utility from following the circadian cycle

    U is total utility
    '''
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


def runsim(T, mparams):
    Hhist = np.zeros(T+1)
    yhist = np.zeros(T+1)
    Ahist = np.zeros(T+1)
    zhist = np.zeros(T+1)
    Chist = np.zeros(T)
    bhist = np.zeros(T)
    Uhist = np.zeros(T)
    epshist = sig*np.random.normal(0., 1., T+1)
    
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
            Uhist[t] = Usleep
        else:
            Ahist[t+1] = 0
            Hhist[t+1] = Hpwake
            Chist[t] = Cwake
            Uhist[t] = Uwake
    
    return Hhist, yhist, Ahist, Chist, bhist, Uhist, zhist


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
muS = -1.5         # sleeping asymptote for homeostatic process
lambd = .2         # wake/sleep switching cost
chiS =  10.        # utility weight on circadian cycle
eta = 1.           # curvature of b(H) functuion
mu = 0.           # additive term for b(H) function
xi = 6.4           # scaling factor for b(H) function
gamma = .8         # curvature of consumption utility
sig = 0.           # standard deviation of z innovations
rho = .9**(24/q)   # autocorrelation of z innovations


# set up sine way for circadian cycle
yvect = np.linspace(0., 2*np.pi, num = q+1)
yvect = -np.cos(yvect)
#plt.plot(yvect)
#plt.show()

# save to mparams list
mparams = (q, nu_S, nu_W, kappa, muW, muS, lambd, chiS, eta, xi, mu, \
           gamma, yvect, rho, sig)


# Simulate with no shocks to find SS
ndays = 5
T = ndays*q    # number of periods to simulate

Hhist, yhist, Ahist, Chist, bhist, Uhist, zhist = runsim(T, mparams)

# plot typical SS cycle over 2 days
start = T - q - 20*pph
end = T - 20*pph + 1
# plot data
H = Hhist[start:end]
C = Chist[start:end]
A = Ahist[start:end]
y = yhist[start:end]
b = bhist[start:end]
U = Uhist[start:end]
y = yhist[start:end]
t = range(start, end)

plothist(H, C, A, y, b, U, t)

plt.savefig(name + '_SS.pdf', format='pdf', dpi=2000)
plt.show()

time = (4, 8, 12, 16, 20, 24, 4)
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


## Simulate with shocks
## Simulate with no shocks to find SS
#ndays = 100000
#T = ndays*q    # number of periods to simulate
#sig = .13
#
#Hhist, yhist, Ahist, Chist, bhist, Uhist, zhist = runsim(T, mparams)
#
### plot typical SS cycle over full sample
##start = 0 
### plot data
##H = Hhist[start:T]
##C = Chist[start:T]
##A = Ahist[start:T]
##y = yhist[start:T]
##b = bhist[start:T]
##U = Uhist[start:T]
##t = range(start, T)
##
##plothist(H, C, A, y, b, U, t)
##
##plt.savefig(name + '.pdf', format='pdf', dpi=2000)
##plt.show()
#
#HrsSlept = np.zeros(ndays)
#HProcess = np.zeros(ndays)
#for d in range(0,ndays):
#    HrsSlept[d] = np.sum(Ahist[d:d+q-1])/pph
#    HProcess[d] = np.mean(Ahist[d:d+q-1])
#
#HrsMean = np.mean(HrsSlept)
#HrsStd  = np.std(HrsSlept)
#HrsAuto = np.corrcoef(HrsSlept[0:ndays-1],HrsSlept[1:ndays])
#HrsAuto = HrsAuto[0,1]
#
#print('average hours of sleep per day:   ', HrsMean)
#print('st dev of hours of sleep per day: ', HrsStd)
#print('autocorr hours of sleep per day:  ', HrsAuto)
#print(' ')
#data = (HrsMean, HrsStd, HrsAuto)
#pkl.dump(data, open( name + '.pkl', 'wb' ) )