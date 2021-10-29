""" STOCHASTIC INTEREST RATE AND LEARNING
    -------------------------------------
    Code for the paper Stochastic Interest Rate and Learning.
"""


# -------------------------------------------------------------------------------
# IMPORT LIBRARIES
# -------------------------------------------------------------------------------


# Import libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import odeint


# -------------------------------------------------------------------------------
# ECONOMY
# -------------------------------------------------------------------------------


# Utility function u

def utility_u(consumption_, gamma_):

    if gamma_ != 0:

        utility_ = (1 - np.exp(- gamma_ * consumption_)) / gamma_

    else:

        utility_ = consumption_

    return utility_

""" Comments:
    * Dividing by gamma_ guarantees that utility increases in consumption also for risk averse investors.
    * Adding the 1 guarantees that utility starts at zero for every gamma_.
"""

# Plot utility function u

consumption = np.arange(0, 1, 0.01)
gamma = [-1.5, 0.0, 1.5]

utility = pd.DataFrame(columns=gamma)

for g in gamma:

    utility.loc[:, g] = utility_u(consumption, g)

fig_1, ax = plt.subplots(nrows=1, ncols=1, figsize=(5.5, 4))
ax.plot(consumption, utility, label=gamma)
ax.set_ylabel('U(C)', fontsize=10)
ax.set_xlabel('C', fontsize=10)
ax.set_title('CARA Utility',fontsize=10)
ax.legend(loc='upper left', fontsize=10)
ax.set_xlim(xmin=np.min(consumption), xmax=np.max(consumption))
ax.set_ylim(ymin=0, ymax=np.max(utility.values))
fig_1.tight_layout()
fig_1.savefig('images/fig_1.png')


# Utility function h

def utility_h(consumption_, psi_):

    if psi_ != 0:

        utility_ = (1 - np.exp(- 1 / psi_ * consumption_)) * psi_

    else:

        utility_ = consumption_

    return utility_


# Plot utility function h

consumption = np.arange(0, 1, 0.01)
psi = [-1.5, 0.0, 1.5]

utility = pd.DataFrame(columns=psi)

for p in psi:

    utility.loc[:, p] = utility_h(consumption, p)

fig_2, ax = plt.subplots(nrows=1, ncols=1, figsize=(5.5, 4))
ax.plot(consumption, utility, label=psi)
ax.set_ylabel('U(C)', fontsize=10)
ax.set_xlabel('C', fontsize=10)
ax.set_title('Exponential Utility',fontsize=10)
ax.legend(loc='upper left', fontsize=10)
ax.set_xlim(xmin=np.min(consumption), xmax=np.max(consumption))
ax.set_ylim(ymin=0, ymax=np.max(utility.values))
fig_2.tight_layout()
fig_2.savefig('images/fig_2.png')


# -------------------------------------------------------------------------------
# ORDINARY DIFFERENTIAL EQUATIONS
# -------------------------------------------------------------------------------

# Parameters

theta2 = 0.0025
theta1 = - 0.0150
sigmaD = 0.0240
r = 0.0041
delta = 1
p12 = 0.1000
p21 = 0.0167

pi2 = p12 / (p12 + p21)

Gamma_D = 1 / r
Gamma_pi = (theta2 - theta1) / (r * (r + p12 + p21))
Gamma_1 = theta1 / r**2 + (theta2 - theta1) * p12 / (r**2 * (r + p12 + p21))

def q_parameters(pi_, gamma_):

    h = (theta2 - theta1) / sigmaD * pi_ * (1 - pi_)

    Q3 = h**2 / 2
    Q1 = gamma_ * sigmaD * h + r * gamma_ * Gamma_pi * h**2 - (p12 + p21) * (pi2 - pi_)
    Q0 = (gamma_ * r)**2 / 2 * Gamma_pi**2 * h**2 + gamma**2 * r * Gamma_pi * sigmaD * h + r * np.log(delta)

    return Q3, Q1, Q0

# System of ODEs

def model(y, pi, gamma):

    y1 = y[0]
    y2 = y[1]
    Q3, Q1, Q0 = q_parameters(pi, gamma)

    dy1 = y2
    dy2 = y2**2 + y2 * Q1 / Q3 + y1 * r / Q3 + Q0 / Q3

    return dy1, dy2


# Initial condition

y0 = [0.1, 0.1]

# Time points

pi = np.linspace(0.001, 0.999, 999)

# Solve ODE

gamma = 1
x = odeint(model, y0, pi, args=(gamma,))

y1 = x[:, 0]
y2 = x[:, 1]


# Plot

plt.plot(pi, y1)
plt.show()


# ---------------
# First order ODE
# ---------------

def model(y, t, k):

    dydt = -k * y

    return dydt

y0 = 5
t = np.linspace(0,20)
k = 0.3

y = odeint(model, y0, t, args=(k,))

plt.plot(t,y)
plt.xlabel('Time')
plt.ylabel('y(t)')
plt.show()

#



# -------------------------------------------------------------------------------
# PHASE DIAGRAM
# -------------------------------------------------------------------------------


# Gamma = 1 / psi

def parameters(theta2_, theta1_, p12_, p21_, pi_, sigmaD_, gamma_, psi_, rate_, delta_):

    h_ = (theta2_ - theta1_) / sigmaD_ * pi_ * (1 - pi_)
    pi2_ = p12_ / (p12_ + p21_)
    Gamma_pi_ = (theta2_ - theta1_) / (rate_ * (rate_ + p12_ + p21_))
    S_pi_ = 
    
    Q3_ = h_ ** 2 / 2
    Q2_ = gamma_ * psi_ * h_ ** 2 / 2
    Q1_ = gamma_ * sigmaD_ * h_ + rate_ * gamma_ * Gamma_pi_ * h_ ** 2 - (p12_ + p21_) * (pi2_ - pi_) - gamma_ ** 2 * psi_ * rate_ * S_pi_ * h_ ** 2 + gamma_ * rate_ * S_pi_ * h_ ** 2
    Q0_ = (rate_ * gamma_) ** 2 / 2 * Gamma_pi_ ** 2 * h_ ** 2 + rate_ * gamma_ ** 2 * sigmaD_ * Gamma_pi_ * h_ + rate_ * np.log(delta_)

    return Q3_, Q2_, Q1_

def parabola(x2_, Q2_, Q1_, Q0_, rate_):

    x2_locus_ = - x2_ ** 2 * Q2_ / rate_ - x2_ * Q1_ / rate_ - Q0_ / rate_

    return x2_locus_

# Parameters

pi = 0.50
theta2 = 0.05
theta1 = -0.05
p12 = 0.20
p21 = 0.02 #0.012
sigmaD = 0.10
rate = 0.10
delta = 1
gamma = 4.5

# Phase diagram

y2 = np.arange(-5.0, 5.1, 0.1)
y1_lim_zero = y2 * p12 / rate - np.log(delta)
y1_lim_one = - y2 * p21 / rate - np.log(delta)


pi_list = [0.5, 0.05, 0.3, 0.7, 0.95]
y1_locus = pd.DataFrame(columns=pi_list)

for pi in pi_list:

    Q3, Q2, Q1 = Qparams(theta2, theta1, p12, p21, pi, sigmaD, gamma, rate)
    y1_locus.loc[:, pi] = parabola(y2, Q3, Q2, Q1, rate)


# Plot phase diagram

fig_1, ax = plt.subplots(nrows=1, ncols=1, figsize=(5.5, 4))
ax.plot(y2, y1_locus.loc[:, 0.5], color='blue')
ax.plot(y2, y1_locus.iloc[:, 1:], color='blue', linestyle='dashed', linewidth=0.5)
ax.plot(y2, y1_lim_zero, color='red')
ax.plot(y2, y1_lim_one, color='red')
ax.hlines(y=0, xmin=np.min(y2), xmax=np.max(y2), color='black', linestyles='dashed')
ax.vlines(x=0, ymin=np.min(np.min(y1_locus)), ymax=np.max(np.max(y1_lim_zero)), color='black', linestyles='dashed')
ax.arrow(x=2.5, y=2.5*p12/rate, dx=0, dy=-0.5*p12/rate, head_width = 0.1)
ax.arrow(x=2.5, y=2.0*p12/rate, dx=0, dy=-7.5, head_width = 0.1)
ax.arrow(x=2.5, y=-1.5*p12/rate, dx=0, dy=-3.6, head_width = 0.1)
ax.arrow(x=2.8, y=-7.5, dx=0, dy=0.7, head_width = 0.1)
ax.arrow(x=2.8, y=-6.7, dx=0, dy=5.2, head_width = 0.1)
ax.arrow(x=2.8, y=-1.3, dx=0, dy=0.5, head_width = 0.1)
plt.text(3.1, -13.5, '$\psi(0.5,y_2)$', fontsize=8)
plt.text(2.5, 7.8, r'$y_1=\frac{p_{12}}{r} y_2$', fontsize=8)
plt.text(-4.5, 1.5, r'$y_1=-\frac{p_{21}}{r} y_2$', fontsize=8)
plt.text(3.9, 5.2, '$\psi(0.05,y_2)$', fontsize=6)
plt.text(3.9, -3.0, '$\psi(0.95,y_2)$', fontsize=6)
plt.text(3.9, -6, '$\psi(0.30,y_2)$', fontsize=6)
plt.text(3.9, -9.5, '$\psi(0.70,y_2)$', fontsize=6)
ax.set_ylabel('$x_1=f$', fontsize=10)
ax.set_xlabel('$x_2=f^\prime$', fontsize=10)
ax.set_title(r'Phase Diagram $\gamma=\frac{1}{\psi}$',fontsize=10)
ax.set_xlim(xmin=np.min(y2), xmax=np.max(y2))
ax.set_ylim(ymin=np.min(np.min(y1_locus)), ymax=np.max(np.max(y1_lim_zero)))
fig_1.savefig('images/phase_diagram.png')






# -------------------------------------------------------------------------------
# PROPOSITION 1
# -------------------------------------------------------------------------------


# Price

thetas = np.arange(0.00, 0.31, 0.001)
sigmaDs = np.arange(0.00, 0.31, 0.001)
interest = np.arange(0.01, 0.101, 0.001)
gammas = [0.0, 0.5, 1.0, 1.5]

def Pt_1(theta, sigmaD, interest, gamma, dividend):

    Pt = - gamma * sigmaD ** 2 / interest ** 2 + dividend / interest + theta / interest ** 2

    return Pt

Pt_interest = pd.DataFrame(columns=gammas)

for j in gammas:

        Pt_interest.loc[:, j] = Pt_1(0.05, 0.25, interest, j, 1)

plt.plot(interest, Pt_interest)


# Value function


def Jt_1(delta, psi, rate, gamma, sigmaD, wealth):

    Jt = - delta / psi * np.exp(1 - delta / (rate) - np.log(rate) - gamma * sigmaD ** 2 / (2 * psi * rate) - rate / psi * wealth)

    return Jt + 1

delta = 0.05
sigmaD = 0.2

psi = 1.5
gamma = 1.5
rates = np.arange(0.01, 0.101, 0.001)
gammas = [0.0, 0.5, 1.0, 1.5]
psis = [0.5, 1.0, 1.5, 2.0]


Jt_rate_gamma = pd.DataFrame(columns=gammas)

for g in gammas:

    Jt_rate_gamma.loc[:, g] = Jt_1(delta=delta, psi=psi, rate=rates, gamma=g, sigmaD=sigmaD, wealth=1)

Jt_rate_psi = pd.DataFrame(columns=psis)

for p in psis:

    Jt_rate_psi.loc[:, p] = Jt_1(delta=delta, psi=p, rate=rates, gamma=gamma, sigmaD=sigmaD, wealth=1)

fig_2, ax = plt.subplots(nrows=1, ncols=2, figsize=(11, 4))

ax[0].plot(rates, Jt_rate_gamma, label=gammas)
ax[0].set_ylabel('Value function ($J_t$)', fontsize=10)
ax[0].set_xlabel('Interest rate ($r$)', fontsize=10)
ax[0].set_title('Value Function For different levels of ARA',fontsize=10)
ax[0].legend(loc='upper right', fontsize=10)
ax[0].set_xlim(xmin=np.min(rates), xmax=np.max(rates))

ax[1].plot(rates, Jt_rate_psi, label=psis)
ax[1].set_ylabel('Value function ($J_t$)', fontsize=10)
ax[1].set_xlabel('Interest rate ($r$)', fontsize=10)
ax[1].set_title('Value Function as a Function of EIS',fontsize=10)
ax[1].legend(loc='upper right', fontsize=10)
ax[1].set_xlim(xmin=np.min(rates), xmax=np.max(rates))

fig_2.tight_layout()
fig_2.savefig('images/fig_2.png')


delta = 0.05
sigmaD = 0.2

psi = 1.5
gamma = 1.5
rates = np.arange(0.01, 0.101, 0.001)
gammas = [0.0, 0.5, 1.0, 1.5]
psis = [0.5, 1.0, 1.5, 2.0]


Jt_rate_gamma = pd.DataFrame(columns=gammas)

for g in gammas:

    Jt_rate_gamma.loc[:, g] = Jt_1(delta=delta, psi=psi, rate=rates, gamma=g, sigmaD=sigmaD, wealth=1)

Jt_rate_psi = pd.DataFrame(columns=psis)

for p in psis:

    Jt_rate_psi.loc[:, p] = Jt_1(delta=delta, psi=p, rate=rates, gamma=gamma, sigmaD=sigmaD, wealth=1)

fig_2, ax = plt.subplots(nrows=1, ncols=2, figsize=(11, 4))

ax[0].plot(rates, Jt_rate_gamma, label=gammas)
ax[0].set_ylabel('Value function ($J_t$)', fontsize=10)
ax[0].set_xlabel('Interest rate ($r$)', fontsize=10)
ax[0].set_title('Value Function For different levels of ARA',fontsize=10)
ax[0].legend(loc='upper right', fontsize=10)
ax[0].set_xlim(xmin=np.min(rates), xmax=np.max(rates))

ax[1].plot(rates, Jt_rate_psi, label=psis)
ax[1].set_ylabel('Value function ($J_t$)', fontsize=10)
ax[1].set_xlabel('Interest rate ($r$)', fontsize=10)
ax[1].set_title('Value Function as a Function of EIS',fontsize=10)
ax[1].legend(loc='upper right', fontsize=10)
ax[1].set_xlim(xmin=np.min(rates), xmax=np.max(rates))

fig_2.tight_layout()
fig_2.savefig('images/fig_2.png')



