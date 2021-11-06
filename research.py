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
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# Set Up

mpl.rcParams['lines.linewidth'] = 0.9


# -------------------------------------------------------------------------------
# ECONOMY
# -------------------------------------------------------------------------------


# Parameters

consumption = np.arange(0, 1.01, 0.01)
gammas = [-1.5, 0.0, 1.5]
psis = [-1.5, 0.0, 1.5]


# Function: Utility u(c)

def utility_u(Consumption, Gamma):

    if Gamma != 0:

        Utility = (1 - np.exp(- Gamma * Consumption)) / Gamma

    else:

        Utility = Consumption

    return Utility

""" Comments:
    - Dividing by Gamma guarantees that utility increases in consumption also for risk averse investors.
    - Adding the 1 guarantees that utility starts at zero for every Gamma.
"""


# Function: Utility h(c)

def utility_h(Consumption, Psi):

    if Psi != 0:

        Utility = (1 - np.exp(- 1 / Psi * Consumption)) * Psi

    else:

        Utility = Consumption

    return Utility


# Calculate utility for different values of gamma

utility_gamma = pd.DataFrame(columns=gammas)

for gamma in gammas:

    utility_gamma.loc[:, gamma] = utility_u(consumption, gamma)


# Calculate utility for different values of psi

utility_psi = pd.DataFrame(columns=psis)

for psi in psis:

    utility_psi.loc[:, psi] = utility_h(consumption, psi)


# Plot utility for different values of gamma

fig_1, ax = plt.subplots(nrows=1, ncols=2, figsize=(11, 4))

ax[0].plot(consumption, utility_gamma, label=gammas)
ax[0].set_ylabel('Utility', fontsize=10)
ax[0].set_xlabel('Consumption', fontsize=10)
ax[0].set_title('Utility - Risk Preferences',fontsize=10)
ax[0].legend(loc='upper left', fontsize=8)
ax[0].set_xlim(xmin=np.min(consumption), xmax=np.max(consumption))
ax[0].set_ylim(ymin=0, ymax=np.max(utility_gamma.values))
ax[0].grid()

ax[1].plot(consumption, utility_psi, label=psis)
ax[1].set_ylabel('Utility', fontsize=10)
ax[1].set_xlabel('Consumption', fontsize=10)
ax[1].set_title('Utility - Time Preferences',fontsize=10)
ax[1].legend(loc='upper left', fontsize=8)
ax[1].set_xlim(xmin=np.min(consumption), xmax=np.max(consumption))
ax[1].set_ylim(ymin=0, ymax=np.max(utility_psi.values))
ax[1].grid()

fig_1.tight_layout()
fig_1.savefig('images/fig_1.png')


# -------------------------------------------------------------------------------
# PHASE DIAGRAM FOR f(pi)
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

gamma_D = 1 / r
gamma_pi = (theta2 - theta1) / (r * (r + p12 + p21))
gamma_1 = theta1 / r ** 2 + (theta2 - theta1) * p12 / (r ** 2 * (r + p12 + p21))


# Function: Coefficients q(pi) for ODE of f(pi)

def parameters_q(Pi, Gamma, Psi, Theta2, Theta1, SigmaD, P12, P21, Pi2, Gamma_pi, R, Delta):

    H = (Theta2 - Theta1) / SigmaD * Pi * (1 - Pi)

    if Gamma == 1 / Psi:

        Q3 = H ** 2 / 2
        Q1 = Gamma * SigmaD * H + R * Gamma * Gamma_pi * H ** 2 - (P12 + P21) * (Pi2 - Pi)
        Q0 = (Gamma * R) ** 2 / 2 * Gamma_pi ** 2 * H ** 2 + Gamma ** 2 * R * Gamma_pi * SigmaD * H + R * np.log(Delta)

        return Q3, Q1, Q0

    else:

        pass


# Function: point x2 that does not move

def x2_stable(Gamma, Psi, Theta2, Theta1, P12, P21, Gamma_pi, R):

    if Gamma == 1 / Psi:

        X2_hat = - (Gamma ** 2 * R * Gamma_pi * (Theta2 - Theta1)) / (Gamma * (Theta2 - Theta1) + P12 + P21)

        return X2_hat

    else:

        pass


# Locus of x2'(pi)=0 and point x2 that does not move

gamma = 10
psi = 1 / gamma

x2 = np.arange(-2.00, 2.01, 0.01).round(2)
pis = [0.00, 0.05, 0.50, 1.00]

phase_plot = pd.DataFrame(columns=pis)

phase_plot.loc[:, pis[0]] = x2 * p12 / r
phase_plot.loc[:, pis[3]] = - x2 * p21 / r

q3, q1, q0 = parameters_q(pis[1], gamma, psi, theta2, theta1, sigmaD, p12, p21, pi2, gamma_pi, r, delta)
phase_plot.loc[:, pis[1]] = - x2 ** 2 * q3 / r - x2 * q1 / r - q0 / r

q3, q1, q0 = parameters_q(pis[2], gamma, psi, theta2, theta1, sigmaD, p12, p21, pi2, gamma_pi, r, delta)
phase_plot.loc[:, pis[2]] = - x2 ** 2 * q3 / r - x2 * q1 / r - q0 / r

x2_hat = x2_stable(gamma, psi, theta2, theta1, p12, p21, gamma_pi, r)
x1_hat = x2_hat * p12 / r


# Plot phase diagram

fig_2, ax = plt.subplots(nrows=1, ncols=1, figsize=(5.5, 4))

ax.plot(x2, phase_plot.loc[:, pis[0]], color='green', label='$x_1=p_{12}/r x_2$')
ax.plot(x2, phase_plot.loc[:, pis[1]], color='purple', label='$\Psi(0.05,x_2)$')
ax.plot(x2, phase_plot.loc[:, pis[2]], color='orange', label='$\Psi(0.50,x_2)$')
ax.plot(x2, phase_plot.loc[:, pis[3]], color='red', label='$x_1=-p_{21}/r x_2$')
ax.plot(x2_hat, x1_hat, 'o', color='black', mfc='yellow')
ax.hlines(y=0, xmin=np.min(x2), xmax=np.max(x2), color='black', linestyles='dashed')
ax.vlines(x=0, ymin=np.min(phase_plot.values), ymax=np.max(phase_plot.values), color='black', linestyles='dashed')
ax.set_ylabel('$x_1=f(\pi)$', fontsize=10)
ax.set_xlabel('$x_2=f^\prime(\pi)$', fontsize=10)
ax.set_title(r'Phase Diagram of f($\pi$) - $\gamma=\frac{1}{\psi}$',fontsize=10)
ax.legend(loc='upper left', fontsize=8)
ax.set_xlim(xmin=np.min(x2), xmax=np.max(x2))
ax.set_ylim(ymin=np.min(phase_plot.values), ymax=np.max(phase_plot.values))
ax.grid()

fig_2.tight_layout()
fig_2.savefig('images/fig_2.png')


# -------------------------------------------------------------------------------
# SOLUTION FOR ODE OF f(pi)
# -------------------------------------------------------------------------------


# ODE of f(pi)

def ode_f(X, Pi, Gamma, Psi, Theta2, Theta1, SigmaD, P12, P21, Pi2, Gamma_pi, R, Delta):

    X1 = X[0]
    X2 = X[1]

    Q3, Q1, Q0 = parameters_q(Pi, Gamma, Psi, Theta2, Theta1, SigmaD, P12, P21, Pi2, Gamma_pi, R, Delta)

    dX1 = X2
    dX2 = X2 ** 2 + X2 * Q1 / Q3 + X1 * r / Q3 + Q0 / Q3

    return dX1, dX2


# Evaluation points

epsilon = 0.001
pi_f = 0.950
n = int((pi_f - epsilon) * 1000 + 1)
pi_range = np.linspace(epsilon, pi_f, n)


# Solution for ODE of f(pi)

x1_sol = pd.DataFrame(columns=gammas)
x2_sol = pd.DataFrame(columns=gammas)
gammas = [1, 2, 4]

for gamma in gammas:

    psi = 1 / gamma

    # Initial condition

    x2_hat = - (gamma ** 2 * r * gamma_pi * (theta2 - theta1)) / (gamma * (theta2 - theta1) + p12 + p21)
    x2_ic = (x2_hat + 0) / 3
    x1_ic = x2_ic * p12 / r
    x_ic = [x1_ic, x2_ic]

    """ Notice that to compute the initial condition x1_ic we do not compute the coefficients q3, q1, q0 and
        then use the formula for the parabola; rather we use the limit condition as if pi=0.
    """

    # Numerical solution

    x_sol = odeint(ode_f, x_ic, pi_range, args=(gamma, psi, theta2, theta1, sigmaD, p12, p21, pi2, gamma_pi, r, delta))

    x1_sol.loc[:, gamma] = x_sol[:, 0]
    x2_sol.loc[:, gamma] = x_sol[:, 1]


# Normalize to the same starting value

x1_plot = pd.DataFrame(columns=gammas)
x1_plot.loc[:, 1] = x1_sol.iloc[:, 0] - x1_sol.iloc[0, 0] + x1_sol.iloc[0, 2]
x1_plot.loc[:, 2] = x1_sol.iloc[:, 1] - x1_sol.iloc[0, 1] + x1_sol.iloc[0, 2]
x1_plot.loc[:, 4] = x1_sol.iloc[:, 2]

# Plot solution of f(pi)

fig_3, ax = plt.subplots(nrows=1, ncols=1, figsize=(5.5, 4))

ax.plot(pi_range, x1_plot, label=gammas)
ax.set_ylabel('$\pi$', fontsize=10)
ax.set_xlabel('$f(\pi)$', fontsize=10)
ax.set_title(r'Numerical solution of f($\pi$) - $\gamma=\frac{1}{\psi}$',fontsize=10)
ax.legend(loc='upper left', fontsize=8)
ax.set_xlim(xmin=np.min(pi_range), xmax=np.max(pi_range))
ax.set_ylim(ymin=np.min(x1_plot.values)-0.01, ymax=np.max(x1_plot.values))
ax.grid()

fig_3.tight_layout()
fig_3.savefig('images/fig_3.png')


# -------------------------------------------------------------------------------
# PHASE DIAGRAM FOR S(pi)
# -------------------------------------------------------------------------------


# Parameters

gamma = 4
psi = 1 / gamma

""" The remaining parameters are defined as above.
"""


# Coefficients p(pi) for ODE of S(pi)

def parameters_p(Pi, Gamma, Psi, F_pi, Theta2, Theta1, SigmaD, P12, P21, Pi2, Gamma_pi, R):

    H = (Theta2 - Theta1) / SigmaD * Pi * (1 - Pi)

    if Gamma == 1 / Psi:

        P3 = H ** 2 / 2
        P1 = Gamma * R * Gamma_pi * H ** 2 + Gamma * SigmaD * H - (P12 + P21) * (Pi2 - Pi) + F_pi * H ** 2
        P0 = Gamma * R * Gamma_pi ** 2 * H ** 2 + 2 * gamma * Gamma_pi * SigmaD * H + F_pi * Gamma_pi * H ** 2 + F_pi / R * SigmaD * H

        return P3, P1, P0

    else:

        pass


# Locus of x2'(pi)=0

y2 = np.arange(-40.00, 40.01, 0.01).round(2)
pis = [0.00, 0.05, 0.50, 1.00]

phase_plot = pd.DataFrame(columns=pis)

for pi in pis:

    if pi == 0.00:

        phase_plot.loc[:, pi] = y2 * p12 / r

    elif pi == 1.00:

        phase_plot.loc[:, pi] = - y2 * p21 / r

    else:

        f_pi = x2_sol.iloc[pi_range == pi, 2]
        p3, p1, p0 = parameters_p(pi, gamma, psi, f_pi.item(), theta2, theta1, sigmaD, p12, p21, pi2, gamma_pi, r)
        phase_plot.loc[:, pi] = - y2 * p1 / r - p0 / r


# Plot phase diagram

fig_4, ax = plt.subplots(nrows=1, ncols=1, figsize=(5.5, 4))

ax.plot(y2, phase_plot.loc[:, 0.00], color='green', label='$y_1=p_{12}/r y_2$')
ax.plot(y2, phase_plot.loc[:, 0.05], color='purple', label='$\Psi(0.05,y_2)$')
ax.plot(y2, phase_plot.loc[:, 0.50], color='orange', label='$\Psi(0.50,y_2)$')
ax.plot(y2, phase_plot.loc[:, 1.00], color='red', label='$y_1=-p_{21}/r y_2$')
ax.hlines(y=0, xmin=np.min(y2), xmax=np.max(y2), color='black', linestyles='dashed')
ax.vlines(x=0, ymin=np.min(phase_plot.values), ymax=np.max(phase_plot.values), color='black', linestyles='dashed')
ax.set_ylabel('$y_1=S(\pi)$', fontsize=10)
ax.set_xlabel('$y_2=S^\prime(\pi)$', fontsize=10)
ax.set_title(r'Phase Diagram of S($\pi$) - $\gamma=\frac{1}{\psi}$',fontsize=10)
ax.legend(loc='upper left', fontsize=10)
ax.set_xlim(xmin=np.min(y2), xmax=np.max(y2))
ax.set_ylim(ymin=np.min(phase_plot.values), ymax=np.max(phase_plot.values))
ax.grid()

fig_4.tight_layout()
fig_4.savefig('images/fig_4.png')


# -------------------------------------------------------------------------------
# SOLUTION FOR ODE OF f(pi) and S(pi)
# -------------------------------------------------------------------------------


# System of ODEs

def ode_sys(X, Pi, Gamma, Psi, Theta2, Theta1, SigmaD, P12, P21, Pi2, Gamma_pi, R, Delta):

    X1 = X[0]
    X2 = X[1]
    X3 = X[2]
    X4 = X[3]

    Q3, Q1, Q0 = parameters_q(Pi, Gamma, Psi, Theta2, Theta1, SigmaD, P12, P21, Pi2, Gamma_pi, R, Delta)
    P3, P1, P0 = parameters_p(Pi, Gamma, Psi, X2, Theta2, Theta1, SigmaD, P12, P21, Pi2, Gamma_pi, R)

    dX1 = X2
    dX2 = X2 ** 2 + X2 * Q1 / Q3 + X1 * r / Q3 + Q0 / Q3
    dX3 = X4
    dX4 = X4 * P1 / P3 + X3 * r / P3 + P0 / P3

    return dX1, dX2, dX3, dX4


# Evaluation points

epsilon = 0.001
pi_f = 0.950
n = int((pi_f - epsilon) * 1000 + 1)
pi_range = np.linspace(epsilon, pi_f, n)


# Solution of system

x1_sol = pd.DataFrame(columns=gammas)
x2_sol = pd.DataFrame(columns=gammas)
x3_sol = pd.DataFrame(columns=gammas)
x4_sol = pd.DataFrame(columns=gammas)

gammas = [1, 2, 4]

for gamma in gammas:

    psi = 1 / gamma

    # Initial condition

    x2_hat = - (gamma ** 2 * r * gamma_pi * (theta2 - theta1)) / (gamma * (theta2 - theta1) + p12 + p21)
    x2_ic = x2_hat + 0.001 # (x2_hat + 0) / 100
    x1_ic = x2_ic * p12 / r

    x4_hat = - (theta2 - theta1) * 2 * gamma * gamma_pi / (gamma * (theta2 - theta1) + p12 + p21) - x2_ic / (r * (gamma * (theta2 - theta1) + p12 + p21))
    x4_ic = (x4_hat + 0) / 100
    x3_ic = x4_ic * p12 / r

    x_ic = [x1_ic, x2_ic, x3_ic, x4_ic]

    """ Notice that to compute the initial condition x1_ic we do not compute the coefficients q3, q1, q0 and
        then use the formula for the parabola; rather we use the limit condition as if pi=0.
    """

    # Numerical solution

    x_sol = odeint(ode_sys, x_ic, pi_range, args=(gamma, psi, theta2, theta1, sigmaD, p12, p21, pi2, gamma_pi, r, delta))

    x1_sol.loc[:, gamma] = x_sol[:, 0]
    x2_sol.loc[:, gamma] = x_sol[:, 1]
    x3_sol.loc[:, gamma] = x_sol[:, 2]
    x4_sol.loc[:, gamma] = x_sol[:, 3]


# Normalize to the same starting value

x1_plot = pd.DataFrame(columns=gammas)
x1_plot.loc[:, 1] = x1_sol.iloc[:, 0] - x1_sol.iloc[0, 0] + x1_sol.iloc[0, 2]
x1_plot.loc[:, 2] = x1_sol.iloc[:, 1] - x1_sol.iloc[0, 1] + x1_sol.iloc[0, 2]
x1_plot.loc[:, 4] = x1_sol.iloc[:, 2]

x3_plot = pd.DataFrame(columns=gammas)
x3_plot.loc[:, 1] = x3_sol.iloc[:, 0] - x3_sol.iloc[0, 0] + x3_sol.iloc[0, 2]
x3_plot.loc[:, 2] = x3_sol.iloc[:, 1] - x3_sol.iloc[0, 1] + x3_sol.iloc[0, 2]
x3_plot.loc[:, 4] = x3_sol.iloc[:, 2]


# Plot solution of f(pi) and S(pi)

fig_5, ax = plt.subplots(nrows=1, ncols=2, figsize=(11, 4))

ax[0].plot(pi_range, x1_plot, label=gammas)
ax[0].set_ylabel('$\pi$', fontsize=10)
ax[0].set_xlabel('$f(\pi)$', fontsize=10)
ax[0].set_title(r'Numerical solution of f($\pi$) - $\gamma=\frac{1}{\psi}$',fontsize=10)
ax[0].legend(loc='upper left', fontsize=10)
ax[0].set_xlim(xmin=np.min(pi_range), xmax=np.max(pi_range))
ax[0].set_ylim(ymin=np.min(x1_plot.values)-0.01, ymax=np.max(x1_plot.values))
ax[0].grid()

ax[1].plot(pi_range, x3_plot, label=gammas)
ax[1].set_ylabel('$\pi$', fontsize=10)
ax[1].set_xlabel('$S(\pi)$', fontsize=10)
ax[1].set_title(r'Numerical solution of S($\pi$) - $\gamma=\frac{1}{\psi}$',fontsize=10)
ax[1].legend(loc='upper left', fontsize=10)
ax[1].set_xlim(xmin=np.min(pi_range), xmax=np.max(pi_range))
ax[1].set_ylim(ymin=np.min(x3_plot.values)-0.01, ymax=np.max(x3_plot.values))
ax[1].grid()

fig_5.tight_layout()
fig_5.savefig('images/fig_5.png')


# -------------------------------------------------------------------------------
# FINITE DIFFERENCE METHOD
# -------------------------------------------------------------------------------




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
ax[0].set_title('Value Function For different levels of ARA', fontsize=10)
ax[0].legend(loc='upper right', fontsize=10)
ax[0].set_xlim(xmin=np.min(rates), xmax=np.max(rates))

ax[1].plot(rates, Jt_rate_psi, label=psis)
ax[1].set_ylabel('Value function ($J_t$)', fontsize=10)
ax[1].set_xlabel('Interest rate ($r$)', fontsize=10)
ax[1].set_title('Value Function as a Function of EIS', fontsize=10)
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
ax[0].set_title('Value Function For different levels of ARA', fontsize=10)
ax[0].legend(loc='upper right', fontsize=10)
ax[0].set_xlim(xmin=np.min(rates), xmax=np.max(rates))

ax[1].plot(rates, Jt_rate_psi, label=psis)
ax[1].set_ylabel('Value function ($J_t$)', fontsize=10)
ax[1].set_xlabel('Interest rate ($r$)', fontsize=10)
ax[1].set_title('Value Function as a Function of EIS', fontsize=10)
ax[1].legend(loc='upper right', fontsize=10)
ax[1].set_xlim(xmin=np.min(rates), xmax=np.max(rates))

fig_2.tight_layout()
fig_2.savefig('images/fig_2.png')
