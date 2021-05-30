""" VERONESI 2000
    -------------
    Replication of Veronesi (2000).
"""


# -------------------------------------------------------------------------------
# IMPORT LIBRARIES
# -------------------------------------------------------------------------------


# Import libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from stochastic_processes import *


# -------------------------------------------------------------------------------
# INVESTORS
# -------------------------------------------------------------------------------


# Function: utility function

def utility(cons_, gamma_, delta_, t_):

    if gamma_ != 1:

        util_ = np.exp(- delta_ * t_) * cons_ ** (1 - gamma_) / (1 - gamma_)

    else:

        util_ = np.exp(- delta_ * t_) * np.log(cons_)

    return util_

# Simulation

cons = np.arange(0.5, 5, 0.005)
gammas = [0.0, 0.5, 1.0, 1.5, 2.0]

util = pd.DataFrame(columns=gammas)

for g in gammas:

    util.loc[:, g] = utility(cons, g, 0, 0)

# Plot utility function

fig_1, ax = plt.subplots(nrows=1, ncols=1, figsize=(5.5, 4))

ax.plot(cons, util.iloc[:,0], color='r')
ax.plot(cons, util.iloc[:,1:], color='b')
ax.hlines(y=0, xmin=np.min(cons), xmax=np.max(cons), color='black', linestyles='dashed')
plt.text(4.4, 4.9, '$\gamma$=0.0')
plt.text(4.4, 3.9, '$\gamma$=0.5')
plt.text(4.4, 1.1, '$\gamma$=1.0')
plt.text(4.4, -0.7, '$\gamma$=1.5')
plt.text(4.4, -1.4, '$\gamma$=2.0')
ax.set_ylabel('U(C)', fontsize=10)
ax.set_xlabel('C', fontsize=10)
ax.set_title('U(C) for varying $\gamma$',fontsize=10)
ax.set_xlim(xmin=np.min(cons), xmax=np.max(cons))

fig_1.savefig('images/fig_1.png')


# -------------------------------------------------------------------------------
# INVESTMENT OPPORTUNITY SET
# -------------------------------------------------------------------------------


# Dimensions

n = 23
dt = 1
T = 1000
periods = int(T/dt)

# Probability of a change

p = 0.50

# Drift

thetas = np.linspace(-0.0045, 0.0065, 23)
theta_ell = thetas[14]

# Unconditional distribution

freq = [1, 2, 4, 7, 11, 17, 23, 25, 27, 29, 30, 33, 30, 29, 27, 25, 23, 17, 11, 7, 4, 2, 1]
f1 = (freq / np.sum(freq))
f2 = np.full(n, 1) / n

# Prior probability distribution

pis = np.full(n, 1) / n

# Precision

sigma_D = 0.015
sigma_e = 0.1
h_D = 1 / sigma_D
h_e = 1 / sigma_e

# Plot unconditional and prior distributions

plt.bar(thetas,height=pis, width=0.0005, edgecolor='black', color='lightcyan', label='Prior')
plt.bar(thetas, height=f1, width=0.0005, edgecolor='black', color='blue', alpha = 0.6, label='Unconditional')
plt.vlines(theta_ell, ymin=0, ymax=max(f1), color='black', linestyles='dashed')
plt.text(theta_ell+0.0001, 0.08, r'$\theta_\ell$')
plt.xlabel(r'$\theta$', fontsize=10)
plt.ylabel('Probability', fontsize=10)
plt.legend(loc='upper left', fontsize=10)
plt.title('Probability distribution', fontsize=10)

# Risky asset process

drift = np.random.choice(a=thetas, size=1, p=f1)

for t in range(periods-1):

    if np.random.uniform(low=0.0, high=1.0) < p * dt:

        drift = np.append(drift, np.random.choice(a=thetas, size=1, p=f1))

    else:

        drift = np.append(drift, drift[-1])

divid = GeneralizedBrownianMotion(x0=1, mu=drift, sigma=sigma_D, dt=dt, T=T, change=False, seed=False)
dDD = divid.num_simuls

np.ndim(drift) != 0 and np.size(drift, 0) == round(T/dt):



fig3, ax = plt.subplots(nrows=1, ncols=1)

ax.plot(ip_sims, linewidth=0.5)
ax.hlines(y=100, xmin=0, xmax=ip.num_simuls, linewidth=0.5, color='black')
ax.set_title('Ito Process - $\mu$={:.1f}%, $\sigma$={:.1f}%'.format(ip.mu * ip.dt * 100, ip.sigma**2 * ip.dt * 100))
ax.set_xlabel('t')
ax.set_ylabel('$x_t$')
ax.set_xlim(left=0, right=ip.num_simuls)
fig3.tight_layout()

# -------------------------------------------------------------------------------
# EXPECTED DRIFT RATE
# -------------------------------------------------------------------------------


# Function: expected drift rate

def m_theta(pis_):

    m_theta_ = np.sum(pis_ * thetas)

    return m_theta_


# -------------------------------------------------------------------------------
# EVOLUTION OF INVESTORS BELIEFS
# -------------------------------------------------------------------------------


def dpi_1():

    dBD_tilde = h_D * ()
    dBe_tilde
    dpi =





# Risk aversion

gammas_g1 = [1.0, 1.5, 2.0, 2.5]
gammas_s1 = [1.0, 0.5, 0.15, 0.0]
gammas = np.arange(0, 5, 0.0005)

# Discount rate

delta = 0.0033

# Precision

sigma_e = float('inf')
sigma_theta = 0.0011
sigmas_theta = np.arange(0, 0.0015, 0.0005)

# -------------------------------------------------------------------------------
# FUNCTION C(THETA)
# -------------------------------------------------------------------------------


# Function for the constant

def kappa(gamma_f):

    k = 0

    for i in range(n):

        k += f[i] / (delta + p + (gamma_f-1) * thetas[i] + 0.5 * gamma_f * (1-gamma_f) * sigma_D ** 2)

    return k

# Function C(thetas)

def C_theta(gamma_f):

    c = 1 / ((delta + p + (gamma_f-1) * thetas + 0.5 * gamma_f * (1-gamma_f) * sigma_D ** 2) * (1 - p * kappa(gamma_f)))

    return c

# Simulations

C_theta_g1 = pd.DataFrame(columns=gammas_g1)
C_theta_s1 = pd.DataFrame(columns=gammas_s1)

for g_g1, g_s1 in zip(gammas_g1, gammas_s1):

    C_theta_g1.loc[:, g_g1] = C_theta(g_g1)
    C_theta_s1.loc[:, g_s1] = C_theta(g_s1)

# Plot figure

fig_1, ax = plt.subplots(nrows=1, ncols=2, figsize=(11, 4))

ax[0].plot(thetas, C_theta_g1, label=['$\gamma$=1.0', '$\gamma$=1.5', '$\gamma$=2.0', '$\gamma$=2.5'])
ax[0].legend(fontsize=8, loc='upper right')
ax[0].set_ylabel('C($\Theta$)', fontsize=10)
ax[0].set_xlabel('$\Theta$', fontsize=10)
ax[0].set_title('C($\Theta$) for $\gamma > 1$',fontsize=10)
ax[0].set_xlim(xmin=np.min(thetas), xmax=np.max(thetas))

ax[1].plot(thetas, C_theta_s1, label=['$\gamma$=1.0', '$\gamma$=0.5', '$\gamma$=0.15', '$\gamma$=0.0'])
ax[1].legend(fontsize=8, loc='upper left')
ax[1].set_ylabel('C($\Theta$)', fontsize=10)
ax[1].set_xlabel('$\Theta$', fontsize=10)
ax[1].set_title('C($\Theta$) for $\gamma < 1$',fontsize=10)
ax[1].set_xlim(xmin=np.min(thetas), xmax=np.max(thetas))

fig_1.tight_layout()
fig_1.savefig('images/fig_1.png')




# Function v_theta

def v_theta():




# Function mu_R

def mu_r():

    mu = gamma * (sigma_D ** 2 + V_theta)

    return mu




# Posterior distribution

dlog_pi = np.zeros((T, n))
log_pi = np.zeros((T, n))

dpi = np.zeros((T, n))
pi = np.zeros((T, n))
pi[0, :] = np.full(n, 1/n).reshape(1, -1)





k = h_D ** 2 + h_e ** 2

# Stochastic differential equation (6)

for t in range(int(T/dt)):

    dB_D = np.random.normal(0,1)
    dB_e = np.random.normal(0,1)
    m_theta = np.sum(pi[t,:] * thetas)
    print(min(pi[t,:]), max(pi[t,:]), sum(pi[t,:]))

    for i in range(n):

        dlog_pi[t+1, i] = (p * (f[i] - pi[t, i]) / pi[t, i] + k * (thetas[i] - m_theta) * (theta_ell - m_theta)) * dt \
                          + (thetas[i] - m_theta) * (h_D * dB_D + h_e * dB_e) - 0.5 * (thetas[i] - m_theta) ** 2 * (h_D ** 2 + h_e ** 2) * dt

        log_pi[t+1, i] = log_pi[t, i] + dlog_pi[t+1, i]

    pi[t + 1, :] = np.exp(log_pi[t+1, :])
    pi[t + 1, :] = pi[t + 1, :] / np.sum(pi[t + 1, :])

    print(min(dpi[t + 1, :]), max(dpi[t + 1, :]), sum(dpi[t + 1, :]))
    print(min(pi[t+1,:]), max(pi[t+1,:]), sum(pi[t+1,:]))



for t in range(int(T/dt)):

    dB_D = np.random.normal(0,1)
    dB_e = np.random.normal(0,1)
    m_theta = np.sum(pi[t,:] * thetas)
    print(min(pi[t,:]), max(pi[t,:]), sum(pi[t,:]))

    for i in range(n):

        dpi[t+1,i] = (p * (f[i] - pi[t,i]) + k * pi[t,i] * (thetas[i] - m_theta) * (theta_ell - m_theta)) * dt \
                     + pi[t,i] * (thetas[i] - m_theta) * (h_D * dB_D + h_e * dB_e)
        log_pi
        pi[t+1,i] = pi[t,i] + dpi[t+1,i]

    print(min(dpi[t + 1, :]), max(dpi[t + 1, :]), sum(dpi[t + 1, :]))
    print(min(pi[t+1,:]), max(pi[t+1,:]), sum(pi[t+1,:]))




# Parameters

n = 100
f = np.ones(n) / n # (n * (n+1) / 2) * np.arange(1,n+1)

gamma = np.arange(1, 3, 0.5)
p = 0.01

thetas = np.linspace(-0.0045, 0.006, n)



