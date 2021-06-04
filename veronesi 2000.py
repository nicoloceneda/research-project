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

fig_1.tight_layout()
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

thetas = np.linspace(-0.0045, 0.0065, n)

# Unconditional distribution

freq = [1, 2, 4, 7, 11, 17, 23, 25, 27, 29, 30, 33, 30, 29, 27, 25, 23, 17, 11, 7, 4, 2, 1]
f1 = np.reshape(freq / np.sum(freq), newshape=(1,-1))
f2 = np.full(shape=(1, n), fill_value=1/n)

# Prior probability distribution

pis = np.full((1, n), 1) / n

# Precision

sigma_D = 0.10
sigma_e = 0.15
h_D = 1 / sigma_D
h_e = 1 / sigma_e
k = h_D ** 2 + h_e ** 2

# Plot unconditional and prior distributions

fig_2, ax = plt.subplots(nrows=1, ncols=1, figsize=(5.5, 4))

plt.bar(thetas, height=pis.reshape(-1), width=0.0005, edgecolor='black', color='lightcyan', label='Prior')
plt.bar(thetas, height=f1.reshape(-1), width=0.0005, edgecolor='black', color='blue', alpha = 0.6, label='Unconditional')
plt.xlabel(r'$\theta$', fontsize=10)
plt.ylabel('Probability', fontsize=10)
plt.legend(loc='upper left', fontsize=10)
plt.title('Probability distribution', fontsize=10)

fig_2.tight_layout()
fig_2.savefig('images/fig_2.png')

# Stochastic drift

def drift_sim(thetas_, f_, p_):

    drift_ = np.random.choice(a=thetas_, size=1, p=f_.reshape(-1))

    for t in range(periods-1):

        if np.random.uniform(low=0.0, high=1.0) < p_ * dt:

            drift_ = np.append(drift_, np.random.choice(a=thetas_, size=1, p=f_.reshape(-1)))

        else:

            drift_ = np.append(drift_, drift_[-1])

    return drift_

drift = drift_sim(thetas, f1, p)

# Risky asset

divid = ItoProcess(x0=1, mu=drift, sigma=sigma_D, dt=dt, T=T, change=True, seed=987654321)
D_sim, dD_sim = divid.simulate()

dDD_sim = dD_sim / D_sim
dDD_sim = dDD_sim[:-1]

# Signal

signal = GeneralizedBrownianMotion(x0=1, mu=drift, sigma=sigma_e, dt=dt, T=T, change=True, seed=123456789)
e_sim, de_sim = signal.simulate()
de_sim = de_sim[:-1]

# Plot simulated dDD

fig_3, ax = plt.subplots(nrows=1, ncols=2, figsize=(11, 4))

ax[0].plot(D_sim, color='b', label='Dividend')
ax[0].plot(e_sim, color='r', label='Signal')
ax[0].hlines(y=0, xmin=0, xmax=periods, color='black', linestyles='dashed')
ax[0].set_ylabel(r'$D$' + ' and ' + r'$e$', fontsize=10)
ax[0].set_xlabel('t', fontsize=10)
ax[0].set_title(r'$D(t+1)=D(t) + dD(t)$' + ' and ' + r'$e(t+1)=e(t) + de(t)$', fontsize=10)
ax[0].legend(loc='upper left', fontsize=10)
ax[0].set_xlim(xmin=0, xmax=periods)

ax[1].plot(dDD_sim, color='b', label='Dividend', zorder=5)
ax[1].plot(de_sim, color='r', label='Signal', zorder=0)
ax[1].hlines(y=0, xmin=0, xmax=periods, color='black', linestyles='dashed', zorder=10)
ax[1].set_ylabel(r'$\frac{dD}{D}$' + ' and ' + r'$de$', fontsize=10)
ax[1].set_xlabel('t', fontsize=10)
ax[1].set_title(r'$\frac{dD}{D}=\theta dt + \sigma_D dB_D$' + ' and ' + r'$de=\theta dt + \sigma_e dB_e$', fontsize=10)
ax[1].legend(loc='upper left', fontsize=10)
ax[1].set_xlim(xmin=0, xmax=periods)

fig_3.tight_layout()
fig_3.savefig('images/fig_3.png')


# -------------------------------------------------------------------------------
# EXPECTED DRIFT RATE
# -------------------------------------------------------------------------------


# Function: expected drift rate

def mtheta(pis_, thetas_):

    m_theta_ = np.sum(pis_ * thetas_)

    return m_theta_


# -------------------------------------------------------------------------------
# EVOLUTION OF INVESTORS BELIEFS
# -------------------------------------------------------------------------------


# Simulated the evolution of beliefs

dpis_evo_1 = np.zeros(shape=(periods, n))
pis_evo_1 = np.zeros(shape=(periods, n))
pis_evo_1[0,:] = pis

for t in range(periods-1):

    m_theta_1 = mtheta(pis_evo_1[t, :], thetas)
    dBD_tilde = h_D * (dDD_sim[t] - m_theta_1 * dt)
    dBe_tilde = h_e * (de_sim[t] - m_theta_1 * dt)
    dpis_evo_1[t,:] = p * (f1 - pis_evo_1[t,:]) * dt + pis_evo_1[t,:] * (thetas - m_theta_1) * (h_D * dBD_tilde + h_e * dBe_tilde)
    pis_evo_1[t+1,:] = pis_evo_1[t,:] + dpis_evo_1[t,:]

# Plot the simulated evolution of beliefs

fig_4 = plt.figure(constrained_layout=True, figsize=(8, 8))

gs = plt.GridSpec(12, 2, figure=fig_4)

col = 0
row = 0

for i in range(23):

    row = i

    if row > 11:
        col = 1
        row -= 12

    ax = fig_4.add_subplot(gs[row, col])
    ax.plot(pis_evo_1[:, i], color='b', linewidth=0.6, zorder=5)
    ax.hlines(y=f1[0, i], xmin=0, xmax=periods, color='r', linestyles='dashed', linewidth=0.6, zorder=10)
    ax.tick_params(axis='both', labelsize='6')
    plt.text(995, (np.max(pis_evo_1[:, i]) + np.min(pis_evo_1[:, i]))/2, '$\pi {}$'.format(i + 1), fontsize=6)

fig_4.suptitle("Evolution of Investors' Beliefs", fontsize=10)

fig_4.savefig('images/fig_4.png')


# -------------------------------------------------------------------------------
# IMPACT OF SIGNAL PRECISION ON DISPERSION OF INVESTORS BELIEFS
# -------------------------------------------------------------------------------

# Known drift

delta_theta_ell = int(periods/10)
theta_ell = np.full(delta_theta_ell, thetas[10])

for i in range(5):

        theta_ell = np.append(theta_ell, np.full(delta_theta_ell, thetas[11 + i]))

for i in range(4):
    theta_ell = np.append(theta_ell, np.full(delta_theta_ell, thetas[3 + i * 3]))

# Plot the evolution of the known drift

fig_5, ax = plt.subplots(nrows=1, ncols=1, figsize=(5.5, 4))

ax.plot(theta_ell, color='b')
ax.hlines(y=0, xmin=0, xmax=periods, color='black', linestyles='dashed')

for i in range(10):

    loc = 100 * i
    plt.text(loc+15, theta_ell[loc]+0.0001, '{:.4f}'.format(theta_ell[loc]), fontsize=6)

ax.set_ylabel(r'$\theta_\ell$', fontsize=10)
ax.set_xlabel('t', fontsize=10)
ax.set_title('Evolution of ' + r'$\theta_\ell$',fontsize=10)
ax.set_xlim(xmin=0, xmax=periods)

fig_5.tight_layout()
fig_5.savefig('images/fig_5.png')

# Re-simulate the risky asset

divid_2 = ItoProcess(x0=1, mu=theta_ell, sigma=sigma_D, dt=dt, T=T, change=True, seed=987654321)
D_sim_2, dD_sim_2 = divid_2.simulate()
dDD_sim_2 = dD_sim_2 / D_sim_2
dDD_sim_2 = dDD_sim_2[:-1]

# Re-simulate the signal

signal_2 = GeneralizedBrownianMotion(x0=1, mu=theta_ell, sigma=sigma_e, dt=dt, T=T, change=True, seed=123456789)
e_sim_2, de_sim_2 = signal_2.simulate()
de_sim_2 = de_sim_2[:-1]

# Simulated the evolution of beliefs

dpis_evo_2 = np.zeros(shape=(periods, n))
pis_evo_2 = np.zeros(shape=(periods, n))
pis_evo_2[0, :] = pis

for t in range(periods-1):

    m_theta_2 = mtheta(pis_evo_2[t, :], thetas)
    dBD = h_D * (dDD_sim_2[t] - theta_ell[t] * dt)
    dBe = h_e * (de_sim_2[t] - theta_ell[t] * dt)
    dpis_evo_2[t,:] = (p * (f1 - pis_evo_2[t,:]) + k * pis_evo_2[t,:] * (thetas - m_theta_2) * (theta_ell[t] - m_theta_2)) * dt + \
                       pis_evo_2[t,:] * (thetas - m_theta_2) * (h_D * dBD + h_e * dBe)
    pis_evo_2[t+1,:] = pis_evo_2[t,:] + dpis_evo_2[t,:]

# Plot the simulated evolution of beliefs

fig_6 = plt.figure(constrained_layout=True, figsize=(8, 8))

gs = plt.GridSpec(12, 2, figure=fig_6)

col = 0
row = 0

for i in range(23):

    row = i

    if row > 11:
        col = 1
        row -= 12

    ax = fig_6.add_subplot(gs[row, col])
    ax.plot(pis_evo_2[:, i], color='b', linewidth=0.6, zorder=5)
    ax.hlines(y=f1[0, i], xmin=0, xmax=periods, color='r', linestyles='dashed', linewidth=0.6, zorder=10)
    ax.tick_params(axis='both', labelsize='6')
    plt.text(995, (np.max(pis_evo_2[:, i]) + np.min(pis_evo_2[:, i]))/2, '$\pi {}$'.format(i + 1), fontsize=6)

fig_6.suptitle("Evolution of Investors' Beliefs - True Drift Rate Known", fontsize=10)

fig_6.savefig('images/fig_6.png')

# -------------------------------------------------------------------------------
# FUNCTION C(THETA)
# -------------------------------------------------------------------------------


# Risk aversion

gammas_g1 = [1.0, 1.5, 2.0, 2.5]
gammas_s1 = [1.0, 0.5, 0.15, 0.0]

# Parameters

delta = 0.0033
p = 0.0167
sigma_D = 0.015

# Function: constant K

def kappa(f_, gamma_, thetas_):

    k = 0

    for i in range(n):

        k += f_[0,i] / (p + delta + (gamma_ - 1) * thetas_[i] - 0.5 * gamma_ * (gamma_ - 1) * sigma_D ** 2)

    return k

# Function: C(theta)

def Ctheta(f_, gamma_, thetas_):

    c = 1 / ((p + delta + (gamma_ - 1) * thetas_ - 0.5 * gamma_ * (gamma_ - 1) * sigma_D ** 2) * (1 - p * kappa(f_, gamma_, thetas_)))

    return c

# Calculate values of C(theta)

C_theta_g1 = pd.DataFrame(columns=gammas_g1)
C_theta_s1 = pd.DataFrame(columns=gammas_s1)

for g_g1, g_s1 in zip(gammas_g1, gammas_s1):

    C_theta_g1.loc[:, g_g1] = Ctheta(f1, g_g1, thetas)
    C_theta_s1.loc[:, g_s1] = Ctheta(f1, g_s1, thetas)

# Plot figure

fig_t1, ax = plt.subplots(nrows=1, ncols=2, figsize=(11, 4))

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

fig_t1.tight_layout()
fig_t1.savefig('images/fig_t1.png')


# -------------------------------------------------------------------------------
# MU_R (1)
# -------------------------------------------------------------------------------


# Generate thetas with varying standard deviation

desired_sigma = np.arange(0.0001, 0.0012, 0.0001)
n = 50
n_sigmas = desired_sigma.shape[0]
thetas_2 = np.zeros((n, desired_sigma.shape[0]))
pis_2 = np.ones((1, n)) / n
f3 = np.ones((1, n)) / n

for i in range(n_sigmas):

    b = 0.0002 + 0.0002 * i
    a = b - np.sqrt(12) * desired_sigma[i]
    thetas_2[:,i] = [a + (b - a)/n * i for i in range(n-1)] + [b]

thetas_2_std = np.std(thetas_2, axis=0) # TODO: wrong way to compute std

# Volatility of signal

sigma_e = np.inf
h_e = 1 / sigma_e

# Risk aversion

gammas_g2 = [2.0, 3.5, 5.0]
gammas_2 = np.arange(0,5,0.005)

# Function: vtheta

def vtheta(pis_, thetas_, f_, gamma_):

    pis_star = pis_ * Ctheta(f_, gamma_, thetas_) / np.sum(pis_ * Ctheta(f_, gamma_, thetas_))
    vtheta_ = mtheta(pis_star, thetas_) - mtheta(pis_, thetas_)

    return vtheta_

# Function: expected excess return

def mur(pis_, thetas_, f_, gamma_):

    mur_r = gamma_ * (sigma_D ** 2 + vtheta(pis_, thetas_, f_, gamma_))

    return mur_r

# Simulate expected excess return

mur_g1 = pd.DataFrame(columns=gammas_g2)

for g_g1 in gammas_g2:

    for t_t2 in range(n_sigmas):

        mur_g1.loc[t_t2, g_g1] = mur(pis_2, thetas_2[:,t_t2], f3, g_g1)


mur_2 = np.array([mur(pis_2, thetas_2[:,-1], f3, g) for g in gammas_2])

# Plot simulated expected excess return

fig_t2, ax = plt.subplots(nrows=1, ncols=2, figsize=(11, 4))

ax[0].plot(thetas_2_std, mur_g1, label=['$\gamma$=2', '$\gamma$=3.5', '$\gamma$=5'])
ax[0].legend(fontsize=8, loc='upper right')
ax[0].set_ylabel('$\mu_R$', fontsize=10)
ax[0].set_xlabel(r'$\sigma_\theta$', fontsize=10)
ax[0].hlines(y=0, xmin=np.min(thetas_2_std), xmax=np.max(thetas_2_std), color='black', linestyles='dashed')
ax[0].set_title('$\mu_R$ for $\gamma > 1$',fontsize=10)
ax[0].set_xlim(xmin=np.min(thetas_2_std), xmax=np.max(thetas_2_std))

ax[1].plot(gammas_2, mur_2)
ax[1].set_ylabel('$\mu_R$', fontsize=10)
ax[1].set_xlabel('$\gamma$', fontsize=10)
ax[1].hlines(y=0, xmin=np.min(gammas_2), xmax=np.max(gammas_2), color='black', linestyles='dashed')
ax[1].set_title('$\mu_R$ for varying $\gamma$',fontsize=10)
ax[1].set_xlim(xmin=np.min(gammas_2), xmax=np.max(gammas_2))

fig_t2.tight_layout()
fig_t2.savefig('images/fig_t2.png')


# -------------------------------------------------------------------------------
# MU_R (2)
# -------------------------------------------------------------------------------


# Dimensions

n = 29
dt = 1
T = 1000
periods = int(T/dt)

# Probability of a change

p = 0.0167

# Drift

thetas = np.linspace(-0.0045, 0.0065, n)

# Unconditional distribution

freq = [1, 2, 4, 7, 11, 17, 23, 25, 27, 29, 30, 33, 37, 42, 49, 42, 37, 33, 30, 29, 27, 25, 23, 17, 11, 7, 4, 2, 1]
f1 = np.reshape(freq / np.sum(freq), newshape=(1,-1))

# Prior probability distribution

pis = np.full((1, n), 1) / n

# Risk aversion

gammas = [1, 3, 4, 5]

# Precision

sigma_D = 0.025
sigma_e = np.inf
h_D = 1 / sigma_D
h_e = 1 / sigma_e
k = h_D ** 2 + h_e ** 2

# Drift

drift = drift_sim(thetas, f1, p)

# Risky asset

divid = ItoProcess(x0=1, mu=drift, sigma=sigma_D, dt=dt, T=T, change=True, seed=1234)
D_sim, dD_sim = divid.simulate()

dDD_sim = dD_sim / D_sim
dDD_sim = dDD_sim[:-1]

# Theta standard deviation

def thetastd(pis_, thetas_):

    thetas_std_ = np.sqrt(np.sum(pis_ * (thetas_ - mtheta(pis_, thetas_)) ** 2))

    return thetas_std_

# Simulated the evolution of beliefs

dpis_evo = np.zeros(shape=(periods, n))
pis_evo = np.zeros(shape=(periods, n))
pis_evo[0,:] = pis
theta_std = np.zeros(shape=periods)

for t in range(periods-1):

    m_theta = mtheta(pis_evo[t, :], thetas)
    dBD_tilde = h_D * (dDD_sim[t] - m_theta * dt)
    dBe_tilde = h_e
    dpis_evo[t,:] = p * (f1 - pis_evo[t,:]) * dt + pis_evo[t,:] * (thetas - m_theta) * (h_D * dBD_tilde + h_e * dBe_tilde)
    pis_evo[t+1,:] = pis_evo[t,:] + dpis_evo[t,:]
    theta_std[t] = thetastd(pis_evo[t,:], thetas)

# Expected excess return

mu_r = pd.DataFrame(columns=gammas)

for g in gammas:

    for t in range(periods):

        mu_r.loc[t, g] = mur(pis_evo[t, :], thetas, f1, g)

# Plot simulated expected excess return

lag1 = 50
lag2 = 10

fig_t3, ax = plt.subplots(nrows=1, ncols=2, figsize=(11, 4))

ax[0].plot(range(periods - lag1 - lag2), mu_r.iloc[lag1:-lag2,:], label=['$\gamma$=1', '$\gamma$=3', '$\gamma$=4', '$\gamma$=5'])
ax[0].legend(fontsize=8, loc='center right')
ax[0].set_ylabel('$\mu_R$', fontsize=10)
ax[0].set_xlabel('t', fontsize=10)
ax[0].set_title('$\mu_R$ over Time',fontsize=10)
ax[0].set_xlim(xmin=0, xmax=periods-lag1-lag2)

ax[1].plot(range(periods - lag1 - lag2), theta_std[lag1:-lag2])
ax[1].set_ylabel(r'$\sigma_\theta$', fontsize=10)
ax[1].set_xlabel('t', fontsize=10)
ax[1].set_title(r'$\sigma_\theta$ over time',fontsize=10)
ax[1].set_xlim(xmin=0, xmax=periods-lag1-lag2)

fig_t3.tight_layout()
fig_t3.savefig('images/fig_t3.png')

# -------------------------------------------------------------------------------
# GENERAL
# -------------------------------------------------------------------------------


# Show plots

plt.show()
