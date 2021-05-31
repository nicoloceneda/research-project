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

thetas = np.linspace(-0.0045, 0.0065, 23)

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

drift = np.random.choice(a=thetas, size=1, p=f1.reshape(-1))

for t in range(periods-1):

    if np.random.uniform(low=0.0, high=1.0) < p * dt:

        drift = np.append(drift, np.random.choice(a=thetas, size=1, p=f1.reshape(-1)))

    else:

        drift = np.append(drift, drift[-1])

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

def mtheta(pis_):

    m_theta_ = np.sum(pis_ * thetas)

    return m_theta_


# -------------------------------------------------------------------------------
# EVOLUTION OF INVESTORS BELIEFS
# -------------------------------------------------------------------------------


# Simulated the evolution of beliefs

dpis_evo_1 = np.zeros(shape=(periods, n))
pis_evo_1 = np.zeros(shape=(periods, n))
pis_evo_1[0,:] = pis

for t in range(periods-1):

    m_theta_1 = mtheta(pis_evo_1[t, :])
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

    m_theta_2 = mtheta(pis_evo_2[t, :])
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

# Discount rate

delta = 0.0033
p = 0.0167
sigma_D = 0.015 #  # sigma_D = 0.10

# Function: constant K

def kappa(gamma_):

    k = 0

    for i in range(n):

        k += f1[0,i] / (p + delta + (gamma_ - 1) * thetas[i] - 0.5 * gamma_ * (gamma_ - 1) * sigma_D ** 2)

    return k

# Function: C(theta)

def Ctheta(gamma_):

    c = 1 / ((p + delta + (gamma_ - 1) * thetas - 0.5 * gamma_ * (gamma_ - 1) * sigma_D ** 2) * (1 - p * kappa(gamma_)))

    return c

# Calculate values of C(theta)

C_theta_g1 = pd.DataFrame(columns=gammas_g1)
C_theta_s1 = pd.DataFrame(columns=gammas_s1)

for g_g1, g_s1 in zip(gammas_g1, gammas_s1):

    C_theta_g1.loc[:, g_g1] = Ctheta(g_g1)
    C_theta_s1.loc[:, g_s1] = Ctheta(g_s1)

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

#################### DELETE ###############################


C_theta_g2 = pd.DataFrame(columns=[0.01, 0.04, 0.06, 0.08, 0.12])

for sigma_D in [0.01, 0.04, 0.06, 0.08, 0.12]:

    def kappa2(gamma_):
        k = 0
        for i in range(n):
            k += f1[0, i] / (p + delta + (gamma_ - 1) * thetas[i] - 0.5 * gamma_ * (gamma_ - 1) * sigma_D ** 2)
        return k


    def Ctheta2(gamma_):
        c = 1 / ((p + delta + (gamma_ - 1) * thetas - 0.5 * gamma_ * (gamma_ - 1) * sigma_D ** 2) * (1 - p * kappa2(gamma_)))
        return c

    C_theta_g2.loc[:, sigma_D] = Ctheta2(2.5)

# Plot figure

fig_t1, ax = plt.subplots(nrows=1, ncols=1, figsize=(11, 4))

ax.plot(thetas, C_theta_g2, label=['sigma_D=0.01', 'sigma_D=0.04', 'sigma_D=0.06', 'sigma_D=0.08', 'sigma_D=0.12'])
ax.legend(fontsize=8, loc='upper right')
ax.set_ylabel('C($\Theta$)', fontsize=10)
ax.set_xlabel('$\Theta$', fontsize=10)
ax.set_title('C($\Theta$) for $\gamma > 1$',fontsize=10)
ax.set_xlim(xmin=np.min(thetas), xmax=np.max(thetas))


# -------------------------------------------------------------------------------
# GENERAL
# -------------------------------------------------------------------------------


# Show plots

plt.show()






# fig3, ax = plt.subplots(nrows=1, ncols=1)

# ax.plot(ip_sims, linewidth=0.5)
# ax.hlines(y=100, xmin=0, xmax=ip.num_simuls, linewidth=0.5, color='black')
# ax.set_title('Ito Process - $\mu$={:.1f}%, $\sigma$={:.1f}%'.format(ip.mu * ip.dt * 100, ip.sigma**2 * ip.dt * 100))
# ax.set_xlabel('t')
# ax.set_ylabel('$x_t$')
# ax.set_xlim(left=0, right=ip.num_simuls)
# fig3.tight_layout()




# Stochastic differential equation (6)

# for t in range(int(T/dt)):

#     dB_D = np.random.normal(0,1)
#     dB_e = np.random.normal(0,1)
#     m_theta_1 = np.sum(pi[t,:] * thetas)
#     print(min(pi[t,:]), max(pi[t,:]), sum(pi[t,:]))

#     for i in range(n):

#         dlog_pi[t+1, i] = (p * (f[i] - pi[t, i]) / pi[t, i] + k * (thetas[i] - m_theta_1) * (theta_ell - m_theta_1)) * dt \
#                           + (thetas[i] - m_theta_1) * (h_D * dB_D + h_e * dB_e) - 0.5 * (thetas[i] - m_theta_1) ** 2 * (h_D ** 2 + h_e ** 2) * dt

#         log_pi[t+1, i] = log_pi[t, i] + dlog_pi[t+1, i]

#     pi[t + 1, :] = np.exp(log_pi[t+1, :])
#     pi[t + 1, :] = pi[t + 1, :] / np.sum(pi[t + 1, :])

#     print(min(dpi[t + 1, :]), max(dpi[t + 1, :]), sum(dpi[t + 1, :]))
#     print(min(pi[t+1,:]), max(pi[t+1,:]), sum(pi[t+1,:]))



# for t in range(int(T/dt)):

#     dB_D = np.random.normal(0,1)
#     dB_e = np.random.normal(0,1)
#     m_theta_1 = np.sum(pi[t,:] * thetas)
#     print(min(pi[t,:]), max(pi[t,:]), sum(pi[t,:]))

#     for i in range(n):

#         dpi[t+1,i] = (p * (f[i] - pi[t,i]) + k * pi[t,i] * (thetas[i] - m_theta_1) * (theta_ell - m_theta_1)) * dt \
#                      + pi[t,i] * (thetas[i] - m_theta_1) * (h_D * dB_D + h_e * dB_e)
#         pi[t+1,i] = pi[t,i] + dpi[t+1,i]

#     print(min(dpi[t + 1, :]), max(dpi[t + 1, :]), sum(dpi[t + 1, :]))
#     print(min(pi[t+1,:]), max(pi[t+1,:]), sum(pi[t+1,:]))





