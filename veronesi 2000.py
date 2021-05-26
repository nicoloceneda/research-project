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


# -------------------------------------------------------------------------------
# PARAMETERS
# -------------------------------------------------------------------------------


# Drift
theta = np.arange(-0.004, 0.0065, 0.0005)

# Unconditional distribution
freq = [1, 2, 4, 7, 11, 17, 23, 25, 27, 29, 30, 29, 27, 25, 23, 17, 11, 7, 4, 2, 1]
f = (freq / np.sum(freq))

# Probability of a change
p = 0.0167

# Risk aversion
gamma_g1 = [1.0, 1.5, 2.0, 2.5]
gamma_s1 = [0.0, 0.15, 0.5, 1.0]

# Discount rate
delta = 0.01

# Precision
sigmaD = 0.015

# Dimensions
n = theta.shape[0]

# -------------------------------------------------------------------------------
# FUNCTION C(THETA)
# -------------------------------------------------------------------------------

# Function for the constant

def kappa(gamma_f):

    k = 0

    for i in range(n):

        k += f[i] / (delta + p + (gamma_f-1) * theta[i] + 0.5 * gamma_f * (1-gamma_f) * sigmaD ** 2)

    return k

# Function C(theta)

def C_theta(gamma_f):

    c = 1 / ((delta + p + (gamma_f-1) * theta + 0.5 * gamma_f * (1-gamma_f) * sigmaD ** 2) * (1 - p * kappa(gamma_f)))

    return c

# Simulations

C_theta_g1 = pd.DataFrame(columns=gamma_g1)

for g in gamma_g1:

    C_theta_g1.loc[:,g] = C_theta(g)


plt.plot(theta, C_theta_g1)






theta_ell = theta[14]

dt = 0.001
T = 10

# Posterior distribution

dlog_pi = np.zeros((T, n))
log_pi = np.zeros((T, n))

dpi = np.zeros((T, n))
pi = np.zeros((T, n))
pi[0, :] = np.full(n, 1/n).reshape(1, -1)




sigma_e = 0.1
h_D = 1 / sigma_D
h_e = 1 / sigma_e
k = h_D ** 2 + h_e ** 2

# Stochastic differential equation (6)

for t in range(int(T/dt)):

    dB_D = np.random.normal(0,1)
    dB_e = np.random.normal(0,1)
    m_theta = np.sum(pi[t,:] * theta)
    print(min(pi[t,:]), max(pi[t,:]), sum(pi[t,:]))

    for i in range(n):

        dlog_pi[t+1, i] = (p * (f[i] - pi[t, i]) / pi[t, i] + k * (theta[i] - m_theta) * (theta_ell - m_theta)) * dt \
                          + (theta[i] - m_theta) * (h_D * dB_D + h_e * dB_e) - 0.5 * (theta[i] - m_theta) ** 2 * (h_D ** 2 + h_e ** 2) * dt

        log_pi[t+1, i] = log_pi[t, i] + dlog_pi[t+1, i]

    pi[t + 1, :] = np.exp(log_pi[t+1, :])
    pi[t + 1, :] = pi[t + 1, :] / np.sum(pi[t + 1, :])

    print(min(dpi[t + 1, :]), max(dpi[t + 1, :]), sum(dpi[t + 1, :]))
    print(min(pi[t+1,:]), max(pi[t+1,:]), sum(pi[t+1,:]))



for t in range(int(T/dt)):

    dB_D = np.random.normal(0,1)
    dB_e = np.random.normal(0,1)
    m_theta = np.sum(pi[t,:] * theta)
    print(min(pi[t,:]), max(pi[t,:]), sum(pi[t,:]))

    for i in range(n):

        dpi[t+1,i] = (p * (f[i] - pi[t,i]) + k * pi[t,i] * (theta[i] - m_theta) * (theta_ell - m_theta)) * dt \
                     + pi[t,i] * (theta[i] - m_theta) * (h_D * dB_D + h_e * dB_e)
        log_pi
        pi[t+1,i] = pi[t,i] + dpi[t+1,i]

    print(min(dpi[t + 1, :]), max(dpi[t + 1, :]), sum(dpi[t + 1, :]))
    print(min(pi[t+1,:]), max(pi[t+1,:]), sum(pi[t+1,:]))

# Plot unconditional and prior distributions

plt.bar(theta,height=pi[0,:], width=0.05, edgecolor='black', color='lightcyan', label='Prior')
plt.bar(theta, height=f, width=0.0005, edgecolor='black', color='blue', alpha = 0.6, label='Unconditional')
plt.vlines(theta_ell, ymin=0, ymax=max(f), color='black', linestyles='dashed', label='True ' + r'$\theta$')
plt.xlabel(r'$\theta$', fontsize=10)
plt.ylabel('Probability', fontsize=10)
plt.legend(loc='upper left', fontsize=10)
plt.title('Probability distribution', fontsize=10)


# Parameters

n = 100
f = np.ones(n) / n # (n * (n+1) / 2) * np.arange(1,n+1)

gamma = np.arange(1, 3, 0.5)
p = 0.01

theta = np.linspace(-0.0045, 0.006, n)




for g in gamma:

    C_theta[g] = list(C(g))

C_theta = pd.DataFrame(C_theta)

C_theta.plot(kind='line')
plt.legend(loc='upper right')


