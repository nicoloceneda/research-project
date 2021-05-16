# Import libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Drift

theta = np.arange(-0.004, 0.0065, 0.0005)
theta_ell = theta[14]

# Unconditional distribution

freq = [1, 2, 4, 7, 11, 17, 23, 25, 27, 29, 30, 29, 27, 25, 23, 17, 11, 7, 4, 2, 1]
f = (freq / np.sum(freq))

# Dimensions

n = theta.shape[0]
dt = 0.001
T = 10

# Posterior distribution

dpi = np.zeros((T, n))
pi = np.zeros((T, n))
pi[0, :] = np.full(n, 1/n).reshape(1, -1)

# Probability of a change

p = 0.5

# Precision
sigma_D = 0.2
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
delta = 0.02
gamma = np.arange(1, 3, 0.5)
p = 0.01
sigmaD_2 = 0.015
theta = np.linspace(-0.0045, 0.006, n)

def K(gamma_f):
    k = 0
    for i in range(n):
        k += f[i] / (delta + p + (gamma_f-1) * theta[i] + 0.5 * gamma_f * (1-gamma_f) * sigmaD_2)
    return k

def C(gamma_f):
    c = 1 / ((delta + p + (gamma_f-1) * theta + 0.5 * gamma_f * (1-gamma_f) * sigmaD_2) * (1 - p * K(gamma_f)))
    return c

C_theta = {}
for g in gamma:

    C_theta[g] = list(C(g))

C_theta = pd.DataFrame(C_theta)

C_theta.plot(kind='line')
plt.legend(loc='upper right')


