import numpy as np
import matplotlib.pyplot as plt

# Business cycle parameters

beta = 0.9896
tau = 2
theta = 0.357
alpha = 0.4
delta = 0.0196

# Regime switching parameters

a_H = 0.0035
a_L = 0.0035
p = 0.975
q = 0.975
zeta = 0.719

sigma = a_H + a_L
lambda_0 = 1 - q
lambda_1 = p + q - 1
xi_0 = (a_H + a_L) * lambda_0 + lambda_1 * a_L - a_L
xi_1 = lambda_1
sigma2_e = p * (1 - p) * lambda_0 / (1 - lambda_1) + q * (1 - q) * (1 - lambda_0 / (1 - lambda_1)) + (zeta ** 2) * (1 + lambda_1 ** 2)

# Steady state

ze_ss = 0
z_ss = 0
k_ss = 23.14
c_ss = 1.288
l_ss = 0.311
y_ss = 1.742

# Simulations

num_sim = 250
len_sim = 200
lambda_HP = 1600

z = np.zeros(len_sim)
k = np.zeros(len_sim)
y = np.zeros(len_sim)


for t in range(1,len_sim):

    z[t] = xi_0 + xi_1 * z[t-1] + sigma * epsilon[t]
    k[t] = (1 - delta) * k[t-1] + i[t]
    y[t] = np.exp(z[t]) * (k[t] ** alpha) * (l[t] ** (1 - alpha))


# Precision
sigma_D = 0.2
sigma_e = 0.1
h_D = 1 / sigma_D
h_e = 1 / sigma_e
k = h_D ** 2 + h_e ** 2

# Drift
theta = np.arange(-0.45,0.50,0.05)

# Unconditional distribution
freq = [1, 2, 4, 7, 11, 17, 23, 27, 29, 30,
     29, 27, 23, 17, 11, 7, 4, 2, 1]
f = freq / np.sum(freq)

# Plot unconditional distribution
plt.plot(f)
plt.bar(theta,height=f, width=0.05)

# Probability of a change
p = 0.5

# Posterior distribution
pi = np.full(19, 1/19)

# Time step
dt = 0.01

dp = np.zeros(19)

dp_store = list()
pi_store = list()

for t in range(100):

    dB_D = np.random.normal(0, 1 / h_D)
    dB_e = np.random.normal(0, 1 / h_e)
    m_theta = np.sum(pi * theta)
    theta_ell = theta[np.random.randint(19)]

    for i in range(len(theta)):

        dp[i] = (p * (f[i] - pi[i]) + k * pi[i] * (theta[i] - m_theta) * (theta_ell - m_theta)) * dt + pi[i] * (theta[i] - m_theta) * (h_D * dB_D + h_e * dB_e)
        pi[i] = pi[i]+ dp[i]

    dp_store.append(dp)
    pi_store.append(pi)

dp_store = np.array(dp_store)
pi_store = np.array(pi_store)