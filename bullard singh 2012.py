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

