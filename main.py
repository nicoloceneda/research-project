import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Parameters

n = 10
f = np.ones(n) / (n * (n+1) / 2) * np.arange(1,n+1)
delta = 0.02
gamma = np.arange(1, 3, 0.5)
p = 0.5
sigmaD_2 = 0.02
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
for gamma_f in gamma:

    C_theta[gamma_f] = list(C(gamma_f))

C_theta = pd.DataFrame(C_theta)
plt.plot(C_theta)
