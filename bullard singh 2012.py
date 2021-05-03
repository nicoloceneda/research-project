import numpy as np
import matplotlib.pyplot as plt


a_H = 0.15
a_L = 0.10
zeta = 0.40
eta_t = np.random.normal(loc=0, scale=1, size=100)

def Z(s_t):
    z = (a_H + a_L) * (s_t + zeta * eta_t) - a_L
    return z

plt.plot(Z(0))
plt.plot(Z(1))