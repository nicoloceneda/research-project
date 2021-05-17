""" SIMULATIONS
    -----------
    Simulation of stochastic processes.
"""


# -------------------------------------------------------------------------------
# IMPORT LIBRARIES
# -------------------------------------------------------------------------------


import numpy as np
import matplotlib.pyplot as plt
from stochastic_processes import *


# -------------------------------------------------------------------------------
# BROWNIAN MOTION
# -------------------------------------------------------------------------------


# Simulate brownian motions

bm = BrownianMotion(x0=0, dt=0.1, T=100, seed=0)

bm_sims = []

for i in range(10):

    bm_sims.append(bm.sim())

bm_sims = np.array(bm_sims).T

fig1, ax = plt.subplots(nrows=1, ncols=1)

ax.plot(bm_sims, linewidth=0.5)
ax.hlines(y=0, xmin=0, xmax=bm.num_sim, linewidth=0.5, color='black')
ax.set_title('Brownian Motion - $\mu$=0, $\sigma^2$={:.1f}'.format(bm.dt))
ax.set_xlabel('t')
ax.set_ylabel('$x_t$')
ax.set_xlim(left=0, right=bm.num_sim)
fig1.tight_layout()


# -------------------------------------------------------------------------------
# GENERALIZED BROWNIAN MOTION
# -------------------------------------------------------------------------------


# Simulate generalized brownian motions

gbm = GeneralizedBrownianMotion(x0=0, mu=0.2, sigma=1.0, dt=0.1, T=100, seed=0)

gbm_sims = []

for i in range(10):

    gbm_sims.append(gbm.sim())

gbm_sims = np.array(gbm_sims).T

fig2, ax = plt.subplots(nrows=1, ncols=1)

ax.plot(gbm_sims, linewidth=0.5)
ax.hlines(y=0, xmin=0, xmax=gbm.num_sim, linewidth=0.5, color='black')
ax.set_title('Generalized Brownian Motion - $\mu$={:.1f}, $\sigma^2$={:.1f}'.format(gbm.mu * gbm.dt, gbm.sigma**2 * gbm.dt))
ax.set_xlabel('t')
ax.set_ylabel('$x_t$')
ax.set_xlim(left=0, right=gbm.num_sim)
fig2.tight_layout()

# Display figures

plt.show()