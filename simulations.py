""" SIMULATIONS
    -----------
    Simulation of stochastic processes.
"""


# -------------------------------------------------------------------------------
# IMPORT LIBRARIES
# -------------------------------------------------------------------------------


from stochastic_processes import *
import matplotlib.pyplot as plt


# -------------------------------------------------------------------------------
# BROWNIAN MOTION
# -------------------------------------------------------------------------------


# Simulate brownian motions

bm = BrownianMotion(x0=0, dt=0.1, T=100, seed=0)

bm_sims = []

for i in range(10):

    bm_sims.append(bm.simulate())

bm_sims = np.array(bm_sims).T

fig1, ax = plt.subplots(nrows=1, ncols=1)

ax.plot(bm_sims, linewidth=0.5)
ax.hlines(y=0, xmin=0, xmax=bm.num_simuls, linewidth=0.5, color='black')
ax.set_title('Brownian Motion - $\mu$=0%, $\sigma^2$={:.1f}%'.format(bm.dt * 100))
ax.set_xlabel('t')
ax.set_ylabel('$x_t$')
ax.set_xlim(left=0, right=bm.num_simuls)
fig1.tight_layout()


# -------------------------------------------------------------------------------
# GENERALIZED BROWNIAN MOTION
# -------------------------------------------------------------------------------


# Simulate generalized brownian motions

gbm = GeneralizedBrownianMotion(x0=0, mu=0.5, sigma=1.0, dt=0.1, T=100, seed=0)

gbm_sims = []

for i in range(10):

    gbm_sims.append(gbm.simulate())

gbm_sims = np.array(gbm_sims).T

fig2, ax = plt.subplots(nrows=1, ncols=1)

ax.plot(gbm_sims, linewidth=0.5)
ax.hlines(y=0, xmin=0, xmax=gbm.num_simuls, linewidth=0.5, color='black')
ax.set_title('Generalized Brownian Motion - $\mu$={:.1f}%, $\sigma^2$={:.1f}%'.format(gbm.mu * gbm.dt * 100, gbm.sigma**2 * gbm.dt * 100))
ax.set_xlabel('t')
ax.set_ylabel('$x_t$')
ax.set_xlim(left=0, right=gbm.num_simuls)
fig2.tight_layout()


# -------------------------------------------------------------------------------
# ITO PROCESS
# -------------------------------------------------------------------------------


# Simulate an Ito Process

ip = ItoProcess(x0=100, mu=0.15, sigma=0.3, dt=1/52, T=10, seed=0)

ip_sims = []

for i in range(10):

    ip_sims.append(ip.simulate())

ip_sims = np.array(ip_sims).T

fig3, ax = plt.subplots(nrows=1, ncols=1)

ax.plot(ip_sims, linewidth=0.5)
ax.hlines(y=100, xmin=0, xmax=ip.num_simuls, linewidth=0.5, color='black')
ax.set_title('Ito Process - $\mu$={:.1f}%, $\sigma$={:.1f}%'.format(ip.mu * ip.dt * 100, ip.sigma**2 * ip.dt * 100))
ax.set_xlabel('t')
ax.set_ylabel('$x_t$')
ax.set_xlim(left=0, right=ip.num_simuls)
fig3.tight_layout()

# Display figures

plt.show()


