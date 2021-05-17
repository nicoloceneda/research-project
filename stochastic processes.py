""" STOCHASTIC PROCESSES
    --------------------
    Simulation of stochastic processes.
"""


# -------------------------------------------------------------------------------
# IMPORT LIBRARIES
# -------------------------------------------------------------------------------


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -------------------------------------------------------------------------------
# GEOMETRIC BROWNIAN MOTION
# -------------------------------------------------------------------------------


# Design the brownian motion

class BrownianMotion:

    """ Brownian motion

            Parameters:
            ----------
            x0 : float
                Starting value
            dt : float (> 0)
                Time interval
            T : float
                Ending time (> 0)


            Attributes:
            ----------
            params : string
                Parameters

            Methods:
            -------
            sim :
        """

    def __init__(self, x0=0, dt=1, T=100, seed=1):

        self.x0 = x0
        self.dt = dt
        self.T = T
        self.num_sim = round(T/dt)
        self.seed = seed
        self.param = 'Brownian motion: x0: {}; dt: {}; T: {}, num_sim: {}, seed: {}'.format(x0, dt, T, self.num_sim, self.seed)

    def sim(self):

        if self.seed == 1
        np.random.seed(987654321)

        xt = self.x0
        x_sim = np.array(xt)

        for i in range(self.num_sim):

            epsilon = np.random.normal(loc=0, scale=1)
            dx = epsilon * np.sqrt(self.dt)
            xt = xt + dx
            x_sim = np.append(x_sim, xt)

        return x_sim

    def sim_stats(self):

        sim = self.sim
        sim_mean = np.mean(sim)
        sim_variance = np.var(sim)

        return sim_mean, sim_variance


bm = BrownianMotion()

print(bm.param)
x_simul = bm.sim()
print(np.mean(x_simul))
print(np.var(x_simul))
print(np.var(x_simul))
plt.plot(x_simul)

# Design the Geometric brownian motion


