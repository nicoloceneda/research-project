""" STOCHASTIC PROCESSES
    --------------------
    Stochastic processes.
"""


# -------------------------------------------------------------------------------
# IMPORT LIBRARIES
# -------------------------------------------------------------------------------


import numpy as np
import matplotlib.pyplot as plt


# -------------------------------------------------------------------------------
# BROWNIAN MOTION
# -------------------------------------------------------------------------------


# Design the brownian motion

class Stochastic:

    """ Stochastic Process

            Parameters:
            ----------
            x0 : float
                Starting value
            dt : float
                Time interval
            T : float
                Ending time
            seed : integer
                1 if a seed is set; 0 otherwise

            Attributes:
            ----------
            num_simuls : integer
                Number of simulated periods

            Methods:
            -------
            simulate : array [num_simuls, ]
                Simulated values
            sim_stats : string
    """

    def __init__(self, x0=0.0, dt=0.1, T=100, seed=1):

        self.x0 = x0
        self.dt = dt
        self.T = T
        self.seed = seed
        self.num_simuls = round(T/dt)


class BrownianMotion(Stochastic):

    """ Brownian Motion

            Parameters:
            ----------
            x0 : float
                Starting value
            dt : float
                Time interval
            T : float
                Ending time
            seed : integer
                1 if a seed is set; 0 otherwise
    """

    def __init__(self, x0=0.0, dt=0.1, T=100, seed=1):

        super().__init__(x0, dt, T, seed)

    def simulate(self):

        if self.seed == 1:
            np.random.seed(987654321)

        xt = self.x0
        x_sim = np.array(xt)

        for i in range(self.num_simuls):
            dx = np.random.normal(0,1) * np.sqrt(self.dt)
            xt = xt + dx
            x_sim = np.append(x_sim, xt)

        return x_sim

class GeneralizedBrownianMotion(Stochastic):

    """ Generalized Brownian Motion

            Parameters:
            ----------
            mu : float
                drift rate
            sigma : float
                variance rate
            x0 : float
                Starting value
            dt : float
                Time interval
            T : float
                Ending time
            seed : integer
                1 if a seed is set; 0 otherwise
    """

    def __init__(self, mu=0.1, sigma=1.0, x0=0.0, dt=0.1, T=100, seed=1):

        super().__init__(x0, dt, T, seed)
        self.mu = mu
        self.sigma = sigma

    def simulate(self):

        if self.seed == 1:
            np.random.seed(987654321)

        xt = self.x0
        x_sim = np.array(xt)

        for i in range(self.num_simuls):
            dx = self.mu * self.dt + self.sigma * np.random.normal(0,1) * np.sqrt(self.dt)
            xt = xt + dx
            x_sim = np.append(x_sim, xt)

        return x_sim

class ItoProcess(Stochastic):

    """ Ito Process

            Parameters:
            ----------
            mu : float
                drift rate
            sigma : float
                variance rate
            x0 : float
                Starting value
            dt : float
                Time interval
            T : float
                Ending time
            seed : integer
                1 if a seed is set; 0 otherwise
    """

    def __init__(self, mu=0.15, sigma=0.3, x0=100, dt=0.02, T=100, seed=1):

        super().__init__(x0, dt, T, seed)
        self.mu = mu
        self.sigma = sigma

        if self.x0 == 0:
            print("x0 can't be zero")

    def simulate(self):

        if self.seed == 1:
            np.random.seed(987654321)

        xt = self.x0
        x_sim = np.array(xt)

        for i in range(self.num_simuls):
            dx = self.mu * xt * self.dt + self.sigma * xt * np.random.normal(0,1) * np.sqrt(self.dt)
            xt = xt + dx
            x_sim = np.append(x_sim, xt)

        return x_sim






