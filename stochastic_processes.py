""" STOCHASTIC PROCESSES
    --------------------
    Stochastic processes.
"""


# -------------------------------------------------------------------------------
# IMPORT LIBRARIES
# -------------------------------------------------------------------------------


import numpy as np


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
            change : boolean
                Returns the changes if set to true
            seed : integer
                True if a seed is set; False otherwise

            Attributes:
            ----------
            num_simuls : integer
                Number of simulated periods

            Methods:
            -------
            simulate : array [num_simuls, ]
                Simulated values
    """


    def __init__(self, x0=0.0, dt=0.1, T=100, change=False, seed=True):

        self.x0 = x0
        self.dt = dt
        self.T = T
        self.change = change
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
            change : boolean
                Returns the changes if set to true
            seed : integer
                True if a seed is set; False otherwise
    """

    def __init__(self, x0=0.0, dt=0.1, T=100, change=False, seed=True):

        super().__init__(x0, dt, T, change, seed)

    def simulate(self):

        if self.seed:
            np.random.seed(987654321)

        xt = self.x0
        dx_sim = []
        x_sim = [xt]

        for i in range(self.num_simuls):
            dx = np.random.normal(0,1) * np.sqrt(self.dt)
            xt = xt + dx
            dx_sim.append(dx)
            x_sim.append(xt)

        dx_sim.append(0.0)
        dx_sim = np.array(dx_sim)
        x_sim = np.array(x_sim)

        if self.change:

            return x_sim, dx_sim

        else:

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
            change : boolean
                Returns the changes if set to true
            seed : integer
                True if a seed is set; False otherwise
    """

    def __init__(self, mu=0.1, sigma=1.0, x0=0.0, dt=0.1, T=100, change=False, seed=True):

        super().__init__(x0, dt, T, change, seed)
        self.mu = mu
        self.sigma = sigma

        if np.ndim(self.mu) == 0:
            self.drift = np.full(self.num_simuls, self.mu)

        if np.ndim(self.mu) != 0 and np.size(self.mu, 0) == self.num_simuls:
            self.drift = self.mu

        if np.ndim(self.mu) != 0 and np.size(self.mu, 0) != self.num_simuls:
            print("The size of the vector array should be equal to T/dt")

    def simulate(self):

        if self.seed:
            np.random.seed(987654321)

        xt = self.x0
        dx_sim = []
        x_sim = [xt]

        for i in range(self.num_simuls):
            dx = self.drift[i] * self.dt + self.sigma * np.random.normal(0,1) * np.sqrt(self.dt)
            xt = xt + dx
            dx_sim.append(dx)
            x_sim.append(xt)

        dx_sim.append(0.0)
        dx_sim = np.array(dx_sim)
        x_sim = np.array(x_sim)

        if self.change:

            return x_sim, dx_sim

        else:

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
            change : boolean
                Returns the changes if set to true
            seed : integer
                True if a seed is set; False otherwise
    """

    def __init__(self, mu=0.15, sigma=0.3, x0=100, dt=0.02, T=100, change=False, seed=True):

        super().__init__(x0, dt, T, change, seed)
        self.mu = mu
        self.sigma = sigma

        if self.x0:
            print("x0 can't be zero")

        if np.ndim(self.mu) == 0:
            self.drift = np.full(self.num_simuls, self.mu)

        if np.ndim(self.mu) != 0 and np.size(self.mu, 0) == self.num_simuls:
            self.drift = self.mu

        if np.ndim(self.mu) != 0 and np.size(self.mu, 0) != self.num_simuls:
            print("The size of the vector array should be equal to T/dt")

    def simulate(self):

        if self.seed == 1:
            np.random.seed(987654321)

        xt = self.x0
        dx_sim = []
        x_sim = [xt]

        for i in range(self.num_simuls):
            dx = self.drift[i] * xt * self.dt + self.sigma * xt * np.random.normal(0,1) * np.sqrt(self.dt)
            xt = xt + dx
            dx_sim.append(dx)
            x_sim.append(xt)

        dx_sim.append(0.0)
        dx_sim = np.array(dx_sim)
        x_sim = np.array(x_sim)

        if self.change:

            return x_sim, dx_sim

        else:

            return x_sim
