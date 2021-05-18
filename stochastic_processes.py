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

class BrownianMotion:

    """ Brownian motion

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
            num_sim : integer
                Number of simulated periods

            Methods:
            -------
            sim : array [num_sim, ]
                Simulated values
            sim_stats : string

        """

    def __init__(self, x0=0.0, dt=0.1, T=100, seed=1):

        self.x0 = x0
        self.dt = dt
        self.T = T
        self.seed = seed

        self.num_sim = round(T/dt)

    def sim(self):

        if self.seed == 1:

            np.random.seed(987654321)

        xt = self.x0
        x_sim = np.array(xt)

        for i in range(self.num_sim):

            epsilon = np.random.normal(loc=0, scale=1)
            dx = epsilon * np.sqrt(self.dt)
            xt = xt + dx
            x_sim = np.append(x_sim, xt)

        return x_sim

    def sim_stats(self, message=0):

        x_sim = self.sim()
        dx = x_sim[1:] - x_sim[:-1]
        sim_mean = np.mean(dx)
        sim_variance = np.var(dx)

        if message == 1:

            print('\nStatistics:'\
                  '\n----------'\
                  '\nmean: {}'\
                  '\nvariance: {}'.format(sim_mean, sim_variance))
        else:

            stats = [sim_mean, sim_variance]

            return stats


# -------------------------------------------------------------------------------
# GENERALIZED BROWNIAN MOTION
# -------------------------------------------------------------------------------


# Design the brownian motion

class GeneralizedBrownianMotion:

    """ Generalized brownian motion

            Parameters:
            ----------
            x0 : float
                Starting value
            mu : float
                drift rate
            sigma : float
                variance rate
            dt : float
                Time interval
            T : float
                Ending time
            seed : integer
                1 if a seed is set; 0 otherwise

            Attributes:
            ----------
            num_sim : integer
                Number of simulated periods

            Methods:
            -------
            sim : array [num_sim, ]
                Simulated values
            sim_stats : string

        """

    def __init__(self, x0=0.0, mu=0.0, sigma=1.0, dt=0.1, T=100, seed=1):

        self.x0 = x0
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.T = T
        self.seed = seed

        self.num_sim = round(T/dt)

    def sim(self):

        if self.seed == 1:

            np.random.seed(987654321)

        xt = self.x0
        x_sim = np.array(xt)

        for i in range(self.num_sim):

            epsilon = np.random.normal(loc=0, scale=1)
            dx = self.mu * self.dt + self.sigma * epsilon * np.sqrt(self.dt)
            xt = xt + dx
            x_sim = np.append(x_sim, xt)

        return x_sim

    def sim_stats(self, message=0):

        x_sim = self.sim()
        dx = x_sim[1:] - x_sim[:-1]
        sim_mean = np.mean(dx)
        sim_variance = np.var(dx)

        if message == 1:

            print('\nStatistics:'\
                  '\n----------'\
                  '\nmean: {}'\
                  '\nvariance: {}'.format(sim_mean, sim_variance))
        else:

            stats = [sim_mean, sim_variance]

            return stats


# -------------------------------------------------------------------------------
# ITO PROCESS
# -------------------------------------------------------------------------------


# Design the brownian motion

class Stochastic:

    """ Ito Process

            Parameters:
            ----------
            x0 : float
                Starting value
            mu : float
                drift rate
            sigma : float
                variance rate
            dt : float
                Time interval
            T : float
                Ending time
            seed : integer
                1 if a seed is set; 0 otherwise

            Attributes:
            ----------
            num_sim : integer
                Number of simulated periods

            Methods:
            -------
            sim : array [num_sim, ]
                Simulated values
            sim_stats : string

        """

    # Initialization

    def __init__(self, process='BrownianMotion', x0=1.0, mu=0.2, sigma=1.0, dt=0.1, T=100, seed=1):

        self.process = process
        self.x0 = x0
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.T = T
        self.seed = seed

        self.num_sim = round(T/dt)

    # List of processes

    def sim(self):

        if self.seed == 1:

            np.random.seed(987654321)

        xt = self.x0
        x_sim = np.array(xt)

        if self.process == 'BrownianMotion':



            dx = epsilon * np.sqrt(self.dt)
            dx = self.mu * self.dt + self.sigma * epsilon * np.sqrt(self.dt)
            dx = self.mu * xt * self.dt + self.sigma * xt * epsilon * np.sqrt(self.dt)

        for i in range(self.num_sim):

            epsilon = np.random.normal(loc=0, scale=1)
            dx = self.mu * xt * self.dt + self.sigma * xt * epsilon * np.sqrt(self.dt)

            xt = xt + dx
            x_sim = np.append(x_sim, xt)

        return x_sim

    def sim_stats(self, message=0):

        x_sim = self.sim()
        dx = x_sim[1:] - x_sim[:-1]
        sim_mean = np.mean(dx)
        sim_variance = np.var(dx)

        if message == 1:

            print('\nStatistics:'\
                  '\n----------'\
                  '\nmean: {}'\
                  '\nvariance: {}'.format(sim_mean, sim_variance))
        else:

            stats = [sim_mean, sim_variance]

            return stats


def process1(x):

    return x + 1

def process2(x):

    return x + 2

s=1
if s==1:

    myf = process1(x)
    x = myf