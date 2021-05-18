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
            num_simulations : integer
                Number of simulated periods

            Methods:
            -------
            simulate : array [num_simulations, ]
                Simulated values
            sim_stats : string
    """

    def __init__(self, x0=0.0, dt=0.1, T=100, seed=1):

        self.x0 = x0
        self.dt = dt
        self.T = T
        self.seed = seed
        self.num_simulations = round(T/dt)

    def simulate(self):

        if self.seed == 1:

            np.random.seed(987654321)

        xt = self.x0
        x_sim = np.array(xt)

        for i in range(self.num_simulations):

            epsilon = np.random.normal(loc=0, scale=1)
            dx = epsilon * np.sqrt(self.dt)
            xt = xt + dx
            x_sim = np.append(x_sim, xt)

        return x_sim


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

    def __init__(self, mu=0.1, sigma=1.0, x0=0.0, dt=0.1, T=100, seed=1):

        super().__init__(x0, dt, T, seed)
        self.mu = mu
        self.sigma = sigma






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


# Design the generalized brownian motion

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
            num_simulations : integer
                Number of simulated periods

            Methods:
            -------
            sim : array [num_simulations, ]
                Simulated values
            sim_stats : string

        """

    def __init__(self, x0=0.0, mu=0.1, sigma=1.0, dt=0.1, T=100, seed=1):

        self.x0 = x0
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.T = T
        self.seed = seed

        self.num_simulations = round(T/dt)

    def sim(self):

        if self.seed == 1:

            np.random.seed(987654321)

        xt = self.x0
        x_sim = np.array(xt)

        for i in range(self.num_simulations):

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


# Design the Ito process

class ItoProcess:

    """ Ito process

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
            num_simulations : integer
                Number of simulated periods

            Methods:
            -------
            sim : array [num_simulations, ]
                Simulated values
            sim_stats : string

        """

    def __init__(self, x0=0.0, mu=0.1, sigma=1.0, dt=0.1, T=100, seed=1):

        self.x0 = x0
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.T = T
        self.seed = seed

        self.num_simulations = round(T/dt)

    def sim(self):

        if self.seed == 1:

            np.random.seed(987654321)

        xt = self.x0
        x_sim = np.array(xt)

        for i in range(self.num_simulations):

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



