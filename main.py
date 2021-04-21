import numpy as np
import matplotlib.pyplot as plt

# Generate Distribution:
random_num = np.random.normal(loc=0, scale=3, size=100000)
random_int = np.round(random_num)

# Plot:
axis = np.arange(start=min(random_int), stop=max(random_int)+1)
d_pi = 0
plt.hist(random_int, bins=axis)

f =
delta = 0.9
p = 0.5
K = 0.7
sigma_D = 1
theta = np.arange(-4000, 7000, 1000)

def kappa

    k =
def c_theta(x):


    c = 1 / ((delta + p + (theta - 1) * x + 0.5 * x * (1 - x) * sigma_D ** 2) * (1 - p * K))

    return c

for gamma in np.arange(1,3,0.5):

    plt.plot(c_theta(gamma))