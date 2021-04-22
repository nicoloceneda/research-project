import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



# Plot:
axis = np.arange(start=min(f), stop=max(f)+1)
d_pi = 0
plt.hist(f, bins=axis)

f =
delta = 0.9
p = 0.5
K = 0.7
sigma_D = 1

# Parameters

theta = np.arange(-0.01, 0.01, 0.001)

random_int = np.random.randint(low=1, high=10, size=20)
f = random_int / np.sum(random_int)




range = pd.cut(random_num, np.arange(len(theta)))

df.groupby('range')['value'].count().reset_index(name='Count').to_dict(orient='records')

def kappa

    k =
def c_theta(x):


    c = 1 / ((delta + p + (theta - 1) * x + 0.5 * x * (1 - x) * sigma_D ** 2) * (1 - p * K))

    return c

for gamma in np.arange(1,3,0.5):

    plt.plot(c_theta(gamma))