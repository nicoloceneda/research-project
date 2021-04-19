import numpy as np
import matplotlib.pyplot as plt

# Generate Distribution:
random_num = np.random.normal(loc=0, scale=3, size=100000)
random_int = np.round(random_num)

# Plot:
axis = np.arange(start=min(random_int), stop = max(random_int) + 1)
plt.hist(random_int, bins = axis)