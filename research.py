""" VERONESI 2000
    -------------
    Replication of Veronesi (2000).
"""


# -------------------------------------------------------------------------------
# IMPORT LIBRARIES
# -------------------------------------------------------------------------------


# Import libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -------------------------------------------------------------------------------
# UTILITY
# -------------------------------------------------------------------------------


# Utility

def utility_fun(consumption_, gamma_):

    if gamma_ != 0:

        utility_ = (1 - np.exp(- gamma_ * consumption_)) / gamma_

    else:

        utility_ = consumption_

    return utility_


# Plot utility function

consumption = np.arange(0, 5, 0.05)
gammas = [0.0, 0.5, 1.0, 1.5]

utility = pd.DataFrame(columns=gammas)

for g in gammas:

    utility.loc[:, g] = utility_fun(consumption, g)

fig_1, ax = plt.subplots(nrows=1, ncols=1, figsize=(5.5, 4))
ax.plot(consumption, utility, label=gammas)
ax.set_ylabel('U(C)', fontsize=10)
ax.set_xlabel('C', fontsize=10)
ax.set_title('CARA Utility',fontsize=10)
ax.legend(loc='upper left', fontsize=10)
ax.set_xlim(xmin=np.min(consumption), xmax=np.max(consumption))
ax.set_ylim(ymin=0, ymax=np.max(utility.iloc[:,0]))
fig_1.tight_layout()
fig_1.savefig('images/fig_1.png')
