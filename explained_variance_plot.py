"""
Author: Lev Kakasenko
Description:
Plots the cumulative variance in the training data explained by the POD modes,
along with a dashed line representing the minimum number of POD modes needed to 
explain a given threshold of variance (as specified by the thres parameter).

If you use this code in any form, please cite "Bridging the Gap Between Deterministic and
Probabilistic Approaches to State Estimation" by Lev Kakasenko,  Alen Alexanderian,
Mohammad Farazmand, and Arvind Krishna Saibaba (2025)
"""

import numpy as np
import random
from data_matrices import fourier, turbulence
import matplotlib.pyplot as plt

# Parameters
data_matrix = 'turbulence'
num_features = turbulence().shape[0]
num_samples = turbulence().shape[1]
split_idx = 750
thres = .95 # Minimum portion of variance in the training data that we need our modal
            # basis to explain.  Must be between 0 and 1.
seed = 0

# set the seed for randomly generated values
random.seed(seed)
np.random.seed(seed)

if data_matrix == 'fourier':
    x = fourier(num_features, num_samples)
elif data_matrix == 'turbulence':
    x = turbulence()
else:
    raise Exception("Invalid data_matrix")

X_train = x[:, :split_idx]
row_means = np.mean(X_train, axis=1)
X_train -= row_means[:, np.newaxis]

s = np.linalg.svd(X_train, full_matrices=True, compute_uv=False)
s_sqrd = s**2
var = s_sqrd / sum(s_sqrd)
cum_var = np.cumsum(var)
indices = np.where(cum_var > thres)[0]
idx = indices[0] if indices.size > 0 else None
idx = idx + 1
print(f'minimum number of modes to explain {thres} of variance: ' + str(idx))
if data_matrix == 'turbulence':
    print('variance explained by the first 100 modes: ' + str(round(cum_var[99],4)))

plt.figure()
cum_var = cum_var[:200]
plt.plot(list(range(1,len(cum_var)+1)), cum_var, linewidth=2)
plt.xlabel('number of POD modes', fontsize=14)
plt.ylabel('explained var.', fontsize=14)
plt.title('Explained variance', fontsize=18)
plt.axvline(x=idx, color='g', linestyle='--')
plt.grid()
plt.show()
