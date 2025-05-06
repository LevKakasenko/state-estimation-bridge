"""
Author: Lev Kakasenko
Description:
Plots the singular value decay of the training data matrix.

If you use this code in any form, please cite "Bridging the Gap Between Deterministic and
Probabilistic Approaches to State Estimation" by Lev Kakasenko,  Alen Alexanderian,
Mohammad Farazmand, and Arvind Krishna Saibaba (2025)
"""

import numpy as np
import random
from data_matrices import fourier, turbulence
import matplotlib.pyplot as plt

# Parameters
data_matrix = 'fourier' # Dataset to use, which can be set to:
                        #   'fourier' (the Fourier dataset used in Section 5.1)
                        #   'turbulence' (the turbulence dataset used in Section 5.2)
num_features = 40 # Number of features in a single data sample.
                  # Each feature corresponds to another grid point on which a function is evaluated.
                  # Set to turbulence().shape[0] for the turbulence data.
num_samples = 1000 # Number of data samples (i.e. snapshots) for training and testing.
                   # Set to turbulence().shape[1] for the turbulence data.
split_idx = 750 # The index at which the data matrix columns are split to generate the train and 
                # test matrices.  If, for example, this index is set to 750, then the first 750
                # samples are used for training, and the remaining samples are used for testing.
seed = 0 # Random seed

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

s = np.linalg.svd(x, full_matrices=True, compute_uv=False)

plt.figure()
plt.semilogy(list(range(1,len(s)+1)), s, linewidth=2)
plt.xlabel('index', fontsize=14)
plt.ylabel('singular values', fontsize=14)
plt.title('Singular Values', fontsize=18)
plt.grid()
plt.show()
