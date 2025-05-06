"""
Author: Lev Kakasenko
Description:
Plots the smallest singular value of the matrix S^T Phi with respect to
the number of POD modes.

If you use this code in any form, please cite "Bridging the Gap Between Deterministic and
Probabilistic Approaches to State Estimation" by Lev Kakasenko,  Alen Alexanderian,
Mohammad Farazmand, and Arvind Krishna Saibaba (2025)
"""

import numpy as np
import random
from data_matrices import generate_data_matrix, turbulence
import matplotlib.pyplot as plt

# Parameters
# (a full description of these parameters can be found in compute_error.py)
data_matrix = 'fourier'
num_features = 40
num_samples = 1000
split_idx = 750
mode_range = range(1, 11)
num_sensors = 5
seed = 0


# set the seed for randomly generated values
random.seed(seed)
np.random.seed(seed)

# generate the matrix of data
x = generate_data_matrix(data_matrix, num_features, num_samples)

# split the data into train and test sets
X_train = x[:, :split_idx]

# center the data
row_means = np.mean(X_train, axis=1)
X_train -= row_means[:, np.newaxis]

# perform SVD on training data
U, _, _ = np.linalg.svd(X_train, full_matrices=True)

# randomly select sensor locations
sensor_indices = np.random.choice(num_features,size=num_sensors,replace=False)
sensor_indices.sort()

# initialize the list of smallest singular values
s_lst = []

# compute the smallest singular value of S^T Phi over a range of modes
for num_modes in mode_range:
    U_r = U[:, :num_modes]
    A = U_r[sensor_indices]
    s = np.linalg.svd(A, compute_uv=False)
    s_min = s[-1]
    s_lst.append(s_min)

# plot results
plt.title(f'number of sensors={num_sensors}', fontsize=14)
plt.semilogy(mode_range, s_lst, color='blue')
plt.axvline(x=num_sensors, color='black', linestyle='--')
plt.xlabel('number of modes', fontsize=14)
plt.ylabel(r'$\sigma_{\text{min}}(S^T \Phi)$', fontsize=14)
plt.show()
