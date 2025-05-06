"""
Author: Lev Kakasenko
Description:
Computes the average noise-induced relative error in a test data sample.

If you use this code in any form, please cite "Bridging the Gap Between Deterministic and
Probabilistic Approaches to State Estimation" by Lev Kakasenko,  Alen Alexanderian,
Mohammad Farazmand, and Arvind Krishna Saibaba (2025)
"""

import random
import numpy as np
from data_matrices import generate_data_matrix, turbulence, add_gaussian_noise
from errors import error_rel_columnwise

# Parameters
# (a full description of these parameters can be found in compute_error.py)
data_matrix = 'turbulence'
num_features = turbulence().shape[0] # Set to turbulence().shape[0] for turbulence data.
num_samples = turbulence().shape[1] # Set to turbulence().shape[1] for turbulence data.
noise = .3
split_idx = 750
seed = 0


# set the seed for randomly generated values
random.seed(seed)
np.random.seed(seed)

# generate the data matrix
x = generate_data_matrix(data_matrix, num_features, num_samples)

# split data into train and test sets
X_train = x[:, :split_idx]
X_test = x[:, split_idx:]

# center the data
row_means = np.mean(X_train, axis=1)
X_train -= row_means[:, np.newaxis]
X_test -= row_means[:, np.newaxis]

# add noise to the test observations
X_test_obs = add_gaussian_noise(X_test, noise)

# compute the average noise-induced relative error in the test data
noise_error, _ = error_rel_columnwise(X_test, X_test_obs)

print('average measurement noise-induced relative error in a test sample: ' 
      + str(round(noise_error, 4)))
