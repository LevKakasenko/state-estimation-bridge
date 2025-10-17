"""
Author: Lev Kakasenko
Description:
Plots the MAP and DEIM reconstruction errors with respect to number of modes given
randomly selected sensor locations.

If you use this code in any form, please cite "Bridging the Gap Between Deterministic and
Probabilistic Approaches to State Estimation" by Lev Kakasenko,  Alen Alexanderian,
Mohammad Farazmand, and Arvind Krishna Saibaba (2025)
"""

import random
import numpy as np
import matplotlib.pyplot as plt

from heuristics import random_select
from reconstructors import deim, map
from errors import error_rel_columnwise
from data_matrices import generate_data_matrix, turbulence, add_gaussian_noise

### Parameters
data_matrix = 'turbulence'
num_features = turbulence().shape[0]
num_samples = turbulence().shape[1]
noise = .3
sigma_noise = noise
mode_range = range(5, 101, 5)
num_sensors = 25
split_idx = 750
seed = 0


# Set the random seed
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

# perform SVD on training data
U, s, Vh = np.linalg.svd(X_train, full_matrices=True)

# select the random sensor locations
random_indices = random_select(num_features=num_features, num_sensors=num_sensors)

# initialize the lists containing the reconstruction error means
error_mean_deim_lst = []
error_mean_map_lst = []

# initialize the lists containing the reconstruction error standard deviations
error_var_deim_lst = []
error_var_map_lst = []

# compute reconstruction error over the range of POD modes
for num_modes in mode_range:
    # generate a matrix with the allowed number of POD modes (as columns)
    U_r = U[:, :num_modes]
    Sigma = np.diag(s[:num_modes])
    Gamma_prior = (Sigma**2) / (Vh.shape[1] - 1)
    Gamma_prior_inv = np.diag(1 / np.diag(Gamma_prior)) # ONLY VALID FOR DIAGONAL MATRICES
    
    # compute the DEIM reconstructions
    X_test_hat_deim = deim(X_test, random_indices, U_r)
    X_test_hat_map = map(X_test, random_indices, U_r, sigma_noise, Gamma_prior_inv)

    # compute reconstruction error
    error_mean_deim, error_var_deim = error_rel_columnwise(X_test=X_test, 
                                                           X_test_hat=X_test_hat_deim)
    error_mean_map, error_var_map = error_rel_columnwise(X_test=X_test, 
                                                           X_test_hat=X_test_hat_map)

    # append errors and error variances to their respective lists
    error_mean_deim_lst.append(error_mean_deim)
    error_mean_map_lst.append(error_mean_map)
    error_var_deim_lst.append(error_var_deim)
    error_var_map_lst.append(error_mean_map)

# convert the lists into arrays, and compute standard deviation from variance
error_mean_deim_lst = np.array(error_mean_deim_lst)
error_std_deim_lst = np.array(error_var_deim_lst)**.5
error_mean_map_lst = np.array(error_mean_map_lst)
error_std_map_lst = np.array(error_var_map_lst)**.5

# plot the results
plt.figure(dpi=200)
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['text.usetex'] = True
plt.text(num_sensors, 7.1, f'sensors = {num_sensors}', fontsize=14, 
        horizontalalignment='center', color='black') # 7.1
plt.errorbar(mode_range, error_mean_deim_lst, error_std_deim_lst, 
             label='Random-DEIM', linestyle='solid', marker='*', 
             markersize=7, linewidth=2, capsize=5, color='black', alpha=1)
plt.errorbar(mode_range, error_mean_map_lst, error_std_map_lst, 
             label='Random-MAP', linestyle='solid', marker='*', 
             markersize=7, linewidth=2, capsize=5, color='darkgoldenrod', alpha=1)
plt.xlabel('Number of modes', fontsize=14)
plt.ylabel('Relative error', fontsize=14)
plt.tick_params(axis='both', which='major', labelsize=14, length=5)
plt.ylim(0, 7)
plt.yticks(range(0, 8, 1))
plt.legend(fontsize=14)
plt.grid()
plt.show()
