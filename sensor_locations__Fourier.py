"""
Author: Lev Kakasenko
Description:
Plots the optimal sensor locations for the Fourier data (optimal in
terms of the reconstruction error of the DEIM and MAP estimates), in addition
to the sensor locations outputted by other sensor placement algorithms.

If you use this code in any form, please cite "Bridging the Gap Between Deterministic and
Probabilistic Approaches to State Estimation" by Lev Kakasenko,  Alen Alexanderian,
Mohammad Farazmand, and Arvind Krishna Saibaba (2025)
"""

import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
from data_matrices import generate_data_matrix, add_gaussian_noise
from heuristics import optimal_map_select, optimal_deim_select, \
                    cpqr_select, dopt_select, dopt_greedy_select
from utils import evenly_spaced_rows
from math import pi

# Parameters
data_matrix = 'fourier' # should always be set to 'fourier'
num_features_full = 10**3 # number of points in the dense grid on which the function is evaluated 
                          # (for a smooth plotted curve)
num_features = 40 # number of rows in the data matrix, each of which is eligible to receive a sensor;
                  # num_features must be less than or equal to num_features_full 
num_samples = 1000 # number of columns in the data matrix (each corresponding to a randomly generated function)
split_idx = 750 # The index at which the data matrix columns are split to generate the train and 
                # test matrices.  If, for example, this index is set to 750, then the first 750
                # samples are used for training, and the remaining samples are used for testing.
num_modes = 20 # number of POD modes to retain
num_sensors = 5 # number of sensors to select
noise = .1 # standard deviation of uncorrelated Gaussian noise added to the test data
sigma_noise = noise # Model parameter representing a measurement noise standard deviation.
heuristics = [
             'opt_map',
             'opt_deim',
             'dopt',
             'dopt_greedy',
             'cpqr'
             ] # for a full description of available heuristics, see compute_error.py
seed = 0 # Random seed

# Explanation of num_features_full and num_features (as specified above):
######################################################################################################
# When plotting our harmonic function (to be reconstructed), we want the grid of angles
# over which the harmonic function is evaluated to be sufficiently dense (so that the plotted function
# is smooth).  The number of angles in this dense grid is specified by num_features_full.  However,
# to make computations tractable, we want to keep the number of available sensor locations 
# (as specified by num_features) reasonably small.  Thus, given the dense grid with num_features_full grid 
# points, we take a subsample of num_features evenly-spaced points from this dense grid as the set of 
# available sensor locations (where num_features << num_features_full).
######################################################################################################


markersize = 100 # for plotting

# set the seed for randomly generated values
random.seed(seed)
np.random.seed(seed)

# Generate the matrix of train and test data, where each column corresponds to a randomly generated 
# harmonic function evaluated on equally spaced points between 0 and 2*pi.
x_full = generate_data_matrix(data_matrix, num_features_full, num_samples)
x = evenly_spaced_rows(x_full, num_features)
X_train = x[:, :split_idx]
X_test = x[:, split_idx:]

# center the data
row_means = np.mean(X_train, axis=1)
X_train -= row_means[:, np.newaxis]
X_test -= row_means[:, np.newaxis]

# add noise to test data
X_test_obs = add_gaussian_noise(X_test, noise)

# generate the modal basis, prior covariance matrix, and matrices derived from the 
# prior covariance matrix
U, s, Vh = np.linalg.svd(X_train, full_matrices=True)
U_r = U[:, :num_modes]
Sigma = np.diag(s[:num_modes])
Gamma_prior = (Sigma**2) / (Vh.shape[1] - 1)
Gamma_prior_sqrt = np.sqrt(Gamma_prior)
Gamma_prior_inv = np.linalg.inv(Gamma_prior)

# generate angles for plotting
angles_full = np.linspace(0, 2 * np.pi, num_features_full, endpoint=False)
angles = evenly_spaced_rows(angles_full, num_features)
angles = np.append(angles, 2*pi) # to move sensors at 0 to 2*pi

# plot the optimal sensor locations
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)

f = plt.figure()
ax = f.add_subplot(111)
ax.yaxis.tick_right()

plt.xlabel(r'$x$', fontsize=18)
label_lst = []
plt.xticks(size=14)
plt.tick_params(axis='y', which='both', length=0)
height = .8

# initialize the list of all sensor locations
sensor_locations = []

# initialize the plot colors and markers
colors = ['black', 'purple', 'blue', 'green', 'darkgoldenrod']
markers = ['s', 's', 'o', '^', 'P']

# compute the reconstructions and reconstruction errors
plot_idx = -1
for heur in heuristics:
    if heur == 'opt_map':
        idx, _, _, _ = optimal_map_select(U_r, X_test, X_test_obs, num_sensors, sigma_noise, 
                                          Gamma_prior_inv)
        label = 'Optimal (MAP)'
    elif heur == 'opt_deim':
        idx, _, _, _ = optimal_deim_select(U[:, :num_sensors], X_test, X_test_obs, num_sensors)
        label = 'Optimal (DEIM)'
    elif heur == 'cpqr':
        cpqr_prior = np.eye(num_sensors)
        idx = cpqr_select(U[:, :num_sensors], num_sensors, cpqr_prior)
        idx = list(idx) # cast to list so that idx can be appended to sensor_locations
        label = 'CPQR' # label for plotting
    elif heur == 'dopt':
        idx = dopt_select(U_r, sigma_noise, Gamma_prior_sqrt, num_sensors)
        label = 'D-Optimal' # label for plotting
    elif heur == 'dopt_greedy':
        idx = dopt_greedy_select(U_r, sigma_noise, Gamma_prior_sqrt, num_sensors)
        label = 'Greedy D-Optimal' # label for plotting
    else:
        raise Exception("Invalid heuristic")

    label_lst.append(label)

    # plot the sub-optimal sensor locations
    plot_idx = plot_idx + 1
    plt.scatter(angles[idx], [height]*len(idx), s=markersize, 
                marker=markers[plot_idx], color=colors[plot_idx])
    height -= .05

    # update list of all sensor locations
    sensor_locations = list(set(sensor_locations + idx))

plt.yticks([.8 - .05 * i for i in range(len(label_lst))], label_lst, size=14)

# plot gray dashed lines for all sensor locations
for i in sensor_locations:
    plt.axvline(x=angles[i], color='gray', linestyle='--', linewidth=1.5, alpha=.4)

plt.rcParams['text.usetex'] = True
plt.xlim(0, 6.4)
plt.tight_layout()
plt.show()
