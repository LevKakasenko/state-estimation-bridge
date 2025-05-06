"""
Author: Lev Kakasenko
Description:
Plots the reconstructions of a single harmonic function (i.e. a single sample of the Fourier data).

If you use this code in any form, please cite "Bridging the Gap Between Deterministic and
Probabilistic Approaches to State Estimation" by Lev Kakasenko,  Alen Alexanderian,
Mohammad Farazmand, and Arvind Krishna Saibaba (2025)
"""

import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
from data_matrices import generate_data_matrix, add_gaussian_noise
from heuristics import cpqr_select, dopt_select, aopt_select, \
                        dopt_greedy_select, aopt_greedy_select
from reconstructors import deim, map
from utils import evenly_spaced_rows
from errors import error_rel_columnwise

# Parameters
data_matrix = 'fourier' # should always be set to 'fourier'
test_idx = 10 # index of the sample to be reconstructed (i.e. index of a column of the test data matrix)
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
sigma_noise = noise # sigma_noise parameter for Bayesian approach
# The following two parameters (heur_list, reconstructors) are lists of sensor placement
# heuristics and reconstructors of the full state. Both lists should be of 
# the same length, with index i of a list corresponding to a single combination of heuristic
# and reconstructor. For example, if heuristics[0] is 'dopt_greedy' and reconstructors[0] is 'map',
# then our plot will include a greedy D-MAP reconstruction.
heuristics = [
             'dopt',
             'dopt_greedy',
             'cpqr'
             ]
reconstructors = ['map', 
                  'map',
                  'deim']
seed = 0 # Random seed.

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


# set the seed for randomly generated values
random.seed(seed)
np.random.seed(seed)

# Generate the 'full' (i.e. dense) data for plotting
x_full = generate_data_matrix(data_matrix, num_features_full, num_samples)

# split the full data into train and test sets
X_train_full = x_full[:, :split_idx]
X_test_full = x_full[:, split_idx:]

# center the full data
row_means_full = np.mean(X_train_full, axis=1)
X_test_full -= row_means_full[:, np.newaxis]

# generate array of angles over which to plot
angles_full = np.linspace(0, 2 * np.pi, num_features_full, endpoint=False)

# plot the smooth harmonic function
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)

plt.figure()
#plt.rcParams['font.family'] = 'Times New Roman'
plt.plot(angles_full, X_test_full[:,test_idx], label='True data', color='black', 
         linewidth=2)

# Select evenly spaced rows of the full data which are available to receive sensors:
x = evenly_spaced_rows(x_full, num_features)

# split the data into train & test sets
X_train = x[:, :split_idx]
X_test_all_samples = x[:, split_idx:]
X_test_single_sample = x[:, (split_idx + test_idx):(split_idx + test_idx + 1)]
row_means = np.mean(X_train, axis=1)
X_train -= row_means[:, np.newaxis]
X_test_single_sample -= row_means[:, np.newaxis]

# add noise to test data
X_test_single_sample_obs = add_gaussian_noise(X_test_single_sample, noise)
X_test_all_samples_obs = add_gaussian_noise(X_test_all_samples, noise)

# create a list of angles (for plotting)
angles = evenly_spaced_rows(angles_full, num_features)

# generate the modal basis
U, s, Vh = np.linalg.svd(X_train, full_matrices=True)
U_r = U[:, :num_modes]
U_s = U[:, :num_sensors]
Sigma = np.diag(s[:num_modes])
Gamma_prior = (Sigma**2) / (Vh.shape[1] - 1)
Gamma_prior_sqrt = np.sqrt(Gamma_prior)
Gamma_prior_inv = np.linalg.inv(Gamma_prior)

# specify the plot colors
colors = ['blue', 'green', 'darkgoldenrod', 'purple']
color_idx = -1

# compute reconstructions and errors of the sensor placement heuristics
for i in range(len(heuristics)):
    heur = heuristics[i]
    reconstructor = reconstructors[i]
    if heur == 'cpqr':
        cpqr_prior = np.eye(num_sensors)
        idx = cpqr_select(U_s, num_sensors, cpqr_prior)
        label = 'Q-' # label for plotting
    elif heur == 'dopt':
        idx = dopt_select(U_r, sigma_noise, Gamma_prior_sqrt, num_sensors)
        label = 'D-' # label for plotting
    elif heur == 'dopt_greedy':
        idx = dopt_greedy_select(U_r, sigma_noise, Gamma_prior_sqrt, num_sensors)
        label = 'Greedy D-' # label for plotting
    elif heur == 'aopt':
        idx = aopt_select(U_r, sigma_noise, Gamma_prior_inv, num_sensors)
        label = 'A-Optimal' # label for plotting
    elif heur == 'aopt_greedy':
        idx = aopt_greedy_select(U_r, sigma_noise, Gamma_prior_inv, num_sensors)
        label = 'Greedy A-' # label for plotting
    else:
        raise Exception("Invalid heuristic")
    
    if reconstructor == 'deim':
        reconstruction_single_sample = deim(X_test_single_sample_obs, idx, U_s)
        reconstruction_all_samples = deim(X_test_all_samples_obs, idx, U_s)
        label += 'DEIM'
    elif reconstructor == 'map':
        reconstruction_single_sample = map(X_test_single_sample_obs, idx, U_r, sigma_noise, Gamma_prior_inv)
        reconstruction_all_samples = map(X_test_all_samples_obs, idx, U_r, sigma_noise, Gamma_prior_inv)
        label += 'MAP'
    else:
        raise Exception("Invalid reconstructor")

    # compute the reconstruction errors on a single test sample and averaged across all test samples
    error_single_sample = error_rel_columnwise(X_test_single_sample, reconstruction_single_sample)[0]
    error_all_samples = error_rel_columnwise(X_test_all_samples, reconstruction_all_samples)[0]
    print(label + ' error (single test sample): ' + str(round(error_single_sample,4)))
    print(label + ' error (all test samples): ' + str(round(error_all_samples,4)))
    print(label + ' locations: ' + str(angles[idx]))
    print('--------')
    
    reconstruction_to_plot = reconstruction_single_sample[:, 0]

    color_idx = color_idx + 1
    plt.plot(angles,reconstruction_to_plot,label=label,marker='o', markersize=4,
             color=colors[color_idx])
        
# finish plotting the results
plt.xlabel(r'$x$', fontsize=18)
plt.ylabel(r'$f(x)$', fontsize=18)
plt.legend(fontsize=14, loc='upper left', bbox_to_anchor=(1, 1))
plt.tick_params(axis='both', which='major', labelsize=14)
plt.tick_params(axis='both', which='major', length=5)
plt.tight_layout()
plt.show()
