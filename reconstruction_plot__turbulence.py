"""
Author: Lev Kakasenko
Description:
Plots the reconstructions of a turbulence snapshot.

If you use this code in any form, please cite "Bridging the Gap Between Deterministic and
Probabilistic Approaches to State Estimation" by Lev Kakasenko,  Alen Alexanderian,
Mohammad Farazmand, and Arvind Krishna Saibaba (2025)
"""
import numpy as np
import random
from utils import load_data
from data_matrices import turbulence, generate_data_matrix, add_gaussian_noise
from reconstructors import reconstruct
from priors import generate_prior
from errors import error_rel_columnwise
from utils import heur_to_string
import matplotlib.pyplot as plt
from matplotlib import rc
from mpl_toolkits.axes_grid1 import ImageGrid

# Parameters
# Note: Some of these parameters (like plot_type) are immaterial to the 
#       plots generated by this program.  We nonetheless include these
#       parameters to allow for data to be loaded in from the logs.
data_matrix = 'turbulence' # should always be set to 'turbulence'
num_features = turbulence().shape[0] # should always be set to turbulence().shape[0]
num_samples = turbulence().shape[1] # should always be set to turbulence().shape[1]
noise = .3 # Standard deviation of uncorrelated Gaussian noise added to the test data.
sigma_noise = noise # Model parameter representing a measurement noise standard deviation.
split_idx = 750 # The index at which the data matrix columns are split to generate the train and 
                # test matrices.  If, for example, this index is set to 750, then the first 750
                # samples are used for training, and the remaining samples are used for testing.
num_modes = 100 # number of POD modes to use in the reconstruction
num_sensors = 50 # number of sensors to use in the reconstruction
plot_type = 'error_vs_sensors' # should always be set to 'error_vs_sensors' 
                               # (necessary only for loading in data from the logs)
# The following three parameters (heuristics, reconstructors, priors) are lists of sensor placement
# heuristics, reconstructors of the full state, and prior formulas.  All three lists should be of 
# the same length, with index i of the lists corresponding to a single combination of heuristic, 
# reconstructor, and prior to be plotted.  For example, if heuristics[0] is 'dopt_greedy', 
# reconstructors[0] is 'map', and priors[0] is 'natural', then the outputted plot will include
# the Greedy D-MAP reconstruction with our proposed 'natural' prior (from Section 3.1).
heuristics = ['dopt_greedy', 'cpqr_fair', 'cpqr']
reconstructors = ['map', 'deim', 'map']
priors = ['natural', 'identity', 'natural']
seed = 0 # Random seed.
sample = 125 # index of the test data sample to be reconstructed (i.e. index of a column of the test data matrix)


mode_range = range(num_modes, num_modes + 1)
sensor_range = range(num_sensors, num_sensors + 1)


# Scrape in data from the logs.
log = load_data(heuristics, reconstructors, priors, plot_type, mode_range, 
                sensor_range, num_features, num_samples, split_idx, 
                data_matrix, noise, sigma_noise, seed)

# set the seed for randomly generated values
random.seed(seed)
np.random.seed(seed)

# Fix the number of sensors (modes) as the first entry of sensor_range (mode_range).
# Also determine if the plot_type is valid.
if plot_type == 'error_vs_modes':
    parameter_range = mode_range
    sensors = sensor_range[0]
elif plot_type == 'error_vs_sensors':
    parameter_range = sensor_range
    modes = mode_range[0]
else:
    raise Exception('Invalid plot_type.')


# generate the data matrix
x = generate_data_matrix(data_matrix, num_features, num_samples)

# split data into train and test sets
X_train = x[:, :split_idx]
X_test = x[:, split_idx:]

# center the data
row_means = np.mean(X_train, axis=1)
X_train -= row_means[:, np.newaxis]
X_test -= row_means[:, np.newaxis]

# take a column of X_test as the snapshot to be reconstructed
X_test = X_test[:, sample:sample+1]

# add noise to the test snapshot
X_test_obs = add_gaussian_noise(X_test, noise)

# perform SVD on training data
U, s, Vh = np.linalg.svd(X_train, full_matrices=True)

# generate a matrix with the allowed number of POD modes (as columns)
U_r = U[:, :num_modes]

# generate a dictionary to be populated with 128 by 128 reconstructions
reconstruction_dict = {}

for idx in range(len(heuristics)):
    heur = heuristics[idx]
    prior_type = priors[idx]
    reconstruction_type = reconstructors[idx]
    heur_name = (heur, prior_type, reconstruction_type)
    results_dict_heur = log[heur_name]

    # generate matrices for the OED approach (namely the prior)
    Gamma_prior = generate_prior(prior_type=prior_type, s_vals=s,
                                 num_samples=X_train.shape[1], num_modes=num_modes)
    Gamma_prior_inv = np.diag(1 / np.diag(Gamma_prior)) # ONLY VALID FOR DIAGONAL PRIORS
    
    # read in data on sensor index locations, then perform and plot the reconstructions
    if plot_type == 'error_vs_modes':
        modes_varied = True

        for modes in mode_range: # there should only be one iteration of this loop
            if heur == "cpqr_fair":
                values = results_dict_heur[(modes, modes)]
            else:
                values = results_dict_heur[(modes, sensors)]
            sensor_indices = list(values[0])
    
    elif plot_type == 'error_vs_sensors':
        modes_varied = False

        for sensors in sensor_range: # there should only be one iteration of this loop
            if heur == "cpqr_fair":
                values = results_dict_heur[(sensors, sensors)]
            else:
                values = results_dict_heur[(modes, sensors)]
            sensor_indices = list(values[0])
    
    else:
        raise Exception('Invalid plot_type.')
    

    # compute the reconstruction
    X_test_hat = reconstruct(reconstruction_type=reconstruction_type, heur=heur, 
                             num_sensors=num_sensors, num_modes=num_modes, X_test=X_test_obs,
                             sensor_indices=sensor_indices, U=U, U_r=U_r, 
                             Gamma_prior_inv=Gamma_prior_inv, 
                             modes_varied=modes_varied, sigma_noise=sigma_noise)
    
    # compute the reconstruction error
    error, _ = error_rel_columnwise(X_test, X_test_hat)

    # reshape X_test and X_test_hat to their original dimensions (128 by 128)
    X_test_hat = X_test_hat.reshape((128,128))

    # add results to reconstruction_dict
    reconstruction_dict[heur_name] = (X_test_hat, error)


# Plot the results
heur_names = list(reconstruction_dict.keys()) # names of reconstructions
# Determine the common vmin and vmax for all arrays in reconstruction_dict
vmin = X_test.min()
vmax = X_test.max()
for heur_name in heur_names:
    vmin_ = reconstruction_dict[heur_name][0].min()
    vmax_ = reconstruction_dict[heur_name][0].max()
    if vmin_ < vmin:
        vmin = vmin_
    if vmax_ > vmax:
        vmax = vmax_

# Create a figure and an AxesGrid for subplots
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)
fig = plt.figure(figsize=(12, 6))
plt.rcParams['font.family'] = 'Times New Roman'
grid = ImageGrid(fig, 111, nrows_ncols=(2, 2), axes_pad=0.1, 
                 cbar_mode='single', cbar_location='right', cbar_pad=0.1)

# Plot the heatmap for X_test
X_test = X_test.reshape((128,128)) # reshape X_test to its original dimensions
im = grid[0].imshow(X_test, cmap='jet', vmin=vmin, vmax=vmax)
grid[0].contour(X_test, colors='black')
grid[0].get_yaxis().set_ticks([])
grid[0].get_xaxis().set_ticks([])

# Plot the heatmaps for the heuristics
i = 1 # index for grid
for heur_name in heur_names:
    X_test_hat = reconstruction_dict[heur_name][0]
    error = reconstruction_dict[heur_name][1]

    im = grid[i].imshow(X_test_hat, cmap='jet', vmin=vmin, vmax=vmax)
    grid[i].contour(X_test_hat, colors='black')
    grid[i].get_yaxis().set_ticks([])
    grid[i].get_xaxis().set_ticks([])
    #title = heur_to_string(heur_name[0], heur_name[2])
    #grid[i].set_title(title, fontsize=18)
    
    print(heur_to_string(heur_name[0], heur_name[2]) + ' relative error (on this snapshot)=' + str(round(error,4)))
    
    i += 1

# Add a common colorbar
cbar = grid.cbar_axes[0].colorbar(im)
cbar.ax.tick_params(labelsize=14)

# Adjust layout to prevent clipping of colorbars
plt.tight_layout()

# Show the plot
plt.show()