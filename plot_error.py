"""
Author: Lev Kakasenko
Description:
Plots the reconstruction error of different pairs of sensor placement 
heuristics (Greedy D-Optimal, CPQR, etc.) and reconstructors (MAP or DEIM) with 
respect to the number of sensors or modes.  Data is loaded in from the logs file.

If you use this code in any form, please cite "Bridging the Gap Between Deterministic and
Probabilistic Approaches to State Estimation" by Lev Kakasenko,  Alen Alexanderian,
Mohammad Farazmand, and Arvind Krishna Saibaba (2025)
"""

import random
import numpy as np
import matplotlib.pyplot as plt
from utils import heur_to_string, load_data, check_missing_data, \
    generate_heur_key
from data_matrices import turbulence, generate_data_matrix

# Parameters
# (a full description of additional parameters can be found in compute_error.py;
# before running plot_error.py with a set of parameters, you should run compute_error.py 
# with the corresponding set of parameters)
data_matrix = 'fourier'
num_features = 40
num_samples = 1000
noise = .1
sigma_noise = noise
split_idx = 750
mode_range = range(1, 21)
sensor_range = range(5, 6)
plot_type = 'error_vs_modes'
        # Set plot_type to one of the following values:
        #   'error_vs_modes' (plots reconstruction error with respect to the number of modes; use if the length of sensor_range is 1)
        #   'error_vs_sensors' (plots reconstruction error with respect to the number of sensors; use if the length of mode_range is 1)
# The following three parameters (heuristics, reconstructors, priors) are lists of sensor placement
# heuristics, reconstructors of the full state, and prior formulas.  All three lists should be of 
# the same length, with index i of the lists corresponding to a single combination of heuristic, 
# reconstructor, and prior to be plotted.  For example, if heuristics[0] is 'dopt_greedy', 
# reconstructors[0] is 'map', and priors[0] is 'natural', then the outputted plot will include
# the reconstruction error of Greedy D-MAP with our proposed 'natural' prior (from Section 3.1).
heuristics = ['aopt_greedy', 'aopt']
reconstructors = ['map', 'map']
priors = ['natural', 'natural']
seed = 0 # Random seed.
         # Set to 0 for the turbulence data (as an arbitrary default value since the data isn't 
         # randomly generated).


# Load in data from the logs.
log = load_data(heuristics, reconstructors, priors, plot_type, mode_range, 
                sensor_range, num_features, num_samples, split_idx, 
                data_matrix, noise, sigma_noise, seed)

# Check if there is any missing data.
check_missing_data(log=log, heuristics=heuristics, priors=priors, reconstructors=reconstructors, 
                   plot_type=plot_type, mode_range=mode_range, sensor_range=sensor_range)

# Set the seed for randomly generated values.
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

# Set the plot colors and parameters.
line_colors = ['green', 'darkgoldenrod', 'blue', 'red', 'purple']
color_idx = -1

plt.figure(dpi=200)
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['text.usetex'] = True
# plt.axvline(x=sensor_range[0], color='black', linestyle='--', 
#            alpha=.8, linewidth=2) # plot vertical dashed axis
# plt.text(sensor_range[0], 1.22, f'sensors = {sensor_range[0]}', fontsize=14, 
#         horizontalalignment='center', color='black')
for idx in range(len(heuristics)):
    heur = heuristics[idx]
    prior = priors[idx]
    reconstructor = reconstructors[idx]
    heur_key = generate_heur_key(heur, prior, reconstructor)
    log_heur = log[heur_key]
    
    error_mean_lst = [] # contains the mean of the numerically computed error
    error_var_lst = [] # contains the variance of the numerically computed error

    if plot_type == 'error_vs_modes':
        for modes in mode_range:
            if heur == "cpqr_fair":
                values = log_heur[(modes, modes)]
            else:
                values = log_heur[(modes, sensors)]
            error_mean_lst.append(values[1])
            error_var_lst.append(values[2])
    
    elif plot_type == 'error_vs_sensors':
        for sensors in sensor_range:
            if heur == "cpqr_fair":
                values = log_heur[(sensors, sensors)]
            else:
                values = log_heur[(modes, sensors)]
            error_mean_lst.append(values[1])
            error_var_lst.append(values[2])
    
    error_mean_lst = np.array(error_mean_lst)
    error_std_lst = np.array(error_var_lst)**.5

    # plot results
    heur_name = heur_to_string(heur, reconstructor)

    upper_error = error_mean_lst + error_std_lst
    lower_error = error_mean_lst - error_std_lst

    color_idx = color_idx + 1
    
    if heur_name == 'Q-MAP':
        plt.errorbar(parameter_range, error_mean_lst, error_std_lst, label=heur_name, 
                linestyle=':', marker='*', markersize=7, linewidth=2, capsize=5,
                color=line_colors[color_idx], alpha=1)
    else:
        plt.errorbar(parameter_range, error_mean_lst, error_std_lst, label=heur_name, 
            linestyle='solid', marker='*', markersize=7, linewidth=2, capsize=5,
            color=line_colors[color_idx], alpha=1)


if plot_type == 'error_vs_modes':
    plt.xlabel('Number of modes', fontsize=14)
elif plot_type == 'error_vs_sensors':
    plt.xlabel('Number of sensors', fontsize=14)
    plt.title(f'modes={modes}', style= "italic", fontsize=14)

plt.ylabel('Relative error', fontsize=14)
plt.tick_params(axis='both', which='major', labelsize=14, length=5)
plt.legend()
plt.grid()
plt.show()
