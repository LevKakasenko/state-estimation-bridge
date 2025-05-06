"""
Author: Lev Kakasenko
Description:
Computes the reconstruction error of different pairs of sensor placement 
heuristics (Greedy D-Optimal, CPQR, etc.) and reconstructors (MAP or DEIM) with respect 
to the number of sensors or modes. Output is stored in the logs file.

If you use this code in any form, please cite "Bridging the Gap Between Deterministic and
Probabilistic Approaches to State Estimation" by Lev Kakasenko,  Alen Alexanderian,
Mohammad Farazmand, and Arvind Krishna Saibaba (2025)
"""

import warnings
import numpy as np
import random

from utils import check_logs, save_log
from data_matrices import generate_data_matrix, turbulence, add_gaussian_noise
from reconstructors import reconstruct
from heuristics import sensor_select, greedy_sensor_select
from priors import generate_prior
from errors import error_rel_columnwise

### Parameters
data_matrix = 'fourier' # Dataset to use, which can be set to:
                        #   'fourier' (the Fourier dataset used in Section 5.1)
                        #   'turbulence' (the turbulence dataset used in Section 5.2)
num_features = 40  # Number of features in a single data sample.
                   # Each feature corresponds to another grid point on which a function is evaluated.
                   # Set to turbulence().shape[0] for the turbulence data.
num_samples = 1000 # Number of data samples (i.e. snapshots) for training and testing.
                   # Set to turbulence().shape[1] for the turbulence data.
noise = .1   # Standard deviation of uncorrelated Gaussian noise added to the test data.
sigma_noise = noise # Model parameter representing a measurement noise standard deviation.
split_idx = 750 # The index at which the data matrix columns are split to generate the train and 
                # test matrices.  If, for example, this index is set to 750, then the first 750
                # samples are used for training, and the remaining samples are used for testing.
mode_range = range(1, 21) # Range of number of POD modes over which to evaluate.
sensor_range = range(5, 6) # Range of number of sensors over which to evaluate.
# Note: For cpqr_fair, if the number of modes (sensors) is varied and sensors (modes) held constant,
#       then sensor_range (mode_range) is irrelevant.  In the case of cpqr_fair, if mode_range 
#       (sensor_range) has a length greater than 1, then sensor_range (mode_range) should have a 
#       length equal to 1.
heur = 'aopt'
        # The heur parameter specifies the sensor placement algorithm (i.e. heuristic), and can be set 
        # to any of the following values: 
        #   'cpqr' (the CPQR algorithm)
        #   'cpqr_fair' (the CPQR algorithm, with the number of modes set equal to the number of sensors)
        #   'dopt' (the brute-force D-optimal algorithm)
        #   'dopt_greedy' (the greedy D-optimal algorithm using Algorithm 1)
        #   'aopt' (the brute-force A-optimal algorithm)
        #   'aopt_greedy' (the greedy A-optimal algorithm using Algorithm 1)
recons = 'map'
        # The recons parameter specifies the reconstruction formula to use, and can be set to
        # any of the following values:
        #   'map' (the MAP reconstructor)
        #   'deim' (the DEIM reconstructor)
prior = 'natural'
        # The method for computing the prior covariance matrix used to place sensors and 
        # reconstruct the full state; set prior_type to any of the following values:
        #   'natural' (computes the prior using the method we propose in Section 3.1)
        #   'identity' (sets the prior to the identity matrix)
seed = 0 # Random seed.
         # Set to 0 for the turbulence data (as an arbitrary default value since the data isn't 
         # randomly generated).


save_interval = 1 # How frequently the log file should be updated. 
                  # Setting this parameter too small can result in an EOFError.
                  # Set to 1 when computations are slow.

### Explanation of cpqr_fair:
###########################################################################################
# When using the cpqr_fair condition, the number of sensors always equals the
# number of modes for CPQR sensor placement and DEIM reconstruction.
# If mode_range (sensor_range) is of length 1, while sensor_range (mode_range)
# if of length > 1, then we vary the number of sensors and modes based
# on sensor_range (mode_range) while disregarding mode_range (sensor_range).  
# For example, if mode_range=range(2,21) and 
# sensor_range=range(5,6), then we disregard the sensor_range parameter and 
# compute the Q-DEIM reconstruction for (number of sensors)=(number of modes)=2, 
# (number of sensors)=(number of modes)=3, ..., (number of sensors)=(number of modes)=20.
###########################################################################################


# metadata that uniquely identifies the log file
metadata = {'num_features': num_features, 'num_samples': num_samples, 'split_idx': split_idx,
            'heur': heur, 'reconstruction': recons, 
            'data_matrix': data_matrix, 'prior_type': prior, 
            'noise': noise, 'sigma_noise': sigma_noise, 'seed': seed}

# Checks the log files in logs_error to determine whether the log with specified metadata 
# already exists; creates new log if log doesn't already exist.
index, log, total_tasks, mode_list, modes_sensors \
    = check_logs(metadata, mode_range, sensor_range)

# set the seed for randomly generated values
random.seed(seed)
np.random.seed(seed)

# determine whether modes (or sensors) are varied in the case of 'cpqr_fair'
modes_varied = False
if heur == "cpqr_fair":
    if len(sensor_range) == 1:
        modes_varied = True
    elif len(mode_range) == 1:
        modes_varied = False
    else:
        raise Exception("Invalid variation of modes/sensors for cpqr_fair.")


# Perform computations.
with warnings.catch_warnings():
    warnings.simplefilter("ignore")

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
    
    # compute the noise error (which can optionally be printed)
    noise_error, _ = error_rel_columnwise(X_test, X_test_obs)

    # perform SVD on training data
    U, s, Vh = np.linalg.svd(X_train, full_matrices=True)
    
    # initialize save_idx for saving the log at given intervals (and for progress bar)
    save_idx = 0

    for num_modes in mode_list:
        # generate list of sensors based on num_modes
        sensor_list = [combo[1] for combo in modes_sensors if num_modes==combo[0]]

        # generate a matrix with the allowed number of POD modes (as columns)
        U_r = U[:, :num_modes]

        # generate matrices for the OED approach (namely the prior)
        Gamma_prior = generate_prior(prior_type=prior, s_vals=s,
                                     num_samples=X_train.shape[1], num_modes=num_modes)
        Gamma_prior_inv = np.diag(1 / np.diag(Gamma_prior)) # ONLY VALID FOR DIAGONAL MATRICES
        Gamma_prior_sqrt = np.sqrt(Gamma_prior) # ONLY VALID FOR DIAGONAL MATRICES
        
        # generate list of sensor indices (i.e. locations) for the greedy heuristics
        max_sensors = max(sensor_list)
        greedy_sensor_list = greedy_sensor_select(heur=heur, U_r=U_r, num_sensors=max_sensors, 
                                                  Gamma_prior_sqrt=Gamma_prior_sqrt, 
                                                  Gamma_prior_inv=Gamma_prior_inv,
                                                  sigma_noise=sigma_noise)

        for num_sensors in sensor_list:
            # select the sensor indices (i.e. locations)
            sensor_indices = sensor_select(heur=heur, U=U, U_r=U_r, num_sensors=num_sensors, 
                                num_modes=num_modes, Gamma_prior_sqrt=Gamma_prior_sqrt,
                                Gamma_prior_inv=Gamma_prior_inv, sigma_noise=sigma_noise, 
                                greedy_sensor_list=greedy_sensor_list,
                                modes_varied=modes_varied)
            
            # generate reconstructions
            X_test_hat = reconstruct(reconstruction_type=recons, heur=heur, 
                                     num_sensors=num_sensors, num_modes=num_modes, X_test=X_test_obs,
                                     sensor_indices=sensor_indices, U=U, U_r=U_r, 
                                     Gamma_prior_inv=Gamma_prior_inv, 
                                     sigma_noise=sigma_noise, modes_varied=modes_varied)

            # compute reconstruction error
            error_mean, error_var = error_rel_columnwise(X_test, X_test_hat)

            # append result to log with the index (num_modes, num_sensors)
            if heur == "cpqr_fair":
                if modes_varied:
                    log[(num_modes, num_modes)] = (sensor_indices, error_mean, error_var)
                if not modes_varied:
                    log[(num_sensors, num_sensors)] = (sensor_indices, error_mean, error_var)
            else:
                log[(num_modes, num_sensors)] = (sensor_indices, error_mean, error_var)

            # save data to file at at each save_interval
            if save_idx % save_interval == 0:
                save_log(log, index)
            
            save_idx += 1

            # display progress
            progress_percent = round(save_idx/total_tasks,4)*100
            print(f"Progress: {save_idx}/{total_tasks}, ({progress_percent}%)")
    
    # save data for the final time
    save_log(log, index)
