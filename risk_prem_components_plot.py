"""
Author: Lev Kakasenko
Description:
Computes and plots the delta_prior and delta_noise components of the risk premium,
along with their respective upper bounds, with respect to the number of modes.

If you use this code in any form, please cite "Bridging the Gap Between Deterministic and
Probabilistic Approaches to State Estimation" by Lev Kakasenko,  Alen Alexanderian,
Mohammad Farazmand, and Arvind Krishna Saibaba (2025)
"""
import numpy as np
import random
from data_matrices import generate_data_matrix, turbulence, add_gaussian_noise
from priors import generate_prior
from risk_prem import risk_prem
import matplotlib.pyplot as plt

# Parameters
# (a full description of these parameters can be found in compute_error.py)
data_matrix = 'turbulence'
num_features = turbulence().shape[0]
num_samples = turbulence().shape[1]
noise = .3
sigma_noise = noise
split_idx = 750
mode_range = range(1, 101)
num_sensors = 25
prior = 'natural'
seed = 0


# set the seed for randomly generated values
random.seed(seed)
np.random.seed(seed)

# generate the data matrix
x = generate_data_matrix(data_matrix, num_features, num_samples)

# split the data into train and test sets
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

# randomly select sensor locations
sensor_indices = np.random.choice(num_features, size=num_sensors, replace=False)
sensor_indices.sort()

# initialize lists
delta_prior_lst = []
delta_noise_lst = []
delta_prior_ub_lst = []
delta_noise_ub_lst = []

# compute the risk premium components and their upper bounds for each mode in mode_range
for num_modes in mode_range:
    U_r = U[:, :num_modes]

    # compute the prior
    Gamma_prior = generate_prior(prior_type=prior, s_vals=s,
                                 num_samples=X_train.shape[1], num_modes=num_modes)
    Gamma_prior_inv = np.diag(1 / np.diag(Gamma_prior)) # ONLY VALID FOR DIAGONAL MATRICES
    Gamma_prior_sqrt = np.sqrt(Gamma_prior) # ONLY VALID FOR DIAGONAL MATRICES

    # compute the risk premium components and their upper bounds
    delta_prior, delta_noise, delta_prior_ub, delta_noise_ub = risk_prem(U_r=U_r, sensor_indices=sensor_indices, Gamma_prior=Gamma_prior, 
                                                                Gamma_prior_inv=Gamma_prior_inv, sigma_noise=sigma_noise)
    
    # append the computed quantities to their respective lists
    delta_prior_lst.append(delta_prior)
    delta_noise_lst.append(delta_noise)
    delta_prior_ub_lst.append(delta_prior_ub)
    delta_noise_ub_lst.append(delta_noise_ub)

# plot the results
plt.figure(dpi=200)
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['text.usetex'] = True
#plt.ticklabel_format(axis='y', style='sci', scilimits=(4,4))
plt.axvline(x=num_sensors, color='black', linestyle='--') # plot vertical dashed axis
plt.text(num_sensors, 21.6*(10**4), f'sensors = {num_sensors}', fontsize=14, 
        horizontalalignment='center', color='black') # 21.2*(10**4)  #186
plt.plot(mode_range, delta_prior_lst, label=r'$\delta_{\textrm{prior}}$', color='blue') 
         #marker='o', markersize=4)
plt.plot(mode_range, delta_prior_ub_lst, label=r'$\delta_{\textrm{prior}}$ u.b.', 
         linestyle='--', color='blue')
plt.plot(mode_range, delta_noise_lst, label=r'$\delta_{\textrm{noise}}$', color='darkorange') 
         #marker='o', markersize=4)
plt.plot(mode_range, delta_noise_ub_lst, label=r'$\delta_{\textrm{noise}}$ u.b.', 
         linestyle='--', color='saddlebrown')
plt.xlabel('Number of modes', fontsize=14)
plt.ylabel('Risk prem. components', fontsize=14)
plt.tick_params(axis='both', which='major', labelsize=14)
# plt.xticks(range(0, 21, 5))
plt.legend(fontsize=14, loc='upper right')
plt.show()
