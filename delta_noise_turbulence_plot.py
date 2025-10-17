"""
Author: Lev Kakasenko
Description:
Computes and plots the delta_noise component of the risk premium,
along with its upper bound, on the turbulence training data with respect to the number of modes.

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
from heuristics import cpqr_select, dopt_greedy_select

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

# compute the prior for greedy D-optimal sensor placement (i.e. when number of modes equals number of sensors)
Gamma_prior_sp = generate_prior(prior_type=prior, 
                                s_vals=s, 
                                num_samples=X_train.shape[1], 
                                num_modes=num_sensors)
Gamma_prior_sp_sqrt = np.sqrt(Gamma_prior_sp) # ONLY VALID FOR DIAGONAL MATRICES

# randomly select sensor locations
U_s = U[:, :num_sensors]
random_indices = np.random.choice(num_features, size=num_sensors, replace=False)
random_indices.sort()
cpqr_indices = cpqr_select(U=U_s, 
                           num_sensors=num_sensors, 
                           Gamma_prior_sqrt=np.eye(num_sensors))
cpqr_indices.sort()
greedyDopt_indices = dopt_greedy_select(U=U_s, 
                                        sigma_noise=sigma_noise, 
                                        Gamma_prior_sqrt=Gamma_prior_sp_sqrt, 
                                        num_sensors=num_sensors)
greedyDopt_indices.sort()
qmap_indices = cpqr_select(U=U_s, 
                           num_sensors=num_sensors, 
                           Gamma_prior_sqrt=Gamma_prior_sp_sqrt)
qmap_indices.sort()

# initialize lists
delta_noise_random_lst = []
delta_noise_ub_random_lst = []
delta_noise_cpqr_lst = []
delta_noise_ub_cpqr_lst = []
delta_noise_greedyDopt_lst = []
delta_noise_ub_greedyDopt_lst = []
delta_noise_qmap_lst = []
delta_noise_ub_qmap_lst = []

# compute the risk premium components and their upper bounds for each mode in mode_range
for num_modes in mode_range:
    U_r = U[:, :num_modes]

    # compute the prior used in the risk premium
    Gamma_prior = generate_prior(prior_type=prior, s_vals=s,
                                 num_samples=X_train.shape[1], num_modes=num_modes)
    Gamma_prior_inv = np.diag(1 / np.diag(Gamma_prior)) # ONLY VALID FOR DIAGONAL MATRICES

    # compute the risk premium components and their upper bounds
    _, delta_noise_random, _, delta_noise_ub_random = risk_prem(U_r=U_r, sensor_indices=random_indices, Gamma_prior=Gamma_prior, 
                                                  Gamma_prior_inv=Gamma_prior_inv, sigma_noise=sigma_noise)
    _, delta_noise_cpqr, _, delta_noise_ub_cpqr = risk_prem(U_r=U_r, sensor_indices=cpqr_indices, Gamma_prior=Gamma_prior, 
                                                  Gamma_prior_inv=Gamma_prior_inv, sigma_noise=sigma_noise)
    _, delta_noise_greedyDopt, _, delta_noise_ub_greedyDopt = risk_prem(U_r=U_r, 
                                                                        sensor_indices=greedyDopt_indices, 
                                                                        Gamma_prior=Gamma_prior, 
                                                                        Gamma_prior_inv=Gamma_prior_inv, 
                                                                        sigma_noise=sigma_noise)
    _, delta_noise_qmap, _, delta_noise_ub_qmap = risk_prem(U_r=U_r, 
                                                            sensor_indices=qmap_indices, 
                                                            Gamma_prior=Gamma_prior, 
                                                            Gamma_prior_inv=Gamma_prior_inv, 
                                                            sigma_noise=sigma_noise)
    
    # append the computed quantities to their respective lists
    delta_noise_random_lst.append(delta_noise_random)
    delta_noise_ub_random_lst.append(delta_noise_ub_random)
    delta_noise_cpqr_lst.append(delta_noise_cpqr)
    delta_noise_ub_cpqr_lst.append(delta_noise_ub_cpqr)
    delta_noise_greedyDopt_lst.append(delta_noise_greedyDopt)
    delta_noise_ub_greedyDopt_lst.append(delta_noise_ub_greedyDopt)
    delta_noise_qmap_lst.append(delta_noise_qmap)
    delta_noise_ub_qmap_lst.append(delta_noise_ub_qmap)

# plot the results
plt.figure(dpi=200)
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['text.usetex'] = True
plt.ticklabel_format(axis='y', style='sci', scilimits=(4,4))
plt.axvline(x=num_sensors, color='black', linestyle='--') # plot vertical dashed axis
plt.text(num_sensors, 50*(10**4), f'sensors = {num_sensors}', fontsize=14, 
        horizontalalignment='center', color='black') # 21.2*(10**4)  #186

plt.semilogy(mode_range, delta_noise_random_lst, 
             label=r'$\delta_{\textrm{noise}}$ (random)', color='black') 
plt.semilogy(mode_range, delta_noise_ub_random_lst, 
             label=r'$\delta_{\textrm{noise}}$ u.b. (random)', linestyle='--', color='black') 

alpha = .75
plt.semilogy(mode_range, delta_noise_cpqr_lst, alpha=1,
             label=r'$\delta_{\textrm{noise}}$ (CPQR)', color='darkgoldenrod') 
plt.semilogy(mode_range, delta_noise_ub_cpqr_lst, alpha=1,
             label=r'$\delta_{\textrm{noise}}$ u.b. (CPQR)', linestyle='--', color='darkgoldenrod')
plt.semilogy(mode_range, delta_noise_greedyDopt_lst, alpha=alpha,
             label=r'$\delta_{\textrm{noise}}$ (Greedy D-opt.)', color='green') 
plt.semilogy(mode_range, delta_noise_ub_greedyDopt_lst, alpha=alpha,
             label=r'$\delta_{\textrm{noise}}$ u.b. (Greedy D-opt.)', linestyle='--', color='green')
plt.semilogy(mode_range, delta_noise_qmap_lst, alpha=alpha,
             label=r'$\delta_{\textrm{noise}}$ (Q-MAP)', color='blue') 
plt.semilogy(mode_range, delta_noise_ub_qmap_lst, alpha=alpha,
             label=r'$\delta_{\textrm{noise}}$ u.b. (Q-MAP)', linestyle='--', color='blue')  

plt.xlabel('Number of modes', fontsize=14)
plt.ylabel(r'$\delta_{\textrm{noise}}$', fontsize=14)
plt.tick_params(axis='both', which='major', labelsize=14)
# plt.xticks(range(0, 21, 5))
plt.legend(fontsize=14, loc='upper right', bbox_to_anchor=(2, 1))
plt.show()