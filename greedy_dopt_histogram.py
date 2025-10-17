"""
Author: Lev Kakasenko
Description:
Plots a histogram of the information gain (computed using Eq. (46))
over all possible sensor permutations, in addition to vertical lines representing the greedy 
information gain, maximum attained information gain, (1-1/e)*(maximum attained information gain),
and the additional upper and lower bounds from Eq. (49) of the paper.

If you use this code in any form, please cite "Bridging the Gap Between Deterministic and
Probabilistic Approaches to State Estimation" by Lev Kakasenko,  Alen Alexanderian,
Mohammad Farazmand, and Arvind Krishna Saibaba (2025)
"""
import random
import numpy as np
import matplotlib.pyplot as plt
import math
from tqdm import tqdm
from itertools import combinations
from heuristics import dopt_greedy_select, cpqr_select
from data_matrices import generate_data_matrix, add_gaussian_noise
from utils import evenly_spaced_rows

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
f = 2 # parameter of sRRQR
seed = 0 # random seed


# set the seed for randomly generated values
random.seed(seed)
np.random.seed(seed)

# Generate the matrix of train and test data, where each column corresponds to a randomly generated 
# harmonic function evaluated on equally spaced points between 0 and 2*pi.
x_full = generate_data_matrix(data_matrix, num_features_full, num_samples)
x = evenly_spaced_rows(x_full, num_features)
X_train = x[:, :split_idx]

# center the data
row_means = np.mean(X_train, axis=1)
X_train -= row_means[:, np.newaxis]

# generate the modal basis, prior covariance matrix, and matrices derived from the 
# prior covariance matrix
U, s__X_train, Vh = np.linalg.svd(X_train, full_matrices=True)
U_r = U[:, :num_modes]
Sigma__X_train = np.diag(s__X_train[:num_modes])
Gamma_prior = (Sigma__X_train**2) / (Vh.shape[1] - 1)
Gamma_prior_sqrt = np.sqrt(Gamma_prior) # ONLY VALID FOR DIAGONAL MATRICES

# Compute the lower and upper bounds from Eq. (42).
G = Gamma_prior_sqrt @ U_r.T / sigma_noise
s__G = np.linalg.svd(G, compute_uv=False)
Sigma__G = np.diag(s__G[:num_sensors])
q_cpqr = np.sqrt(num_features - num_sensors) * (2**num_sensors)
q_srrqr = np.sqrt(1+(f**2)*num_sensors*(num_features-num_sensors))
I_sensors = np.eye(num_sensors)
lower_bound_qmap_cpqr = np.log(np.linalg.det(I_sensors + (Sigma__G / q_cpqr)**2))
lower_bound_qmap_srrqr = np.log(np.linalg.det(I_sensors + (Sigma__G / q_srrqr)**2))
upper_bound_qmap = np.log(np.linalg.det(I_sensors + Sigma__G**2))

# Select the greedy D-optimal and Q-MAP sensor locations.
idx_dopt_greedy = dopt_greedy_select(U=U_r, sigma_noise=sigma_noise, 
                                     Gamma_prior_sqrt=Gamma_prior_sqrt, 
                                     num_sensors=num_sensors)
idx_qmap = cpqr_select(U=U_r, num_sensors=num_sensors, Gamma_prior_sqrt=Gamma_prior_sqrt)

if set(idx_dopt_greedy) == set(idx_qmap):
    print('The sensor locations of the greedy D-optimal and Q-MAP algorithms are identical.')
else:
    print('The sensor locations of the greedy D-optimal and Q-MAP algorithms are NOT identical.')

F_dopt_greedy = U_r[list(idx_dopt_greedy)] @ Gamma_prior_sqrt
H_tilde_dopt_greedy = (sigma_noise**(-2))*(F_dopt_greedy.T @ F_dopt_greedy)

# Compute the information gain of the greedy D-optimal and Q-MAP sensor locations.
info_gain_dopt_greedy = np.log(np.linalg.det(H_tilde_dopt_greedy + np.identity(num_modes)))

# Compute the information gain of all possible sensor permutations, including the
# maximum possible information gain.
idx_lst = list(range(num_features))
idx_select_lst = list(combinations(idx_lst, num_sensors))
info_gain_lst = []
max_info_gain = np.nan
I_modes = np.identity(num_modes)

for idx in tqdm(idx_select_lst):
    F = U_r[list(idx)] @ Gamma_prior_sqrt
    H_tilde = (sigma_noise**(-2))*(F.T @ F)
    info_gain = np.log(np.linalg.det(H_tilde + I_modes))
    info_gain_lst.append(info_gain)

    if np.isnan(max_info_gain):
        max_info_gain = info_gain
    elif info_gain > max_info_gain:
        max_info_gain = info_gain

# Print the lower bounds on the Q-MAP information gain (using CPQR and sRRQR).
print('Lower bound on Q-MAP information gain (using CPQR): ' + str(round(lower_bound_qmap_cpqr,2)))
print(f'Lower bound on Q-MAP information gain (using sRRQR, f={f}): ' + str(round(lower_bound_qmap_srrqr,2)))

# Print the greedy D-optimal information gain.
print('Greedy D-optimal information gain: ' + str(round(info_gain_dopt_greedy,2)))

# Print the maximum and minimum information gain (over all sensor permutations).
min_info_gain = min(info_gain_lst)
print('Minimum attained information gain: ' + str(round(min_info_gain,2)))
print('Maximum attained information gain: ' + str(round(max_info_gain,2)))

# Print the lower bound on the greedy D-optimal information gain.
lower_bound_dopt_greedy = (1-1/math.e)*max_info_gain
print('Lower bound on greedy D-optimal information gain (computed as (1-1/e)*(max. attained info. gain)): ' + str(round(lower_bound_dopt_greedy,2)))

# Print the percentile that the greedy D-optimal information gain falls into 
# (over all sensor permutations).
greedy_dopt_perc = 1 - sum([i > info_gain_dopt_greedy for i in info_gain_lst]) / len(info_gain_lst)
print('Greedy D-Optimal information gain percentile (over all sensor permutations): ' +
      str(round(greedy_dopt_perc*100,2)) + '%')

# Create histogram
plt.figure(dpi=200)
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['text.usetex'] = True

plt.hist(info_gain_lst, bins=50, density=True, edgecolor='black') # edgecolor='black'

# Plot vertical lines for D-optimal and greedy D-optimal sensor permutations, along
# with the lower bound on the greedy D-optimal solution.
plt.axvline(x=lower_bound_dopt_greedy, color='magenta', linestyle='--', linewidth=1.5, label=r'$(1-1/e)J_D(\xi_{\textrm{D-opt}})$')
plt.axvline(x=info_gain_dopt_greedy, color='red', linestyle='--', linewidth=1.5, label=r'$J_D(\tilde{\xi}_{\textrm{greedy D-opt}})$')
plt.axvline(x=max_info_gain, color='black', linestyle='--', linewidth=1.5, label=r'$J_D(\xi_{\textrm{D-opt}})$')
#plt.axvline(x=lower_bound_qmap_cpqr, color='darkgoldenrod', linestyle='--', linewidth=1.5, label=r'$\log{ \det{ (I + \frac{1}{q(N,k)^2} \Sigma_k^2} )}$ (CPQR)')
#plt.axvline(x=lower_bound_qmap_srrqr, color='purple', linestyle='--', linewidth=1.5, label=r'$\log{ \det{ (I + \frac{1}{q(N,k)^2} \Sigma_k^2} )}$ (sRRQR; ' + f'$f=${f})')
#plt.axvline(x=upper_bound_qmap, color='green', linestyle='--', linewidth=1.5, label=r'$\log{ \det{ (I + \Sigma_k^2} )}$')

# Add labels and title
plt.xlabel(r'$J_D(\xi)$', fontsize=14)
plt.ylabel('Density')
plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.35), ncol=3)
plt.yticks(np.arange(0, .9, step=0.2))
plt.tight_layout()

# Show plot
plt.show()
