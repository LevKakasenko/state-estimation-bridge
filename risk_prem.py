"""
Author: Lev Kakasenko
Description:
Provides the function for computing the risk premium, its prior and
noise components, and the upper bound on these prior and noise components.

If you use this code in any form, please cite "Bridging the Gap Between Deterministic and
Probabilistic Approaches to State Estimation" by Lev Kakasenko,  Alen Alexanderian,
Mohammad Farazmand, and Arvind Krishna Saibaba (2025)
"""

import numpy as np
from numpy.linalg import pinv, inv, matrix_rank, eigvals
from numpy import trace

def risk_prem(U_r: np.ndarray, sensor_indices: np.ndarray, Gamma_prior: np.ndarray, 
              Gamma_prior_inv: np.ndarray, sigma_noise: float):
    """
    Computes the risk premium components and their respective upper bounds.

    Parameters:
        U_r (ndarray): The modal basis truncated to the first r columns.
        sensor_indices (ndarray): One dimensional array of sensor indices.
        Gamma_prior (ndarray): The prior covariance matrix.
        Gamma_prior_inv (ndarray): The inverse of the prior covariance matrix.
        sigma_noise (float): Model parameter representing a measurement noise standard deviation.

    Return:
        The two risk premium components (delta_prior and delta_noise) and their 
        respective upper bounds (delta_prior_ub and delta_noise_ub).
    """
    A = U_r[sensor_indices]
    A_pinv = pinv(A)
    I = np.eye(Gamma_prior.shape[0])
    Gamma_post = inv(Gamma_prior_inv + (sigma_noise**(-2)) * A.T @ A)
    
    # compute delta_prior, the delta_noise upper bound, and delta_noise
    delta_prior = trace((I - A_pinv @ A) @ (Gamma_prior - Gamma_post))
    delta_noise_ub = trace((sigma_noise**2) * A_pinv @ A_pinv.T)
    delta_noise = delta_noise_ub - trace(A_pinv @ A @ Gamma_post)

    # compute relevant eigenvalues
    evals = eigvals(Gamma_prior - Gamma_post)
    idx = evals.argsort()[::-1]
    evals = evals[idx] # sort in descending order

    # compute the delta_prior upper bound
    n = A.shape[1]
    rank_A = matrix_rank(A)
    nullity_A = n - rank_A
    delta_prior_ub = sum(evals[:nullity_A])

    return delta_prior, delta_noise, delta_prior_ub, delta_noise_ub
