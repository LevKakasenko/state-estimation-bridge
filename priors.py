"""
Author: Lev Kakasenko
Description:
Contains the function that generates prior covariance matrices.

If you use this code in any form, please cite "Bridging the Gap Between Deterministic and
Probabilistic Approaches to State Estimation" by Lev Kakasenko,  Alen Alexanderian,
Mohammad Farazmand, and Arvind Krishna Saibaba (2025)
"""

import numpy as np

def generate_prior(prior_type: str, s_vals: np.ndarray, num_samples: int, num_modes: int):
    """
    Generates the prior covariance matrix (i.e. Gamma_prior) of 
    the inversion parameter with respect to the basis of POD modes.

    Parameters:
        prior_type (str): The type of prior to generate 
                          ('natural': computed using a change of basis of the sample covariance matrix,
                          'identity': the identity matrix)
        s_vals (ndarray): The 1-dimensional array of singular values sorted in 
                          descending order.
        num_samples (int): The number of samples used to generate the training data
                           that generates Gamma_prior (which should be set to the 
                           number of columns of X_train).
        num_modes (int): The number of modes in the modal basis (which determines 
                         the number of rows/columns of Gamma_prior).
    
    Returns:
        Gamma_prior (ndarray): The prior covariance matrix.
    """
    Sigma = np.diag(s_vals[:num_modes])
    if prior_type == 'natural':
        Gamma_prior = (Sigma**2) / (num_samples - 1)
    elif prior_type == 'identity':
        Gamma_prior = np.eye(num_modes)
    else:
        raise Exception("Invalid prior_type.")
    return Gamma_prior
