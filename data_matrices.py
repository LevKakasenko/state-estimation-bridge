"""
Author: Lev Kakasenko
Description:
Generates or loads in matrices of data from specified distributions/processes.  
Each column of a 'data matrix' corresponds to a single sample (i.e. snapshot)
of data.

If you use this code in any form, please cite "Bridging the Gap Between Deterministic and
Probabilistic Approaches to State Estimation" by Lev Kakasenko,  Alen Alexanderian,
Mohammad Farazmand, and Arvind Krishna Saibaba (2025)
"""

import numpy as np
from numpy import sin, pi
import pickle

def add_gaussian_noise(X_test: np.ndarray, stdev: float) -> np.ndarray:
    """
    Adds uncorrelated Gaussian noise to data (to simulate measurement noise).

    Parameters:
        X_test (ndarray): The matrix to which measurement noise should be added.
        stdev (float): The measurement noise standard deviation.
    
    Returns:
        The original matrix X_test with added measurement noise.
    """
    return X_test + np.random.randn(X_test.shape[0], X_test.shape[1]) * stdev


def generate_data_matrix(data_matrix: str, num_features: int, num_samples: int) -> np.ndarray:
    """
    Generates a matrix of data, where each matrix column corresponds to a single
    data sample.

    Parameters:
        data_matrix (str): Type of data matrix to be generated ('fourier', 'turbulence').
        num_features (int): Number of rows (i.e. features) of the data matrix.
        num_samples (int): Number of samples (i.e. columns) of the data matrix.
    
    Returns:
        x (ndarray): A matrix of data of the specified type, with the specified number of rows 
                     (i.e. features) and columns (i.e. samples).
    """

    if data_matrix == 'fourier':
        x = fourier(num_features, num_samples)
    elif data_matrix == 'turbulence':
        x = turbulence()
    else:
        raise Exception("Invalid data_matrix")
    return x


def sine_sum(x: float, J = 20, L = 2*pi, gap_loc = 20) -> np.ndarray:
    """
    This function returns the sum of sine functions, where the amplitude of each sine 
    function is random, the period is deterministic, and the phase is random.

    Parameters:
        x (ndarray): Array of angles.
        J (int): Number of sine terms to sum over.
        L (float): Period of the sine functions.
        gap_loc (int): Index of the last mode before the 1st spectral gap.

    Returns:
        y (float): Sum of all the sine terms.
    """

    y = 0
    for j in range(1, J + 1):
        if j <= gap_loc / 2: #This condition introduces a spectral gap.
            y += np.random.randn() * (1 / j) * sin(2 * pi * j * x / L + np.random.uniform(0, 2*pi))
        else:
            y += np.random.randn() * (1 / j**3) * sin(2 * pi * j * x / L + np.random.uniform(0, 2*pi))
    return y


def fourier(num_angles: int, num_samples: int) -> np.ndarray:
    """
    Generates a data matrix of sine waves with a spectral gap added.

    Parameters:
        num_angles (int): Number of angles over which function is evaluated. Equal to the 
                        number of rows of the data matrix.
        num_samples (int): Number of realizations of the function with random parameters.  
                        Equal to the number of columns of the data matrix.

    Returns:
        x (ndarray): Data matrix of harmonic waves (where each column is a sample).
    """
    x = np.empty(shape=(0, num_angles))
    angles = np.linspace(0, 2 * np.pi, num_angles, endpoint=False)
    for _ in range(num_samples):
        x_new = sine_sum(angles)
        x = np.vstack((x, x_new))
    x = x.T

    return x


def turbulence() -> np.ndarray:
    """
    Loads in the turbulence data.

    Returns:
        data (ndarray): The turbulence data as a matrix (with each column corresponding to a 
                        snapshot over a discretized grid).
    """
    with open('2DTurbData/2DTurbData.pkl', 'rb') as file:
        data = pickle.load(file)
    return data
