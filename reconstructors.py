"""
Author: Lev Kakasenko
Description:
Provides different formulas for reconstructing data samples from 
partial, noisy observations.

If you use this code in any form, please cite "Bridging the Gap Between Deterministic and
Probabilistic Approaches to State Estimation" by Lev Kakasenko,  Alen Alexanderian,
Mohammad Farazmand, and Arvind Krishna Saibaba (2025)
"""
import numpy as np

def reconstruct(reconstruction_type: str, heur: str, num_sensors: int, num_modes: int, 
                X_test: np.ndarray, sensor_indices: int, U: np.ndarray, U_r: np.ndarray, 
                Gamma_prior_inv: np.ndarray, sigma_noise: float, modes_varied: bool):
    """
    Reconstructs test data from partial, noisy observations.

    Parameters:
        reconstruction_type (str): Type of reconstruction to use (deim, map).
        heur (str): Sensor placement heuristic (relevant in the case of cpqr_fair).
        num_sensors (int): Number of sensors with which to reconstruct.
        num_modes (int): Number of modes with which to reconstruct.
        X_test (ndarray): Matrix of test data to be reconstructed, with each column 
                          representing a data sample/snapshot.
        sensor_indices (int): List of indices of X_test that are observable 
                              (i.e. the sensor locations).
        U (ndarray): The modal basis.
        U_r (ndarray): The modal basis truncated to the first r columns.
        Gamma_prior_inv (ndarray): The inverse of the prior covariance matrix.
        sigma_noise (float): The magnitude of a measurement noise standard deviation.
        modes_varied (bool): Whether the modes are being varied (used in the case of 
                             cpqr_fair to ensure that modes and sensors are equal).
    
    Returns:
        The reconstruction X_test_test of the noisy, partially-observed set of 
        data samples X_test.
    """
    if reconstruction_type == 'deim' and heur == 'cpqr_fair':
        # switches between num_modes (when varying modes) & num_sensors (when varying sensors)
        if modes_varied:
            X_test_hat = deim(X_test, sensor_indices, U[:, :num_modes]) 
        elif not modes_varied:
            X_test_hat = deim(X_test, sensor_indices, U[:, :num_sensors])
    elif reconstruction_type == 'deim':
        X_test_hat = deim(X_test, sensor_indices, U_r)
    elif reconstruction_type == 'map':
        X_test_hat = map(X_test, sensor_indices, U_r, sigma_noise, Gamma_prior_inv)
    
    return X_test_hat


def deim(X_test: np.ndarray, idx: int, U: np.ndarray):
    """
    Uses the discrete empirical interpolation method (DEIM) formula
    as proposed in "Nonlinear Model Reduction via Discrete Empirical Interpolation"
    by Saifon Chaturantabut and Danny C. Sorensen (2010)
    to estimate the inversion parameter, which is then used to return
    the reconstructed full-state (X_test).

    Parameters:
        X_test (ndarray): Test data to be reconstructed.
        idx (int): List of sensor locations.
        U (ndarray): Matrix of POD modes.

    Returns:
        X_test_hat (ndarray): DEIM reconstruction of X_test.
    """
    
    X_test_hat = U @ (np.linalg.pinv(U[idx]) @ (X_test[idx]))
    return X_test_hat


def map(X_test: np.ndarray, idx: int, U: np.ndarray, sigma_noise: float, 
               Gamma_prior_inv: np.ndarray):
    """
    Finds the maximum-a-posteriori (MAP) estimate of the inversion parameter, 
    which is then used to reconstruct the full-state (X_test).

    Parameters:
        X_test (ndarray): Test data to be reconstructed.
        idx (int): List of sensor locations.
        U (ndarray): Matrix of POD modes.
        sigma_noise (float): The magnitude of a measurement noise standard deviation.
        Gamma_prior_inv (ndarray): Inverse of the prior covariance matrix.

    Returns:
        X_test_hat (ndarray): MAP reconstruction of X_test.
    """

    C = np.identity(X_test.shape[0])[idx]
    y = C @ X_test
    F = C @ U
    H = (sigma_noise**(-2)) * F.T @ F + Gamma_prior_inv
    k = (sigma_noise**(-2)) * F.T @ y
    m_map = np.linalg.solve(H, k)
    X_test_hat = U @ m_map
    return X_test_hat
