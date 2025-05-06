""" 
Author: Lev Kakasenko
Description:
Contains different heuristic and brute-force algorithms for selecting sensor locations.

If you use this code in any form, please cite "Bridging the Gap Between Deterministic and
Probabilistic Approaches to State Estimation" by Lev Kakasenko,  Alen Alexanderian,
Mohammad Farazmand, and Arvind Krishna Saibaba (2025)
"""
from itertools import combinations
import numpy as np
from scipy.linalg import qr
from errors import error_fro
from reconstructors import deim, map
from tqdm import tqdm


def sensor_select(heur, U, U_r, num_sensors, num_modes, Gamma_prior_sqrt, 
                  Gamma_prior_inv, sigma_noise, greedy_sensor_list, modes_varied):
    """
    Computes the sensor locations given a specified heuristic and other relevant information.

    Parameters:
        heur (str): The name of the heuristic algorithm used to place sensors.
        U (ndarray): The modal basis.
        U_r (ndarray): The modal basis truncated to the first r columns.
        num_sensors (int):  The number of sensors to place.
        num_modes (int):  The number of modes (used in the case of cpqr_fair to ensure
                          that modes and sensors are equal).
        Gamma_prior_sqrt (ndarray): The square root of the prior covariance matrix.
        Gamma_prior_inv (ndarray): The inverse of the prior covariance matrix.
        sigma_noise (float): The magnitude of a measurement noise standard deviation.
        greedy_sensor_list: A 1-dimensional numpy array containing 
                            the list of sensors selected by the heuristic if the heuristic
                            is greedy (where the first entry is the first location selected
                            by the heuristic, the second entry is the second location selected
                            by the heuristic, etc.), to be truncated to a length of num_sensors; 
                            -1 if the heuristic is not greedy.
        modes_varied (bool): Whether the modes are being varied (used in the case of cpqr_fair to ensure
                             that modes and sensors are equal).
    
    Returns:
        The sensor locations at which to take measurements.
    """
    if heur == "cpqr": 
        idx = greedy_sensor_list[:num_sensors]
    elif heur == "cpqr_fair":
        if modes_varied:
            idx = cpqr_select(U[:, :num_modes], num_modes, 
                              Gamma_prior_sqrt[:num_modes, :num_modes])
        elif not modes_varied:
            idx = cpqr_select(U[:, :num_sensors], num_sensors, 
                              Gamma_prior_sqrt[:num_sensors, :num_sensors])
    elif heur == 'dopt':
        idx = dopt_select(U_r, sigma_noise, Gamma_prior_sqrt, num_sensors)
    elif heur == 'dopt_greedy':
        idx = greedy_sensor_list[:num_sensors]
    elif heur == 'aopt':
        idx = aopt_select(U_r, sigma_noise, Gamma_prior_inv, num_sensors)
    elif heur == 'aopt_greedy':
        idx = greedy_sensor_list[:num_sensors]
    else:
        raise Exception("Invalid heuristic")
    
    return idx


def greedy_sensor_select(heur, U_r, num_sensors, Gamma_prior_sqrt, Gamma_prior_inv,
                         sigma_noise):
    """
    Returns a list of sensors selected by a given greedy algorithm.

    Parameters:
        heur (str): The name of the greedy heuristic algorithm used to place sensors.
        U_r (ndarray): The modal basis truncated to the first r columns.
        num_sensors (int):  The number of sensors to place.
        Gamma_prior_sqrt (ndarray): The square root of the prior covariance matrix.
        Gamma_prior_inv (ndarray): The inverse of the prior covariance matrix.
        sigma_noise (float): The magnitude of a measurement noise standard deviation.
    Returns:
        An ordered list of sensors selected by the given greedy algorithm, with the 
        1st list entry being the 1st selected sensor, the 2nd list entry being the 
        2nd selected sensors, etc.
    """
    if heur == "cpqr":
        p = cpqr_select(U_r, num_sensors, Gamma_prior_sqrt)
        return p
    elif heur == 'dopt_greedy':
        p = dopt_greedy_select(U_r, sigma_noise, Gamma_prior_sqrt, num_sensors)
        return p
    elif heur == 'aopt_greedy':
        p = aopt_greedy_select(U_r, sigma_noise, Gamma_prior_inv, num_sensors)
        return p
    return  -1


def optimal_deim_select(U: np.ndarray, X_test: np.ndarray, X_test_obs: np.ndarray,
                        num_sensors: int):
    """
    Determines the optimal sensor locations that minimize the DEIM reconstruction error on 
    the test data via brute-force.

    Parameters:
        U (ndarray): Matrix of POD modes.
        X_test (ndarray): Test data to be reconstructed.
        X_test_obs (ndarray): X_test with added measurement noise.
        num_sensors (int): Number of available sensors.

    Returns:
        idx_opt (tuple): Optimal sensor locations.
        min_error (float): Optimal (i.e. minimized) reconstruction error.
        max_error (float): The maximum reconstruction error encountered during the 
                        brute-force search of all sensor permutations.
        X_test_hat_opt (ndarray): Optimal reconstructed X_test (i.e. matrix that is "closest" to X_test).
    """

    idx_lst = list(range(X_test.shape[0]))
    idx_select_lst = list(combinations(idx_lst, num_sensors))

    error_lst = []
    min_error = np.nan
    idx_opt = []
    X_test_hat_opt = np.nan

    # switch the line below to "for idx in tqdm(idx_select_lst):" for progress bar
    for idx in idx_select_lst:
        X_test_hat = deim(X_test_obs, list(idx), U)
        error = error_fro(X_test, X_test_hat)
        error_lst.append(error)
        if np.isnan(min_error):
            min_error = error
            max_error = error
            idx_opt = idx
            X_test_hat_opt = X_test_hat
        elif error < min_error:
            min_error = error
            idx_opt = idx
            X_test_hat_opt = X_test_hat
        elif error > max_error:
            max_error = error

    min_error = min_error / np.linalg.norm(X_test)
    max_error = max_error / np.linalg.norm(X_test)

    return (list(idx_opt), min_error, max_error, X_test_hat_opt)


def optimal_map_select(U: np.ndarray, X_test: np.ndarray, X_test_obs: np.ndarray,
                       num_sensors: int, sigma_n: float, Gamma_prior_inv: np.ndarray):
    """
    Determines the optimal sensor locations that minimize the 
    MAP reconstruction error on the test data via brute-force.

    Parameters:
        U (ndarray): Matrix of POD modes.
        X_test (ndarray): Test data to be reconstructed.
        X_test_obs (ndarray): X_test with added measurement noise.
        num_sensors (int): Number of available sensors.
        sigma_n (float): Magnitude of a measurement noise standard deviation.
        Gamma_prior_inv (ndarray): Inverse of the prior covariance matrix.

    Returns:
        idx_opt (tuple): Optimal sensor locations.
        min_error (float): Optimal (i.e. minimized) reconstruction error.
        max_error (float): The maximum reconstruction error encountered during the 
                           brute-force search of all sensor permutations.
        X_test_hat_opt (ndarray): Optimal reconstructed X_test (i.e. matrix that is 
                                  "closest" to X_test).
    """

    idx_lst = list(range(X_test.shape[0]))
    idx_select_lst = list(combinations(idx_lst, num_sensors))

    error_lst = []
    min_error = np.nan
    idx_opt = []
    X_test_hat_opt = np.nan

    # switch the line below to "for idx in tqdm(idx_select_lst):" for progress bar
    for idx in idx_select_lst:
        X_test_hat = map(X_test_obs, list(idx), U, sigma_n, Gamma_prior_inv)
        error = error_fro(X_test, X_test_hat)
        error_lst.append(error)
        if np.isnan(min_error):
            min_error = error
            max_error = error
            idx_opt = idx
            X_test_hat_opt = X_test_hat
        elif error < min_error:
            min_error = error
            idx_opt = idx
            X_test_hat_opt = X_test_hat
        elif error > max_error:
            max_error = error

    min_error = min_error / np.linalg.norm(X_test)
    max_error = max_error / np.linalg.norm(X_test)

    return (list(idx_opt), min_error, max_error, X_test_hat_opt)


def cpqr_select(U: np.ndarray, num_sensors: int, Gamma_prior_sqrt: np.ndarray):
    """
    Selects sensor locations based on the column-pivoted QR (CPQR) algorithm
    proposed in "A New Selection Operator for the Discrete Empirical Interpolation 
    Method" by Zlatko Drmač and Serkan Gugercin (2016).  The number of columns
    of the basis must be greater than or equal to num_sensors.

    Parameters:
        U (ndarray): Matrix whose columns are basis vectors of the data (i.e. POD modes).
        num_sensors (int): Number of available sensors.
        Gamma_prior_sqrt (ndarray): Square root of the prior covariance matrix.  Note that this matrix
                                    is always symmetric since it is the square root of a symmetric
                                    positive semi-definite matrix.

    Returns:
        idx_list (ndarray): The sensor locations as determined by the CPQR algorithm.
    """
    
    U = U @ Gamma_prior_sqrt
    _, _, p = qr(U.T, mode='full', pivoting=True)
    idx_list = p[:num_sensors]
    return idx_list


def dopt_select(U: np.ndarray, sigma_noise: float, Gamma_prior_sqrt: np.ndarray, num_sensors: int):
    """
    Selects sensor locations according to the D-optimal criterion.

    Parameters:
        U (ndarray): Matrix whose columns are basis vectors of the data.
        sigma_noise (float): Model parameter representing a measurement noise standard deviation.
        Gamma_prior_sqrt (ndarray): Square root of the prior covariance matrix.
        num_sensors (int): Number of available sensors.
    
    Returns:
        idx_opt (list): List of D-optimal sensor locations as determined by brute-force.
    """
    idx_lst = list(range(U.shape[0]))
    idx_select_lst = list(combinations(idx_lst, num_sensors))

    min_error_heur = np.nan
    idx_opt = []

    # switch line below to "for idx in tqdm(idx_select_lst):" for progress bar
    for idx in idx_select_lst:
        F = U[list(idx)] @ Gamma_prior_sqrt
        H_tilde = (sigma_noise**(-2))*(F.T @ F)
        error = -np.linalg.det(H_tilde + np.identity(U.shape[1]))
        if np.isnan(min_error_heur):
            min_error_heur = error
            idx_opt = idx
        elif error < min_error_heur:
            min_error_heur = error
            idx_opt = idx

    idx_opt = list(idx_opt)
    return idx_opt


def aopt_select(U: np.ndarray, sigma_noise: float, Gamma_prior_inv: np.ndarray, num_sensors: int):
    """
    Selects sensor locations according to the A-optimal criterion.

    Parameters:
        U (ndarray): Matrix whose columns are basis vectors of the data.
        sigma_noise (float): Model parameter representing a measurement noise standard deviation.
        Gamma_prior_inv (ndarray): Inverse of the prior covariance matrix.
        num_sensors (int): Number of available sensors.

    Returns:
        idx_opt (list): List of A-optimal sensor locations as determined by brute-force.
    """
    idx_lst = list(range(U.shape[0]))
    idx_select_lst = list(combinations(idx_lst, num_sensors))

    min_error_heur = np.nan
    idx_opt = []

    # switch line below to "for idx in tqdm(idx_select_lst):" for progress bar
    for idx in idx_select_lst:
        F = U[list(idx)]
        H = (sigma_noise**(-2)) * F.T @ F
        error = np.trace(np.linalg.inv(H + Gamma_prior_inv))
        if np.isnan(min_error_heur):
            min_error_heur = error
            idx_opt = idx
        elif error < min_error_heur:
            min_error_heur = error
            idx_opt = idx

    idx_opt = list(idx_opt)
    return idx_opt


def dopt_greedy_select(U: np.ndarray, sigma_noise: float,
                        Gamma_prior_sqrt: np.ndarray, num_sensors: int):
    """
    Greedily selects D-optimal sensors using Algorithm 1.

    Parameters:
        U (ndarray): Matrix whose columns are basis vectors of the data.
        sigma_noise (float): Model parameter representing a measurement noise standard deviation.
        Gamma_prior_sqrt (ndarray): Square root of the prior covariance matrix.
        num_sensors (int): Number of available sensors.

    Returns:
        idx_opt (list): List of greedy D-optimal sensor locations.
    """

    idx_lst = list(range(U.shape[0]))
    idx_select_lst = []

    id1 = np.identity(U.shape[1])

    for _ in tqdm(range(num_sensors)):
        error__opt = np.nan
        idx_opt = np.nan
        for idx in idx_lst:
            F = U[idx_select_lst + [idx]] @ Gamma_prior_sqrt
            H_tilde = (sigma_noise**(-2))*(F.T @ F)
            error = -np.linalg.det(H_tilde + id1)

            if np.isnan(error__opt):
                error__opt = error
                idx_opt = idx
            elif error < error__opt:
                error__opt = error
                idx_opt = idx
        idx_select_lst = idx_select_lst + [idx_opt]
        idx_lst.remove(idx_opt)

    return idx_select_lst


def aopt_greedy_select(U: np.ndarray, sigma_noise: float, Gamma_prior_inv: np.ndarray, 
                        num_sensors: int):
    """
    Greedily selects A-optimal sensors using Algorithm 1.

    Parameters:
        U (ndarray): Matrix whose columns are basis vectors of the data.
        sigma_noise (float): Model parameter representing a measurement noise standard deviation.
        Gamma_prior_int (ndarray): Inverse of the prior covariance matrix.
        num_sensors (int): Number of available sensors.

    Returns:
        idx_select_lst (list): List of greedy A-optimal sensor locations.
    """

    idx_lst = list(range(U.shape[0]))
    idx_select_lst = []

    for _ in tqdm(range(num_sensors)):
        min_error = np.nan
        idx_opt = np.nan
        for idx in idx_lst:
            F = U[idx_select_lst + [idx]]
            H = (sigma_noise**(-2)) * (F.T @ F)
            error = np.trace(np.linalg.inv(H + Gamma_prior_inv))
            if np.isnan(min_error):
                min_error = error
                idx_opt = idx
            elif error < min_error:
                min_error = error
                idx_opt = idx
        idx_select_lst = idx_select_lst + [idx_opt]
        idx_lst.remove(idx_opt)

    return idx_select_lst
