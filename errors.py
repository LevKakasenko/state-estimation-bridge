"""
Author: Lev Kakasenko
Description:
Provides formulas for computing errors between samples and their approximations.

If you use this code in any form, please cite "Bridging the Gap Between Deterministic and
Probabilistic Approaches to State Estimation" by Lev Kakasenko,  Alen Alexanderian,
Mohammad Farazmand, and Arvind Krishna Saibaba (2025)
"""

import numpy as np

def error_rel_columnwise(X_test: np.ndarray, X_test_hat: np.ndarray):
    """
    Returns the columnwise relative error between X_test_hat and X_test
    computed in the 2-norm.

    Parameters:
        X_test (ndarray): Data matrix (whose columns are samples) to be reconstructed.
        X_test_hat (ndarray): Estimate of X_test.

    Returns:
        error (float): Columnwise relative error between X_test_hat and X_test.
    """
    diff_norms = np.linalg.norm(X_test - X_test_hat, axis=0, ord=2)
    X_test_norms = np.linalg.norm(X_test, axis=0, ord=2)
    rel_errors = diff_norms / X_test_norms
    return np.mean(rel_errors), np.var(rel_errors, ddof=0)


def error_fro(X_test: np.ndarray, X_test_hat: np.ndarray) -> float:
    """
    Returns the absolute error in the Frobenius norm between X_test_hat and X_test.

    Parameters:
        X_test (ndarray): Data matrix (whose columns are samples) to be reconstructed.
        X_test_hat (ndarray): Estimate of X_test.
        norm (float): Norm used to compute the error.

    Returns:
        error (float): Absolute error in the Frobenius norm between X_test_hat and X_test.
    """

    error = np.linalg.norm(X_test - X_test_hat, ord='fro')
    return error
