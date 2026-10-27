import numpy as np


def is_negative_definite(a: np.ndarray) -> bool:
    """
    a: a 2D array

    Returns:
        True if a is square, symmetric (np.allclose) and every eigenvalue
        is strictly negative, otherwise False.
    """
    # TODO: Mirror the test from 01-positive-definite.
    if a.shape[0] == a.shape[1] and np.allclose(a, a.T, atol=10e-8):
        evals = np.linalg.eigvalsh(a)
        for eval in evals:
            if eval >= 0:
                return False
        return True
    else:
        return False


def is_negative_semidefinite(a: np.ndarray, tol: float = 1e-10) -> bool:
    """
    a: a 2D array
    tol: eigenvalues up to +tol count as zero

    Returns:
        True if a is square, symmetric (np.allclose) and no eigenvalue is
        above tol, otherwise False.
    """
    # TODO: Mirror the test from 02-positive-semidefinite.
    
    if a.ndim == 2 and a.shape[0] == a.shape[1] and np.allclose(a, a.T, atol=tol):
        evals = np.linalg.eigvalsh(a)
        for eval in evals:
            if eval > tol:
                return False
        return True

    else:
        return False


def quadratic_maximizer(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    a: n x n symmetric array
    b: 1D array of length n

    Returns:
        The point x that maximizes 0.5 * x^T a x + b^T x, as a 1D array.

    Raises:
        ValueError: if a is not negative definite, so no unique highest
        point exists.
    """
    # TODO: Find where the gradient from Theory is zero.
    if not is_negative_definite(a):
        raise ValueError

    return -np.linalg.solve(a, b)
