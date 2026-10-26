import numpy as np


def is_positive_semidefinite(a: np.ndarray, tol: float = 1e-10) -> bool:
    """
    a: a 2D array
    tol: eigenvalues down to -tol count as zero

    Returns:
        True if a is square, symmetric (np.allclose) and has no
        eigenvalue below -tol, otherwise False.
    """
    # TODO: Loosen the strict eigenvalue test from 01-positive-definite.
    if len(a) == len(a[0]) and np.allclose(a, a.T, atol=tol):
        for ev in np.linalg.eigvalsh(a):
            if ev < -tol:
                return False
        return True
    else:
        return False
    

def gram_matrix(x: np.ndarray) -> np.ndarray:
    """
    x: m x d array whose rows are vectors

    Returns:
        The m x m matrix of all pairwise dot products of the rows of x.
    """
    # TODO: Build the Gram matrix from Theory.
    return x @ x.T


def nearest_psd(a: np.ndarray) -> np.ndarray:
    """
    a: a square 2D array

    Returns:
        The symmetric positive semi-definite matrix closest to
        (a + a.T) / 2 in Frobenius norm, same shape as a. a must not be
        modified.
    """
    # TODO: Clip the negative eigenvalues of the symmetric part, as in Theory.
    a_sym = (a + a.T) / 2.0

    evals, evecs = np.linalg.eigh(a_sym)
    evals_clipped = np.maximum(evals, 0)

    return evecs @ np.diag(evals_clipped) @ evecs.T
