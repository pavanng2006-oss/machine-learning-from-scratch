import numpy as np


def determinant(matrix: np.ndarray) -> float:
    """
    Return the determinant of square matrix `matrix`, using
    np.linalg.det.
    """
    return np.linalg.det(matrix)


def is_invertible(matrix: np.ndarray) -> bool:
    """
    Return True if `matrix` has a non-zero determinant (and is
    therefore invertible), False otherwise. Use a small tolerance
    (e.g. abs(det) > 1e-10) rather than exact equality to 0, since
    floating-point computation of the determinant may not produce
    an exact zero even for a genuinely singular matrix.
    """
    return not np.isclose(determinant(matrix), 0, atol=10e-8)


def safe_inverse(matrix: np.ndarray):
    """
    Return the inverse of `matrix` using np.linalg.inv if it is
    invertible (check with is_invertible first), or None if it
    is not — do not let np.linalg.inv's error propagate out of
    this function for a singular matrix.
    """
    if is_invertible(matrix):
        return np.linalg.inv(matrix)
    return None


def verify_inverse(matrix: np.ndarray, inverse: np.ndarray) -> bool:
    """
    Return True if matrix @ inverse is approximately equal to
    the identity matrix of the corresponding size (use
    np.allclose for the comparison, to allow for floating-point
    imprecision), False otherwise.
    """
    id = np.eye(len(matrix))
    return np.allclose(matrix @ inverse, id, atol=10e-8)
