import numpy as np


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    a, b: 1-D vectors of equal length.

    Returns:
        the cosine of the angle between a and b, or 0.0 if either has
        zero length.
    """
    # TODO: Dot product divided by the product of the norms.
    la = np.sqrt(sum(a**2))
    lb = np.sqrt(sum(b**2))
    if la * lb == 0:
        return 0.0
    return a @ b /(la * lb)
