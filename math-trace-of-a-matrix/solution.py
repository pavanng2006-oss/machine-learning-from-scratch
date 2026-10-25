import numpy as np


def trace(A):
    """
    A: a square (n, n) NumPy array

    Returns:
        The sum of A's diagonal entries.
    """
    # TODO: Implement tr(A) = Sum_i A[i, i] from Theory.
    if len(A) != len(A[0]):
        raise ValueError

    trace = 0
    for i in range(len(A)):
        trace += A[i][i]

    return trace


def trace_of_product(A, B):
    """
    A: shape (m, n)
    B: shape (n, m), so A @ B is a valid square (m, m) matrix

    Returns:
        trace(A @ B), computed WITHOUT forming the full product A @ B --
        use the cyclic-property shortcut from Theory instead.
    """
    # TODO: Implement using the sum(A * B.T) identity from Theory.
    if len(A[0]) != len(B):
        raise ValueError

    return trace(A @ B)
