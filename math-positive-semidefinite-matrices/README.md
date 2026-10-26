# Positive Semi-Definite Matrices

Intermediate | linear-algebra | matrices | optimization

### The problem, from first principles

`01-positive-definite` asks that a quadratic form be strictly positive in every direction, so the bowl it describes has a single lowest point. Many matrices that matter in practice only meet a weaker bar: the form is never negative, but is allowed to be exactly zero along some direction, like a trough with a flat floor instead of a bowl. Covariance matrices and every matrix of the shape `X X^T` behave this way, and so does a Hessian at a flat spot. This question builds the weaker test, the family of matrices that always pass it, and a repair for matrices that should pass it and narrowly fail because of noise.

### From theory to code

Implement `is_positive_semidefinite(a, tol)`, which tests the weaker condition, then `gram_matrix(x)`, which builds a matrix that is positive semi-definite by construction, then `nearest_psd(a)`, which returns the closest positive semi-definite matrix to a given one. The signatures and docstrings are already in the editor.

### Constraints

- A matrix must be square and symmetric (within `np.allclose`) to be positive semi-definite. A non-symmetric matrix is not.
- Eigenvalues are only accurate to rounding error, so `is_positive_semidefinite` accepts an eigenvalue as zero if it is at least `-tol`. A genuinely negative eigenvalue such as `-1e-3` must still be rejected with the default `tol = 1e-10`.
- `gram_matrix(x)` takes an `m x d` array whose rows are vectors and returns the `m x m` matrix of all pairwise dot products of those rows.
- `nearest_psd(a)` returns a symmetric positive semi-definite matrix of the same shape. A non-symmetric input is first replaced by its symmetric part `(a + a.T) / 2`. The result must be the closest positive semi-definite matrix in Frobenius norm to that symmetric part.
- Inputs must not be modified.

### Hints

<details>
<summary>Hint 1</summary>

Positive semi-definiteness is positive-definiteness with the strict inequality on the eigenvalues loosened to allow zero. The tolerance exists because an eigenvalue that is zero in exact arithmetic often comes out as a tiny negative number.

</details>

<details>
<summary>Hint 2</summary>

For `gram_matrix`, the quadratic form is `v^T (X X^T) v = ||X^T v||^2`, a squared length, which can never be negative. That is the whole reason this construction works.

</details>

<details>
<summary>Hint 3</summary>

Break the symmetric matrix into eigenvectors and eigenvalues. Replacing every negative eigenvalue with zero and rebuilding the matrix removes exactly the directions with negative curvature and changes nothing else.

</details>
