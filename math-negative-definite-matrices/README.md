# Negative Definite & Semi-Definite Matrices

Intermediate | linear-algebra | matrices | optimization

### The problem, from first principles

A bowl has a lowest point. Turn it upside down and it has a highest point, a hilltop where every direction curves down. The matrices that describe an upside-down bowl are the mirror image of the ones in `01-positive-definite` and `02-positive-semidefinite`: the quadratic form is negative in every direction, or at worst zero along some of them. Maximizing a smooth function, as in maximum-likelihood estimation or a reward objective, is exactly the problem of finding such a hilltop. This question builds the two tests and uses them to find the peak of a quadratic function, which exists and is unique only when the curvature has this shape.

### From theory to code

Implement `is_negative_definite(a)` and `is_negative_semidefinite(a, tol)`, then `quadratic_maximizer(a, b)`, which returns the single highest point of `f(x) = 0.5 * x^T a x + b^T x` when one exists. The signatures and docstrings are already in the editor.

### Constraints

- Both tests require a square, symmetric matrix (within `np.allclose`). A non-symmetric matrix is neither.
- `is_negative_definite` uses a strict test: every eigenvalue is strictly below zero. `is_negative_semidefinite` allows eigenvalues up to `tol`, so an eigenvalue of `1e-12` counts as zero while `1e-3` does not, with the default `tol = 1e-10`.
- `quadratic_maximizer(a, b)` takes a square symmetric `a` and a 1D `b` of matching length. It raises `ValueError` unless `a` is negative definite, because otherwise the function has no unique highest point.
- Inputs must not be modified.

### Hints

<details>
<summary>Hint 1</summary>

Flipping every sign of a matrix flips every sign of its quadratic form, so `a` is negative definite exactly when `-a` is positive definite. Both tests are the earlier tests applied to `-a`.

</details>

<details>
<summary>Hint 2</summary>

The highest point is where the gradient `a x + b` is zero. Solve that linear system rather than inverting the matrix.

</details>
