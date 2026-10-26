# Positive-definite Matrices

Intermediate | linear-algebra | matrices | optimization

### The problem, from first principles

Stand at the bottom of a bowl and every direction you step curves back up, you're at a minimum. Stand on a mountain pass and some directions curve up, others curve down, a saddle. Stand at the top of a hill and every direction curves down, a maximum. A single matrix, the Hessian (second derivatives, the next question in this track), tells you which of these three situations you're actually in at any point where a function's slope is zero, and "positive-definite" is the precise, checkable condition for "every direction curves up", i.e. a genuine minimum.

This matters concretely: every training loop in this curriculum stops when the gradient is (near) zero, but zero gradient alone doesn't tell you whether you've found a good minimum, a bad maximum, or a saddle point stalling training. Positive-definiteness is the test that distinguishes them.

### From theory to code

Theory defines positive-definiteness via a quadratic form (`x^T A x > 0` for every nonzero `x`) that's impossible to check exhaustively, and gives an equivalent, checkable condition instead: symmetric, with every eigenvalue strictly positive. Implement the quadratic form directly (so you can see the definition compute a real number), then the checkable eigenvalue-based test.

Implement `is_symmetric(a)`, `quadratic_form(a, x)` and `is_positive_definite(a)` against that reasoning. The signatures and docstrings are already in the editor.

### Constraints

- Positive-definiteness is only checked for square matrices, and requires symmetry first.
- Use strict `> 0` on eigenvalues, not `>= 0` (that weaker condition is a different, related property, positive _semi_-definiteness).
- `is_symmetric` should tolerate floating-point noise (`np.allclose`, not exact equality).

### Hints

Open one at a time. Each gives away a little more than the last.

<details>
<summary>Hint 1</summary>

`quadratic_form` is a direct translation of `x^T @ A @ x`, no cleverness needed.

</details>

<details>
<summary>Hint 2</summary>

`06-eigenvalues-eigenvectors`'s own `np.linalg.eigvalsh` sibling gives you eigenvalues without the eigenvectors, exactly what the checkable condition needs.

</details>
