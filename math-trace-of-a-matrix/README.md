# Trace of a Matrix

Beginner | linear-algebra

### The problem, from first principles

The **trace** is the smallest, cheapest-to-compute summary a square matrix has — just its diagonal entries, added up. It sounds almost too simple to matter, but it shows up constantly: as a fast way to compute the sum of a matrix's eigenvalues without finding any of them, inside the KL divergence formula for two multivariate Gaussians, and inside regularization terms that penalize a weight matrix's overall scale.

### From theory to code

Implement `trace(A)`, summing `A`'s diagonal entries, then `trace_of_product(A, B)`, computing `trace(A @ B)` — but doing it in a way that reveals a useful shortcut identity, covered in Theory. The signatures and docstrings are already in the editor.

### Constraints

- `A` is a square 2D NumPy array (`n × n`).
- For `trace_of_product`, `A` is `(m, n)` and `B` is `(n, m)`, so `A @ B` is a valid `(m, m)` square matrix.

### Hints

<details>
<summary>Hint 1</summary>

`trace(A)` is `Σ_{i} A[i, i]` — `01-summation-notation`'s `Σ` applied directly to the diagonal, or equivalently `np.diag(A).sum()`.

</details>

<details>
<summary>Hint 2</summary>

`trace(A @ B)` and `trace(B @ A)` are always equal even when `A @ B` and `B @ A` are entirely different-shaped matrices (or not even both defined) — you don't need to actually form the full product `A @ B` to compute its trace; summing `(A * B.T)` element-wise and then summing that result computes the same number more directly. Try implementing it the direct way (`np.trace(A @ B)`) first, then think about why the shortcut works.

</details>
