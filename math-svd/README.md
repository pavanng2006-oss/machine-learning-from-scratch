# Singular Value Decomposition

Advanced | linear-algebra | matrices

### The problem, from first principles

`06-eigenvalues-eigenvectors` only works for square matrices, but most real data isn't square: a dataset is `(num_samples, num_features)`, an image is `(height, width)`, neither is guaranteed to have equal dimensions. You still want the same kind of insight, "what are this matrix's most important directions, and can I throw away the unimportant ones", just without requiring squareness.

Singular Value Decomposition answers exactly that, for any matrix at all. And it comes with a genuinely useful guarantee: if you keep only the most important handful of directions it finds, you get the mathematically _best possible_ approximation of the original matrix at that size, not just a reasonable one.

### From theory to code

Theory gives the factorization `A = U @ diag(sigma) @ V^T` with singular values sorted descending by construction. Implement the factorization itself, a function that rebuilds the original from the three pieces, and a compression step that keeps only the top `k` singular values.

Implement `svd(a)`, `reconstruct_from_svd(u, singular_values, vt)` and `low_rank_approximation(a, k)` against that reasoning. The signatures and docstrings are already in the editor.

### Constraints

- `a` is `(m, n)`, any shape, not required to be square.
- Use the reduced ("economy") form: `u` is `(m, r)`, `vt` is `(r, n)`, `r = min(m, n)`.
- `low_rank_approximation` keeps the top `k` singular values, since `np.linalg.svd` already sorts them descending, "top k" is just "first k", no explicit sorting needed.

### Hints

Open one at a time. Each gives away a little more than the last.

<details>
<summary>Hint 1</summary>

`np.linalg.svd` with `full_matrices=False` returns exactly the three pieces this question asks for, in the shapes Theory describes.

</details>

<details>
<summary>Hint 2</summary>

`low_rank_approximation` is `svd` followed by slicing each of the three returned pieces down to their first `k` entries/columns/rows, then calling `reconstruct_from_svd` on the truncated pieces.

</details>
