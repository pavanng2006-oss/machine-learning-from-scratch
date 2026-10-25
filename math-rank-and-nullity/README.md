# Rank & Nullity of a Matrix

Intermediate | linear-algebra

### The problem, from first principles

`05-matrix-inverse` established that some matrices simply have no inverse. **Rank** is the precise number that tells you exactly why, and exactly how badly: it counts how many of a matrix's rows (or, equivalently, columns) are genuinely independent, rather than being a combination of the others. A square matrix has an inverse exactly when its rank equals its size; anything less, and the matrix is throwing away information that can never be recovered, which is exactly what a missing inverse means.

### From theory to code

Implement `rank(A)`, computing the number of independent rows of `A`, then `nullity(A)`, computing `A`'s number of columns minus its rank — the size of the space of vectors `A` maps to zero. The signature and docstring are already in the editor.

### Constraints

- `A` is an `m × n` NumPy array (not necessarily square).
- Compute rank via row-reduction (`10-gaussian-elimination`'s approach, extended to handle non-square, rank-deficient inputs by skipping a column entirely when every remaining candidate pivot in it is zero), not by calling `np.linalg.matrix_rank` directly — the point is to see rank fall directly out of row-reduction, not to treat it as a library black box.
- Use a small numerical tolerance (e.g. `abs(value) < 1e-10` counts as zero) rather than exact equality, since floating-point row-reduction rarely produces an exact `0.0`.

### Hints

<details>
<summary>Hint 1</summary>

Row-reduce `A` to echelon form exactly as in `10-gaussian-elimination`, but track a **pivot column pointer** separately from the row you're working on — when a column has no usable (non-zero, above tolerance) pivot in any remaining row, move on to the next column without advancing the row pointer, since that column contributed no new independent direction.

</details>

<details>
<summary>Hint 2</summary>

The rank is simply the number of pivots actually found by the time row-reduction finishes — count them as you go, rather than trying to infer the count from the shape of the final result afterward.

</details>
