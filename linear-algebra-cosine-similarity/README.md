# Cosine Similarity

Beginner | mathematics | linear-algebra | similarity

### The problem, from first principles

Two vectors can be far apart in distance yet point in the same direction (a document and a copy of it repeated ten times). Cosine similarity compares only direction, ignoring length, which is why it is the default way to compare embeddings.

### From theory to code

Implement `cosine_similarity(a, b)` for two one-dimensional vectors.

### Constraints

- `a` and `b` are one-dimensional float arrays of equal length.
- Return a Python `float` in `[-1, 1]`.
- If either vector has zero length, return `0.0` rather than dividing by zero.

### Hints

Open one at a time. Each gives away a little more than the last.

<details><summary>Hint 1</summary>

The dot product measures alignment; dividing by both norms removes the effect of length.

</details>

<details><summary>Hint 2</summary>

`np.linalg.norm(v)` gives a vector's length.

</details>
