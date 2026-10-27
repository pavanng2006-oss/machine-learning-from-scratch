# Telescoping Series

Intermediate | calculus | series

### The problem, from first principles

`01-geometric` found a closed form by multiplying a sum by `r` and subtracting so that almost every term cancelled. Some series cancel without any trick at all: each term contains a piece that the next term subtracts away, like the sections of a collapsing spyglass sliding into each other until only the two ends remain. Spotting that structure turns a sum of a million terms into a single subtraction. This question implements it, and applies it to a series that does not look like it cancels until it is rewritten.

### From theory to code

Implement `telescoping_partial_sum(f, n)`, the sum of the first `n` terms of the series with terms `f(k) - f(k + 1)`, then `telescoping_infinite_sum(f_first, f_limit)`, the sum of the infinite series, then `sum_reciprocal_products(n)`, the sum of `1 / (k * (k + 1))` for `k` from 1 to `n`. The signatures and docstrings are already in the editor.

### Constraints

- `f` is a function taking an integer and returning a float. The series is `(f(1) - f(2)) + (f(2) - f(3)) + ...`.
- `telescoping_partial_sum` takes an integer `n >= 0`, calls `f` exactly twice for any `n >= 1`, and never loops over the terms. With `n = 0` it returns `0.0` and does not call `f`.
- `telescoping_infinite_sum` receives the value `f(1)` and the limit of `f(k)` as `k` grows. It does not need `f` itself.
- `sum_reciprocal_products` takes an integer `n >= 0` and returns a float. It must not loop over the terms and must return exactly `0.0` for `n = 0`.

### Hints

<details>
<summary>Hint 1</summary>

Write out the first four terms of `(f(1) - f(2)) + (f(2) - f(3)) + ...` and cross out whatever appears once with each sign. Only two values survive, however many terms there are.

</details>

<details>
<summary>Hint 2</summary>

`1 / (k * (k + 1))` is not written as a difference of consecutive values yet, but it splits into two simple fractions with denominators `k` and `k + 1`. Find the two numerators, and the series becomes `f(k) - f(k + 1)` for a very simple `f`.

</details>
