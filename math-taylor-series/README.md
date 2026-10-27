# Taylor Series

Intermediate | calculus

### The problem, from first principles

`05-hessian` computes a function's curvature at a point, but never connects that number to what it's actually _for_: predicting the function's value nearby without re-evaluating it. Taylor series is that connection — a way to approximate any smooth function, near a chosen point, using only its derivatives at that one point. It also explains, precisely, why gradient descent (`08-gradient-descent`) only ever needs the gradient: gradient descent is quietly using just the **first-order** Taylor approximation and discarding everything else.

### From theory to code

Implement `taylor_first_order(f, f_prime, a, x)`, the tangent-line approximation of `f` at `x`, centered at `a`, then `taylor_second_order(f, f_prime, f_double_prime, a, x)`, adding the next term. The signatures and docstrings are already in the editor.

### Constraints

- `f`, `f_prime`, `f_double_prime` are functions taking a single float and returning a float.
- `a` is the expansion point; `x` is where the approximation is evaluated (may equal `a`, or be arbitrarily far from it — accuracy degrades with distance, but the formula itself is defined everywhere).
- Both approximations must equal `f(a)` exactly when `x == a`.

### Hints

<details>
<summary>Hint 1</summary>

The first-order term is exactly `05-hessian`'s tangent-plane idea in one dimension: `f(a) + f'(a) * (x - a)` — the function's value at `a`, plus a linear correction based on the slope there.

</details>

<details>
<summary>Hint 2</summary>

The second-order term adds `(f''(a) / 2) * (x - a)²` — note the `/ 2`, which comes directly from the general Taylor series formula's `n!` denominator (`03-factorial-and-binomial-coefficient`) at `n = 2`.

</details>
