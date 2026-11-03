# Chain Rule

Intermediate | calculus

### The problem, from first principles

A thermostat controls a heater's power, and the heater's power controls the room's temperature. If you want to know "how much does turning the thermostat dial one degree change the room's temperature," you can't answer that from either relationship alone, you need to combine "how much does the dial change the heater's power" with "how much does the heater's power change the temperature." Multiply those two sensitivities together and you get the answer for the whole chain.

That multiplication is the chain rule, and it is not a niche calculus trick, it is the single mechanism behind every gradient this entire curriculum computes. A neural network is nothing but a long chain of composed functions (linear, activation, linear, activation, ...), and "backpropagation" is just this rule, applied once per link in that chain, walking from the output back to the input.

### From theory to code

Theory gives the exact rule for two composed functions: the outer function's derivative, evaluated at the inner function's output, times the inner function's own derivative. Implement a way to compose two functions into one, and a way to compute that combined derivative directly from the pieces (not by composing then finite-differencing).

Implement `compose(f, g)` and `chain_rule_derivative(f_prime, g, g_prime, x)` against that reasoning. The signatures and docstrings are already in the editor.

### Constraints

- `compose(f, g)` returns a callable, `h`, such that `h(x) == f(g(x))`.
- `chain_rule_derivative` computes the derivative analytically from `f_prime`, `g` and `g_prime`, it does not call `compose` or approximate with finite differences.
- `f` itself is never needed by `chain_rule_derivative`, only `f_prime`.

### Hints

Open one at a time. Each gives away a little more than the last.

<details>
<summary>Hint 1</summary>

`compose` is one line: a lambda (or a small nested function) that calls `g` first, then feeds the result into `f`.

</details>

<details>
<summary>Hint 2</summary>

You need `g(x)` before you can evaluate `f_prime` at the right point, `f_prime` is evaluated at the INNER function's output, not at `x` itself.

</details>
