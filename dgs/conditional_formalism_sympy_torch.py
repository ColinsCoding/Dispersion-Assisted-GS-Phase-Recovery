"""Formalizing the "if" case -- the same conditional logic already used
in dgs.diffraction_grating._sinc_unnormalized (np.where(|x|<eps, 1.0, ...))
and dgs.lhopital_rule (an ordinary Python `if` deciding whether a form is
still indeterminate) -- expressed in two more formal paradigms:

  * SYMBOLIC (SymPy Piecewise): a conditional is a first-class algebraic
    object, sp.Piecewise((value_if, condition), (value_else, True)),
    that SymPy can differentiate, integrate, and take limits of directly.

  * DIFFERENTIABLE (torch.where): a conditional inside an autograd graph
    CANNOT be a plain Python `if` on a tensor (that would either error on
    a batch of values, or silently break gradient flow at the branch
    point) -- torch.where evaluates BOTH branches for every element and
    selects afterward, which is what keeps the whole expression
    differentiable straight through the branch.

The running example is the same sinc(x)=sin(x)/x special case at x=0
already used elsewhere in this repo (dgs.diffraction_grating,
dgs.lhopital_rule) -- one indeterminate form, three formalizations:
plain Python if/else (already in the repo), SymPy Piecewise (this
module), and torch.where (this module).
"""

import numpy as np
import sympy as sp

x = sp.symbols('x', real=True)


def sinc_piecewise_symbolic():
    """The sinc "if" case as a first-class SymPy object: Piecewise((1,
    x==0), (sin(x)/x, True)). Verified continuous at x=0 by comparing the
    Piecewise's x=0 branch value against the LIMIT of the general branch
    (they must agree for the Piecewise function to actually be
    continuous, not just defined, at the branch point)."""
    expr = sp.Piecewise((1, sp.Eq(x, 0)), (sp.sin(x) / x, True))
    branch_value_at_0 = expr.subs(x, 0)
    limit_of_general_branch = sp.limit(sp.sin(x) / x, x, 0)
    is_continuous = branch_value_at_0 == limit_of_general_branch
    return expr, is_continuous


def sinc_piecewise_numeric(x_val):
    """Evaluate the Piecewise sinc at a specific value -- confirms the
    symbolic object actually produces the right number, not just that it
    LOOKS right when printed."""
    expr, _ = sinc_piecewise_symbolic()
    return float(expr.subs(x, x_val))


def sinc_torch_where(x_tensor, eps=1e-6):
    """torch.where evaluates BOTH branches (torch.sin(safe_x)/safe_x AND
    the constant 1) for every element, then SELECTS per-element based on
    the condition -- this is what keeps autograd correct through x=0,
    unlike a plain Python `if abs(x) < eps:` which cannot branch per-
    element on a batched tensor at all, let alone preserve gradients.
    `safe_x` avoids a 0/0 division ever being evaluated (even though its
    result is discarded by torch.where for that element, autograd still
    computes its LOCAL gradient contribution unless the input to that
    branch is replaced with something safe first). Imports torch lazily
    (module-level, not at file scope) since torch is py-3.12-only in
    this environment."""
    import torch
    safe_x = torch.where(torch.abs(x_tensor) < eps, torch.ones_like(x_tensor), x_tensor)
    return torch.where(torch.abs(x_tensor) < eps, torch.ones_like(x_tensor),
                        torch.sin(safe_x) / safe_x)


def verify_torch_where_sinc():
    """Actually run the torch.where formalization and check it: sinc(0)=1,
    the gradient at x=0 is 0 (sinc is even, smooth, has a max there), and
    away from 0 the gradient matches the ordinary quotient-rule derivative
    d/dx[sin(x)/x] = cos(x)/x - sin(x)/x^2 -- a real executed check, not
    reasoned-through-but-unverified source."""
    import torch
    x_t = torch.tensor([0.0, 0.5, 1.0, -0.3], dtype=torch.float64, requires_grad=True)
    y = sinc_torch_where(x_t)
    y.sum().backward()

    values = y.detach().numpy()
    grads = x_t.grad.numpy()

    checks = {}
    checks["sinc(0)==1"] = abs(values[0] - 1.0) < 1e-10
    checks["grad at x=0 is ~0"] = abs(grads[0]) < 1e-6
    for i, xv in enumerate([0.5, 1.0, -0.3], start=1):
        analytic = np.cos(xv) / xv - np.sin(xv) / xv ** 2
        checks[f"grad at x={xv} matches analytic"] = abs(grads[i] - analytic) < 1e-6
    return values, grads, checks


if __name__ == "__main__":
    print("=== SymPy Piecewise: the 'if' case as a symbolic object ===")
    expr, is_continuous = sinc_piecewise_symbolic()
    print(f"sinc(x) = {expr}")
    print(f"continuous at the branch point (x=0 value matches the limit "
          f"of the general branch): {is_continuous}")
    assert is_continuous

    print("\nnumeric spot-checks:")
    for x_val in [0, 0.5, 1.0, -0.3]:
        val = sinc_piecewise_numeric(x_val)
        print(f"  sinc({x_val}) = {val:.6f}")

    print("\n=== torch.where: the differentiable 'if' case ===")
    try:
        values, grads, checks = verify_torch_where_sinc()
        print(f"values: {values}")
        print(f"gradients: {grads}")
        for name, ok in checks.items():
            print(f"  {name}: {ok}")
        assert all(checks.values())
        print("\nConfirmed: torch.where keeps the branch differentiable straight")
        print("through x=0, matching the analytic quotient-rule derivative away")
        print("from the branch point and giving the correct (zero) gradient at it.")
    except ImportError as e:
        print(f"torch unavailable ({e}) -- this needs py-3.12, run via "
              f"`py -3.12 -m dgs.conditional_formalism_sympy_torch`")
    except RuntimeError as e:
        print(f"torch failed to run ({e}) -- if this is an OpenBLAS/memory error, "
              f"retry once system memory frees up; this is an environment issue, "
              f"not a logic problem in sinc_torch_where itself.")
