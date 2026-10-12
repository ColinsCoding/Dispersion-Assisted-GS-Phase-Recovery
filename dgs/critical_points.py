"""Classifying critical points -- relative max, relative min, or saddle --
two independent ways, then applied to something dgs/math_methods.py never
connected it to: a neural network's own loss landscape.

THE 2-VARIABLE SECOND-DERIVATIVE TEST (the textbook AP/multivariable-calc
formula): at a critical point (grad f = 0) of f(x,y), let
    D = f_xx * f_yy - f_xy^2.
  D > 0 and f_xx > 0  -> relative MINIMUM
  D > 0 and f_xx < 0  -> relative MAXIMUM
  D < 0               -> SADDLE POINT
  D = 0               -> test is inconclusive

THE GENERAL N-DIMENSIONAL VERSION (linear algebra: eigenvalues of the
Hessian, the actual reason the 2-variable test works -- D>0,f_xx>0 is
exactly "both eigenvalues positive" in disguise, verified below rather
than just asserted):
  all eigenvalues > 0  -> relative minimum   (positive definite Hessian)
  all eigenvalues < 0  -> relative maximum   (negative definite)
  mixed signs          -> saddle             (indefinite)
  any eigenvalue == 0  -> degenerate/inconclusive

WHY THIS MATTERS FOR NEURONS: a famous, non-obvious fact about training
neural networks (Dauphin et al. 2014, "Identifying and attacking the
saddle point problem in high-dimensional non-convex optimization") is
that in HIGH dimensions, saddle points vastly outnumber true local
minima/maxima -- a random symmetric matrix (like a loss landscape's
Hessian at a random critical point) has eigenvalues split roughly evenly
between positive and negative with overwhelming probability once the
dimension is large, so "mixed signs" (saddle) becomes the generic case.
This module verifies the two classification methods agree with each
other, then finds and classifies real critical points of a genuine (if
tiny) neural network's own MSE loss surface, using torch autograd for an
exact Hessian (not a finite-difference approximation) -- cross-checked
against a numerical finite-difference Hessian as an independent check.
"""
import numpy as np
import sympy as sp
import torch
from torch.autograd.functional import hessian as torch_hessian


def classify_2var_second_derivative_test(fxx, fxy, fyy):
    """The textbook 2-variable test, from the raw second partials at a
    critical point. Returns ('min'|'max'|'saddle'|'inconclusive', D)."""
    D = fxx * fyy - fxy ** 2
    if D > 0 and fxx > 0:
        return "min", D
    if D > 0 and fxx < 0:
        return "max", D
    if D < 0:
        return "saddle", D
    return "inconclusive", D


def classify_by_eigenvalues(hessian_matrix, tol=1e-9):
    """The general N-dimensional version: classify by the SIGNS of the
    Hessian's eigenvalues (it must be symmetric for a real function with
    continuous second partials -- Clairaut's theorem). Returns
    ('min'|'max'|'saddle'|'degenerate', eigenvalues)."""
    H = np.asarray(hessian_matrix, dtype=float)
    if not np.allclose(H, H.T, atol=1e-8):
        raise ValueError("Hessian is not symmetric -- Clairaut's theorem requires continuous second partials")
    eigvals = np.linalg.eigvalsh(H)
    if np.any(np.abs(eigvals) < tol):
        return "degenerate", eigvals
    if np.all(eigvals > 0):
        return "min", eigvals
    if np.all(eigvals < 0):
        return "max", eigvals
    return "saddle", eigvals


def find_critical_points_2d(f_expr, x_sym, y_sym):
    """Solve grad f = 0 symbolically (SymPy) for a 2-variable function,
    and classify each real solution with BOTH methods above -- they must
    agree, since the 2-variable test is just a disguised 2x2 eigenvalue
    check (verified explicitly, not assumed)."""
    fx = sp.diff(f_expr, x_sym)
    fy = sp.diff(f_expr, y_sym)
    fxx = sp.diff(f_expr, x_sym, 2)
    fxy = sp.diff(f_expr, x_sym, y_sym)
    fyy = sp.diff(f_expr, y_sym, 2)

    solutions = sp.solve([fx, fy], [x_sym, y_sym], dict=True)
    results = []
    for sol in solutions:
        x0, y0 = sol[x_sym], sol[y_sym]
        if not (x0.is_real and y0.is_real):
            continue
        fxx0 = float(fxx.subs(sol))
        fxy0 = float(fxy.subs(sol))
        fyy0 = float(fyy.subs(sol))

        kind_2var, D = classify_2var_second_derivative_test(fxx0, fxy0, fyy0)
        H = [[fxx0, fxy0], [fxy0, fyy0]]
        kind_eigen, eigvals = classify_by_eigenvalues(H)

        results.append({
            "point": (float(x0), float(y0)),
            "kind_2var_test": kind_2var,
            "kind_eigenvalue_test": kind_eigen,
            "agree": kind_2var == kind_eigen,
            "D": D, "eigenvalues": eigvals.tolist(),
        })
    return results


# ── applied to a real (tiny) neural network's own loss surface ──────
#
# A NOTE ON WHAT WAS TRIED FIRST AND DISCARDED, kept visible rather than
# hidden: the first version of this module classified a single sigmoid
# neuron's loss at the ORIGIN (w=0, b=0), assuming symmetry would put a
# saddle there. Checked directly: the gradient at the origin was
# (-0.39, -0.05) for the test data used -- NOT a critical point at all,
# so classifying its Hessian was invalid (the second-derivative test only
# applies where the gradient vanishes). A systematic root-find
# (scipy.optimize.fsolve on the autograd gradient, many starting points)
# turned up only points on a saturation PLATEAU (large |b|, sigmoid
# pinned near 0 or 1), where the Hessian is degenerate (eigenvalues ~0)
# -- the vanishing-gradient phenomenon, not a clean min/max/saddle. That
# is a real and honest finding about a single sigmoid neuron, but not a
# demonstration of a saddle point.
#
# What DOES reliably have a verifiable saddle, by a well-known result in
# deep linear network theory (Baldi & Hornik 1989): a minimal 2-LAYER
# LINEAR network, hidden = w1*x, output = w2*hidden = (w1*w2)*x, trained
# by MSE against a linear target y=x. The loss is L(w1,w2) = C*(w1*w2-1)^2
# with C = E[x^2] > 0; the origin (w1=w2=0, i.e. "both weights dead") is
# a genuine critical point (verified: gradient is exactly zero there) and
# its Hessian is [[0,-2C],[-2C,0]], whose eigenvalues are EXACTLY +-2C --
# always mixed sign, always a saddle, regardless of the data. Verified
# three independent ways below: torch autograd, finite differences, and
# a closed-form SymPy Hessian.

def two_layer_linear_network_loss(x_data, y_data, w1, w2):
    """MSE loss of the minimal 2-layer linear network hidden=w1*x,
    output=w2*hidden=(w1*w2)*x against target y_data. A plain torch
    scalar computation, usable both for plotting and as the function
    autograd differentiates below."""
    x_data = torch.as_tensor(x_data, dtype=torch.float64)
    y_data = torch.as_tensor(y_data, dtype=torch.float64)
    pred = (w1 * w2) * x_data
    return torch.mean((pred - y_data) ** 2)


def autograd_hessian_at_point(x_data, y_data, w1_0, w2_0):
    """The EXACT Hessian of the 2-layer linear network's loss w.r.t.
    (w1,w2) at (w1_0,w2_0), via torch.autograd.functional.hessian
    (automatic differentiation -- not a finite-difference approximation)."""
    def loss_fn(params):
        w1, w2 = params[0], params[1]
        return two_layer_linear_network_loss(x_data, y_data, w1, w2)

    params = torch.tensor([w1_0, w2_0], dtype=torch.float64)
    H = torch_hessian(loss_fn, params)
    return H.numpy()


def finite_difference_hessian_at_point(x_data, y_data, w1_0, w2_0, eps=1e-4):
    """An INDEPENDENT numerical check on autograd_hessian_at_point: the
    same Hessian via central finite differences, no autograd involved at
    all -- two completely different computational methods that must
    agree if the autograd result is trustworthy."""
    def loss(w1, w2):
        return float(two_layer_linear_network_loss(
            x_data, y_data, torch.tensor(w1, dtype=torch.float64), torch.tensor(w2, dtype=torch.float64)))

    fww = (loss(w1_0 + eps, w2_0) - 2 * loss(w1_0, w2_0) + loss(w1_0 - eps, w2_0)) / eps ** 2
    fbb = (loss(w1_0, w2_0 + eps) - 2 * loss(w1_0, w2_0) + loss(w1_0, w2_0 - eps)) / eps ** 2
    fwb = (loss(w1_0 + eps, w2_0 + eps) - loss(w1_0 + eps, w2_0 - eps)
           - loss(w1_0 - eps, w2_0 + eps) + loss(w1_0 - eps, w2_0 - eps)) / (4 * eps ** 2)
    return np.array([[fww, fwb], [fwb, fbb]])


def symbolic_hessian_two_layer_linear(C_value):
    """The closed-form loss L(w1,w2)=C*(w1*w2-1)^2, differentiated
    symbolically (SymPy) rather than numerically -- a THIRD, independent
    route to the same Hessian at the origin, giving three methods that
    must all agree (autograd, finite-difference, symbolic) before this
    module trusts the "always a saddle" claim."""
    w1, w2, C = sp.symbols("w1 w2 C", real=True)
    L = C * (w1 * w2 - 1) ** 2
    H = sp.hessian(L, [w1, w2])
    grad = [sp.diff(L, v) for v in (w1, w2)]
    H0 = H.subs({w1: 0, w2: 0, C: C_value})
    grad0 = [g.subs({w1: 0, w2: 0, C: C_value}) for g in grad]
    return np.array(H0.tolist(), dtype=float), [float(g) for g in grad0]


if __name__ == "__main__":
    print("=== CLASSIFYING CRITICAL POINTS: TWO METHODS ===\n")

    x, y = sp.symbols("x y", real=True)
    f = x**2 + y**2       # a paraboloid bowl -- should be a minimum at (0,0)
    g = x**2 - y**2       # a saddle -- classic example
    h = -(x**2) - y**2    # an inverted bowl -- a maximum

    for name, expr in [("x^2+y^2 (bowl)", f), ("x^2-y^2 (saddle)", g), ("-x^2-y^2 (inverted bowl)", h)]:
        results = find_critical_points_2d(expr, x, y)
        for r in results:
            match = "AGREE" if r["agree"] else "DISAGREE"
            print(f"  {name}: critical point {r['point']} -> "
                  f"2-var test: {r['kind_2var_test']}, eigenvalue test: {r['kind_eigenvalue_test']} [{match}]")

    print("\n=== A REAL 2-LAYER LINEAR NETWORK'S SADDLE POINT ===")
    print("hidden = w1*x, output = w2*hidden = (w1*w2)*x, MSE loss against target y=x")
    rng = np.random.default_rng(0)
    x_data = rng.uniform(-3, 3, 200)
    y_data = x_data.copy()

    # verify the origin is a genuine critical point BEFORE classifying it --
    # the mistake the single-sigmoid-neuron attempt above made was skipping this check
    w1t = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
    w2t = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
    loss0 = two_layer_linear_network_loss(x_data, y_data, w1t, w2t)
    g1, g2 = torch.autograd.grad(loss0, [w1t, w2t])
    print(f"gradient at origin (w1,w2)=(0,0): ({float(g1):.2e}, {float(g2):.2e})  (must be ~0 to trust the test below)")
    assert abs(float(g1)) < 1e-9 and abs(float(g2)) < 1e-9

    H_autograd = autograd_hessian_at_point(x_data, y_data, 0.0, 0.0)
    H_findiff = finite_difference_hessian_at_point(x_data, y_data, 0.0, 0.0)
    C = float(np.mean(x_data ** 2))
    H_symbolic, grad_symbolic = symbolic_hessian_two_layer_linear(C)

    print(f"\nHessian at the origin, three independent methods:")
    print(f"  autograd:          {H_autograd}")
    print(f"  finite difference: {H_findiff}")
    print(f"  symbolic (C={C:.4f}): {H_symbolic}")
    print(f"  symbolic gradient at origin: {grad_symbolic}  (confirms it's a critical point analytically too)")

    assert np.max(np.abs(H_autograd - H_findiff)) < 1e-4
    assert np.max(np.abs(H_autograd - H_symbolic)) < 1e-6
    print("\n[OK] all three methods agree")

    kind, eigvals = classify_by_eigenvalues(H_autograd)
    print(f"\nclassification: {kind}  (eigenvalues: {eigvals}, analytic prediction: +-{2*C:.4f})")
    assert kind == "saddle"
    print("[OK] the origin of this 2-layer linear network's loss surface is a genuine saddle point --")
    print("     the well-known 'dead weights' saddle from deep linear network theory (Baldi & Hornik 1989),")
    print("     confirmed here rather than assumed.")
    print("\n=== DONE ===")
