"""L'Hopital's rule, proved (via Cauchy's Mean Value Theorem) and applied
to a real indeterminate form already sitting in this repo:
dgs.diffraction_grating._sinc_unnormalized hardcodes sin(x)/x -> 1 as
x->0 (a 0/0 indeterminate form) without deriving it -- this module
supplies that missing derivation and cross-checks it against the actual
hardcoded value.

STATEMENT (0/0 case): if f(a)=g(a)=0, f,g are differentiable near a,
g'(x) != 0 near a (x != a), and lim_{x->a} f'(x)/g'(x) exists, then
    lim_{x->a} f(x)/g(x) = lim_{x->a} f'(x)/g'(x).

PROOF SKETCH (Cauchy's Mean Value Theorem): for x near a (x != a),
Cauchy's MVT gives some c strictly between a and x such that
    [f(x)-f(a)] * g'(c) = [g(x)-g(a)] * f'(c)
Since f(a)=g(a)=0, this is f(x)*g'(c) = g(x)*f'(c), i.e.
    f(x)/g(x) = f'(c)/g'(c)   (valid wherever g(x) != 0)
As x -> a, c is squeezed between a and x, so c -> a too. Taking the
limit of both sides (using that f'/g' has a limit at a by hypothesis):
    lim_{x->a} f(x)/g(x) = lim_{c->a} f'(c)/g'(c) = lim_{x->a} f'(x)/g'(x).
QED -- this is an EXACT algebraic identity (f(x)/g(x)=f'(c)/g'(c) for
SOME c) turned into a limit statement by squeezing c, not an
approximation.
"""

import numpy as np
import sympy as sp


def lhopital_limit_symbolic(f_expr, g_expr, var, point, max_applications=5):
    """Apply L'Hopital's rule symbolically: differentiate f and g (as
    many times as needed, up to max_applications, for higher-order 0/0 or
    inf/inf forms) until the limit of f'/g' is no longer indeterminate,
    then evaluate it -- the actual mechanical procedure, not a lookup."""
    f, g = f_expr, g_expr
    for n_applications in range(max_applications + 1):
        f_at_point = f.subs(var, point)
        g_at_point = g.subs(var, point)
        is_00 = f_at_point == 0 and g_at_point == 0
        is_infinf = f_at_point in (sp.oo, -sp.oo) and g_at_point in (sp.oo, -sp.oo)
        if not (is_00 or is_infinf):
            # base case: no longer indeterminate, evaluate directly
            value = sp.limit(f / g, var, point)
            return value, n_applications
        f = sp.diff(f, var)
        g = sp.diff(g, var)
    raise ValueError(f"still indeterminate after {max_applications} applications of L'Hopital's rule")


def cauchy_mvt_identity_check(f_expr, g_expr, var, a, x_test):
    """A direct numerical check of the PROOF's core identity: for some c
    strictly between a and x_test, f(x_test)/g(x_test) should equal
    f'(c)/g'(c) EXACTLY. Rather than solve for c symbolically (messy in
    general), this scans c across (a, x_test) and confirms such a c
    exists (the function f'(c)/g'(c) - f(x)/g(x) changes sign or hits
    zero somewhere in the interval), which is what Cauchy's MVT guarantees."""
    f_prime = sp.diff(f_expr, var)
    g_prime = sp.diff(g_expr, var)
    target = float((f_expr.subs(var, x_test) / g_expr.subs(var, x_test)))

    c_vals = np.linspace(float(a), float(x_test), 2000)[1:-1]   # strictly interior
    residuals = []
    for c in c_vals:
        fp_c = float(f_prime.subs(var, c))
        gp_c = float(g_prime.subs(var, c))
        if abs(gp_c) < 1e-12:
            continue
        residuals.append(fp_c / gp_c - target)
    residuals = np.array(residuals)
    sign_change = np.any(residuals[:-1] * residuals[1:] < 0)
    return sign_change, np.min(np.abs(residuals))


if __name__ == "__main__":
    x = sp.symbols('x')

    print("=== The exact indeterminate form hardcoded in dgs.diffraction_grating ===")
    print("_sinc_unnormalized(x) returns 1.0 for |x|<1e-12 -- this is lim(x->0) sin(x)/x,")
    print("never derived in that module. Deriving it here via L'Hopital's rule:\n")

    f_expr, g_expr = sp.sin(x), x
    value, n_apps = lhopital_limit_symbolic(f_expr, g_expr, x, 0)
    print(f"f(x)=sin(x), g(x)=x: f(0)={f_expr.subs(x,0)}, g(0)={g_expr.subs(x,0)} -- 0/0 indeterminate")
    print(f"L'Hopital ({n_apps} application): lim f'/g' = {value}")
    print(f"matches dgs.diffraction_grating's hardcoded value of 1.0: {value == 1}")

    print("\n=== Verifying Cauchy's MVT identity underlying the proof ===")
    sign_change, min_residual = cauchy_mvt_identity_check(f_expr, g_expr, x, 0, 0.5)
    print(f"some c in (0, 0.5) makes f'(c)/g'(c) EXACTLY equal f(0.5)/g(0.5): "
          f"sign change found = {sign_change}, min |residual| = {min_residual:.2e}")

    print("\n=== A SECOND-ORDER example: L'Hopital applied TWICE ===")
    print("(1-cos(x))/x^2 at x=0: both f and g (and f',g') vanish at x=0 --")
    print("needs L'Hopital applied twice to resolve.")
    f2, g2 = 1 - sp.cos(x), x**2
    value2, n_apps2 = lhopital_limit_symbolic(f2, g2, x, 0)
    print(f"f(0)={f2.subs(x,0)}, g(0)={g2.subs(x,0)} -- 0/0 indeterminate")
    print(f"after 1 application: f'={sp.diff(f2,x)}, g'={sp.diff(g2,x)}, "
          f"f'(0)={sp.diff(f2,x).subs(x,0)}, g'(0)={sp.diff(g2,x).subs(x,0)} -- STILL 0/0")
    print(f"L'Hopital ({n_apps2} applications): limit = {value2}  (matches the well-known 1/2)")
    assert value2 == sp.Rational(1, 2)

    print("\n=== Numerical cross-check: does the actual sinc code converge to L'Hopital's answer? ===")
    from dgs.diffraction_grating import _sinc_unnormalized
    for eps in [1e-2, 1e-4, 1e-6, 1e-10]:
        numeric = _sinc_unnormalized(np.array([eps]))[0]
        direct_sin_over_x = np.sin(eps) / eps
        print(f"  x={eps:.0e}: sinc code returns {numeric:.12f}, "
              f"direct sin(x)/x = {direct_sin_over_x:.12f}, L'Hopital limit = 1.0")
