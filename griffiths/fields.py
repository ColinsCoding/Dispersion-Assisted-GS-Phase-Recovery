"""Field classification, potentials, the r-hat/r^2 paradox, and trig substitution.

Backs the grad/curl/div applications notebook: conservative vs solenoidal
fields, scalar potentials by radial path integration, the divergence
"paradox" that delta^3(r) resolves (Griffiths Sec. 1.5.3), and a
step-showing trig-substitution helper.
"""

import sympy as sp

from .vectors import CARTESIAN, _as_vec3, _check_vars, curl, div, grad


def is_conservative(F, vars=CARTESIAN):
    """True iff curl F = 0 identically (so F = -grad V exists on R^3)."""
    return sp.simplify(curl(F, vars)) == sp.zeros(3, 1)


def is_solenoidal(F, vars=CARTESIAN):
    """True iff div F = 0 identically (so F = curl A exists)."""
    return sp.simplify(div(F, vars)) == 0


def scalar_potential(F, vars=CARTESIAN):
    """Scalar potential V with F = -grad V, via the radial path integral

        V(r) = -int_0^1 F(t r) . r dt

    Raises if F is not conservative (curl != 0) -- no potential exists.
    """
    vars = _check_vars(vars)
    F = _as_vec3(F, "F")
    if not is_conservative(F, vars):
        raise ValueError("field is not conservative (curl F != 0); no scalar potential")
    t = sp.Symbol("t", positive=True)
    radial = {v: t * v for v in vars}
    integrand = sum(F[i].subs(radial, simultaneous=True) * vars[i] for i in range(3))
    return sp.simplify(-sp.integrate(integrand, (t, 0, 1)))


def rr2_field(vars=CARTESIAN):
    """The field r-hat / r^2 in Cartesian components."""
    vars = _check_vars(vars)
    r = sp.sqrt(sum(v**2 for v in vars))
    return sp.Matrix(vars) / r**3


def rr2_paradox(vars=CARTESIAN):
    """Griffiths Sec. 1.5.3: return (pointwise_div, total_flux) for r-hat/r^2.

    The divergence is 0 everywhere except the origin, yet the flux through
    any sphere is 4*pi -- so div(r-hat/r^2) = 4*pi*delta^3(r).
    """
    pointwise = sp.simplify(div(rr2_field(vars), vars))
    th, ph = sp.symbols("theta phi", nonnegative=True)
    # (r-hat/r^2) . r-hat dA = (1/r^2) r^2 sin(th) dth dph = sin(th) dth dph
    flux = sp.integrate(sp.sin(th), (th, 0, sp.pi), (ph, 0, 2 * sp.pi))
    return pointwise, flux


def trig_substitution(integrand, var, sub, angle):
    """Carry out the substitution var = sub(angle) inside an integral.

    Returns the transformed integrand (old integrand with var replaced,
    times the Jacobian d var/d angle), trig-simplified.  E.g.

        trig_substitution(sqrt(a**2 - x**2), x, a*sin(th), th)

    The caller supplies symbols with the right assumptions (a > 0, etc.)
    so sqrt(a^2 cos^2) collapses to a*cos cleanly.
    """
    sub = sp.sympify(sub)
    if angle not in sub.free_symbols:
        raise ValueError(f"substitution {sub} does not involve the angle {angle}")
    jac = sp.diff(sub, angle)
    new = sp.sympify(integrand).subs(var, sub) * jac
    return sp.trigsimp(sp.simplify(new))


def radial_div_theorem(n, R=None):
    """G1.39: check the divergence theorem for v = r^n r-hat over a sphere.

    Returns (divergence, volume_integral, surface_integral) where
    divergence = (1/r^2) d/dr (r^(n+2)) = (n+2) r^(n-1).  Both integrals
    equal 4 pi R^(n+2) whenever n > -2; n = -2 is the delta-function case
    (volume integral of the naive divergence is 0, surface gives 4 pi) --
    see rr2_paradox().
    """
    n = sp.sympify(n)
    R = sp.Symbol("R", positive=True) if R is None else sp.sympify(R)
    if R.is_number and R <= 0:
        raise ValueError(f"sphere radius R must be positive, got {R}")
    r = sp.Symbol("r", positive=True)
    divergence = sp.simplify(sp.diff(r**2 * r**n, r) / r**2)
    th, ph = sp.symbols("theta phi", nonnegative=True)
    volume = sp.integrate(divergence * r**2 * sp.sin(th),
                          (r, 0, R), (th, 0, sp.pi), (ph, 0, 2 * sp.pi))
    surface = sp.integrate(R**n * R**2 * sp.sin(th),
                           (th, 0, sp.pi), (ph, 0, 2 * sp.pi))
    return divergence, sp.simplify(volume), sp.simplify(surface)


def sifting_property(c=None, var=None):
    """The sifting property as a sympy Eq with a generic test function.

    int f(x) delta(x - c) dx = f(c)
    """
    var = sp.Symbol("x", real=True) if var is None else var
    c = sp.Symbol("c", real=True) if c is None else sp.sympify(c)
    f = sp.Function("f")
    lhs = sp.Integral(f(var) * sp.DiracDelta(var - c), (var, -sp.oo, sp.oo))
    return sp.Eq(lhs, lhs.doit().simplify())


# ── line integrals: the calculus that *is* Kirchhoff's voltage law ──
def line_integral(F, path, t, t0, t1, vars=CARTESIAN):
    """Work integral int F . dr along r(t) = path, for t in [t0, t1].

    F    : 3-vector field in `vars` (x, y, z).
    path : [x(t), y(t), z(t)] -- the parametrized curve.
    Returns the (symbolic) integral. This is the calculus line integral; in a
    circuit it is the EMF/voltage drop along a branch, int E . dl.
    """
    F = _as_vec3(F)
    subs = {vars[i]: path[i] for i in range(3)}
    dr = [sp.diff(p, t) for p in path]
    integrand = sum(F[i].subs(subs) * dr[i] for i in range(3))
    return sp.simplify(sp.integrate(integrand, (t, t0, t1)))


def circulation(F, path, t, vars=CARTESIAN):
    """Closed-loop line integral oint F . dl over one period t in [0, 2*pi].

    For a CONSERVATIVE field this is 0 -- that is exactly Kirchhoff's voltage law
    (sum of voltage drops around a loop = 0). For a NON-conservative field
    (curl != 0, e.g. a Faraday-induced E) it equals the flux of curl F through
    the loop (Stokes), the EMF that forces KVL to carry an extra term.
    """
    return line_integral(F, path, t, 0, 2 * sp.pi, vars)
