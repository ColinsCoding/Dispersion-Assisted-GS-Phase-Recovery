"""Vector-calculus engine for Griffiths Ch. 1 (Problems 1.13, 1.21-1.25).

Everything is SymPy-symbolic, so "check the product rule" means proving it
for arbitrary functions, not spot-checking numbers.
"""

import sympy as sp

# Field point (x, y, z) and source point (x', y', z') -- Griffiths' notation.
x, y, z = sp.symbols("x y z", real=True)
xp, yp, zp = sp.symbols("x' y' z'", real=True)
CARTESIAN = (x, y, z)


def _as_vec3(v, name="vector"):
    """Coerce to a 3x1 sympy Matrix with a clear error if it isn't one."""
    try:
        m = sp.Matrix(v)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            f"{name} must be a sequence of 3 sympy expressions, "
            f"got {type(v).__name__}"
        ) from exc
    if m.shape == (1, 3):
        m = m.T
    if m.shape != (3, 1):
        raise ValueError(
            f"{name} must have exactly 3 components, got shape {m.shape}"
        )
    return m


def _check_vars(vars):
    if len(vars) != 3:
        raise ValueError(f"vars must be 3 coordinate symbols, got {len(vars)}")
    return tuple(vars)


def separation_vector(source=(xp, yp, zp), field=(x, y, z)):
    """Griffiths' script-r: vector from source (x',y',z') to field point (x,y,z)."""
    return _as_vec3(field, "field") - _as_vec3(source, "source")


def separation_length(source=(xp, yp, zp), field=(x, y, z)):
    """Magnitude of the separation vector (script-r in Prob. 1.13)."""
    r = separation_vector(source, field)
    return sp.sqrt(sum(c**2 for c in r))


def grad(f, vars=CARTESIAN):
    """Gradient of a scalar field as a 3x1 Matrix."""
    vars = _check_vars(vars)
    return sp.Matrix([sp.diff(f, v) for v in vars])


def div(A, vars=CARTESIAN):
    """Divergence of a 3-component vector field."""
    vars = _check_vars(vars)
    A = _as_vec3(A, "A")
    return sum(sp.diff(A[i], vars[i]) for i in range(3))


def curl(A, vars=CARTESIAN):
    """Curl of a 3-component vector field as a 3x1 Matrix."""
    vars = _check_vars(vars)
    A = _as_vec3(A, "A")
    return sp.Matrix([
        sp.diff(A[2], vars[1]) - sp.diff(A[1], vars[2]),
        sp.diff(A[0], vars[2]) - sp.diff(A[2], vars[0]),
        sp.diff(A[1], vars[0]) - sp.diff(A[0], vars[1]),
    ])


def a_dot_del(A, B, vars=CARTESIAN):
    """(A . del)B -- Problem 1.22(a).

    Component j is  Ax dBj/dx + Ay dBj/dy + Az dBj/dz : the derivative of B
    along the direction of A, scaled by |A|.  Not a divergence of anything --
    (A . del) is a scalar differential operator applied to each component of B.
    """
    vars = _check_vars(vars)
    A = _as_vec3(A, "A")
    B = _as_vec3(B, "B")
    return sp.Matrix([
        sum(A[i] * sp.diff(B[j], vars[i]) for i in range(3))
        for j in range(3)
    ])


def grad_r_power(n):
    """Problem 1.13: return (computed, predicted) for  del(r^n) = n r^(n-1) rhat,

    where r is the separation length from (x',y',z') to (x,y,z).  Both entries
    are simplified 3x1 Matrices; they should be identical for any n != 0.
    """
    if n == 0:
        raise ValueError("n=0 gives del(1) = 0 identically; nothing to compare")
    rvec = separation_vector()
    rlen = separation_length()
    computed = sp.simplify(grad(rlen**n))
    predicted = sp.simplify(n * rlen**(n - 1) * (rvec / rlen))
    return computed, predicted


def _generic_fields(vars):
    """Two generic scalar fields and two generic vector fields of (x,y,z)."""
    f = sp.Function("f")(*vars)
    g = sp.Function("g")(*vars)
    A = sp.Matrix([sp.Function(f"A_{v}")(*vars) for v in "xyz"])
    B = sp.Matrix([sp.Function(f"B_{v}")(*vars) for v in "xyz"])
    return f, g, A, B


# Griffiths' six product rules, Eq. 1.41-1.46. Each entry maps the rule
# number to (lhs, rhs) builders taking (f, g, A, B, vars).
PRODUCT_RULES = {
    "i":   "grad(f*g) = f grad(g) + g grad(f)",
    "ii":  "grad(A.B) = A x (curl B) + B x (curl A) + (A.del)B + (B.del)A",
    "iii": "div(f*A)  = f (div A) + A . grad(f)",
    "iv":  "div(A x B) = B . (curl A) - A . (curl B)",
    "v":   "curl(f*A) = f (curl A) - A x grad(f)",
    "vi":  "curl(A x B) = (B.del)A - (A.del)B + A (div B) - B (div A)",
}

QUOTIENT_RULES = {
    "grad": "grad(f/g) = (g grad(f) - f grad(g)) / g^2",
    "div":  "div(A/g)  = (g (div A) - A . grad(g)) / g^2",
    "curl": "curl(A/g) = (g (curl A) + A x grad(g)) / g^2",
}


def _rule_sides(rule, f, g, A, B, vars):
    if rule == "i":
        return grad(f * g, vars), f * grad(g, vars) + g * grad(f, vars)
    if rule == "ii":
        lhs = grad(A.dot(B), vars)
        rhs = (A.cross(curl(B, vars)) + B.cross(curl(A, vars))
               + a_dot_del(A, B, vars) + a_dot_del(B, A, vars))
        return lhs, rhs
    if rule == "iii":
        lhs = div(f * A, vars)
        rhs = f * div(A, vars) + A.dot(grad(f, vars))
        return sp.Matrix([lhs]), sp.Matrix([rhs])
    if rule == "iv":
        lhs = div(A.cross(B), vars)
        rhs = B.dot(curl(A, vars)) - A.dot(curl(B, vars))
        return sp.Matrix([lhs]), sp.Matrix([rhs])
    if rule == "v":
        return curl(f * A, vars), f * curl(A, vars) - A.cross(grad(f, vars))
    if rule == "vi":
        lhs = curl(A.cross(B), vars)
        rhs = (a_dot_del(B, A, vars) - a_dot_del(A, B, vars)
               + A * div(B, vars) - B * div(A, vars))
        return lhs, rhs
    raise ValueError(
        f"rule must be one of {sorted(PRODUCT_RULES)}, got {rule!r}"
    )


def check_product_rule(rule, f=None, g=None, A=None, B=None, vars=CARTESIAN):
    """Verify Griffiths product rule i/ii/iii/iv/v/vi.

    With no operands supplied, the rule is proven for fully generic
    functions of (x,y,z).  Pass concrete f/g/A/B (e.g. Problem 1.25) to
    check each side term-by-term.  Returns (lhs, rhs, holds: bool).
    """
    vars = _check_vars(vars)
    gf, gg, gA, gB = _generic_fields(vars)
    f = gf if f is None else sp.sympify(f)
    g = gg if g is None else sp.sympify(g)
    A = gA if A is None else _as_vec3(A, "A")
    B = gB if B is None else _as_vec3(B, "B")
    lhs, rhs = _rule_sides(rule, f, g, A, B, vars)
    diff = sp.simplify(sp.expand(lhs - rhs))
    holds = diff == sp.zeros(*diff.shape)
    return sp.simplify(lhs), sp.simplify(rhs), holds


def check_quotient_rule(which, f=None, g=None, A=None, vars=CARTESIAN):
    """Verify the three quotient rules of Problem 1.24. Returns (lhs, rhs, holds)."""
    vars = _check_vars(vars)
    gf, gg, gA, _ = _generic_fields(vars)
    f = gf if f is None else sp.sympify(f)
    g = gg if g is None else sp.sympify(g)
    A = gA if A is None else _as_vec3(A, "A")
    if which == "grad":
        lhs = grad(f / g, vars)
        rhs = (g * grad(f, vars) - f * grad(g, vars)) / g**2
    elif which == "div":
        lhs = sp.Matrix([div(A / g, vars)])
        rhs = sp.Matrix([(g * div(A, vars) - A.dot(grad(g, vars))) / g**2])
    elif which == "curl":
        lhs = curl(A / g, vars)
        rhs = (g * curl(A, vars) + A.cross(grad(g, vars))) / g**2
    else:
        raise ValueError(
            f"which must be one of {sorted(QUOTIENT_RULES)}, got {which!r}"
        )
    diff = sp.simplify(sp.together(sp.expand(lhs - rhs)))
    holds = diff == sp.zeros(*diff.shape)
    return sp.simplify(lhs), sp.simplify(rhs), holds
