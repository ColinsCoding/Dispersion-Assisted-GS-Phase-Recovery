"""Gauge invariance in electrodynamics: the scalar potential V and vector
potential A are not unique -- you can shift them by a gauge transformation
without changing the physical E and B fields.

    A -> A + grad(lambda)
    V -> V - d(lambda)/dt

E = -grad(V) - dA/dt  and  B = curl(A) are both UNCHANGED by this shift,
because curl(grad(lambda)) = 0 and the two time-derivative terms cancel.
This module verifies that symbolically for an explicit lambda(x,y,z,t), and
implements the two standard gauge choices (Coulomb, Lorenz) used to simplify
Maxwell's equations into wave equations for V and A.
"""
import sympy as sp


x, y, z, t = sp.symbols('x y z t', real=True)
c = sp.Symbol('c', positive=True)
mu0, eps0 = sp.symbols('mu_0 epsilon_0', positive=True)


def gauge_transform(V, A, lam):
    """Apply A -> A + grad(lam), V -> V - d(lam)/dt.
    V: scalar function of (x,y,z,t). A: sp.Matrix([Ax,Ay,Az]) of (x,y,z,t).
    lam: scalar gauge function lambda(x,y,z,t)."""
    grad_lam = sp.Matrix([sp.diff(lam, x), sp.diff(lam, y), sp.diff(lam, z)])
    A_new = A + grad_lam
    V_new = V - sp.diff(lam, t)
    return V_new, A_new


def E_field(V, A):
    """E = -grad(V) - dA/dt."""
    grad_V = sp.Matrix([sp.diff(V, x), sp.diff(V, y), sp.diff(V, z)])
    dA_dt = sp.Matrix([sp.diff(a, t) for a in A])
    return -grad_V - dA_dt


def B_field(A):
    """B = curl(A)."""
    Ax, Ay, Az = A
    return sp.Matrix([
        sp.diff(Az, y) - sp.diff(Ay, z),
        sp.diff(Ax, z) - sp.diff(Az, x),
        sp.diff(Ay, x) - sp.diff(Ax, y),
    ])


def verify_gauge_invariance(V, A, lam):
    """Check that E and B computed from (V,A) match those from the gauge-
    transformed (V',A') -- i.e. the physical fields don't change."""
    E_before = E_field(V, A)
    B_before = B_field(A)
    V_new, A_new = gauge_transform(V, A, lam)
    E_after = sp.simplify(E_field(V_new, A_new))
    B_after = sp.simplify(B_field(A_new))
    E_match = bool(sp.simplify(E_before - E_after) == sp.zeros(3, 1))
    B_match = bool(sp.simplify(B_before - B_after) == sp.zeros(3, 1))
    return {"E_before": E_before, "E_after": E_after, "E_invariant": E_match,
            "B_before": B_before, "B_after": B_after, "B_invariant": B_match}


def coulomb_gauge_condition():
    """Coulomb gauge: div(A) = 0. Simplifies V's equation to Poisson's
    equation (instantaneous, non-physical-looking but mathematically fine
    because A carries the retardation)."""
    Ax, Ay, Az = sp.symbols('A_x A_y A_z', cls=sp.Function)
    div_A = sp.diff(Ax(x, y, z, t), x) + sp.diff(Ay(x, y, z, t), y) + sp.diff(Az(x, y, z, t), z)
    return sp.Eq(div_A, 0)


def lorenz_gauge_condition():
    """Lorenz gauge: div(A) + (1/c^2) dV/dt = 0. Decouples Maxwell's
    equations into two symmetric wave equations, one for V and one for A --
    the gauge of choice for radiation/dispersion problems."""
    Ax, Ay, Az = sp.symbols('A_x A_y A_z', cls=sp.Function)
    V = sp.Function('V')
    div_A = sp.diff(Ax(x, y, z, t), x) + sp.diff(Ay(x, y, z, t), y) + sp.diff(Az(x, y, z, t), z)
    return sp.Eq(div_A + sp.diff(V(x, y, z, t), t) / c**2, 0)


def gauge_invariance_sympy_5():
    """Five symbolic equations: gauge transform, E invariance statement,
    B invariance statement, Coulomb gauge, Lorenz gauge."""
    lam = sp.Function('lambda')
    A_sym = sp.Function('A')
    V_sym = sp.Function('V')
    return {
        "Gauge_transform_A":
            sp.Eq(sp.Symbol('A_prime'), A_sym(x, y, z, t) + sp.Symbol('grad_lambda')),
        "Gauge_transform_V":
            sp.Eq(sp.Symbol('V_prime'), V_sym(x, y, z, t) - sp.diff(lam(x, y, z, t), t)),
        "E_field_definition":
            sp.Eq(sp.Symbol('E'), -sp.Symbol('grad_V') - sp.Symbol('dA_dt')),
        "Coulomb_gauge": coulomb_gauge_condition(),
        "Lorenz_gauge": lorenz_gauge_condition(),
    }


if __name__ == "__main__":
    print("=== Gauge invariance: plane-wave potentials, lambda = x*y*t ===")
    k = sp.Symbol('k', positive=True)
    omega = sp.Symbol('omega', positive=True)
    V0 = sp.cos(k * x - omega * t)
    A0 = sp.Matrix([sp.sin(k * x - omega * t), 0, 0])
    lam = x * y * t

    res = verify_gauge_invariance(V0, A0, lam)
    print(f"  E invariant under gauge transform: {res['E_invariant']}")
    print(f"  B invariant under gauge transform: {res['B_invariant']}")

    print("\n=== Coulomb gauge condition ===")
    print(f"  {coulomb_gauge_condition()}")

    print("\n=== Lorenz gauge condition ===")
    print(f"  {lorenz_gauge_condition()}")

    print("\n=== SymPy 5 ===")
    for k_, eq in gauge_invariance_sympy_5().items():
        print(f"  {k_}: {eq}")
