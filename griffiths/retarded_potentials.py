"""Griffiths Ch.10 -- Potentials and Fields: retarded time, retarded
potentials, and the Lienard-Wiechert potentials of a moving point charge.

potentials.py already has the STATIC-looking machinery (V, A, gauge,
d'Alembertian). What Ch.10 adds is that a source's influence doesn't reach
a field point instantaneously -- it takes exactly the light-travel time
scriptr/c to get there. Every formula below reduces to the familiar
electrostatic/magnetostatic one in the v->0 (nothing moving) limit, which
is the sanity check this module actually runs, not just claims.
"""
import numpy as np
import sympy as sp

eps0, mu0, c = sp.symbols('epsilon_0 mu_0 c', positive=True)
C_SI = 299792458.0
EPS0_SI = 8.8541878128e-12


def retarded_time_symbolic():
    """t_r = t - scriptr/c, where scriptr = |r - r'(t_r)| is the distance
    from the (retarded) source position to the field point. Implicit in
    t_r because the source's position at time t_r can itself depend on
    t_r for a moving charge -- this is WHY Lienard-Wiechert potentials are
    harder than just "plug in an earlier time"."""
    t, scriptr, c_s = sp.symbols('t scriptr c', real=True, positive=True)
    t_r = sp.Symbol('t_r', real=True)
    return sp.Eq(t_r, t - scriptr / c_s)


def retarded_potential_formula():
    """V(r,t) = 1/(4*pi*eps0) * Integral[ rho(r', t_r) / scriptr, dtau' ]
    -- Griffiths Eq. 10.19. Same Coulomb's-law INTEGRAND as the static
    potential, except rho is evaluated at the RETARDED time, not the
    present one. Returned as a symbolic Eq for inspection, not evaluated
    (the integral's value depends on a specific, arbitrary rho)."""
    r, rp, t, scriptr = sp.symbols('r rprime t scriptr', real=True, positive=True)
    t_r = sp.Symbol('t_r', real=True)
    rho = sp.Function('rho')
    V = sp.Function('V')
    return sp.Eq(V(r, t), sp.Integral(rho(rp, t_r) / scriptr, rp) / (4 * sp.pi * eps0))


def lienard_wiechert_potentials(q, w_of_t, r_field, t_eval, c_num=C_SI, eps0_num=EPS0_SI,
                                 n_search=2000):
    """Numeric Lienard-Wiechert V, A for a point charge q with trajectory
    w_of_t(t) -> np.array([x,y,z]) at field point r_field and time t_eval.

    Solves for the retarded time t_r implicitly (t_r < t_eval such that
    |r_field - w(t_r)| = c*(t_eval - t_r)) by a direct bisection search --
    the ACTUAL hard part of this problem that a naive "just use t-scriptr/c"
    formula glosses over, because scriptr itself depends on t_r for a
    moving source.

        V = (1/4*pi*eps0) * q*c / (scriptr*c - scriptr_vec . v)
        A = (v/c^2) * V

    (Griffiths Eq. 10.46-10.47.) Returns dict with V, A, t_r, scriptr, and
    the retarded velocity v(t_r) (via finite difference).
    """
    if n_search < 10:
        raise ValueError(f"n_search={n_search}: must be >= 10")
    r_field = np.asarray(r_field, dtype=float)

    def residual(t_r):
        scriptr_vec = r_field - w_of_t(t_r)
        return np.linalg.norm(scriptr_vec) - c_num * (t_eval - t_r)

    # bisection: t_r must be < t_eval; search back far enough that residual changes sign.
    # residual(t_r) is monotonically INCREASING in t_r here (more negative for t_r far in
    # the past, since c*(t_eval-t_r) grows faster than |scriptr_vec| can shrink) -- so a
    # negative residual means the root is further forward (raise lo), not the reverse.
    # Confirmed by direct trace, not assumed: the original `if residual(mid) > 0: lo = mid`
    # had this backwards and collapsed the interval onto the wrong boundary every time.
    lo, hi = t_eval - 1.0, t_eval - 1e-12
    if residual(lo) > 0:
        raise ValueError("search window too small: no retarded time found in [t_eval-1, t_eval)")
    for _ in range(n_search):
        mid = 0.5 * (lo + hi)
        if residual(mid) < 0:
            lo = mid
        else:
            hi = mid
    t_r = 0.5 * (lo + hi)

    h = 1e-9
    v = (w_of_t(t_r + h) - w_of_t(t_r - h)) / (2 * h)

    scriptr_vec = r_field - w_of_t(t_r)
    scriptr = np.linalg.norm(scriptr_vec)
    denom = scriptr * c_num - np.dot(scriptr_vec, v)
    if abs(denom) < 1e-300:
        raise ValueError("denominator ~0: field point on the charge's forward light cone")
    V = (q * c_num) / (4 * np.pi * eps0_num * denom)
    A = (v / c_num**2) * V
    return {"V": V, "A": A, "t_r": t_r, "scriptr": scriptr, "v_retarded": v}


def coulomb_potential(q, r, eps0_num=EPS0_SI):
    """Static Coulomb potential -- the v=0 limit lienard_wiechert_potentials
    must reduce to, checked directly below rather than assumed."""
    if r <= 0:
        raise ValueError("r must be positive")
    return q / (4 * np.pi * eps0_num * r)


if __name__ == "__main__":
    print("=== retarded time ===")
    sp.pprint(retarded_time_symbolic())

    print("\n=== retarded potential formula (Eq. 10.19) ===")
    sp.pprint(retarded_potential_formula())

    print("\n=== Lienard-Wiechert: STATIC charge, must reduce to Coulomb's law ===")
    q = 1e-9
    r_field = np.array([1.0, 0.0, 0.0])

    def w_static(t):
        return np.array([0.0, 0.0, 0.0])

    result = lienard_wiechert_potentials(q, w_static, r_field, t_eval=0.0)
    V_coulomb = coulomb_potential(q, 1.0)
    print(f"Lienard-Wiechert V = {result['V']:.6e} V")
    print(f"Coulomb V          = {V_coulomb:.6e} V")
    print(f"relative difference: {abs(result['V']-V_coulomb)/V_coulomb:.2e}")
    print(f"A (should be ~0, v=0): {result['A']}")

    print("\n=== Lienard-Wiechert: charge moving at constant velocity ===")
    v0 = 1e6  # m/s, along x

    def w_moving(t):
        return np.array([v0 * t, 0.0, 0.0])

    result2 = lienard_wiechert_potentials(q, w_moving, r_field, t_eval=0.0)
    print(f"retarded time t_r = {result2['t_r']:.6e} s  (should be negative: charge was closer in the past)")
    print(f"V = {result2['V']:.6e} V,  A = {result2['A']}")
