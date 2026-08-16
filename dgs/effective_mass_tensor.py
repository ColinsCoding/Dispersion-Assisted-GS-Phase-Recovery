"""effective_mass_tensor.py -- Newton's second law, generalized for a
crystal: a_i = SUM_j (m*^-1)_ij F_j, a TENSOR equation, not the scalar
a=F/m every intro course teaches. This is not an abstraction for its own
sake -- it's the actual reason silicon's conduction band (six ANISOTROPIC
valleys along the <100> directions, each an ellipsoid with a heavy
longitudinal mass m_l~0.98*m0 and light transverse mass m_t~0.19*m0) needs
a real device model, while GaAs's single, ISOTROPIC conduction-band
minimum (m*~0.067*m0) reduces exactly back to scalar F=ma -- and why
"strained silicon" CMOS deliberately breaks the valley symmetry to
selectively enhance electron mobility in the channel direction.

SEMICLASSICAL DERIVATION (the reason the tensor shows up at all):
  hbar * dk/dt = F                      (crystal momentum responds to force)
  v_i = (1/hbar) * dE/dk_i              (group velocity of the wave packet)
  => dv_i/dt = (1/hbar) * SUM_j (d^2E/dk_i dk_j) * dk_j/dt
             = SUM_j (1/hbar^2) * (d^2E/dk_i dk_j) * F_j
             = SUM_j (m*^-1)_ij * F_j
The inverse effective-mass tensor (m*^-1)_ij = (1/hbar^2) * d^2E/dk_i dk_j
is literally the (scaled) Hessian of the band structure E(k) -- curvature
of the energy band IS the (inverse) mass. A flat band -> heavy, sluggish
carriers; a sharply curved band -> light, fast carriers.
"""

from __future__ import annotations
import numpy as np
import sympy as sp

HBAR = 1.054571817e-34   # J*s
M_ELECTRON = 9.1093837015e-31   # kg, free-electron rest mass


def _validate_positive(**kwargs) -> None:
    for name, value in kwargs.items():
        if value <= 0:
            raise ValueError(f"{name} must be > 0, got {value}")


# ── 1. Inverse effective-mass tensor from band curvature ────────────────────

def inverse_mass_tensor_numeric(E_func, k0: np.ndarray, h: float = 1e6) -> np.ndarray:
    """(m*^-1)_ij = (1/hbar^2) * d^2E/dk_i dk_j, via a central-difference
    numerical Hessian -- works for ANY E(k), not just an analytic model.
    `h` is the finite-difference step in k-space (rad/m); default matches
    a reasonable fraction of a typical Brillouin-zone-boundary k value
    (~1e10 rad/m for silicon), i.e. h ~ 1e-4 of that scale."""
    k0 = np.asarray(k0, dtype=float)
    n = len(k0)
    hessian = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            kpp, kpm, kmp, kmm = k0.copy(), k0.copy(), k0.copy(), k0.copy()
            kpp[i] += h; kpp[j] += h
            kpm[i] += h; kpm[j] -= h
            kmp[i] -= h; kmp[j] += h
            kmm[i] -= h; kmm[j] -= h
            hessian[i, j] = (E_func(kpp) - E_func(kpm) - E_func(kmp) + E_func(kmm)) / (4 * h * h)
    return hessian / HBAR**2


def silicon_valley_energy_symbolic():
    """The standard anisotropic parabolic model for ONE of silicon's six
    conduction-band valleys (along +/-kx here, WLOG): heavy along the
    valley axis (m_l), light transverse to it (m_t) -- an ellipsoidal
    constant-energy surface, not a sphere. Returns (E, (kx,ky,kz), (kx0,ml,mt))."""
    kx, ky, kz, kx0, ml, mt = sp.symbols("k_x k_y k_z k_x0 m_l m_t", positive=True)
    E = (sp.Rational(1, 2) * sp.Symbol("hbar", positive=True)**2
         * ((kx - kx0)**2 / ml + ky**2 / mt + kz**2 / mt))
    return E, (kx, ky, kz), (kx0, ml, mt)


def inverse_mass_tensor_symbolic(E, k_vars) -> sp.Matrix:
    """(m*^-1)_ij via SymPy's Hessian -- the exact symbolic tensor, for
    cross-checking inverse_mass_tensor_numeric's finite-difference result."""
    hbar_sym = sp.Symbol("hbar", positive=True)
    H = sp.hessian(E, k_vars)
    return sp.simplify(H / hbar_sym**2)


def verify_silicon_valley_is_diagonal(m_l: float = 0.98, m_t: float = 0.19) -> bool:
    """CHECKED: the silicon-valley model's inverse mass tensor is EXACTLY
    diag(1/m_l, 1/m_t, 1/m_t) -- not merely "approximately anisotropic,"
    the specific diagonal values match the input parameters exactly."""
    E, k_vars, params = silicon_valley_energy_symbolic()
    inv_mass = inverse_mass_tensor_symbolic(E, k_vars)
    kx0_s, ml_s, mt_s = params
    subs = {kx0_s: 0, ml_s: m_l, mt_s: m_t}
    inv_mass_numeric = inv_mass.subs(subs)
    expected = sp.diag(sp.Rational(1, 1) / m_l, sp.Rational(1, 1) / m_t, sp.Rational(1, 1) / m_t)
    diff = sp.simplify(inv_mass_numeric - sp.Matrix(expected))
    if diff != sp.zeros(3, 3):
        raise AssertionError(f"silicon valley inverse mass tensor not diagonal as expected: leftover {diff}")
    return True


# ── 2. The generalized Newton's second law: a tensor equation ───────────────

def tensor_acceleration(F: np.ndarray, inv_mass_tensor: np.ndarray) -> np.ndarray:
    """a = (m*^-1) . F -- the ANISOTROPIC generalization of a=F/m.
    inv_mass_tensor must be a real, symmetric NxN matrix (units 1/kg)."""
    F = np.asarray(F, dtype=float)
    inv_mass_tensor = np.asarray(inv_mass_tensor, dtype=float)
    if inv_mass_tensor.shape != (len(F), len(F)):
        raise ValueError(f"inv_mass_tensor shape {inv_mass_tensor.shape} doesn't match F length {len(F)}")
    if not np.allclose(inv_mass_tensor, inv_mass_tensor.T, atol=1e-30 * max(np.abs(inv_mass_tensor).max(), 1)):
        raise ValueError("inv_mass_tensor must be symmetric (a real band-structure Hessian always is)")
    return inv_mass_tensor @ F


def acceleration_is_parallel_to_force(F: np.ndarray, inv_mass_tensor: np.ndarray,
                                      cos_atol: float = 1e-9) -> bool:
    """Whether a is parallel to F -- TRUE for isotropic (scalar) mass
    (a=F/m*, always parallel by construction), generally FALSE for an
    anisotropic tensor unless F happens to align with a principal axis.
    Checked via the cosine-of-the-angle-between-them test, not assumed
    from "it's anisotropic so it must differ."

    BUG THIS SESSION CAUGHT: an earlier version reused `cos_atol` as BOTH
    the cosine tolerance (dimensionless, ~1e-9 is fine) AND the "is F/a
    numerically zero" magnitude guard (units of Newtons/(m/s^2) --
    comparing a real force in Newtons, e.g. ~1e-18 N at atomic scale,
    against a 1e-9 threshold made the zero-guard fire by accident on every
    realistic force, silently returning True regardless of the real
    angle. The zero-guard now uses a fixed, scale-appropriate epsilon
    instead of reusing the caller-facing cosine tolerance."""
    F = np.asarray(F, dtype=float)
    a = tensor_acceleration(F, inv_mass_tensor)
    F_norm, a_norm = np.linalg.norm(F), np.linalg.norm(a)
    zero_guard = 1e-300   # only trips for a genuinely (numerically) zero vector
    if F_norm < zero_guard or a_norm < zero_guard:
        return True
    cos_angle = np.dot(F, a) / (F_norm * a_norm)
    return bool(abs(cos_angle - 1.0) < cos_atol)


# ── 3. GaAs comparison: the isotropic special case ───────────────────────────

def gaas_inverse_mass_tensor(m_star: float = 0.067) -> np.ndarray:
    """GaAs has a single, ISOTROPIC conduction-band minimum at the Gamma
    point -- (m*^-1) = (1/m*) * Identity exactly, recovering scalar F=ma
    as a special case of the general tensor equation (not a different
    equation, the SAME one with a degenerate Hessian)."""
    _validate_positive(m_star=m_star)
    return np.eye(3) / (m_star * M_ELECTRON)


def silicon_inverse_mass_tensor(m_l: float = 0.98, m_t: float = 0.19) -> np.ndarray:
    """Numeric (kg^-1) version of the silicon valley's inverse mass tensor,
    for use with tensor_acceleration."""
    _validate_positive(m_l=m_l, m_t=m_t)
    return np.diag([1.0 / (m_l * M_ELECTRON), 1.0 / (m_t * M_ELECTRON), 1.0 / (m_t * M_ELECTRON)])


# ── 4. Semiclassical self-consistency: verify via an independent trajectory ─

def verify_semiclassical_trajectory(E_func, k0: np.ndarray, F: np.ndarray,
                                    dt: float = 1e-16, atol_rel: float = 1e-3) -> bool:
    """CHECKED independently of the Hessian formula: simulate hbar*dk/dt=F
    for a short time step (k(t)=k0+F*t/hbar exactly, since F is constant),
    compute the group velocity v=grad_k(E)/hbar at two nearby times via
    finite differences, and confirm the resulting dv/dt matches
    tensor_acceleration's prediction -- an independent numerical check of
    the same physics, not a restatement of the Hessian algebra."""
    k0 = np.asarray(k0, dtype=float)
    F = np.asarray(F, dtype=float)
    n = len(k0)

    def group_velocity(k, h=1e5):
        v = np.zeros(n)
        for i in range(n):
            kp, km = k.copy(), k.copy()
            kp[i] += h
            km[i] -= h
            v[i] = (E_func(kp) - E_func(km)) / (2 * h) / HBAR
        return v

    k_t0 = k0
    k_t1 = k0 + F * dt / HBAR
    v_t0 = group_velocity(k_t0)
    v_t1 = group_velocity(k_t1)
    a_finite_diff = (v_t1 - v_t0) / dt

    inv_mass = inverse_mass_tensor_numeric(E_func, k0)
    a_predicted = tensor_acceleration(F, inv_mass)

    rel_err = np.linalg.norm(a_finite_diff - a_predicted) / max(np.linalg.norm(a_predicted), 1e-30)
    if rel_err > atol_rel:
        raise AssertionError(f"trajectory-based acceleration disagrees with tensor formula: "
                             f"finite-diff={a_finite_diff}, predicted={a_predicted}, rel_err={rel_err:.2e}")
    return True


if __name__ == "__main__":
    print("=== 1. Silicon valley: inverse mass tensor is exactly diagonal, checked ===")
    ok_diag = verify_silicon_valley_is_diagonal()
    print(f"  diag(1/m_l, 1/m_t, 1/m_t) confirmed symbolically: {ok_diag}")

    Si_inv_mass = silicon_inverse_mass_tensor()
    GaAs_inv_mass = gaas_inverse_mass_tensor()

    print("\n=== 2. Force applied off-axis: acceleration NOT parallel to force in silicon ===")
    F_applied = np.array([1e-18, 1e-18, 0.0])   # N, at 45 deg in the (kx,ky) plane
    a_Si = tensor_acceleration(F_applied, Si_inv_mass)
    a_GaAs = tensor_acceleration(F_applied, GaAs_inv_mass)
    parallel_Si = acceleration_is_parallel_to_force(F_applied, Si_inv_mass)
    parallel_GaAs = acceleration_is_parallel_to_force(F_applied, GaAs_inv_mass)
    print(f"  F = {F_applied} N")
    print(f"  Si (anisotropic):   a = {a_Si} m/s^2   a parallel to F: {parallel_Si}")
    print(f"  GaAs (isotropic):   a = {a_GaAs} m/s^2   a parallel to F: {parallel_GaAs}")

    print("\n=== 3. Semiclassical trajectory check: independent of the Hessian formula ===")
    E_si, k_vars, params = silicon_valley_energy_symbolic()
    kx0_s, ml_s, mt_s = params
    # ml_s, mt_s are DIMENSIONLESS ratios (m/m0) in the symbolic model --
    # substitute the actual kg masses (0.98*M_ELECTRON, not bare 0.98) so
    # E_func has correct SI energy units throughout, with no hacky
    # post-multiply needed to compensate
    E_si_numeric = E_si.subs({kx0_s: 0, ml_s: 0.98 * M_ELECTRON, mt_s: 0.19 * M_ELECTRON,
                              sp.Symbol("hbar", positive=True): HBAR})
    E_si_func = sp.lambdify(k_vars, E_si_numeric, "numpy")
    E_func_vec = lambda k: float(E_si_func(k[0], k[1], k[2]))

    inv_mass_from_numeric_hessian = inverse_mass_tensor_numeric(E_func_vec, np.array([0.0, 0.0, 0.0]))
    print(f"  numeric Hessian inverse mass tensor matches silicon_inverse_mass_tensor(): "
          f"{np.allclose(inv_mass_from_numeric_hessian, Si_inv_mass, rtol=1e-3)}")

    ok_traj = verify_semiclassical_trajectory(E_func_vec, np.array([0.0, 0.0, 0.0]), F_applied)
    print(f"  finite-difference trajectory acceleration matches tensor formula: {ok_traj}")

    print("\nThis is why 'effective mass' isn't a fudge factor: it's the curvature of the")
    print("actual band structure, and when that curvature is anisotropic (silicon's")
    print("valleys), acceleration genuinely does NOT point along the applied force --")
    print("a real consequence device engineers design around (strained-Si mobility enhancement).")
