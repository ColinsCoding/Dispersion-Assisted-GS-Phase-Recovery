"""Radiation patterns, equipotential surfaces, unit sphere geometry.

Griffiths Chapter 11 + EC ENGR 279AS (RF/Microwave/Photonics) material.

PHYSICS:
  Electric dipole radiation (oscillating charge q*d*cos(omega*t)):
    E_theta = (mu_0 * q*omega^2 * d * sin(theta)) / (4*pi) * cos(omega*(t-r/c)) / r
    B_phi   = E_theta / c
    <S>     = (mu_0 * omega^4 * p0^2 * sin^2(theta)) / (32*pi^2 * c) * r_hat / r^2

  Power pattern:  P(theta) = sin^2(theta)   <- the doughnut shape
  Total radiated: P_total = mu_0*omega^4*p0^2 / (12*pi*c)
  Larmor formula: P = mu_0*q^2*a^2 / (6*pi*c)

  UNIT SPHERE: integrating sin^2(theta) over 4pi steradians
    integral_0^pi sin^2(theta) * 2*pi*sin(theta)*d(theta) = 8*pi/3
    -> normalization factor 3/2 for gain pattern (directive gain D=3/2)

  EQUIPOTENTIAL SURFACES:
    For static dipole:  phi = (1/4*pi*eps0) * p*cos(theta)/r^2
    Equipotentials:     cos(theta)/r^2 = const -> r = sqrt(cos(theta)/C)
    These are the "onion" shaped surfaces surrounding the dipole
"""

from __future__ import annotations
import numpy as np
from typing import Dict, List, Tuple

# Physical constants
MU_0   = 4e-7 * np.pi    # H/m
EPS_0  = 8.854e-12        # F/m
C_LIGHT = 2.998e8         # m/s
ETA_0  = np.sqrt(MU_0 / EPS_0)   # free-space impedance ~377 Ohm


# ── Dipole radiation pattern ──────────────────────────────────────────────────
def dipole_power_pattern(theta: np.ndarray) -> np.ndarray:
    """Normalized radiation intensity pattern for Hertzian dipole.

    P(theta) = sin^2(theta)   (doughnut symmetric about z-axis)

    theta : polar angle from z-axis (dipole axis), radians
    """
    return np.sin(theta)**2


def dipole_gain(theta: np.ndarray) -> np.ndarray:
    """Directive gain of Hertzian dipole (normalized to isotropic = 1).

    G(theta) = (3/2) * sin^2(theta)
    Maximum gain = 3/2 = 1.76 dBi at theta = pi/2 (equatorial plane)
    """
    return 1.5 * np.sin(theta)**2


def dipole_poynting(theta: np.ndarray,
                    r: float,
                    omega: float,
                    p0: float) -> Dict:
    """Time-averaged Poynting vector for electric dipole radiation.

    <S> = (mu_0 * omega^4 * p0^2) / (32*pi^2 * c) * sin^2(theta) / r^2

    Parameters
    ----------
    theta  : polar angle (rad)
    r      : observation distance (m)
    omega  : angular frequency (rad/s)
    p0     : dipole moment amplitude (C*m)

    Returns magnitude <|S|> and total radiated power.
    """
    prefactor = MU_0 * omega**4 * p0**2 / (32 * np.pi**2 * C_LIGHT)
    S_mag = prefactor * np.sin(theta)**2 / r**2
    # Total power: P = integral <S> * dA over sphere of radius r
    # = prefactor * (8*pi/3) = mu_0*omega^4*p0^2 / (12*pi*c)
    P_total = MU_0 * omega**4 * p0**2 / (12 * np.pi * C_LIGHT)
    return {
        "S_r":       S_mag,
        "P_total_W": P_total,
        "prefactor": prefactor,
    }


def larmor_radiated_power(q: float, a: float) -> float:
    """Larmor formula: power radiated by accelerating charge.

    P = (mu_0 * q^2 * a^2) / (6*pi * c)
      = (q^2 * a^2) / (6*pi * eps_0 * c^3)

    q : charge (C)
    a : acceleration (m/s^2)
    """
    return MU_0 * q**2 * a**2 / (6 * np.pi * C_LIGHT)


# ── Unit sphere integration ───────────────────────────────────────────────────
def unit_sphere_integral(f_theta_phi,
                          n_theta: int = 200,
                          n_phi: int = 400) -> Dict:
    """Integrate a function f(theta, phi) over the unit sphere.

    Uses Gaussian quadrature via uniform theta,phi grid with proper
    Jacobian: dOmega = sin(theta) * d(theta) * d(phi)

    Returns the solid angle integral and directive gain D_max.
    """
    theta = np.linspace(0, np.pi, n_theta)
    phi   = np.linspace(0, 2*np.pi, n_phi)
    TH, PH = np.meshgrid(theta, phi, indexing="ij")

    F = f_theta_phi(TH, PH)
    # Jacobian: sin(theta) * d_theta * d_phi
    dtheta = theta[1] - theta[0]
    dphi   = phi[1] - phi[0]
    jacobian = np.sin(TH) * dtheta * dphi
    integral = float(np.sum(F * jacobian))
    F_max = float(np.max(F))
    # Directive gain: D = 4*pi * F_max / integral
    D = 4 * np.pi * F_max / integral if integral > 0 else 0
    return {
        "integral":   integral,
        "F_max":      F_max,
        "D_gain":     D,
        "D_dBi":      10 * np.log10(D) if D > 0 else -np.inf,
    }


def sphere_integral_dipole() -> Dict:
    """Verify: integral of sin^2(theta) over sphere = 8*pi/3."""
    # Analytical: int_0^pi sin^3(theta) d(theta) * int_0^{2pi} d(phi)
    #           = 4/3 * 2*pi = 8*pi/3
    analytical = 8 * np.pi / 3
    res = unit_sphere_integral(lambda th, ph: np.sin(th)**2)
    return {
        "numerical":   res["integral"],
        "analytical":  analytical,
        "error_pct":   abs(res["integral"] - analytical) / analytical * 100,
        "D_gain":      res["D_gain"],   # should be 1.5
        "D_dBi":       res["D_dBi"],    # should be 1.76 dBi
    }


# ── Equipotential surfaces ────────────────────────────────────────────────────
def dipole_equipotentials(C_values: List[float],
                           n_theta: int = 200) -> Dict:
    """Compute equipotential surface radii for static electric dipole.

    Electrostatic dipole potential:
        phi(r, theta) = (1/4*pi*eps_0) * p*cos(theta) / r^2

    Equipotential phi = phi_0:
        r^2 = (p / (4*pi*eps_0*phi_0)) * cos(theta)  = K * cos(theta)

    So r(theta) = sqrt(K * cos(theta))  for cos(theta) > 0 (upper hemisphere)

    C_values : list of K = p/(4*pi*eps_0*phi_0) values (m^3)
               (positive = equipotential in upper hemisphere, theta < pi/2)
    """
    theta = np.linspace(0, np.pi/2 - 0.01, n_theta)  # upper hemisphere
    surfaces = {}
    for K in C_values:
        if K > 0:
            r = np.sqrt(K * np.cos(theta))
            x = r * np.sin(theta)
            z = r * np.cos(theta)
            surfaces[K] = {"theta": theta, "r": r, "x": x, "z": z}
    return {"surfaces": surfaces, "theta": theta}


def electric_field_dipole(x: np.ndarray,
                            z: np.ndarray,
                            p: float = 1.0) -> Dict:
    """Electric field of static point dipole p*z_hat at origin.

    E = (p / (4*pi*eps_0)) * [2*cos(theta)*r_hat + sin(theta)*theta_hat] / r^3

    In Cartesian (xz-plane, phi=0):
        E_x = (p / 4*pi*eps_0) * (3xz) / r^5
        E_z = (p / 4*pi*eps_0) * (2z^2 - x^2) / r^5
    """
    r = np.sqrt(x**2 + z**2)
    r = np.where(r < 1e-10, 1e-10, r)  # avoid division by zero
    prefactor = p / (4 * np.pi * EPS_0)
    Ex = prefactor * 3 * x * z / r**5
    Ez = prefactor * (2*z**2 - x**2) / r**5
    phi = prefactor * z / r**3   # potential (p*cos(theta)/r^2)
    return {
        "Ex": Ex, "Ez": Ez,
        "E_mag": np.sqrt(Ex**2 + Ez**2),
        "phi": phi,
        "r": r,
    }


# ── 3D potential energy surfaces ─────────────────────────────────────────────
def potential_energy_surface(V_func,
                              x_range: Tuple[float, float] = (-3, 3),
                              y_range: Tuple[float, float] = (-3, 3),
                              n: int = 200) -> Dict:
    """Compute a 2D potential energy surface V(x, y).

    Returns grid and stability info: minima, saddle points, maxima.

    Example V functions:
      Double-well: V = (x^2 - 1)^2 + 0.1*y^2
      Mexican hat:  V = (x^2 + y^2 - 1)^2 + 0.1*(x^2+y^2)
      Harmonic:     V = 0.5*(kx*x^2 + ky*y^2)
    """
    x = np.linspace(x_range[0], x_range[1], n)
    y = np.linspace(y_range[0], y_range[1], n)
    X, Y = np.meshgrid(x, y)
    V = V_func(X, Y)

    # Gradient and Hessian for stability
    dVdx = np.gradient(V, x[1]-x[0], axis=1)
    dVdy = np.gradient(V, y[1]-y[0], axis=0)
    grad_mag = np.sqrt(dVdx**2 + dVdy**2)

    return {
        "X": X, "Y": Y, "V": V,
        "grad_mag": grad_mag,
        "V_min": float(V.min()),
        "V_max": float(V.max()),
        "x": x, "y": y,
    }


def double_well_stability(a: float = 1.0, b: float = 0.1) -> Dict:
    """Double-well potential V = (x^2-a)^2 + b*y^2.

    Minima at x=±sqrt(a), y=0 (stable equilibria).
    Saddle at x=0, y=0 (unstable — Noether: broken symmetry at saddle).
    """
    surf = potential_energy_surface(
        lambda x, y: (x**2 - a)**2 + b * y**2)
    x_min = np.sqrt(a)
    V_min = 0.0    # by construction
    V_saddle = a**2
    # Hessian at (±sqrt(a), 0): d^2V/dx^2 = 4*(3x^2-a), d^2V/dy^2 = 2b
    d2V_dx2 = 4 * (3*a - a)   # = 8a > 0 (stable in x)
    d2V_dy2 = 2 * b             # stable in y
    return {
        **surf,
        "minima_x":    [x_min, -x_min],
        "V_min":       V_min,
        "V_saddle":    V_saddle,
        "V_barrier":   V_saddle - V_min,  # = a^2
        "omega_x":     np.sqrt(d2V_dx2),  # small oscillation frequency
        "omega_y":     np.sqrt(d2V_dy2),
        "stable":      True,
    }


# ── Noether charge for radiation ──────────────────────────────────────────────
def noether_charge_EM() -> Dict:
    """Noether charges for electromagnetic field.

    Symmetry             -> Conserved quantity
    Time translation     -> Energy  U = (eps_0/2)*(E^2 + c^2*B^2) * Vol
    Space translation    -> Momentum p = eps_0 * (E x B) * Vol = S/c^2
    Rotation             -> Angular momentum L = r x p
    U(1) gauge symmetry  -> Electric charge Q = integral rho dV

    Radiation connection:
      The emitted Poynting flux <S> carries energy and momentum.
      Larmor: dU/dt = P = mu_0*q^2*a^2/(6*pi*c)
      Radiation reaction force (Abraham-Lorentz):
        F_rad = mu_0*q^2/(6*pi*c) * da/dt
      This is the back-reaction of emitted Noether charge on the source.
    """
    import sympy as sp

    t, omega_s, p0, q, a = sp.symbols("t omega p_0 q a", positive=True)
    mu0, eps0, c = sp.Symbol("mu_0"), sp.Symbol("eps_0"), sp.Symbol("c")
    r, theta = sp.Symbol("r", positive=True), sp.Symbol("theta", real=True)

    # Larmor formula
    P_larmor = mu0 * q**2 * a**2 / (6 * sp.pi * c)
    # Dipole radiated power
    P_dipole = mu0 * omega_s**4 * p0**2 / (12 * sp.pi * c)
    # Poynting magnitude on sphere
    S_theta = (mu0 * omega_s**4 * p0**2 * sp.sin(theta)**2 /
               (32 * sp.pi**2 * c * r**2))
    # U(1) charge conservation
    charge_conserv = sp.Symbol("dQ_dt")

    return {
        "P_Larmor":      sp.Eq(sp.Symbol("P_Larmor"), P_larmor),
        "P_dipole":      sp.Eq(sp.Symbol("P_dipole"), P_dipole),
        "Poynting_S":    sp.Eq(sp.Symbol("S(r,theta)"), S_theta),
        "charge_consv":  sp.Eq(charge_conserv, 0),
        "D_max_dBi":     sp.Rational(3, 2),   # = 1.76 dBi
        "sphere_integral": sp.Rational(8, 1) * sp.pi / 3,
    }


# ── SymPy: 5 key radiation equations ─────────────────────────────────────────
def radiation_sympy_5() -> Dict:
    """5 key radiation equations for sp.init_printing."""
    import sympy as sp

    theta, r, omega_s = sp.Symbol("theta"), sp.Symbol("r"), sp.Symbol("omega")
    p0, q, a = sp.Symbol("p_0", positive=True), sp.Symbol("q"), sp.Symbol("a")
    mu0, c = sp.Symbol("mu_0"), sp.Symbol("c")

    # 1. Time-averaged Poynting vector (dipole)
    S = mu0 * omega_s**4 * p0**2 * sp.sin(theta)**2 / (32 * sp.pi**2 * c * r**2)
    eq1 = sp.Eq(sp.Symbol("<S>"), S)

    # 2. Larmor formula
    P = mu0 * q**2 * a**2 / (6 * sp.pi * c)
    eq2 = sp.Eq(sp.Symbol("P_Larmor"), P)

    # 3. Dipole gain pattern G(theta) = (3/2)*sin^2(theta)
    eq3 = sp.Eq(sp.Symbol("G(theta)"), sp.Rational(3, 2) * sp.sin(theta)**2)

    # 4. Sphere integral: int_0^{4pi} sin^2(theta) dOmega = 8*pi/3
    eq4 = sp.Eq(sp.Symbol("Integral_sin2_dOmega"),
                sp.Rational(8, 1) * sp.pi / 3)

    # 5. Equipotential surface of static dipole
    K = sp.Symbol("K", positive=True)
    r_eq = sp.sqrt(K * sp.cos(theta))
    eq5 = sp.Eq(sp.Symbol("r_equip(theta)"), r_eq)

    return {
        "Poynting_dipole":   eq1,
        "Larmor":            eq2,
        "Gain_pattern":      eq3,
        "Sphere_integral":   eq4,
        "Equipotential_r":   eq5,
    }


if __name__ == "__main__":
    import sympy as sp

    print("=== Dipole Radiation: Unit Sphere Integration ===")
    sph = sphere_integral_dipole()
    print(f"  Integral of sin^2 over sphere: {sph['numerical']:.5f}")
    print(f"  Analytical:                    {sph['analytical']:.5f}")
    print(f"  Error:                         {sph['error_pct']:.4f}%")
    print(f"  Directive gain D:              {sph['D_gain']:.4f}  (expected 1.5)")
    print(f"  Gain in dBi:                   {sph['D_dBi']:.4f}  (expected 1.76)")

    print("\n=== Larmor Radiation: Electron at a=1e15 m/s^2 ===")
    P = larmor_radiated_power(1.602e-19, 1e15)
    print(f"  P_Larmor = {P:.3e} W")

    print("\n=== Dipole Poynting at r=1m, theta=pi/2 ===")
    omega_demo = 2 * np.pi * 1e9   # 1 GHz
    p0_demo    = 1e-9              # 1 nC*m
    sp_res = dipole_poynting(np.pi/2, 1.0, omega_demo, p0_demo)
    print(f"  <S>(r=1m, theta=90deg) = {sp_res['S_r']:.3e} W/m^2")
    print(f"  P_total = {sp_res['P_total_W']:.3e} W")

    print("\n=== Double-Well Potential Stability ===")
    dw = double_well_stability(a=1.0, b=0.1)
    print(f"  Minima at x = ±{dw['minima_x'][0]:.3f}")
    print(f"  V_barrier = {dw['V_barrier']:.3f}")
    print(f"  omega_x = {dw['omega_x']:.3f} rad/s (small oscillation)")
    print(f"  omega_y = {dw['omega_y']:.3f} rad/s")

    print("\n=== Noether Charges (EM field) ===")
    nc = noether_charge_EM()
    for k, v in nc.items():
        print(f"  {k}: {v}")

    print("\n=== 5 SymPy Equations ===")
    sp.init_printing(use_latex=False)
    for name, eq in radiation_sympy_5().items():
        print(f"  [{name}]  {eq}")
