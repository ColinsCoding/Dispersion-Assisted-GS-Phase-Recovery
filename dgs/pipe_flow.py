"""Pipe flow (Poiseuille / Hagen-Poiseuille) — calculus 2 in cylindrical coords.

CALCULUS 2 CONNECTION:
  The velocity profile in a pipe is found by integrating the Navier-Stokes
  equation in cylindrical coordinates:

    (1/r) * d/dr (r * dv/dr) = (1/mu) * dP/dz  = const = -G/mu

  Solution: v(r) = (G/4mu) * (R^2 - r^2)   (parabolic Poiseuille profile)

  VOLUME FLOW RATE uses the SHELL METHOD (calculus 2):
    Q = int_0^R v(r) * 2*pi*r dr
      = 2*pi * int_0^R (G/4mu)(R^2 - r^2) * r dr
      = 2*pi * (G/4mu) * [R^2*r^2/2 - r^4/4]_0^R
      = 2*pi * (G/4mu) * R^4/4
      = pi*G*R^4 / (8*mu)            <- Hagen-Poiseuille Law

  The r*dr in the shell integral = Jacobian of cylindrical coordinates.
  This is the SAME integral as the Gaussian integral in polar coords:
    int_0^inf exp(-r^2) * 2*pi*r dr = pi   (Gaussian normalization)

EM + OPTICS ANALOG:
  The parabolic velocity profile v(r) = v_max*(1 - r^2/R^2) is identical
  in form to:
    - Gaussian beam intensity I(r) ~ exp(-2r^2/w^2)  (Gaussian approx)
    - LP01 fiber mode E(r) ~ J_0(u*r/R)              (exact Bessel)
    - GVD phase accumulation phi(omega) ~ omega^2     (parabolic)
  All governed by Laplacian in their respective coordinate systems.

WIRE RESISTANCE ANALOG:
  Electrical resistance: R_elec = rho*L/A = rho*L/(pi*R^2)
  Fluid resistance:      R_fluid = 8*mu*L/(pi*R^4)
  Both scale as L/R^n — fluid much more sensitive to radius (R^4 vs R^2).
  A 10% reduction in pipe radius -> 34% reduction in flow.
"""

from __future__ import annotations
import numpy as np
import sympy as sp
from typing import Dict


# ════════════════════════════════════════════════════════════════════════════
# §1  POISEUILLE FLOW
# ════════════════════════════════════════════════════════════════════════════

def poiseuille_profile(R: float, mu: float, dP_dz: float,
                        n_pts: int = 300) -> Dict:
    """Parabolic velocity profile v(r) = (G/4mu)(R^2 - r^2).

    R      : pipe radius (m)
    mu     : dynamic viscosity (Pa·s)  water~0.001, air~1.8e-5
    dP_dz  : pressure gradient (Pa/m)  negative for flow in +z direction
    """
    G  = -dP_dz   # G > 0 drives flow
    r  = np.linspace(0, R, n_pts)
    v  = (G / (4 * mu)) * (R**2 - r**2)
    v_max = G * R**2 / (4 * mu)
    v_avg = v_max / 2          # average = half max for parabola

    # Volume flow rate — SHELL METHOD (calculus 2)
    # Q = int_0^R v(r) * 2*pi*r dr
    Q_numerical = float(np.trapezoid(v * 2 * np.pi * r, r))
    Q_exact     = np.pi * G * R**4 / (8 * mu)    # Hagen-Poiseuille

    # Fluid resistance (Hagen-Poiseuille)
    # R_fluid = Delta_P / Q = 8*mu / (pi*R^4) per unit length
    R_fluid_per_L = 8 * mu / (np.pi * R**4)

    # Reynolds number (need velocity and density)
    # Re = rho*v_avg*D/mu; for water rho=1000
    rho_water = 1000.0
    Re = rho_water * v_avg * 2 * R / mu

    # Wall shear stress: tau_w = mu * |dv/dr|_{r=R} = G*R/2
    tau_wall = G * R / 2

    return {
        "r":              r,
        "v":              v,
        "v_max":          v_max,
        "v_avg":          v_avg,
        "Q_numerical":    Q_numerical,
        "Q_exact":        Q_exact,
        "Q_error_pct":    abs(Q_numerical - Q_exact) / Q_exact * 100,
        "R_fluid_per_L":  R_fluid_per_L,
        "Re":             Re,
        "laminar":        Re < 2300,
        "tau_wall":       tau_wall,
        "G":              G,
        "R":              R,
        "mu":             mu,
    }


def shell_method_demo(R: float = 0.01, mu: float = 1e-3,
                       dP_dz: float = -100.0) -> Dict:
    """Explicit shell method integration: Q = int_0^R v(r)*2*pi*r dr.

    Demonstrates the calculus 2 technique step by step:
    shell volume element dV = 2*pi*r*dr*dz  (cylindrical shell)
    dQ = v(r) * dA = v(r) * 2*pi*r dr

    Also: compare with disk method (wrong for this problem — shows WHY
    shell method is natural for cylindrical geometry).
    """
    G = -dP_dz
    n_shells = np.array([5, 10, 50, 200, 1000])
    results = {}
    Q_exact = np.pi * G * R**4 / (8 * mu)

    for n in n_shells:
        r_shells = np.linspace(0, R, n+1)
        r_mid    = 0.5 * (r_shells[:-1] + r_shells[1:])
        dr       = r_shells[1] - r_shells[0]
        v_mid    = (G / (4*mu)) * (R**2 - r_mid**2)
        # Shell: dQ = v(r) * 2*pi*r*dr
        dQ       = v_mid * 2 * np.pi * r_mid * dr
        Q_approx = float(dQ.sum())
        results[n] = {
            "Q_approx":  Q_approx,
            "error_pct": abs(Q_approx - Q_exact) / Q_exact * 100,
        }

    return {
        "Q_exact":    Q_exact,
        "n_shells":   list(n_shells),
        "results":    results,
        "convergence": "O(dr^2) — second order because midpoint rule",
        "formula":    "Q = pi*G*R^4 / (8*mu)",
    }


# ════════════════════════════════════════════════════════════════════════════
# §2  ENTRY LENGTH, TURBULENCE, ANNULAR FLOW
# ════════════════════════════════════════════════════════════════════════════

def entry_length(R: float, v_avg: float, mu: float,
                  rho: float = 1000.0) -> Dict:
    """Hydrodynamic entry length: distance for Poiseuille profile to develop.

    L_entry = 0.06 * Re * D  (laminar)
    L_entry = 4.4 * Re^(1/6) * D  (turbulent)
    """
    D  = 2 * R
    Re = rho * v_avg * D / mu
    L_lam  = 0.06 * Re * D
    L_turb = 4.4 * Re**(1/6) * D if Re > 4000 else None
    return {
        "Re": Re, "D": D,
        "L_entry_laminar":    L_lam,
        "L_entry_turbulent":  L_turb,
        "laminar": Re < 2300,
        "transitional": 2300 <= Re <= 4000,
        "turbulent": Re > 4000,
    }


def annular_flow(R_inner: float, R_outer: float,
                  mu: float, dP_dz: float) -> Dict:
    """Poiseuille flow in annular pipe (r_i < r < r_o).

    Solution: v(r) = G/(4*mu) * [R_o^2 - r^2 + (R_o^2 - R_i^2)/ln(R_o/R_i) * ln(r/R_o)]

    This is the coaxial cable / fiber cladding geometry.
    """
    G  = -dP_dz
    ro, ri = R_outer, R_inner
    k  = (ro**2 - ri**2) / np.log(ro/ri)
    r  = np.linspace(ri, ro, 300)
    v  = G/(4*mu) * (ro**2 - r**2 + k * np.log(r/ro))
    Q  = float(2*np.pi * np.trapezoid(v * r, r))
    Q_exact = np.pi*G/(8*mu) * (ro**4 - ri**4 - (ro**2 - ri**2)**2 / np.log(ro/ri))
    return {
        "r": r, "v": v,
        "Q": Q, "Q_exact": Q_exact,
        "Q_error_pct": abs(Q - Q_exact) / abs(Q_exact) * 100,
        "v_max_r": float(r[np.argmax(v)]),
    }


# ════════════════════════════════════════════════════════════════════════════
# §3  REAL VS COMPLEX ANALYSIS IN PIPE FLOW
# ════════════════════════════════════════════════════════════════════════════

def complex_potential_flow(n_pts: int = 60) -> Dict:
    """Complex potential for 2D ideal (inviscid) flow.

    In 2D: velocity field (u, v) derives from complex potential:
      w(z) = phi(x,y) + i*psi(x,y)
      u - iv = dw/dz   (holomorphic!)

    Uniform flow:     w(z) = U*z          -> u=U, v=0
    Flow past circle: w(z) = U*(z + R^2/z)  -> circle theorem
    Line vortex:      w(z) = -i*Gamma/(2*pi) * log(z)  -> branch cut!

    The branch cut of log(z) in potential flow:
      The vortex potential w = -i*Gamma/(2*pi)*Log(z) has a branch cut.
      Going around the vortex: circulation Gamma = closed integral v·dl = 2*pi*Gamma
      Stokes theorem + singularity: the vortex IS the branch point.
    """
    x = np.linspace(-2, 2, n_pts)
    y = np.linspace(-2, 2, n_pts)
    X, Y = np.meshgrid(x, y)
    Z    = X + 1j*Y
    eps  = 1e-3   # avoid singularity

    # Uniform flow
    U = 1.0
    w_uniform = U * Z

    # Flow past circle R=0.5
    R_circle = 0.5
    mask = np.abs(Z) > R_circle + eps
    w_circle = np.where(mask, U * (Z + R_circle**2 / (Z + eps*(~mask))), 0j)

    # Line vortex at origin (branch cut on negative real axis)
    Gamma = 1.0
    w_vortex = -1j * Gamma / (2*np.pi) * np.log(np.abs(Z) + eps)

    # Streamlines = constant Im[w]
    psi_uniform = np.imag(w_uniform)
    psi_circle  = np.imag(w_circle)

    return {
        "X":          X, "Y":          Y,
        "psi_uniform": psi_uniform,
        "psi_circle":  psi_circle,
        "w_vortex":    w_vortex,
        "connection":  "Vortex branch cut = Poiseuille vorticity at wall",
        "real_part":   "velocity potential phi (potential energy)",
        "imag_part":   "stream function psi (mass conservation)",
    }


def real_vs_complex_numbers_summary() -> Dict:
    """Taxonomy: where real vs complex numbers appear in physics.

    REAL numbers:
      - Physical observables (energy, position, probability)
      - Eigenvalues of Hermitian operators (quantum mechanics)
      - Flow velocities in real pipe (Poiseuille v(r) is real)
      - Power spectrum |E(omega)|^2 (what TS-DFT measures directly)

    COMPLEX numbers:
      - Wave amplitudes E(omega) = |E|*exp(i*phi) (amplitude AND phase)
      - Complex potential w(z) = phi + i*psi (potential flow)
      - Transfer function H(s) = 1/(ms^2+cs+k) (Laplace domain)
      - GVD phase factor exp(i*beta2*L*omega^2/2) (what GS recovers)
      - Branch cuts of log: needed when phase is multivalued

    THE FUNDAMENTAL THEOREM connecting real and complex:
      Real observable = Re[complex field] or |complex field|^2
      Phase information (lost in intensity measurement) = Im[log E]
      GS algorithm: recovers Im[log E] from |E|^2 alone.
    """
    return {
        "real_observables": ["energy", "position", "probability", "|E|^2", "v(r)"],
        "complex_fields": ["E(omega)", "w(z)", "H(s)", "exp(i*phi)", "psi(x)"],
        "bridge": "E_real = Re[E_complex] or |E_complex|^2",
        "GS_bridge": "GS recovers Im[log E] (phase) from |E|^2 (intensity)",
        "branch_cut_role": "phase multivaluedness = log branch cut; unwrapping = Riemann sheet choice",
    }


# ════════════════════════════════════════════════════════════════════════════
# §4  SYMPY: 5 EQUATIONS
# ════════════════════════════════════════════════════════════════════════════

def pipe_flow_sympy_5() -> Dict:
    """5 key equations: Poiseuille, shell method, Hagen-Poiseuille, complex potential, Re."""
    r, R, mu, G = sp.symbols("r R mu G", positive=True)
    z, U, Gamma = sp.symbols("z U Gamma", complex=True)

    # 1. Poiseuille velocity profile
    eq1 = sp.Eq(sp.Symbol("v(r)"),
                G / (4*mu) * (R**2 - r**2))

    # 2. Shell method: flow rate integral
    v_r = G / (4*mu) * (R**2 - r**2)
    Q_sym = sp.integrate(v_r * 2 * sp.pi * r, (r, 0, R))
    eq2 = sp.Eq(sp.Symbol("Q"),
                sp.simplify(Q_sym))

    # 3. Hagen-Poiseuille law
    eq3 = sp.Eq(sp.Symbol("Q_HP"),
                sp.pi * G * R**4 / (8 * mu))

    # 4. Complex potential (uniform flow)
    eq4 = sp.Eq(sp.Symbol("w(z)"),
                U * z + U * sp.Symbol("R_c")**2 / z)

    # 5. Reynolds number
    rho, v_avg, D = sp.symbols("rho v_avg D", positive=True)
    eq5 = sp.Eq(sp.Symbol("Re"),
                rho * v_avg * D / mu)

    return {
        "Poiseuille_profile":    eq1,
        "Shell_method_Q":        eq2,
        "Hagen_Poiseuille":      eq3,
        "Complex_potential":     eq4,
        "Reynolds_number":       eq5,
    }


if __name__ == "__main__":
    print("=== Poiseuille Flow: water in 1cm pipe ===")
    p = poiseuille_profile(R=0.01, mu=1e-3, dP_dz=-100.0)
    print(f"  v_max = {p['v_max']:.4f} m/s")
    print(f"  v_avg = {p['v_avg']:.4f} m/s")
    print(f"  Q     = {p['Q_exact']*1e6:.4f} mL/s")
    print(f"  Shell method Q error: {p['Q_error_pct']:.4f}%")
    print(f"  Re    = {p['Re']:.1f}  (laminar: {p['laminar']})")
    print(f"  Wall shear: {p['tau_wall']:.4f} Pa")

    print("\n=== Shell Method Convergence ===")
    s = shell_method_demo()
    for n, r in s["results"].items():
        print(f"  n={n:5d}: error = {r['error_pct']:.4f}%")

    print("\n=== Annular Flow (coaxial geometry) ===")
    ann = annular_flow(R_inner=0.003, R_outer=0.010, mu=1e-3, dP_dz=-100.0)
    print(f"  Q = {ann['Q']*1e6:.4f} mL/s  (error {ann['Q_error_pct']:.4f}%)")
    print(f"  v_max at r = {ann['v_max_r']*1e3:.2f} mm")

    print("\n=== Real vs Complex Numbers ===")
    rv = real_vs_complex_numbers_summary()
    print(f"  Bridge: {rv['bridge']}")
    print(f"  GS:     {rv['GS_bridge']}")

    print("\n=== SymPy: Hagen-Poiseuille from Shell Method ===")
    eqs = pipe_flow_sympy_5()
    print(f"  {eqs['Shell_method_Q']}")
    print(f"  {eqs['Hagen_Poiseuille']}")
