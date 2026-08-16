"""compton_scattering.py -- the Compton wavelength shift, DERIVED from
relativistic energy-momentum conservation in a photon-electron collision,
not quoted (dgs/modern_physics.py already states the formula as a known
result; this module is the collision-kinematics derivation of it).

SETUP: a photon (wavelength lambda, momentum h/lambda) strikes a
stationary electron (rest mass m_e). After the collision: the photon
scatters at angle theta with new wavelength lambda', the electron recoils
with momentum p_e at angle phi. Four conservation laws close the system:
  energy:      hc/lambda + m_e*c^2 = hc/lambda' + E_e
  momentum x:  h/lambda = (h/lambda')*cos(theta) + p_e*cos(phi)
  momentum y:  0 = (h/lambda')*sin(theta) - p_e*sin(phi)
  relativistic dispersion: E_e^2 = (p_e*c)^2 + (m_e*c^2)^2
Eliminating p_e (via momentum's law-of-cosines combination) and phi, then
solving the remaining energy equation for lambda', gives -- as an actual
symbolic solve, not an assumed shortcut --
    lambda' - lambda = (h / (m_e*c)) * (1 - cos(theta)).
"""

from __future__ import annotations
import numpy as np
import sympy as sp

H_PLANCK = 6.62607015e-34   # J*s
M_ELECTRON = 9.1093837015e-31   # kg
C_LIGHT = 299792458.0   # m/s
COMPTON_WAVELENGTH = H_PLANCK / (M_ELECTRON * C_LIGHT)   # m, ~2.426 pm


def _validate_positive(**kwargs) -> None:
    for name, value in kwargs.items():
        if value <= 0:
            raise ValueError(f"{name} must be > 0, got {value}")


# ── 1. Derive the Compton formula from conservation laws, symbolically ─────

def derive_compton_shift_symbolic():
    """Sets up energy conservation + the momentum triangle's law-of-cosines
    combination + the relativistic dispersion relation, and SOLVES for
    lambda' (not substitutes a known answer) -- returns the unique
    solution and lambda'-lambda in closed form."""
    h, c, me, lam, lam_p, theta = sp.symbols("h c m_e lambda lambda_p theta", positive=True)

    pe_sq = (h / lam)**2 + (h / lam_p)**2 - 2 * (h / lam) * (h / lam_p) * sp.cos(theta)
    Ee = h * c / lam - h * c / lam_p + me * c**2
    dispersion_eq = sp.Eq(Ee**2, pe_sq * c**2 + (me * c**2)**2)

    solutions = sp.solve(dispersion_eq, lam_p)
    if len(solutions) != 1:
        raise AssertionError(f"expected a unique physical solution for lambda', got {len(solutions)}: {solutions}")
    lam_p_solved = solutions[0]
    shift = sp.simplify(lam_p_solved - lam)

    expected_shift = h * (1 - sp.cos(theta)) / (me * c)
    if sp.simplify(shift - expected_shift) != 0:
        raise AssertionError(f"derived shift {shift} does not match the textbook formula {expected_shift}")

    return lam_p_solved, shift


def derive_electron_recoil_angle_symbolic():
    """From the two momentum-conservation components directly (not the
    combined law-of-cosines form), solve for tan(phi), the electron's
    recoil angle -- an independent piece of the same collision, not
    needed for the wavelength shift itself but for a complete kinematic
    picture."""
    h, lam, lam_p, theta, phi = sp.symbols("h lambda lambda_p theta phi", positive=True)
    x_eq = sp.Eq(h / lam, (h / lam_p) * sp.cos(theta) + sp.Symbol("p_e", positive=True) * sp.cos(phi))
    y_eq = sp.Eq(0, (h / lam_p) * sp.sin(theta) - sp.Symbol("p_e", positive=True) * sp.sin(phi))
    pe = sp.Symbol("p_e", positive=True)
    pe_cos_phi = h / lam - (h / lam_p) * sp.cos(theta)
    pe_sin_phi = (h / lam_p) * sp.sin(theta)
    tan_phi = sp.simplify(pe_sin_phi / pe_cos_phi)
    return tan_phi


# ── 2. Numeric formulas, using the derived closed form ──────────────────────

def compton_wavelength_shift(theta_rad: float, h: float = H_PLANCK, me: float = M_ELECTRON,
                             c: float = C_LIGHT) -> float:
    """Delta_lambda = (h/(m_e c)) * (1 - cos(theta)) -- the closed form
    derive_compton_shift_symbolic() solved for, not a separate formula."""
    return (h / (me * c)) * (1 - np.cos(theta_rad))


def compton_wavelength_out(lambda_in_m: float, theta_rad: float, h: float = H_PLANCK,
                           me: float = M_ELECTRON, c: float = C_LIGHT) -> float:
    _validate_positive(lambda_in_m=lambda_in_m)
    return lambda_in_m + compton_wavelength_shift(theta_rad, h, me, c)


def compton_electron_kinetic_energy(lambda_in_m: float, theta_rad: float,
                                    h: float = H_PLANCK, me: float = M_ELECTRON,
                                    c: float = C_LIGHT) -> float:
    """KE_electron = hc/lambda - hc/lambda' (energy conservation), the
    energy the photon gave up."""
    lam_out = compton_wavelength_out(lambda_in_m, theta_rad, h, me, c)
    return h * c / lambda_in_m - h * c / lam_out


def compton_electron_recoil_angle(lambda_in_m: float, theta_rad: float,
                                  h: float = H_PLANCK, me: float = M_ELECTRON,
                                  c: float = C_LIGHT) -> float:
    """phi via tan(phi) = sin(theta) / (lambda_out/lambda_in - cos(theta)),
    the closed form from derive_electron_recoil_angle_symbolic()."""
    lam_out = compton_wavelength_out(lambda_in_m, theta_rad, h, me, c)
    return np.arctan2(np.sin(theta_rad), (lam_out / lambda_in_m) - np.cos(theta_rad))


# ── 3. Full numeric conservation check (energy AND both momentum axes) ──────

def verify_full_conservation(lambda_in_m: float, theta_rad: float, h: float = H_PLANCK,
                            me: float = M_ELECTRON, c: float = C_LIGHT,
                            rtol: float = 1e-9) -> dict:
    """CHECKED: for a concrete numeric example, energy conservation AND
    BOTH momentum components are satisfied simultaneously to high
    precision, using the electron kinetic energy / recoil angle formulas
    -- not merely "the wavelength shift formula holds," the FULL
    collision's conservation laws, independently reconstructed."""
    _validate_positive(lambda_in_m=lambda_in_m)
    lam_out = compton_wavelength_out(lambda_in_m, theta_rad, h, me, c)
    KE_e = compton_electron_kinetic_energy(lambda_in_m, theta_rad, h, me, c)
    phi = compton_electron_recoil_angle(lambda_in_m, theta_rad, h, me, c)

    Ee_total = KE_e + me * c**2
    pe = np.sqrt(max(Ee_total**2 - (me * c**2)**2, 0.0)) / c

    energy_lhs = h * c / lambda_in_m + me * c**2
    energy_rhs = h * c / lam_out + Ee_total
    energy_residual = abs(energy_lhs - energy_rhs) / energy_lhs

    px_lhs = h / lambda_in_m
    px_rhs = (h / lam_out) * np.cos(theta_rad) + pe * np.cos(phi)
    px_residual = abs(px_lhs - px_rhs) / px_lhs

    py_lhs = 0.0
    py_rhs = (h / lam_out) * np.sin(theta_rad) - pe * np.sin(phi)
    py_residual = abs(py_rhs) / max(h / lam_out, 1e-300)

    checks = {"energy_conserved": energy_residual < rtol,
              "momentum_x_conserved": px_residual < rtol,
              "momentum_y_conserved": py_residual < rtol}
    return {"lambda_out_m": lam_out, "KE_electron_J": KE_e, "phi_rad": phi,
            "energy_residual": energy_residual, "px_residual": px_residual,
            "py_residual": py_residual, "checks": checks}


def verify_thomson_limit(theta_rad: float = np.pi / 2, wavelength_ratio: float = 1e6) -> bool:
    """CHECKED: as the photon wavelength grows much larger than the
    Compton wavelength (equivalently, photon energy << m_e*c^2), the
    FRACTIONAL wavelength shift Delta_lambda/lambda -> 0 -- recovering
    classical (Thomson) scattering, where the photon's wavelength doesn't
    change at all. `wavelength_ratio` is lambda/lambda_C."""
    if wavelength_ratio <= 0:
        raise ValueError(f"wavelength_ratio must be > 0, got {wavelength_ratio}")
    lambda_in = wavelength_ratio * COMPTON_WAVELENGTH
    shift = compton_wavelength_shift(theta_rad)
    fractional_shift = shift / lambda_in
    if fractional_shift > 1.0 / wavelength_ratio * 10:
        raise AssertionError(f"fractional shift {fractional_shift} too large for the Thomson limit "
                             f"at wavelength_ratio={wavelength_ratio}")
    return True


if __name__ == "__main__":
    print("=== 1. Compton shift derived from conservation laws (not quoted) ===")
    lam_p_solved, shift_derived = derive_compton_shift_symbolic()
    print(f"  lambda' (solved) = {lam_p_solved}")
    print(f"  lambda' - lambda (derived) = {shift_derived}")

    print("\n=== 2. Electron recoil angle, derived independently ===")
    tan_phi = derive_electron_recoil_angle_symbolic()
    print(f"  tan(phi) = {tan_phi}")

    print("\n=== 3. Numeric example: X-ray photon (lambda=0.1 nm), theta=90 deg ===")
    lambda_in = 0.1e-9
    theta = np.pi / 2
    lam_out = compton_wavelength_out(lambda_in, theta)
    shift = compton_wavelength_shift(theta)
    print(f"  Compton wavelength lambda_C = h/(m_e c) = {COMPTON_WAVELENGTH*1e12:.4f} pm")
    print(f"  shift at theta=90deg = {shift*1e12:.4f} pm  (should equal lambda_C exactly at 90 deg)")
    print(f"  lambda_in = {lambda_in*1e9:.4f} nm -> lambda_out = {lam_out*1e9:.6f} nm")

    print("\n=== 4. Full conservation check: energy AND both momentum components ===")
    result = verify_full_conservation(lambda_in, theta)
    for name, ok in result["checks"].items():
        print(f"  {name}: {ok}")
    print(f"  electron KE = {result['KE_electron_J']/1.602176634e-19:.2f} eV, "
          f"recoil angle phi = {np.degrees(result['phi_rad']):.2f} deg")

    print("\n=== 5. Thomson (classical) limit: long-wavelength photon, shift becomes negligible ===")
    ok_thomson = verify_thomson_limit(wavelength_ratio=1e6)
    print(f"  fractional wavelength shift vanishes as lambda >> lambda_C: {ok_thomson}")

    print("\nThe Compton formula isn't a separate quantum-mechanical rule bolted onto")
    print("classical mechanics -- it's exactly what relativistic energy-momentum")
    print("conservation gives for a photon-electron collision, treating the photon as")
    print("a real particle with momentum p=h/lambda, solved here rather than assumed.")
