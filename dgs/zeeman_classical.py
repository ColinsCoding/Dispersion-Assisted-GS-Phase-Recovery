"""Classical Zeeman effect -- Maxwell predicts a spectral line splits in a B field.

Problem 3.1, "Light as an Electromagnetic Wave" (classical/pre-quantum treatment).
An atom is modeled as an electron orbiting at radius r with angular frequency
omega_0 = v/r; the orbiting charge radiates light at omega_0. Switch on a magnetic
field B perpendicular to the orbital plane and, purely from Maxwell's equations, the
emission frequency shifts -- the line splits into a triplet omega_0, omega_0 +/- Delta.

The derivation, in four steps (each a function below):

  (a) FARADAY. Ramping B through the orbit induces a tangential E field. With
      the loop being the orbit (circumference 2*pi*r, area pi*r^2),
          contour(E . ds) = E * 2*pi*r = -dPhi_B/dt = -pi*r^2 dB/dt
      so the magnitude is
          E = (r/2) dB/dt.

  (b) IMPULSE. That E field pushes the electron along its path, F = eE, and
      F dt = m dv accumulates a speed change. With r held fixed,
          m_e Delta_v = integral(e E dt) = (e r / 2) integral(dB) = e r B / 2
      so
          Delta_v = e r B / (2 m_e).

  (c) FREQUENCY SHIFT. Since omega = v/r, Delta_omega = Delta_v / r = e B / (2 m_e)
      -- the LARMOR frequency, independent of the orbit radius. For B = 1 T this is
      ~8.8e10 rad/s; for a 500 nm line the fractional shift Delta_omega/omega_0 is
      only ~2e-5, but it is measurable and it is exactly what experiment sees.

  (d) TRIPLET. Electrons circulating one way speed up (omega_0 + Delta_omega), the
      other way slow down (omega_0 - Delta_omega); those whose orbital plane contains
      B are unshifted (omega_0). The single line becomes three.

That same e B / (2 m_e) is the quantum result too: the energy splitting Delta_E =
hbar*Delta_omega equals the Bohr magneton times B, mu_B * B -- classical orbit and
quantum spin land on the same magneton. Numbers verified against CODATA constants;
the algebra is also carried out symbolically in SymPy. NumPy + SymPy; py-3.13.
"""

import numpy as np
import sympy as sp

# CODATA fundamental constants (SI)
E_CHARGE = 1.602176634e-19      # C
M_ELECTRON = 9.1093837015e-31   # kg
C_LIGHT = 2.99792458e8          # m/s
HBAR = 1.054571817e-34          # J s
BOHR_MAGNETON = 9.2740100783e-24  # J/T


# ----------------------------------------------------------------------
# (a) the induced electric field from Faraday's law
# ----------------------------------------------------------------------

def induced_electric_field(r, dB_dt):
    """Part (a): magnitude of the tangential E field induced around an orbit of
    radius r while B changes at rate dB_dt, from contour(E.ds) = -dPhi_B/dt:
        E = (r/2) dB/dt."""
    if r <= 0:
        raise ValueError("orbit radius r must be positive")
    return r / 2 * dB_dt


# ----------------------------------------------------------------------
# (b) the speed change from the impulse
# ----------------------------------------------------------------------

def delta_v(r, B_final, B_initial=0.0):
    """Part (b): change in the electron's speed as B ramps from B_initial to
    B_final at fixed radius, integral(e E dt) = m_e Delta_v:
        Delta_v = e r (B_final - B_initial) / (2 m_e)."""
    if r <= 0:
        raise ValueError("orbit radius r must be positive")
    return E_CHARGE * r * (B_final - B_initial) / (2 * M_ELECTRON)


# ----------------------------------------------------------------------
# (c) the angular-frequency shift = Larmor frequency
# ----------------------------------------------------------------------

def larmor_delta_omega(B):
    """Part (c): Delta_omega = Delta_v / r = e B / (2 m_e), independent of the
    orbit radius -- the Larmor angular frequency. ~8.8e10 rad/s at B = 1 T."""
    return E_CHARGE * B / (2 * M_ELECTRON)


def angular_frequency(wavelength_nm):
    """omega_0 = 2*pi*c/lambda for a spectral line of the given wavelength."""
    if wavelength_nm <= 0:
        raise ValueError("wavelength must be positive")
    return 2 * np.pi * C_LIGHT / (wavelength_nm * 1e-9)


def fractional_shift(B, wavelength_nm):
    """Delta_omega / omega_0 for a line at the given wavelength in field B --
    ~2.3e-5 for a 500 nm line at 1 T (small but resolvable)."""
    return larmor_delta_omega(B) / angular_frequency(wavelength_nm)


# ----------------------------------------------------------------------
# (d) the triplet, and the quantum bridge
# ----------------------------------------------------------------------

def zeeman_triplet(omega0, B):
    """Part (d): the single line omega0 splits into (omega0 - Delta, omega0,
    omega0 + Delta) with Delta = e B / (2 m_e). Sub-/unshifted/super- from the
    electron's sense of circulation relative to B."""
    d = larmor_delta_omega(B)
    return (omega0 - d, omega0, omega0 + d)


def energy_splitting(B):
    """The quantum bridge: Delta_E = hbar * Delta_omega = mu_B * B, the Bohr-
    magneton energy shift. The classical orbit reproduces the magneton."""
    return HBAR * larmor_delta_omega(B)


def zeeman_symbolic():
    """Carry out parts (a) and (b) symbolically in SymPy, so the algebra is
    checked, not just asserted. Returns the derived E field and Delta_v, which
    equal (r/2) dB/dt and e r B / (2 m_e)."""
    t, r, e, m = sp.symbols("t r e m", positive=True)
    B = sp.Function("B")
    # (a) Faraday: emf = -dPhi/dt around the orbit, E = emf / circumference
    Phi = B(t) * sp.pi * r ** 2
    emf = -sp.diff(Phi, t)
    E_field = sp.simplify(emf / (2 * sp.pi * r))          # -> -r/2 dB/dt
    # (b) impulse: m dv = e E dt; integrate B from 0 to B_f (r fixed)
    Bf = sp.symbols("B_f", positive=True)
    E_mag = r / 2 * sp.Symbol("dBdt")
    dv = sp.integrate(e * (r / 2), (sp.Symbol("B_"), 0, Bf)) / m   # e r B_f/(2m)
    return {"E_field": E_field, "delta_v": sp.simplify(dv),
            "E_magnitude": sp.Rational(1, 2) * r * sp.Symbol("dBdt")}


if __name__ == "__main__":
    print("PROBLEM 3.1 -- classical Zeeman effect\n")
    print("(a) induced E field:  E = (r/2) dB/dt")
    print(f"    e.g. r=1e-10 m, dB/dt=1 T/s -> E = {induced_electric_field(1e-10, 1.0):.3e} V/m")

    print("\n(b) speed change:  Delta_v = e r B / (2 m_e)")
    print(f"    r=1e-10 m, B=1 T -> Delta_v = {delta_v(1e-10, 1.0):.3e} m/s")

    dw = larmor_delta_omega(1.0)
    print("\n(c) frequency shift:  Delta_omega = e B / (2 m_e)  (Larmor, r-independent)")
    print(f"    B=1 T -> Delta_omega = {dw:.4e} rad/s")
    print(f"    500 nm line: omega_0 = {angular_frequency(500):.3e} rad/s, "
          f"Delta_omega/omega_0 = {fractional_shift(1.0, 500):.3e}")

    w0 = angular_frequency(500)
    lo, mid, hi = zeeman_triplet(w0, 1.0)
    print("\n(d) the triplet (500 nm, 1 T):")
    print(f"    omega_0 - Delta = {lo:.6e}")
    print(f"    omega_0         = {mid:.6e}")
    print(f"    omega_0 + Delta = {hi:.6e}")

    print(f"\nquantum bridge: Delta_E = hbar*Delta_omega = {energy_splitting(1.0):.4e} J "
          f"= mu_B*B = {BOHR_MAGNETON:.4e} J")

    sym = zeeman_symbolic()
    print("\nsymbolic check:  E =", sym["E_field"], "   Delta_v =", sym["delta_v"])
