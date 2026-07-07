"""Where does the energy go? -- storage, release, and dissipation in RLC.

A charged capacitor and a current-carrying inductor are ENERGY TANKS:
    E_C = 1/2 C v^2      (energy in the electric field between the plates)
    E_L = 1/2 L i^2      (energy in the magnetic field of the coil)
Wire them to a resistor and the stored energy is RELEASED -- and a resistor
only ever removes it, at rate p = i^2 R (Joule heating, always >= 0). Three
canonical releases, in rising order of drama:
  * RC discharge: the cap dumps into R, no oscillation. Every joule that
    leaves the field is dissipated: E_R(inf) = 1/2 C v0^2, EXACTLY.
  * LC (R=0): energy SLOSHES between cap and coil forever, E_C + E_L fixed.
    The electrical twin of a pendulum trading PE for KE -- nothing lost.
  * RLC: the cap discharges through R and L; underdamped it RINGS while R
    bleeds the energy away, and at every instant
        E_C(t) + E_L(t) + E_dissipated(t) = E_0.
    That running total is conservation of energy, checkable to O(h^2).

This module supplies the energy bookkeeping; the transient itself comes from
dgs.spice.rlc_step_response (driven with V=0 and a nonzero initial cap
voltage = the natural/discharge response). Integrals reuse
dgs.numerical_methods.trapezoid. The ring frequency f_d connects to the FFT
work in dgs.strided_slicing / dgs.eye_diagram: the spectrum of a decaying
ring peaks at the damped natural frequency. NumPy. Education.
"""

import numpy as np
from dgs import numerical_methods as nm


# ----------------------------------------------------------------------
# The two energy tanks
# ----------------------------------------------------------------------

def capacitor_energy(C, v):
    """E_C = 1/2 C v^2 -- energy stored in a capacitor's electric field."""
    if C <= 0:
        raise ValueError("C must be positive")
    return 0.5 * C * np.asarray(v, float) ** 2


def inductor_energy(L, i):
    """E_L = 1/2 L i^2 -- energy stored in an inductor's magnetic field."""
    if L <= 0:
        raise ValueError("L must be positive")
    return 0.5 * L * np.asarray(i, float) ** 2


def dissipated_energy(iL, R, t):
    """Cumulative energy turned to heat in the resistor: integral of
    p(t) = i(t)^2 R up to each time, by the trapezoid rule. Monotonically
    non-decreasing because i^2 R >= 0 -- a resistor never returns energy."""
    if R < 0:
        raise ValueError("R must be non-negative")
    iL, t = np.asarray(iL, float), np.asarray(t, float)
    power = iL ** 2 * R
    # running trapezoid integral so E_R(t) can be compared instant by instant
    cum = np.concatenate([[0.0], np.cumsum(0.5 * (power[1:] + power[:-1]) * np.diff(t))])
    return cum


# ----------------------------------------------------------------------
# RC discharge -- the clean closed form to check numerics against
# ----------------------------------------------------------------------

def rc_discharge(R, C, v0, t):
    """Analytic capacitor discharge through a resistor: v(t) = v0 e^(-t/RC),
    i(t) = (v0/R) e^(-t/RC). Returns (v, i). The time constant tau = RC is
    the 1/e point; after ~5 tau essentially all the energy is gone."""
    if R <= 0 or C <= 0:
        raise ValueError("R and C must be positive")
    t = np.asarray(t, float)
    tau = R * C
    v = v0 * np.exp(-t / tau)
    return v, v / R      # i = v/R -> (v0/R) e^(-t/RC), the same exponential


def rc_energy_released(R, C, v0, t):
    """Closed form for energy dissipated by an RC discharge up to time t:
        E_R(t) = 1/2 C v0^2 (1 - e^(-2 t / RC)),
    reaching the FULL stored energy 1/2 C v0^2 as t -> inf. Independent of R:
    a bigger resistor just takes longer to release the same joules."""
    if R <= 0 or C <= 0:
        raise ValueError("R and C must be positive")
    t = np.asarray(t, float)
    return 0.5 * C * v0 ** 2 * (1 - np.exp(-2 * t / (R * C)))


# ----------------------------------------------------------------------
# Full energy audit of an RLC natural response
# ----------------------------------------------------------------------

def energy_audit(R, L, C, t, vc, iL):
    """Given an RLC transient (vc, iL sampled on t -- e.g. from
    dgs.spice.rlc_step_response with V=0), return the complete energy ledger:
      E_C, E_L      stored in the two fields at each instant,
      E_R           cumulative heat in the resistor,
      total         E_C + E_L + E_R  (should be flat = E_0),
      E0            the initial stored energy 1/2 C vc0^2 + 1/2 L iL0^2,
      max_drift     max |total - E_0| / E_0  -- the conservation residual.
    A small max_drift is first-law bookkeeping closing to O(h^2)."""
    vc, iL, t = np.asarray(vc, float), np.asarray(iL, float), np.asarray(t, float)
    E_C = capacitor_energy(C, vc)
    E_L = inductor_energy(L, iL)
    E_R = dissipated_energy(iL, R, t)
    total = E_C + E_L + E_R
    E0 = float(E_C[0] + E_L[0])
    drift = 0.0 if E0 == 0 else float(np.max(np.abs(total - E0)) / E0)
    return {"E_C": E_C, "E_L": E_L, "E_R": E_R, "total": total,
            "E0": E0, "max_drift": drift}


def damped_ring_frequency(R, L, C):
    """The frequency an underdamped RLC actually rings at:
        f_d = sqrt(1/(LC) - (R/2L)^2) / (2 pi)   [Hz],
    the undamped f0 = 1/(2 pi sqrt(LC)) pulled DOWN by damping. Returns 0.0
    if the circuit is critically- or over-damped (no ring). This is the peak
    an FFT of the decaying transient lands on (see dgs.eye_diagram spectra)."""
    if L <= 0 or C <= 0 or R < 0:
        raise ValueError("need L, C > 0 and R >= 0")
    w0_sq = 1.0 / (L * C)
    alpha = R / (2 * L)
    wd_sq = w0_sq - alpha ** 2
    return 0.0 if wd_sq <= 0 else float(np.sqrt(wd_sq) / (2 * np.pi))


if __name__ == "__main__":
    from dgs import spice

    # RC discharge: numerical heat integral vs the closed form
    R, C, v0 = 1e3, 1e-6, 5.0
    t = np.linspace(0, 5 * R * C, 4000)
    _, i = rc_discharge(R, C, v0, t)
    E_num = dissipated_energy(i, R, t)[-1]
    E_ana = rc_energy_released(R, C, v0, t)[-1]
    E_stored = capacitor_energy(C, v0)
    print(f"RC: stored 1/2 C v0^2 = {E_stored*1e6:.2f} uJ, "
          f"dissipated (numeric) {E_num*1e6:.2f} uJ, (formula) {E_ana*1e6:.2f} uJ")

    # RLC natural response: discharge a charged cap through R, L; audit energy
    L, R = 1e-3, 20.0
    t = np.linspace(0, 2e-3, 20000)
    vc, iL = spice.rlc_step_response(R, L, C, t, V=0.0, vc0=v0, il0=0.0)
    audit = energy_audit(R, L, C, t, vc, iL)
    print(f"RLC ring: E0 = {audit['E0']*1e6:.2f} uJ, "
          f"conservation drift = {audit['max_drift']:.2e}, "
          f"f_d = {damped_ring_frequency(R, L, C):.0f} Hz "
          f"(f0 = {spice.resonant_frequency(L, C):.0f} Hz)")

    # lossless LC: energy sloshes but the total is fixed
    vc, iL = spice.rlc_step_response(0.0, L, C, t, V=0.0, vc0=v0, il0=0.0)
    audit = energy_audit(0.0, L, C, t, vc, iL)
    print(f"LC (R=0): E_C+E_L constant? drift = {audit['max_drift']:.2e} "
          f"(energy sloshes, none lost)")
