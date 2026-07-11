"""Scattering by a potential step: the quantum version of a Fresnel interface.

A particle of energy E travels in from the left and hits a step in the potential from 0 up to U at
x=0 (Griffiths Example 7.4). Left of the step the wave is incident + reflected, right of it either
propagates or decays:

    x<0:  Psi = A e^{ikx} + B e^{-ikx},        k  = sqrt(2mE)/hbar
    x>0:  Psi = C e^{ik'x}  (E>U, k'=sqrt(2m(E-U))/hbar)   or
          Psi = C e^{-kappa x}  (E<U, kappa=sqrt(2m(U-E))/hbar).

Matching Psi and Psi' at x=0 gives, for E>U,
        B/A = (k-k')/(k+k'),   R = |B/A|^2 = ((k-k')/(k+k'))^2,   T = 4 k k'/(k+k')^2,   R+T=1.
That reflection coefficient is EXACTLY the Fresnel formula R=((n1-n2)/(n1+n2))^2 for light crossing an
index step, with the wavevector k in the role of the refractive index n. Quantum step scattering and
optical reflection at a dielectric boundary are the same boundary-matching problem -- the "jalali
photonics" connection: a potential step is an index step.

For E<U the transmitted wavevector turns imaginary: k'->i*kappa. The wave beyond the step becomes
EVANESCENT (decaying, penetration depth 1/kappa) and R=|B/A|^2=1 -- TOTAL reflection, but with a phase
shift on the reflected wave and a nonzero field leaking into the forbidden region. That is TOTAL
INTERNAL REFLECTION: the R=1 with an evanescent tail is exactly frustrated-TIR / optical tunneling
(put a second interface a finite distance away and the tail couples across -- the dgs.square_barrier
tunneling). Real potentials in / evanescent out; the imaginary wavevector is the decay.

Verified against Fresnel, flux conservation R+T=1, total reflection for E<U, the E->U crossover, and
quantum reflection off a step DOWN. NumPy; py-3.13. Complements dgs.square_barrier (finite width) and
dgs.wave_reflection (impedance form).
"""

import numpy as np


def wavevector(E, m=1.0, hbar=1.0):
    """Incident wavevector k = sqrt(2mE)/hbar (E measured from the left-side potential = 0)."""
    if E <= 0:
        raise ValueError("E must be > 0")
    return np.sqrt(2 * m * E) / hbar


def transmitted_wavevector(E, U, m=1.0, hbar=1.0):
    """Wavevector past the step, k' = sqrt(2m(E-U))/hbar. Requires E > U (propagating region)."""
    if E <= U:
        raise ValueError("k' is real only for E > U; use decay_constant for E < U")
    return np.sqrt(2 * m * (E - U)) / hbar


def decay_constant(E, U, m=1.0, hbar=1.0):
    """Evanescent decay constant kappa = sqrt(2m(U-E))/hbar for E < U (classically forbidden)."""
    if E >= U:
        raise ValueError("kappa is real only for E < U")
    return np.sqrt(2 * m * (U - E)) / hbar


def step_amplitudes(E, U, m=1.0, hbar=1.0):
    """Reflected (B/A) and transmitted/evanescent (C/A) amplitudes with the incident A=1, from
    matching Psi and Psi' at the step. Returns a dict; amplitudes are complex for E<U."""
    if E <= 0:
        raise ValueError("E must be > 0")
    k = wavevector(E, m, hbar)
    if E > U:
        kp = transmitted_wavevector(E, U, m, hbar)
        return {"regime": "propagating", "k": k, "kp": kp,
                "B_over_A": (k - kp) / (k + kp), "C_over_A": 2 * k / (k + kp)}
    if E == U:
        return {"regime": "critical", "k": k, "kp": 0.0,
                "B_over_A": 1.0, "C_over_A": 1.0}          # k'->0: total reflection onset
    kap = decay_constant(E, U, m, hbar)
    # matching with Psi = C e^{-kappa x}:  B/A = (ik+kappa)/(ik-kappa),  C/A = 2ik/(ik-kappa)
    B = (1j * k + kap) / (1j * k - kap)
    C = 2j * k / (1j * k - kap)
    return {"regime": "evanescent", "k": k, "kappa": kap, "B_over_A": B, "C_over_A": C}


def reflection_coefficient(E, U, m=1.0, hbar=1.0):
    """Reflection probability R = |B/A|^2. Equals 1 for E<=U (total reflection)."""
    return float(np.abs(step_amplitudes(E, U, m, hbar)["B_over_A"]) ** 2)


def transmission_coefficient(E, U, m=1.0, hbar=1.0):
    """Transmission probability T = (k'/k)|C/A|^2 (the k'/k is the flux factor). Zero for E<=U."""
    amp = step_amplitudes(E, U, m, hbar)
    if amp["regime"] != "propagating":
        return 0.0
    return float((amp["kp"] / amp["k"]) * np.abs(amp["C_over_A"]) ** 2)


def reflection_phase(E, U, m=1.0, hbar=1.0):
    """Phase of the reflected amplitude arg(B/A). Zero for a step up with E>U, pi for a step down,
    and a continuously varying value in (-pi,0) for E<U (the TIR phase shift)."""
    return float(np.angle(step_amplitudes(E, U, m, hbar)["B_over_A"]))


def penetration_depth(E, U, m=1.0, hbar=1.0):
    """Evanescent penetration depth 1/kappa into the forbidden region (E < U) = hbar/sqrt(2m(U-E))."""
    return 1.0 / decay_constant(E, U, m, hbar)


def fresnel_reflectance(n1, n2):
    """Optical (normal-incidence) reflectance R=((n1-n2)/(n1+n2))^2 -- identical in form to the
    quantum step with wavevector k in the role of refractive index n."""
    if n1 <= 0 or n2 <= 0:
        raise ValueError("indices must be > 0")
    return ((n1 - n2) / (n1 + n2)) ** 2


if __name__ == "__main__":
    m = hbar = 1.0
    print("=== step UP, E > U (partial reflection = Fresnel) ===")
    U = 1.0
    print(" E      R        T        R+T")
    for E in (1.5, 2.0, 4.0, 10.0):
        R, T = reflection_coefficient(E, U), transmission_coefficient(E, U)
        print(f" {E:4.1f}   {R:.4f}   {T:.4f}   {R+T:.4f}")
    k = wavevector(4.0); kp = transmitted_wavevector(4.0, U)
    print(f"  R(E=4) = {reflection_coefficient(4.0,U):.4f}  vs Fresnel((k,k')) = "
          f"{fresnel_reflectance(k, kp):.4f}  (k as refractive index)")

    print("\n=== step UP, E < U (total internal reflection + evanescent tail) ===")
    print(" E      R      penetration 1/kappa   reflected phase")
    for E in (0.2, 0.5, 0.9):
        print(f" {E:4.1f}   {reflection_coefficient(E,U):.3f}      "
              f"{penetration_depth(E,U):.4f}          {reflection_phase(E,U):+.3f} rad")
    amp = step_amplitudes(0.5, U)
    print(f"  evanescent amplitude |C/A|^2 = {abs(amp['C_over_A'])**2:.4f} (>0: field leaks in, "
          f"the basis of frustrated-TIR / tunneling)")

    print("\n=== step DOWN (E > U with U < 0): quantum reflection ===")
    print(f"  E=1, U=-3:  R = {reflection_coefficient(1.0,-3.0):.4f} (>0, though classically "
          f"the particle always passes)")
