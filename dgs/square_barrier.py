"""Square-barrier scattering: solve the joining conditions for the reflection and transmission
coefficients of a rectangular potential barrier (the quantum version of a reflected wave).

A particle of energy E hits a rectangular barrier of height V0 and width L. Its wavefunction is
a plane wave on each side and (for E < V0) an evanescent exp(+-alpha x) inside:

    region I  (x<0):    psi = A e^{ikx} + B e^{-ikx}     (incident + reflected)
    region II (0<x<L):  psi = C e^{-alpha x} + D e^{+alpha x}
    region III(x>L):    psi = F e^{ikx}                  (transmitted only)

with k = sqrt(2mE)/hbar and alpha = sqrt(2m(V0-E))/hbar. Demanding that psi and dpsi/dx are
continuous at x=0 and x=L gives the four JOINING CONDITIONS (Eqs. 7.8):

    A + B = C + D
    ik(A - B) = alpha(D - C)
    C e^{-aL} + D e^{+aL} = F e^{ikL}
    alpha(D e^{+aL} - C e^{-aL}) = ik F e^{ikL}

Four equations, four unknowns (B,C,D,F in terms of A). Solving them gives the transmission
T = |F/A|^2 and reflection R = |B/A|^2, with T + R = 1 (probability is conserved -- no particle
is lost, it either tunnels through or bounces back). This module solves that linear system
directly, so it works in EVERY regime: E < V0 (tunneling, T decays as e^{-2 alpha L}), E = V0,
and E > V0 (over-barrier, where alpha turns imaginary and T oscillates -- hitting T=1 RESONANCES
whenever a half-integer number of wavelengths fits the barrier, the Ramsauer-Townsend effect).

Same physics as dgs.wave_reflection (impedance mismatch -> reflected wave), now with wavevector k
in place of impedance Z. Cross-checks dgs.tunneling.rectangular_barrier_T for E < V0. NumPy; the
matching solver uses complex arithmetic so one code path covers all energies. py-3.13.
"""

import numpy as np


def wavevector(E, m=1.0, hbar=1.0):
    """Propagating wavevector k = sqrt(2 m E)/hbar."""
    if E <= 0:
        raise ValueError("E must be > 0")
    return np.sqrt(2 * m * E) / hbar


def _alpha(E, V0, m, hbar):
    """Barrier decay constant alpha = sqrt(2 m (V0 - E))/hbar, complex so E > V0 (over-barrier)
    comes out imaginary automatically."""
    return np.lib.scimath.sqrt(2 * m * (V0 - E)) / hbar


def barrier_amplitudes(E, V0, L, m=1.0, hbar=1.0):
    """Solve the four joining conditions for the amplitudes, with the incident amplitude A=1.
    Returns a dict with A, B (reflected), C, D (inside), F (transmitted), all complex."""
    if E <= 0 or L <= 0:
        raise ValueError("E and L must be > 0")
    k = wavevector(E, m, hbar)
    if np.isclose(E, V0, rtol=1e-12, atol=1e-12):
        # alpha -> 0 is a degenerate basis (the two exp solutions collapse); the correct
        # region-II basis is {1, x}. Matching gives F = 2 e^{-ikL}/(2 - ikL) exactly.
        eik = np.exp(1j * k * L)
        F = 2 * np.exp(-1j * k * L) / (2 - 1j * k * L)
        D = 1j * k * F * eik
        C = F * eik - D * L
        return {"A": 1.0 + 0j, "B": C - 1.0, "C": C, "D": D, "F": F}
    a = _alpha(E, V0, m, hbar)
    emL, epL, eik = np.exp(-a * L), np.exp(a * L), np.exp(1j * k * L)
    # unknowns x = [B, C, D, F];  A = 1
    M = np.array([
        [1.0,        -1.0,          -1.0,          0.0],           # A+B = C+D
        [-1j * k,     a,            -a,            0.0],           # ik(A-B) = alpha(D-C)
        [0.0,         emL,           epL,         -eik],           # continuity of psi at L
        [0.0,        -a * emL,       a * epL,     -1j * k * eik],  # continuity of psi' at L
    ], dtype=complex)
    rhs = np.array([-1.0, -1j * k, 0.0, 0.0], dtype=complex)       # from A=1
    B, C, D, F = np.linalg.solve(M, rhs)
    return {"A": 1.0 + 0j, "B": B, "C": C, "D": D, "F": F}


def transmission_coefficient(E, V0, L, m=1.0, hbar=1.0):
    """Transmission probability T = |F/A|^2 from the joining conditions -- valid for E < V0,
    E = V0, and E > V0 alike."""
    amp = barrier_amplitudes(E, V0, L, m, hbar)
    return float(np.abs(amp["F"]) ** 2)


def reflection_coefficient(E, V0, L, m=1.0, hbar=1.0):
    """Reflection probability R = |B/A|^2. Together with T it obeys R + T = 1."""
    amp = barrier_amplitudes(E, V0, L, m, hbar)
    return float(np.abs(amp["B"]) ** 2)


def transmission_closed_form(E, V0, L, m=1.0, hbar=1.0):
    """Closed-form transmission, branch by regime:
        E < V0:  T = [1 + V0^2 sinh^2(alpha L)/(4E(V0-E))]^-1   (tunneling)
        E > V0:  T = [1 + V0^2 sin^2(k2 L)/(4E(E-V0))]^-1       (over-barrier)
        E = V0:  T = [1 + m V0 L^2/(2 hbar^2)]^-1               (limit).
    A cross-check on the matching solver."""
    if E <= 0 or L <= 0:
        raise ValueError("E and L must be > 0")
    if np.isclose(E, V0):
        return 1.0 / (1.0 + m * V0 * L ** 2 / (2 * hbar ** 2))
    if E < V0:
        a = np.sqrt(2 * m * (V0 - E)) / hbar
        return 1.0 / (1.0 + V0 ** 2 * np.sinh(a * L) ** 2 / (4 * E * (V0 - E)))
    k2 = np.sqrt(2 * m * (E - V0)) / hbar
    return 1.0 / (1.0 + V0 ** 2 * np.sin(k2 * L) ** 2 / (4 * E * (E - V0)))


def resonance_energies(V0, L, n_max=4, m=1.0, hbar=1.0):
    """Over-barrier resonance energies where the barrier is perfectly transparent (T=1):
    k2 L = n pi  ->  E_n = V0 + (n pi hbar / L)^2 / (2 m),  n = 1, 2, ..."""
    return [V0 + (n * np.pi * hbar / L) ** 2 / (2 * m) for n in range(1, n_max + 1)]


if __name__ == "__main__":
    m = hbar = 1.0
    V0, L = 10.0, 1.0

    print("=== tunneling: E below the barrier (E < V0=10) ===")
    print(" E     T(match)   T(closed)   R        T+R")
    for E in (1.0, 3.0, 5.0, 9.0):
        T = transmission_coefficient(E, V0, L)
        Tc = transmission_closed_form(E, V0, L)
        R = reflection_coefficient(E, V0, L)
        print(f" {E:4.1f}  {T:.3e}  {Tc:.3e}  {R:.5f}  {T+R:.6f}")

    print("\n=== over the barrier: resonances where T = 1 (Ramsauer-Townsend) ===")
    for n, Eres in enumerate(resonance_energies(V0, L, 3), start=1):
        T = transmission_coefficient(Eres, V0, L)
        print(f"  n={n}: E={Eres:6.3f}  T={T:.5f}  (perfect transmission)")
    E_off = (resonance_energies(V0, L, 2)[0] + resonance_energies(V0, L, 2)[1]) / 2
    print(f"  between resonances E={E_off:.3f}: T={transmission_coefficient(E_off,V0,L):.4f} (< 1)")

    print("\n=== reflected-wave analogy ===")
    print("  same as dgs.wave_reflection with Z -> k: the barrier is an impedance mismatch,")
    print("  and T=1 resonances are the quarter/half-wave matched conditions.")
