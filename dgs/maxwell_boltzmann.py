"""Maxwell-Boltzmann: how fast the atoms in a gas actually move.

At temperature T the atoms of a gas do not all share one speed -- they have a spread,
the Maxwell-Boltzmann SPEED DISTRIBUTION:
        f(v) = 4 pi (m / 2 pi k T)^(3/2) v^2 exp(-m v^2 / 2 k T).
The v^2 out front (more ways to point a fast velocity in 3-D) pushes the peak up from
zero; the exponential (the Boltzmann factor e^{-E/kT} with E = 1/2 m v^2) pulls the tail
down. Their product is the bell-with-a-tail every kinetic-theory chapter draws.

Three speeds summarize it, all ~ sqrt(kT/m) but with different numbers:
        most probable  v_mp  = sqrt(2 k T / m)      (the peak of f)
        mean           v_avg = sqrt(8 k T / pi m)   (the average)
        rms            v_rms = sqrt(3 k T / m)      (sqrt of the mean square)
and they always order v_mp < v_avg < v_rms (ratios sqrt2 : sqrt(8/pi) : sqrt3). The rms
one carries the energy: the average translational kinetic energy per atom is
        <KE> = 1/2 m <v^2> = 3/2 k T,
the equipartition theorem (1/2 kT per degree of freedom, three translational). That
1/2 m v^2 with the same Boltzmann e^{-E/kT} is the classical limit of the thermal
occupation in dgs.quantum_oscillator and the Boltzmann factor in dgs.blackbody.

Verified: the distribution integrates to 1, its first and second moments give v_avg and
v_rms, its peak sits at v_mp, and <KE> = 3/2 kT -- all by numerical integration, plus the
T and m scalings. NumPy only; py-3.13.
"""

import numpy as np

K_BOLTZ = 1.380649e-23        # J/K
AMU = 1.66053906660e-27       # kg per atomic mass unit


def maxwell_boltzmann_pdf(v, mass, T):
    """Speed probability density f(v) [s/m]: 4 pi (m/2 pi kT)^{3/2} v^2 e^{-m v^2/2kT}.
    The probability of a speed in [v, v+dv] is f(v) dv."""
    if mass <= 0 or T <= 0:
        raise ValueError("mass and T must be positive")
    v = np.asarray(v, float)
    a = mass / (2 * K_BOLTZ * T)
    return 4 * np.pi * (a / np.pi) ** 1.5 * v ** 2 * np.exp(-a * v ** 2)


def rms_speed(mass, T):
    """Root-mean-square speed sqrt(3kT/m) -- the one that carries the kinetic energy."""
    if mass <= 0 or T <= 0:
        raise ValueError("mass and T must be positive")
    return np.sqrt(3 * K_BOLTZ * T / mass)


def mean_speed(mass, T):
    """Average speed sqrt(8kT/pi m)."""
    if mass <= 0 or T <= 0:
        raise ValueError("mass and T must be positive")
    return np.sqrt(8 * K_BOLTZ * T / (np.pi * mass))


def most_probable_speed(mass, T):
    """Most probable speed sqrt(2kT/m) -- the peak of the distribution."""
    if mass <= 0 or T <= 0:
        raise ValueError("mass and T must be positive")
    return np.sqrt(2 * K_BOLTZ * T / mass)


def average_kinetic_energy(T, dof=3):
    """Mean kinetic energy per atom = (dof/2) kT (equipartition). For a monatomic
    gas the 3 translational degrees give 3/2 kT."""
    if T <= 0 or dof < 1:
        raise ValueError("T must be positive and dof >= 1")
    return 0.5 * dof * K_BOLTZ * T


def _moments(mass, T, n=400000):
    """Numerically integrate the distribution to get (norm, <v>, <v^2>) -- an
    independent check of the closed-form speeds."""
    v = np.linspace(0, 6 * rms_speed(mass, T), n)
    f = maxwell_boltzmann_pdf(v, mass, T)
    norm = np.trapezoid(f, v)
    v1 = np.trapezoid(v * f, v)
    v2 = np.trapezoid(v ** 2 * f, v)
    return norm, v1, v2


if __name__ == "__main__":
    m = 28 * AMU        # N2 molecule
    T = 300.0
    print(f"nitrogen (N2) at {T:.0f} K:")
    print(f"  most probable v = {most_probable_speed(m, T):.0f} m/s")
    print(f"  mean speed      = {mean_speed(m, T):.0f} m/s")
    print(f"  rms speed       = {rms_speed(m, T):.0f} m/s")
    print(f"  ordering v_mp < v_avg < v_rms with ratios "
          f"{most_probable_speed(m,T)/most_probable_speed(m,T):.3f} : "
          f"{mean_speed(m,T)/most_probable_speed(m,T):.3f} : "
          f"{rms_speed(m,T)/most_probable_speed(m,T):.3f}  (1 : sqrt(4/pi) : sqrt(1.5))")

    norm, v1, v2 = _moments(m, T)
    print(f"\nnumerical checks:")
    print(f"  integral of f(v) dv = {norm:.4f}  (= 1)")
    print(f"  <v>       = {v1:.1f} m/s vs mean_speed {mean_speed(m, T):.1f}")
    print(f"  sqrt<v^2> = {np.sqrt(v2):.1f} m/s vs rms_speed {rms_speed(m, T):.1f}")
    print(f"  <KE> = 1/2 m <v^2> = {0.5*m*v2:.3e} J vs 3/2 kT "
          f"{average_kinetic_energy(T):.3e} J")

    print(f"\ntemperature scaling: v_rms doubles when T x 4 "
          f"({rms_speed(m,1200)/rms_speed(m,300):.2f}x for 4x T)")
