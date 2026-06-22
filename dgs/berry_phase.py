"""Berry's phase -- the geometric phase from a loop in parameter space (Griffiths Ch 10).

Cycle a quantum system's Hamiltonian slowly (adiabatically) around a CLOSED loop in
parameter space and the wavefunction returns with an extra phase that depends only on
the GEOMETRY of the loop, not on how fast you went -- the Berry (geometric) phase. For a
spin-1/2 whose magnetic-field direction traces a loop on the unit sphere, the result is
beautifully simple:

    gamma = -(1/2) * Omega ,

where Omega is the SOLID ANGLE the loop encloses. We compute it gauge-invariantly as the
discrete (Pancharatnam) phase gamma = -Im ln( prod <psi_n | psi_{n+1}> ) around the loop,
and check it against -1/2 the solid angle. The same idea is the Pancharatnam phase of
polarized light -- a geometric phase in optics, alongside the dynamical phase the
dispersion-GS receiver recovers. NumPy. Education.
"""

import numpy as np


def spin_eigenstate(theta, phi):
    """Spin-1/2 state aligned with the field direction (theta=colatitude, phi=azimuth):
    |+> = [cos(theta/2), sin(theta/2) e^{i phi}]."""
    return np.array([np.cos(theta / 2.0), np.sin(theta / 2.0) * np.exp(1j * phi)])


def berry_phase(states):
    """Discrete (Pancharatnam) Berry phase around a closed loop of state vectors:
        gamma = -Im ln( prod_n <psi_n | psi_{n+1}> )  (loop closed back to psi_0).
    Gauge-invariant -- independent of each state's arbitrary phase."""
    prod = 1.0 + 0j
    for n in range(len(states) - 1):
        prod *= np.vdot(states[n], states[n + 1])
    prod *= np.vdot(states[-1], states[0])              # close the loop
    return float(-np.angle(prod))


def solid_angle_cone(theta0):
    """Solid angle enclosed by a circle of colatitude theta0 (a cone): 2 pi (1 - cos theta0).
    0 at the pole, 2 pi at the equator (a hemisphere), 4 pi at the far pole."""
    return 2 * np.pi * (1 - np.cos(theta0))


def spin_loop_states(theta0, n=400):
    """The spin eigenstates as the field traces a circle at colatitude theta0 (one full
    loop in azimuth phi). Feed to berry_phase."""
    phi = np.linspace(0, 2 * np.pi, n)
    return [spin_eigenstate(theta0, p) for p in phi]


def berry_phase_spin(theta0, n=2000):
    """Berry phase for spin-1/2 around a cone of half-angle theta0, computed from the
    states; equals -1/2 * solid_angle_cone(theta0) (mod 2 pi)."""
    return berry_phase(spin_loop_states(theta0, n))


if __name__ == "__main__":
    print("  theta0   Berry phase   -Omega/2     match")
    for deg in (30, 60, 90, 120, 150):
        th = np.radians(deg)
        g = berry_phase_spin(th)
        pred = -0.5 * solid_angle_cone(th)
        # compare mod 2 pi
        diff = (g - pred + np.pi) % (2 * np.pi) - np.pi
        print(f"  {deg:4d}    {g:+.4f}      {pred:+.4f}    {abs(diff) < 1e-2}")
    print("\nBerry phase = -1/2 * (solid angle enclosed): purely geometric, speed-independent.")
