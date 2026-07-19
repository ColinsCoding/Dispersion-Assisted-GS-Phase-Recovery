"""Radiation -- Griffiths Ch. 11. Accelerating charges and antennas.

A charge moving at constant velocity makes no radiation; *acceleration* costs
energy that leaves as radiation -- the Larmor formula. An oscillating dipole is
the elemental antenna, with the sin^2(theta) "donut" pattern; the half-wave
dipole is its practical cousin. SymPy/symbolic; the optical version of the same
dipole is an atom emitting a photon, the tie to this repo's photonics.
"""

import sympy as sp

mu0, c, eps0 = sp.symbols("mu_0 c epsilon_0", positive=True)
theta = sp.Symbol("theta", positive=True)


# ── Larmor: accelerating charges radiate ────────────────────────────
def larmor_power(q, a):
    """Total power radiated by a nonrelativistic point charge of acceleration a:
    P = mu0 q^2 a^2 / (6 pi c)  =  q^2 a^2 / (6 pi eps0 c^3)."""
    return sp.simplify(mu0 * q**2 * a**2 / (6 * sp.pi * c))


def larmor_angular(q, a):
    """Angular distribution dP/dOmega = (mu0 q^2 a^2 / 16 pi^2 c) sin^2(theta):
    a charge radiates most broadside to its acceleration, nothing along it."""
    return mu0 * q**2 * a**2 / (16 * sp.pi**2 * c) * sp.sin(theta)**2


# ── the oscillating electric dipole (the antenna) ───────────────────
def dipole_average_power(p0, omega):
    """Time-averaged power of an oscillating dipole p(t)=p0 cos(wt):
    <P> = mu0 p0^2 omega^4 / (12 pi c). The omega^4 is why the sky is blue."""
    return sp.simplify(mu0 * p0**2 * omega**4 / (12 * sp.pi * c))


def dipole_E_theta(p0, omega, r, t):
    """Far-field radiation E_theta of an oscillating dipole (Griffiths 11.18):
    transverse, falls as 1/r (not 1/r^2), carries the energy away."""
    return sp.simplify(-mu0 * p0 * omega**2 / (4 * sp.pi) * sp.sin(theta) / r
                       * sp.cos(omega * (t - r / c)))


def radiation_pattern(kind="hertzian"):
    """Normalised radiated-power pattern P(theta)/P_max.

    'hertzian'  : sin^2(theta)  (short dipole / the donut)
    'half_wave' : [cos((pi/2)cos theta)/sin theta]^2  (practical lambda/2 dipole)
    """
    if kind == "hertzian":
        return sp.sin(theta)**2
    if kind == "half_wave":
        return (sp.cos(sp.pi / 2 * sp.cos(theta)) / sp.sin(theta))**2
    raise ValueError("kind must be 'hertzian' or 'half_wave'")


def directivity(kind="hertzian"):
    """Peak directivity D = P_max / <P>_solid-angle (the antenna 'gain' over isotropic)."""
    pat = radiation_pattern(kind)
    avg = sp.integrate(pat * sp.sin(theta), (theta, 0, sp.pi)) / 2   # phi-averaged /4pi*2pi
    pmax = 1 if kind == "hertzian" else sp.limit(pat, theta, sp.pi / 2)
    return sp.simplify(pmax / avg)


def total_pattern_solid_angle(kind="hertzian"):
    """Integral of the pattern over the full sphere (for normalisation/checks)."""
    pat = radiation_pattern(kind)
    return sp.simplify(sp.integrate(pat * sp.sin(theta), (theta, 0, sp.pi)) * 2 * sp.pi)


# ── parity of the radiation (odd vs even multipoles) ────────────────
def multipole_parity(order):
    """Parity of the 2^order-pole radiation field under r -> -r:
    dipole (order 1) is odd, quadrupole (order 2) even, ... = (-1)^order.
    The leading nonzero (usually odd dipole) dominates -- why antennas are dipoles."""
    if order < 1:
        raise ValueError("multipole order must be >= 1")
    return (-1)**order
