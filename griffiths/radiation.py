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


# ── radiation reaction: the charge pushes back on itself ─────────────
def abraham_lorentz_force(q, da_dt):
    """F_rad = mu0*q^2/(6*pi*c) * da/dt  (Griffiths Eq. 11.80). Radiating
    away energy (Larmor) means something must be doing work AGAINST the
    charge's motion -- this is that self-force, proportional to the JERK
    (da/dt), not the acceleration itself. Same prefactor as larmor_power's
    mu0*q^2/(6*pi*c), one derivative higher."""
    return sp.simplify(mu0 * q**2 / (6 * sp.pi * c) * da_dt)


def verify_abraham_lorentz_does_work_matching_larmor(q, a0, omega, t):
    """For a(t) = a0*cos(omega*t), verify that F_rad's average power
    delivered (-F_rad * v, integrated/averaged) matches larmor_power's
    average radiated power -- the whole POINT of the Abraham-Lorentz force
    is that it's the mechanical bookkeeping for Larmor radiation, checked
    here rather than just asserted. v(t) = a0/omega * sin(omega t) (one
    integral of a(t), zero integration constant -- oscillatory motion)."""
    a = a0 * sp.cos(omega * t)
    da_dt = sp.diff(a, t)
    F_rad = abraham_lorentz_force(q, da_dt)
    v = a0 / omega * sp.sin(omega * t)
    power_delivered_to_charge = sp.simplify(F_rad * v)  # -(-F_rad*v) sign handled by defn below
    T = 2 * sp.pi / omega
    avg_power_lost_by_charge = sp.simplify(-sp.integrate(power_delivered_to_charge, (t, 0, T)) / T)
    P_larmor_avg = sp.simplify(dipole_average_power(sp.Symbol('p0_placeholder'), omega))
    # direct comparison: Larmor for a POINT CHARGE (not a dipole moment) is
    # <P> = mu0*q^2*<a^2>/(6*pi*c) = mu0*q^2*a0^2/(12*pi*c) (a0^2 time-averages to a0^2/2)
    P_larmor_point_charge = sp.simplify(mu0 * q**2 * a0**2 / (12 * sp.pi * c))
    return {
        "avg_power_from_abraham_lorentz": avg_power_lost_by_charge,
        "avg_power_from_larmor": P_larmor_point_charge,
        "match": bool(sp.simplify(avg_power_lost_by_charge - P_larmor_point_charge) == 0),
    }


# ── magnetic dipole radiation ─────────────────────────────────────────
def magnetic_dipole_average_power(m0, omega):
    """Time-averaged power of an oscillating MAGNETIC dipole m(t)=m0*cos(wt)
    (Griffiths Eq. 11.39-equivalent): <P> = mu0*m0^2*omega^4/(12*pi*c^3).
    Same omega^4/(12*pi*c) structure as the electric dipole
    (dipole_average_power), with one extra factor of 1/c^2 -- magnetic
    dipole radiation is weaker than electric dipole radiation by that
    factor for comparable source strengths, which is WHY electric dipole
    radiation dominates almost every practical antenna and atomic
    transition."""
    return sp.simplify(mu0 * m0**2 * omega**4 / (12 * sp.pi * c**3))


def electric_vs_magnetic_dipole_power_ratio():
    """dipole_average_power(p0,w) / magnetic_dipole_average_power(m0,w) with
    p0=m0 (equal-magnitude source moments, an apples-to-apples comparison,
    NOT a claim that p0 and m0 have the same units/physical meaning) --
    isolates the c^2 factor the two formulas differ by."""
    p0, m0, omega = sp.symbols('p0 m0 omega', positive=True)
    ratio = sp.simplify(dipole_average_power(p0, omega) / magnetic_dipole_average_power(m0, omega).subs(m0, p0))
    return ratio


# ── parity of the radiation (odd vs even multipoles) ────────────────
def multipole_parity(order):
    """Parity of the 2^order-pole radiation field under r -> -r:
    dipole (order 1) is odd, quadrupole (order 2) even, ... = (-1)^order.
    The leading nonzero (usually odd dipole) dominates -- why antennas are dipoles."""
    if order < 1:
        raise ValueError("multipole order must be >= 1")
    return (-1)**order
