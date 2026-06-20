"""Magnetostatics -- Griffiths Ch. 5.

The magnetic (Lorentz) force and cyclotron motion, Biot-Savart fields (straight
wire, on-axis loop), Ampere's-law fields (solenoid), the magnetic vector
potential, and the two structural facts div B = 0, curl B = mu0 J. SymPy
throughout; mu0 symbolic. Deliberately parallels griffiths.electrostatics --
the electric ring (Ch. 2) and the magnetic loop here are duals.
"""

import sympy as sp

from .vectors import CARTESIAN, _as_vec3, curl, div

mu0 = sp.Symbol("mu_0", positive=True)


# ── magnetic force and cyclotron motion ─────────────────────────────
def lorentz_force(q, E, v, B):
    """F = q(E + v x B)."""
    E, v, B = _as_vec3(E, "E"), _as_vec3(v, "v"), _as_vec3(B, "B")
    return sp.simplify(q * (E + v.cross(B)))


def cyclotron_frequency(q, m, B):
    """omega_c = qB/m (sign gives sense of rotation)."""
    if m == 0:
        raise ValueError("mass m must be nonzero")
    return sp.simplify(q * B / m)


def cyclotron_radius(m, vperp, q, B):
    """r = m v_perp / (|q| B)."""
    return sp.simplify(m * vperp / (sp.Abs(q) * B))


def cyclotron_period(q, m, B):
    """T = 2 pi m / (|q| B) -- independent of speed (the isochronism that makes
    the cyclotron work)."""
    return sp.simplify(2 * sp.pi * m / (sp.Abs(q) * B))


def cyclotron_trajectory(q, m, B, vperp, vpar, t):
    """Analytic helical path in uniform B = B z-hat, starting at the origin with
    velocity (vperp, 0, vpar). Returns (x(t), y(t), z(t)). Verified to satisfy
    m a = q v x B."""
    w = q * B / m
    x = (vperp / w) * sp.sin(w * t)
    y = (vperp / w) * (sp.cos(w * t) - 1)
    z = vpar * t
    return sp.simplify(x), sp.simplify(y), sp.simplify(z)


def ExB_drift(E, B):
    """Drift velocity of a charge in crossed fields: v_d = (E x B)/B^2,
    independent of charge and mass (the velocity-selector / Hall condition)."""
    E, B = _as_vec3(E, "E"), _as_vec3(B, "B")
    B2 = (B.T * B)[0]
    if B2 == 0:
        raise ValueError("B must be nonzero")
    return sp.simplify(E.cross(B) / B2)


# ── Biot-Savart fields ──────────────────────────────────────────────
def wire_field(I, s):
    """Infinite straight wire: B = mu0 I / (2 pi s), circling the wire."""
    return sp.simplify(mu0 * I / (2 * sp.pi * s))


def loop_field_axis(I, R, z):
    """On-axis field of a circular current loop (radius R, current I):
    B_z = mu0 I R^2 / (2 (R^2 + z^2)^{3/2}). The magnetic dual of the charged
    ring's E_z (Griffiths 2.5)."""
    return sp.simplify(mu0 * I * R**2 / (2 * (R**2 + z**2)**sp.Rational(3, 2)))


def magnetic_dipole_moment(I, R):
    """m = I * area = I pi R^2 for a circular loop."""
    return sp.simplify(I * sp.pi * R**2)


# ── Ampere's law ────────────────────────────────────────────────────
def solenoid_field(n, I):
    """Long solenoid: uniform interior field B = mu0 n I (n = turns per length),
    zero outside."""
    return sp.simplify(mu0 * n * I)


def ampere_enclosed_wire(I, s):
    """Ampere's law on a circle of radius s around a wire: 2 pi s B = mu0 I."""
    B = sp.Symbol("B", positive=True)
    return sp.solve(sp.Eq(2 * sp.pi * s * B, mu0 * I), B)[0]


# ── vector potential and the structural laws ────────────────────────
def B_from_vector_potential(A, vars=CARTESIAN):
    """B = curl A."""
    return curl(_as_vec3(A, "A"), vars)


def is_divergence_free(B, vars=CARTESIAN):
    """Magnetic fields obey div B = 0 (no monopoles). Returns (ok, div_B)."""
    d = sp.simplify(div(_as_vec3(B, "B"), vars))
    return (d == 0), d


def current_from_field(B, vars=CARTESIAN):
    """Ampere (static): J = (curl B) / mu0."""
    return sp.simplify(curl(_as_vec3(B, "B"), vars) / mu0)
