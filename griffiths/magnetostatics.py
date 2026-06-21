"""Magnetostatics -- Griffiths Ch. 5.

The magnetic (Lorentz) force and cyclotron motion, Biot-Savart fields (straight
wire, on-axis loop), Ampere's-law fields (solenoid), the magnetic vector
potential, and the two structural facts div B = 0, curl B = mu0 J. SymPy
throughout; mu0 symbolic. Deliberately parallels griffiths.electrostatics --
the electric ring (Ch. 2) and the magnetic loop here are duals.
"""

import numpy as np
import sympy as sp

from .vectors import CARTESIAN, _as_vec3, curl, div

mu0 = sp.Symbol("mu_0", positive=True)
MU0_SI = 4e-7 * np.pi          # vacuum permeability [T m / A], for numerical Biot-Savart


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


def toroid_field(N, I, s):
    """Toroidal solenoid (N total turns, current I): the field is purely azimuthal
    and lives ENTIRELY inside the windings,
        B = mu0 N I / (2 pi s)   inside the tube,   B = 0   outside,
    with s the distance from the toroid's central axis. Ampere's law gives this in
    one line because the symmetry makes B constant on a circle of radius s -- the
    same move that FAILS for a finite straight solenoid (no constant-B loop exists)."""
    return sp.simplify(mu0 * N * I / (2 * sp.pi * s))


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


# ── numerical Biot-Savart: prove the closed forms by integrating dl x r-hat ──
def biot_savart(path_points, I, field_point):
    """Numerically integrate the Biot-Savart law over an arbitrary wire:
        B(r) = (mu0 I / 4 pi) integral  dl x (r - r') / |r - r'|^3 .
    `path_points` is an (N,3) array tracing the wire; the integral is the sum over
    the N-1 straight segments (midpoint rule). Returns the 3-vector B at
    `field_point` [tesla]. This is the cross-product line integral done by hand --
    run it against the closed forms below to *prove* them rather than trust them."""
    p = np.asarray(path_points, dtype=float)
    r = np.asarray(field_point, dtype=float)
    dl = np.diff(p, axis=0)                       # segment vectors dl
    mid = 0.5 * (p[:-1] + p[1:])                  # segment midpoints r'
    sep = r - mid                                 # r - r'
    dist = np.linalg.norm(sep, axis=1)            # |r - r'|
    dB = np.cross(dl, sep) / dist[:, None] ** 3   # dl x (r-r') / |r-r'|^3
    return MU0_SI * I / (4 * np.pi) * dB.sum(axis=0)


def straight_wire_path(length, n=20001, axis=2, offset=(0.0, 0.0, 0.0)):
    """An (n,3) path for a straight wire of given length centered at `offset`,
    running along `axis` (0=x,1=y,2=z). Use a large length to approximate the
    infinite wire."""
    t = np.linspace(-length / 2, length / 2, n)
    p = np.zeros((n, 3))
    p[:, axis] = t
    return p + np.asarray(offset, float)


def circular_loop_path(R, n=2001, center=(0.0, 0.0, 0.0)):
    """An (n,3) closed path tracing a circular current loop of radius R in the xy
    plane (current flows counterclockwise -> B along +z on axis)."""
    phi = np.linspace(0, 2 * np.pi, n)
    c = np.asarray(center, float)
    return np.column_stack([R * np.cos(phi), R * np.sin(phi), np.zeros(n)]) + c


def biot_savart_multi(loops, I, field_point):
    """Total Biot-Savart field at `field_point` from several wire paths (e.g. the N
    turns of a toroid), summing each loop's contribution."""
    total = np.zeros(3)
    for lp in loops:
        total = total + biot_savart(lp, I, field_point)
    return total


def toroid_loops(R, a, N, n=200):
    """N circular turns (minor/tube radius a) wound around a torus of major radius R,
    centered on the z-axis. Returns a list of N (n,3) loop paths -- feed to
    biot_savart_multi to *measure* the toroid field and confirm the Ampere result
    B = mu0 N I / (2 pi R) inside the tube and ~0 outside. Each turn lies in the
    plane spanned by its radial direction and z."""
    loops = []
    for k in range(N):
        phi = 2 * np.pi * k / N
        er = np.array([np.cos(phi), np.sin(phi), 0.0])     # radial, in the toroid plane
        ez = np.array([0.0, 0.0, 1.0])
        t = np.linspace(0, 2 * np.pi, n)
        pts = R * er + a * np.cos(t)[:, None] * er + a * np.sin(t)[:, None] * ez
        loops.append(pts)
    return loops
