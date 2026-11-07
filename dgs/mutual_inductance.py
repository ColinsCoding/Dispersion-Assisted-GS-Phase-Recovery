"""Mutual inductance in terms of self-inductance: M = k sqrt(L1 L2), derived, computed and measured.

In plain words: two coils near each other share magnetic flux. Current in coil 1 makes flux, and the
part of that flux that also threads coil 2 induces a voltage there. The mutual inductance M measures
that shared flux, the self-inductances L1 and L2 measure each coil's own flux, and the coupling
coefficient k (between 0 and 1) says what fraction is shared. The headline result is

        M = k sqrt(L1 L2),        0 <= k <= 1,

so M can never be bigger than the geometric mean of the two self-inductances. Two independent
arguments give the bound, and this module codes both:

  * Energy. Two coils carrying I1 and I2 store W = 1/2 L1 I1^2 + 1/2 L2 I2^2 + M I1 I2. Stored
    magnetic energy is an integral of B^2 / 2 mu0, so W >= 0 for every choice of currents. Write
    x = I1 / I2: W / I2^2 = 1/2 L1 x^2 + M x + 1/2 L2 is a parabola that must never dip below zero,
    so its discriminant M^2 - L1 L2 must be <= 0. Equivalently the inductance matrix
    [[L1, M], [M, L2]] is positive semidefinite (det >= 0). derive_energy_bound() does this in SymPy.
  * Flux linkage. If coil 1 makes flux Phi1 per turn and a fraction k1 of it reaches coil 2, then
    L1 = N1 Phi1 / I1 and M = N2 k1 Phi1 / I1; the same from coil 2's side gives M = N1 k2 Phi2 / I2.
    Multiply the two expressions: M^2 = k1 k2 L1 L2, so k = sqrt(k1 k2) <= 1.

What is in here (SI units throughout, henries, metres, amperes):

  * mutual_inductance(L1, L2, k), coupling_coefficient(L1, L2, M), energy(...),
    inductance_matrix_is_psd(...): the core relations, with input checks (L > 0, 0 <= k <= 1,
    |M| <= sqrt(L1 L2)).
  * Closed forms. Long coaxial solenoids, inner area A1 inside outer area A2, same length l:
    L = mu0 N^2 A / l and M = mu0 N1 N2 A1 / l, so k = sqrt(A1 / A2), which is 1 for equal areas.
    Coaxial circular loops (radii a, b, axial gap d): Maxwell's elliptic-integral formula
    M = mu0 sqrt(a b) [(2/m - m) K(m) - (2/m) E(m)], m^2 = 4ab / ((a + b)^2 + d^2). The complete
    elliptic integrals K and E are computed with the arithmetic-geometric mean, so no SciPy needed.
    Thin-wire loop self-inductance L = mu0 R (ln(8R / a_w) - 7/4) (uniform current; use -2 when skin
    effect pushes the current to the surface).
  * Neumann's formula M = mu0 / (4 pi) closed-integral closed-integral dl1 . dl2 / |r1 - r2|, as a
    double sum over two polylines (any shape, offset or tilted) and as the 1-D integral it reduces
    to for coaxial loops. These check the elliptic formula numerically.
  * Series connection. Both currents equal: L_aid = L1 + L2 + 2M; one coil reversed: L_opp =
    L1 + L2 - 2M. Subtracting gives the bench recipe M = (L_aid - L_opp) / 4. A second recipe uses
    open- and short-circuit readings on one winding: L_short = L1 (1 - k^2), so
    k = sqrt(1 - L_short / L_open).
  * Transformers. T-equivalent (common ground): series arms L1 - M and L2 - M, shunt arm M. Input
    impedance Z_in = j w L1 + w^2 M^2 / (j w L2 + Z_L) and voltage ratio. At k = 1 the voltage ratio
    is exactly sqrt(L2 / L1) = N2 / N1 for any load, and with L1, L2 -> infinity Z_in -> Z_L L1 / L2:
    the ideal transformer. Leakage inductance is (1 - k) L on each side in the symmetric split.
  * Wireless power. Flat spiral coils modelled as concentric filament turns, so L and M come from
    the same loop formulas; best achievable link efficiency eta_max = x / (1 + sqrt(1 + x))^2 with
    x = k^2 Q1 Q2 (standard result for a resonant inductive link, quoted from memory).

Connects to: dgs.ac_circuits (phasor impedances), dgs.spice (circuit simulation),
griffiths/electrodynamics.py (Faraday's law; Griffiths ch. 7 derives Neumann's formula and the
equality M12 = M21), dgs.error_propagation (the bench-measurement uncertainty in the notebook).
"""
from __future__ import annotations

import math

import numpy as np

MU0 = 4e-7 * math.pi          # close enough to the 2019 SI value (1.25663706212e-6) for teaching


# ---------------------------------------------------------------- input checks
def _check_L(name, L):
    L = float(L)
    if not math.isfinite(L) or L <= 0:
        raise ValueError(f"{name} must be a positive, finite inductance in henries; got {L!r}")
    return L


def _check_k(k):
    k = float(k)
    if not (0.0 <= k <= 1.0):
        raise ValueError(f"coupling coefficient k must satisfy 0 <= k <= 1; got {k!r}")
    return k


def _check_pos(name, x):
    x = float(x)
    if not math.isfinite(x) or x <= 0:
        raise ValueError(f"{name} must be positive and finite; got {x!r}")
    return x


# ---------------------------------------------------------------- core relations
def mutual_inductance(L1, L2, k):
    """M = k sqrt(L1 L2)."""
    return _check_k(k) * math.sqrt(_check_L("L1", L1) * _check_L("L2", L2))


def coupling_coefficient(L1, L2, M, tol=1e-12):
    """k = |M| / sqrt(L1 L2); raises if |M| exceeds sqrt(L1 L2) (that would allow negative energy)."""
    Lmax = math.sqrt(_check_L("L1", L1) * _check_L("L2", L2))
    k = abs(float(M)) / Lmax
    if k > 1.0 + tol:
        raise ValueError(f"|M| = {abs(M):.6g} H exceeds sqrt(L1 L2) = {Lmax:.6g} H: k = {k:.6g} > 1 is "
                         "unphysical (the stored energy could go negative)")
    return min(k, 1.0)


def energy(L1, L2, M, I1, I2):
    """Stored magnetic energy W = 1/2 L1 I1^2 + 1/2 L2 I2^2 + M I1 I2 (joules)."""
    return 0.5 * L1 * I1 ** 2 + 0.5 * L2 * I2 ** 2 + M * I1 * I2


def inductance_matrix(L1, L2, M):
    return np.array([[L1, M], [M, L2]], dtype=float)


def inductance_matrix_is_psd(L1, L2, M, tol=1e-15):
    """True when [[L1, M], [M, L2]] has no negative eigenvalue, i.e. W >= 0 for all currents."""
    return bool(np.linalg.eigvalsh(inductance_matrix(L1, L2, M)).min() >= -tol * max(L1, L2))


def derive_energy_bound():
    """SymPy: minimise W over the current ratio and read off M^2 <= L1 L2.

    Returns a dict with W, the minimising ratio x* = I1/I2, W_min, the discriminant and the
    determinant of the inductance matrix.
    """
    import sympy as sp
    L1, L2 = sp.symbols("L_1 L_2", positive=True)
    M, x, I2 = sp.symbols("M x I_2", real=True)
    W = sp.Rational(1, 2) * L1 * (x * I2) ** 2 + sp.Rational(1, 2) * L2 * I2 ** 2 + M * (x * I2) * I2
    x_star = sp.solve(sp.diff(W, x), x)[0]
    W_min = sp.factor(sp.simplify(W.subs(x, x_star)))
    w = sp.expand(W / I2 ** 2)                                   # 1/2 L1 x^2 + M x + 1/2 L2
    disc = sp.expand(sp.discriminant(w, x))                      # M^2 - L1 L2
    det = sp.Matrix([[L1, M], [M, L2]]).det()
    return {"W": W, "x_star": x_star, "W_min": W_min, "discriminant": disc, "det": det}


# ---------------------------------------------------------------- series connection, bench recipes
def series_aiding(L1, L2, M):
    return _check_L("L1", L1) + _check_L("L2", L2) + 2 * M


def series_opposing(L1, L2, M):
    return _check_L("L1", L1) + _check_L("L2", L2) - 2 * M


def mutual_from_series(L_aid, L_opp):
    """M = (L_aid - L_opp) / 4 from two series LCR readings (one coil flipped between them)."""
    _check_L("L_aid", L_aid), _check_L("L_opp", L_opp)
    return (L_aid - L_opp) / 4.0


def k_from_series(L_aid, L_opp, L1, L2):
    return coupling_coefficient(L1, L2, mutual_from_series(L_aid, L_opp))


def short_circuit_inductance(L1, L2, k):
    """Inductance seen at winding 1 with winding 2 shorted: L1 (1 - k^2) (leakage seen from side 1)."""
    _check_L("L2", L2)
    return _check_L("L1", L1) * (1 - _check_k(k) ** 2)


def k_from_open_short(L_open, L_short):
    """k = sqrt(1 - L_short / L_open) from two readings on the same winding."""
    L_open = _check_L("L_open", L_open)
    L_short = float(L_short)
    if not (0 <= L_short <= L_open):
        raise ValueError(f"need 0 <= L_short <= L_open; got L_short={L_short!r}, L_open={L_open!r}")
    return math.sqrt(1 - L_short / L_open)


# ---------------------------------------------------------------- transformers
def t_equivalent(L1, L2, M):
    """Series arms and shunt arm of the T network (shared ground): (L1 - M, L2 - M, M).

    An arm can come out negative (e.g. a step-down with L2 < M); that is fine for analysis, it just
    is not a single physical inductor.
    """
    coupling_coefficient(L1, L2, M)
    return L1 - M, L2 - M, M


def leakage_and_magnetizing(L1, L2, k):
    """Symmetric split: leakage (1-k) L1 and (1-k) L2, magnetizing k L1 referred to side 1, ratio sqrt(L2/L1)."""
    L1, L2, k = _check_L("L1", L1), _check_L("L2", L2), _check_k(k)
    return {"leak1": (1 - k) * L1, "leak2": (1 - k) * L2, "Lm1": k * L1, "n": math.sqrt(L2 / L1)}


def input_impedance(L1, L2, M, Z_L, omega):
    """Z_in = j w L1 + w^2 M^2 / (j w L2 + Z_L) (lossless windings, phasors)."""
    coupling_coefficient(L1, L2, M)
    w = _check_pos("omega", omega)
    return 1j * w * L1 + (w * M) ** 2 / (1j * w * L2 + Z_L)


def voltage_ratio(L1, L2, M, Z_L, omega):
    """V2 / V1 with load Z_L on winding 2 (dot convention, both currents into the dots)."""
    w = _check_pos("omega", omega)
    return 1j * w * M * Z_L / ((1j * w * L2 + Z_L) * input_impedance(L1, L2, M, Z_L, w))


def link_efficiency_max(k, Q1, Q2):
    """Best efficiency of a resonant inductive link with optimal load: x / (1 + sqrt(1 + x))^2, x = k^2 Q1 Q2."""
    x = _check_k(k) ** 2 * _check_pos("Q1", Q1) * _check_pos("Q2", Q2)
    return x / (1 + math.sqrt(1 + x)) ** 2


# ---------------------------------------------------------------- closed forms: solenoids
def solenoid_inductance(N, area, length):
    """Long solenoid L = mu0 N^2 A / l (ignores end effects; good when length >> radius)."""
    return MU0 * _check_pos("N", N) ** 2 * _check_pos("area", area) / _check_pos("length", length)


def nested_solenoids(N1, A1, N2, A2, length):
    """Inner solenoid (N1, A1) inside outer (N2, A2 >= A1), same length: returns L1, L2, M, k.

    The outer coil's field mu0 N2 I2 / l fills A1, so M = mu0 N1 N2 A1 / l and k = sqrt(A1 / A2).
    """
    if A1 > A2:
        raise ValueError(f"inner area A1={A1!r} must not exceed outer area A2={A2!r}")
    L1 = solenoid_inductance(N1, A1, length)
    L2 = solenoid_inductance(N2, A2, length)
    M = MU0 * N1 * N2 * A1 / length
    return {"L1": L1, "L2": L2, "M": M, "k": M / math.sqrt(L1 * L2)}


# ---------------------------------------------------------------- closed forms: loops
def _agm_KE(m):
    """Complete elliptic integrals K(m), E(m) of modulus m (not parameter m^2), via the AGM."""
    if not (0 <= m < 1):
        raise ValueError(f"elliptic modulus must be in [0, 1); got {m!r}")
    a, b, c = 1.0, math.sqrt(1 - m * m), m
    s, p = 0.5 * c * c, 0.5                       # sum of 2^(n-1) c_n^2, starting n = 0
    while abs(c) > 1e-16:
        a, b, c = 0.5 * (a + b), math.sqrt(a * b), 0.5 * (a - b)
        p *= 2
        s += p * c * c
    K = math.pi / (2 * a)
    return K, K * (1 - s)


def elliptic_K(m):
    return _agm_KE(m)[0]


def elliptic_E(m):
    return _agm_KE(m)[1]


def coaxial_loops_mutual(a, b, d):
    """Maxwell's formula for two coaxial circular filaments, radii a, b, axial separation d (H)."""
    a, b = _check_pos("a", a), _check_pos("b", b)
    d = abs(float(d))
    m = math.sqrt(4 * a * b / ((a + b) ** 2 + d ** 2))
    if m >= 1:
        raise ValueError("coincident filaments (a == b, d == 0): M diverges; give the wire a radius")
    K, E = _agm_KE(m)
    return MU0 * math.sqrt(a * b) * ((2 / m - m) * K - (2 / m) * E)


def loop_self_inductance(R, wire_radius, uniform_current=True):
    """Thin circular loop: mu0 R (ln(8R/a) - 7/4) for uniform current, - 2 for surface current."""
    R, aw = _check_pos("R", R), _check_pos("wire_radius", wire_radius)
    if aw >= R:
        raise ValueError("wire radius must be much smaller than loop radius")
    return MU0 * R * (math.log(8 * R / aw) - (1.75 if uniform_current else 2.0))


def neumann_coaxial(a, b, d, n=2048):
    """Neumann's formula for coaxial loops reduced to one angle: mu0 ab/2 int cos(phi)/r dphi.

    The integrand is smooth and periodic, so the plain trapezoid rule converges exponentially.
    """
    phi = 2 * np.pi * np.arange(n) / n
    r = np.sqrt(a * a + b * b + d * d - 2 * a * b * np.cos(phi))
    return MU0 * a * b / 2 * np.sum(np.cos(phi) / r) * (2 * np.pi / n)


def circle_points(radius, center=(0, 0, 0), normal=(0, 0, 1), n=400):
    """n points on a circle (closed curve; the last point is not repeated)."""
    nrm = np.asarray(normal, float)
    nrm = nrm / np.linalg.norm(nrm)
    helper = np.array([1.0, 0, 0]) if abs(nrm[0]) < 0.9 else np.array([0, 1.0, 0])
    u = np.cross(nrm, helper); u /= np.linalg.norm(u)
    v = np.cross(nrm, u)
    t = 2 * np.pi * np.arange(n) / n
    return np.asarray(center, float) + radius * (np.outer(np.cos(t), u) + np.outer(np.sin(t), v))


def neumann_polyline(P1, P2):
    """M = mu0/(4 pi) sum_i sum_j dl_i . dl_j / |m_i - m_j| over two closed polylines (rows = points).

    Midpoint rule on straight segments; error falls like 1/n^2. Curves must not touch.
    """
    P1, P2 = np.asarray(P1, float), np.asarray(P2, float)
    dl1 = np.roll(P1, -1, axis=0) - P1
    dl2 = np.roll(P2, -1, axis=0) - P2
    m1 = P1 + 0.5 * dl1
    m2 = P2 + 0.5 * dl2
    r = np.linalg.norm(m1[:, None, :] - m2[None, :, :], axis=-1)
    if r.min() == 0:
        raise ValueError("the two curves share a point: Neumann's integral diverges")
    return MU0 / (4 * np.pi) * np.sum((dl1 @ dl2.T) / r)


# ---------------------------------------------------------------- flat spiral coils (wireless power)
def spiral_radii(r_in, r_out, turns):
    """Radii of the filament turns of a flat spiral, evenly spaced from r_in to r_out."""
    turns = int(turns)
    if turns < 1:
        raise ValueError("turns must be >= 1")
    if not (0 < r_in <= r_out):
        raise ValueError("need 0 < r_in <= r_out")
    return np.linspace(r_in, r_out, turns)


def spiral_self_inductance(radii, wire_radius):
    """Sum of each turn's self-inductance plus every pair's mutual (same plane, d = 0)."""
    radii = np.asarray(radii, float)
    L = sum(loop_self_inductance(r, wire_radius) for r in radii)
    for i in range(len(radii)):
        for j in range(i + 1, len(radii)):
            L += 2 * coaxial_loops_mutual(radii[i], radii[j], 0.0)
    return L


def spiral_mutual(radii1, radii2, gap):
    """Mutual inductance of two coaxial flat spirals separated by `gap` (sum over all turn pairs)."""
    return sum(coaxial_loops_mutual(a, b, gap) for a in radii1 for b in radii2)


def spiral_mutual_offset(radii1, radii2, gap, lateral, n=200):
    """Same, but coil 2 shifted sideways by `lateral` (no closed form: Neumann polyline sum)."""
    total = 0.0
    for a in radii1:
        P1 = circle_points(a, (0, 0, 0), n=n)
        for b in radii2:
            total += neumann_polyline(P1, circle_points(b, (lateral, 0, gap), n=n))
    return total
