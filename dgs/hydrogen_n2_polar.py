"""The four n = 2 hydrogen wavefunctions: unit vectors, polar areas, color as a 3rd axis, time as a 4th.

In plain words: hydrogen's second shell has exactly FOUR orbital states -- 2s and the three
2p's (m = -1, 0, +1). They are unit vectors: each has length 1 (total probability 1) and any
two are perpendicular (overlap 0), so they form an orthonormal basis of a 4-dimensional
space, just as x-hat, y-hat, z-hat do for ordinary space. The "length" is an integral, and in
spherical (polar) coordinates the volume element is r^2 sin(theta) dr dtheta dphi: forget the
r^2 and the "area under the curve" comes out wrong. The radial probability P(r) = r^2 |R(r)|^2
is the curve whose area is 1.

Pictures: a 2D slice of |psi|^2 drawn with color is a 3D object (x, z, color). Add time and it
is 4D: a superposition of 1s and 2p0 sloshes up and down at the Bohr frequency
(E2 - E1)/h = 2.47e15 Hz -- the oscillating dipole that emits Lyman-alpha light at 121.6 nm.
2s + 2p0 does NOT move: they have the same energy (hydrogen's accidental degeneracy).

Energy vs frequency spans many decades, so it is plotted on log axes with E = h f, from the
21 cm hyperfine line (1.42 GHz, ~6 micro-eV) to the Lyman limit (3.29e15 Hz, 13.6 eV).

Computing e^{-r/2} by a Taylor series shows a computer-science trap: the alternating series
for e^{-x} loses digits to cancellation for large x; 1/e^{+x} or range reduction fixes it.
The C (math.h) and CUDA versions are in projects/hydrogen_taylor/.

Atomic units: a0 = 1, energies in hartree (1 Ha = 27.211386 eV) unless named _eV.
"""

import math

import numpy as np
import sympy as sp

from dgs import hydrogen_atom as ha

HARTREE_EV = 27.211386245988
H_EV_S = 4.135667696e-15          # Planck constant in eV s
C_LIGHT = 299792458.0
N2_STATES = [(2, 0, 0), (2, 1, -1), (2, 1, 0), (2, 1, 1)]


def _check_state(n, l, m):
    if (n, l, m) not in [(1, 0, 0)] + N2_STATES:
        raise ValueError(f"only 1s and the n=2 states are implemented, got {(n, l, m)}")


# ---------------------------------------------------------------- symbolic (SymPy)

r_s, th_s, ph_s = sp.symbols("r theta phi", positive=True)


def radial_sym(n, l):
    """R_nl(r) in atomic units (a0 = 1)."""
    if (n, l) == (1, 0):
        return 2 * sp.exp(-r_s)
    if (n, l) == (2, 0):
        return 1 / (2 * sp.sqrt(2)) * (2 - r_s) * sp.exp(-r_s / 2)
    if (n, l) == (2, 1):
        return 1 / (2 * sp.sqrt(6)) * r_s * sp.exp(-r_s / 2)
    raise ValueError(f"(n, l) = {(n, l)} not implemented")


def psi_sym(n, l, m):
    """psi_nlm = R_nl(r) Y_lm(theta, phi) (SymPy's Ynm, Condon-Shortley phase)."""
    _check_state(n, l, m)
    return radial_sym(n, l) * sp.Ynm(l, m, th_s, ph_s).expand(func=True)


def inner_product(a, b):
    """<a|b> = integral conj(a) b r^2 sin(theta) dr dtheta dphi over all space."""
    integrand = sp.simplify(sp.conjugate(a) * b * r_s ** 2 * sp.sin(th_s))
    return sp.simplify(sp.integrate(integrand, (ph_s, 0, 2 * sp.pi), (th_s, 0, sp.pi), (r_s, 0, sp.oo)))


def gram_matrix(states=N2_STATES):
    """4x4 matrix of inner products; the identity means 'four orthonormal unit vectors'."""
    psis = [psi_sym(*s) for s in states]
    return sp.Matrix(len(psis), len(psis), lambda i, j: inner_product(psis[i], psis[j]))


def spherical_unit_vectors():
    """r-hat, theta-hat, phi-hat in Cartesian components, and their Gram matrix (should be I)."""
    rh = sp.Matrix([sp.sin(th_s) * sp.cos(ph_s), sp.sin(th_s) * sp.sin(ph_s), sp.cos(th_s)])
    thh = sp.Matrix([sp.cos(th_s) * sp.cos(ph_s), sp.cos(th_s) * sp.sin(ph_s), -sp.sin(th_s)])
    phh = sp.Matrix([-sp.sin(ph_s), sp.cos(ph_s), 0])
    vecs = [rh, thh, phh]
    G = sp.Matrix(3, 3, lambda i, j: sp.simplify(vecs[i].dot(vecs[j])))
    return vecs, G


def radial_area(n, l, with_jacobian=True):
    """Area under P(r) = r^2 R^2 (with_jacobian=True, equals 1) or under R^2 alone (the mistake)."""
    R = radial_sym(n, l)
    f = R ** 2 * (r_s ** 2 if with_jacobian else 1)
    return sp.simplify(sp.integrate(f, (r_s, 0, sp.oo)))


def most_probable_radius(n, l):
    """Peaks of P(r) = r^2 R_nl^2 (solve dP/dr = 0, r > 0)."""
    P = r_s ** 2 * radial_sym(n, l) ** 2
    crit = sp.solve(sp.diff(P, r_s), r_s)
    vals = [cr for cr in crit if cr.is_positive]
    return max(vals, key=lambda v: float(P.subs(r_s, v)))


def radial_nodes(n, l):
    """Positive roots of R_nl (radial nodes; n - l - 1 of them)."""
    return [s_ for s_ in sp.solve(radial_sym(n, l), r_s) if s_.is_positive]


# ---------------------------------------------------------------- numeric (NumPy)

def radial(n, l, r):
    r = np.asarray(r, float)
    if (n, l) == (1, 0):
        return 2 * np.exp(-r)
    if (n, l) == (2, 0):
        return (2 - r) * np.exp(-r / 2) / (2 * np.sqrt(2))
    if (n, l) == (2, 1):
        return r * np.exp(-r / 2) / (2 * np.sqrt(6))
    raise ValueError(f"(n, l) = {(n, l)} not implemented")


def ylm(l, m, theta, phi):
    """Spherical harmonics for l <= 1, Condon-Shortley phase (matches SymPy's Ynm)."""
    theta, phi = np.asarray(theta, float), np.asarray(phi, float)
    if (l, m) == (0, 0):
        return np.full(np.broadcast(theta, phi).shape, 1 / np.sqrt(4 * np.pi), complex)
    if (l, m) == (1, 0):
        return np.sqrt(3 / (4 * np.pi)) * np.cos(theta) + 0j
    if l == 1 and m in (1, -1):
        return -m * np.sqrt(3 / (8 * np.pi)) * np.sin(theta) * np.exp(1j * m * phi)
    raise ValueError(f"(l, m) = {(l, m)} not implemented")


def psi(n, l, m, r, theta, phi):
    _check_state(n, l, m)
    return radial(n, l, r) * ylm(l, m, theta, phi)


def to_spherical(x, y, z):
    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    theta = np.arccos(np.clip(np.divide(z, r, out=np.zeros_like(r), where=r > 0), -1, 1))
    return r, theta, np.arctan2(y, x)


def density_slice_xz(states_coeffs, extent=15.0, n=301, t=0.0):
    """|Psi(x, 0, z, t)|^2 on the y = 0 plane for a superposition {(n,l,m): c}.
    Each state carries e^{-i E_n t} (atomic units). The 2D map + color is the '3D' picture."""
    ax = np.linspace(-extent, extent, n)
    X, Z = np.meshgrid(ax, ax)
    r, th, ph = to_spherical(X, np.zeros_like(X), Z)
    total = np.zeros_like(X, complex)
    for (nn, l, m), c in states_coeffs.items():
        total += c * psi(nn, l, m, r, th, ph) * np.exp(-1j * energy_ha(nn) * t)
    return ax, np.abs(total) ** 2


def energy_ha(n):
    """E_n = -1/(2 n^2) hartree."""
    return -0.5 / n ** 2


def dipole_z(t, c1=1 / np.sqrt(2), c2=1 / np.sqrt(2)):
    """<z>(t) for c1|1s> + c2|2p0>: 2 Re(c1* c2 <1s|z|2p0> e^{-i(E2-E1)t}); <1s|z|2p0> = 128 sqrt(2)/243."""
    zme = 128 * np.sqrt(2) / 243
    w = energy_ha(2) - energy_ha(1)
    return 2 * np.real(np.conj(c1) * c2 * zme * np.exp(-1j * w * np.asarray(t, float)))


def bohr_frequency_hz(n_i, n_f):
    """|E_i - E_f| / h in Hz, using dgs.hydrogen_atom energies (eV)."""
    return abs(ha.energy_level(n_i) - ha.energy_level(n_f)) / H_EV_S


def spectrum_table():
    """Energies (eV) and frequencies (Hz) from radio to UV for a log-log E = h f chart."""
    rows = [("H 21 cm hyperfine", 1420.405751768e6 * H_EV_S),
            ("Balmer alpha (n=3->2), 656 nm", abs(ha.transition_energy(3, 2))),
            ("Lyman alpha (n=2->1), 122 nm", abs(ha.transition_energy(2, 1))),
            ("Lyman limit (ionization), 91 nm", 13.605693122994),
            ("kT at 300 K", 8.617333262e-5 * 300),
            ("1 GHz clock tick", 1e9 * H_EV_S)]
    return [dict(name=nm, energy_eV=float(E), freq_Hz=float(E / H_EV_S), wavelength_m=float(C_LIGHT * H_EV_S / E))
            for nm, E in rows]


# ---------------------------------------------------------------- Taylor series (the C/CUDA reference)

def taylor_exp(x, n_terms):
    """sum_{k<n_terms} x^k / k!, evaluated by Horner's rule (what the C/CUDA code does)."""
    if n_terms < 1:
        raise ValueError("n_terms must be >= 1")
    x = np.asarray(x, float)
    acc = np.ones_like(x)
    for k in range(n_terms - 1, 0, -1):
        acc = 1 + x * acc / k
    return acc


def exp_neg_naive(x, n_terms=60):
    """e^{-x} straight from the alternating Taylor series: suffers cancellation for large x."""
    return taylor_exp(-np.asarray(x, float), n_terms)


def exp_neg_stable(x, n_terms=60):
    """e^{-x} = 1 / e^{+x}: all Taylor terms positive, no cancellation."""
    return 1.0 / taylor_exp(np.asarray(x, float), n_terms)


def exp_range_reduced(x, n_terms=14):
    """e^x = 2^k e^s with s = x - k ln2, |s| <= ln2/2: few terms, full precision (how libm does it)."""
    x = np.asarray(x, float)
    k = np.round(x / math.log(2))
    return np.ldexp(taylor_exp(x - k * math.log(2), n_terms), k.astype(int))


def terms_needed(x, rel_tol=1e-15, max_terms=200):
    """Smallest number of Taylor terms with |taylor_exp(x) / e^x - 1| < rel_tol (positive x)."""
    for n in range(1, max_terms + 1):
        if abs(float(taylor_exp(x, n)) / math.exp(x) - 1) < rel_tol:
            return n
    return None
