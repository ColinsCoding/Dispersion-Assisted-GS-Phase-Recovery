"""SEALS: spectrally encoded angular light scattering, ported from the lab MATLAB.

In plain words: a broadband laser hits a particle, and a pair of gratings sends
each scattering angle to a different wavelength. So one spectrum is really an
angular scattering pattern. This module has the three pieces of that model:

  seals()           wavelength -> beam displacement -> scattering angle   (SEALS.m)
  mie()             exact sphere scattering, intensity and phase           (mie-2.m)
  rayleigh_debye()  approximate scattering for weakly scattering spheres   (rayleighdebye.m)

The original MATLAB is in seals/matlab/. tests/test_seals_vs_matlab.py runs the
same default case through both and checks they agree.

One deliberate difference: rayleigh_debye() here returns (1 + cos^2 theta) * P(u)^2.
The MATLAB returns |P(u) * (1 + cos^2 theta)|, with the form factor not squared.
Against exact Mie theory in the regime where Rayleigh-Debye-Gans is supposed to
hold (|m - 1| << 1), f * P^2 matches and the MATLAB form does not. The test file
checks this too.

A second thing to know: mie() follows mie-2.m, whose I_p + I_s is twice the
unpolarized intensity (see the mie() docstring).

Units: SI throughout (meters, radians unless a name says deg).
"""
import numpy as np
from scipy.special import spherical_jn, spherical_yn

# Default case from the lab's main.m
DEFAULTS = dict(
    dia=9940e-9, npar=1.39, nmed=1.0, r=0.10,
    d=9.0909e-7, D=0.065, a=0.9023, dcorr=-4.2448e-4, P=0.0058, NA=0.70,
    mangle=20.0, lam1=1580e-9, lam2=1600e-9, n_lam=500,
)


# --- SEALS geometry -------------------------------------------------------

def seals(d, D, a, dcorr, P, NA, lamvec):
    """Wavelengths -> (displacement y [m], scattering angle [deg], valid mask).

    Same equations as SEALS.m. The angle does not include the measurement
    offset `mangle`; main.m adds that afterwards.
    """
    lamvec = np.asarray(lamvec, dtype=float)
    arg = lamvec / d - np.sin(a)
    valid = np.abs(arg) < 1.0
    delta = a - np.arcsin(np.clip(arg, -0.9999, 0.9999))
    t = np.tan(delta)
    y = (D / 6.0) * t / (1.0 + t * np.tan(a))
    y = y - y[-1]
    ycenter = (y[0] - y[-1]) / 2.0
    theta = np.degrees(np.arctan((2.0 / P) * (y - ycenter + dcorr) * np.tan(np.arcsin(NA))))
    return y, theta, valid


# --- Rayleigh-Debye-Gans --------------------------------------------------

def form_factor_P(u):
    """Sphere form factor 3(sin u - u cos u)/u^3, with a series near u = 0."""
    u = np.asarray(u, dtype=float)
    out = np.empty_like(u)
    small = np.abs(u) < 0.1
    u2 = u[small] ** 2
    out[small] = 1.0 - u2 / 10.0 + u2 ** 2 / 280.0 - u2 ** 3 / 15120.0
    ul = u[~small]
    out[~small] = 3.0 * (np.sin(ul) - ul * np.cos(ul)) / ul ** 3
    return out


def rayleigh_debye(dia, lam, n_bg, n_sp, theta, R):
    """Unpolarized RDG intensity for unit incident intensity at distance R.

    I = k^4 a^6 |(m^2-1)/(m^2+2)|^2 (1 + cos^2 theta) P(u)^2 / (2 R^2),
    with k = 2 pi n_bg / lam and u = 2 k a sin(theta/2).
    """
    a = dia / 2.0
    k = 2 * np.pi * n_bg / lam
    m = n_sp / n_bg
    u = 2 * k * a * np.sin(np.asarray(theta) / 2.0)
    pol = abs((m ** 2 - 1) / (m ** 2 + 2)) ** 2
    return k ** 4 * a ** 6 * pol * (1 + np.cos(theta) ** 2) * form_factor_P(u) ** 2 / (2 * R ** 2)


def rayleigh_debye_matlab(dia, lam, n_bg, n_sp, theta, R):
    """The lab MATLAB formula, kept only so the tests can show the difference."""
    a = dia / 2.0
    k = 2 * np.pi * n_bg / lam
    m = n_sp / n_bg
    u = 2 * k * a * np.sin(np.asarray(theta) / 2.0)
    Rr = form_factor_P(u) * (1 + np.cos(theta) ** 2)
    return np.abs(Rr * abs((m ** 2 - 1) / (m ** 2 + 2)) ** 2 / (2 * R ** 2) * (2 * np.pi / lam) ** 4 * a ** 6)


# --- Mie --------------------------------------------------------------------

def _bessel_sets(n, x):
    jx = spherical_jn(n.astype(int), x)
    yx = spherical_yn(n.astype(int), x)
    j1 = np.concatenate(([np.sin(x) / x], jx[:-1]))
    y1 = np.concatenate(([-np.cos(x) / x], yx[:-1]))
    return jx, yx, j1, y1


def mie_coefficients(npar, nmed, dia, lam):
    """Mie a_n, b_n. Returns (n, a_n, b_n, size parameter x)."""
    x = np.pi * dia / (lam / nmed)
    m = npar / nmed
    nmax = max(int(round(2 + x + 4 * x ** (1 / 3))), 5)
    n = np.arange(1, nmax + 1, dtype=float)
    z = m * x
    jx, yx, j1x, y1x = _bessel_sets(n, x)
    jz, _, j1z, _ = _bessel_sets(n, z)
    hx, h1x = jx + 1j * yx, j1x + 1j * y1x
    ax, az, ahx = x * j1x - n * jx, z * j1z - n * jz, x * h1x - n * hx
    an = (m * m * jz * ax - jx * az) / (m * m * jz * ahx - hx * az)
    bn = (jz * ax - jx * az) / (jz * ahx - hx * az)
    return n, an, bn, x


def _pi_tau(nmax, u):
    pi = np.zeros((nmax, u.size))
    tau = np.zeros((nmax, u.size))
    pi[0], tau[0] = 1.0, u
    if nmax > 1:
        pi[1], tau[1] = 3 * u, 6 * u ** 2 - 3
    for j in range(2, nmax):
        k = j + 1
        pi[j] = ((2 * k - 1) / (k - 1)) * u * pi[j - 1] - (k / (k - 1)) * pi[j - 2]
        tau[j] = k * u * pi[j] - (k + 1) * pi[j - 1]
    return pi, tau


def mie(npar, nmed, dia, lam, theta, r):
    """Exact Mie scattering, same outputs and order as mie-2.m:
    (sigma_s, I_p, I_s, a_n, b_n, phase_p, phase_s).

    Kept identical to the MATLAB on purpose, including one quirk: both field
    components are multiplied by cos(phi) (the textbook has cos for one and
    sin for the other). With phi = pi that makes I_p + I_s equal to TWICE the
    unpolarized intensity. Use (I_p + I_s) / 2 when comparing to Rayleigh or RDG.
    """
    theta = np.asarray(theta, dtype=float)
    n, an, bn, x = mie_coefficients(npar, nmed, dia, lam)
    k = 2 * np.pi * nmed / lam
    pi, tau = _pi_tau(n.size, np.cos(theta))
    w = ((2 * n + 1) / (n * (n + 1)))[:, None]
    S1 = np.sum(an[:, None] * w * pi + bn[:, None] * w * tau, axis=0)
    S2 = np.sum(an[:, None] * w * tau + bn[:, None] * w * pi, axis=0)
    phase = np.exp(1j * k * r)
    E_par = phase / (-1j * k * r) * np.cos(np.pi) * S2
    E_perp = phase / (1j * k * r) * np.cos(np.pi) * S1
    q_sca = 2 * np.sum((2 * n + 1) * (np.abs(an) ** 2 + np.abs(bn) ** 2)) / x ** 2
    sigma_s = q_sca * np.pi * (dia / 2) ** 2
    return sigma_s, np.abs(E_par) ** 2, np.abs(E_perp) ** 2, an, bn, np.angle(E_par), np.angle(E_perp)
