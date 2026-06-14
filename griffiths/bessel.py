"""Bessel functions and cylindrical/spherical separation of variables.

Where Laplace/Helmholtz in cylindrical coordinates leaves you: the radial
factor obeys Bessel's equation, so the modes are J_m, Y_m (and the modified
I_m, K_m). In spherical coordinates the Helmholtz radial factor gives spherical
Bessel j_l, y_l. This module verifies the ODE symbolically (SymPy) and does the
numerics -- zeros, Fourier-Bessel expansion, a cylinder boundary-value problem,
the infinite spherical well, and step-index optical-fibre LP modes -- with
mpmath (no scipy needed). The fibre modes are the repo's own physics: a
step-index core guides J_m inside, K_m outside.
"""

import mpmath as mp
import sympy as sp

_s = sp.Symbol("s", positive=True)


# ── the ODE the modes solve ─────────────────────────────────────────
def bessel_ode_residual(m):
    """Verify J_m(s) solves Bessel's equation s^2 y'' + s y' + (s^2 - m^2) y = 0.
    Returns the simplified residual (0 confirms it)."""
    y = sp.besselj(m, _s)
    expr = _s**2 * sp.diff(y, _s, 2) + _s * sp.diff(y, _s) + (_s**2 - m**2) * y
    return sp.simplify(expr)


def bessel_zeros(m, n):
    """First n positive zeros of J_m (via mpmath)."""
    if n < 1:
        raise ValueError("n must be >= 1")
    return [float(mp.besseljzero(m, k)) for k in range(1, n + 1)]


# ── Fourier-Bessel expansion on a disk ──────────────────────────────
def fourier_bessel_coeffs(f, a, n_terms, order=0):
    """Expand f(s) on [0, a] in J_order(alpha_n s / a), alpha_n = nth zero of J_order.

    c_n = 2/(a^2 J_{order+1}(alpha_n)^2) * int_0^a f(s) J_order(alpha_n s/a) s ds.
    Returns (zeros, coeffs).
    """
    if n_terms < 1:
        raise ValueError("n_terms must be >= 1")
    zeros, coeffs = [], []
    for k in range(1, n_terms + 1):
        al = mp.besseljzero(order, k)
        Jp1 = mp.besselj(order + 1, al)
        num = mp.quad(lambda x: f(x) * mp.besselj(order, al * x / a) * x, [0, a])
        coeffs.append(float(2 / (a**2 * Jp1**2) * num))
        zeros.append(float(al))
    return zeros, coeffs


def fourier_bessel_eval(zeros, coeffs, s, a, order=0):
    """Reconstruct the Fourier-Bessel series at radius s."""
    return float(sum(c * mp.besselj(order, z * s / a) for z, c in zip(zeros, coeffs)))


# ── cylinder boundary-value problem ─────────────────────────────────
def cylinder_cap_potential(V0, a, L, n_terms):
    """Grounded cylinder (radius a, height L): side and bottom at 0, top cap at V0.

    V(s,z) = sum A_n J_0(alpha_n s/a) sinh(alpha_n z/a)/sinh(alpha_n L/a),
    with A_n the Fourier-Bessel coefficients of the constant V0. Returns a
    callable V(s, z).
    """
    zeros, A = fourier_bessel_coeffs(lambda x: V0, a, n_terms, order=0)

    def V(s, z):
        return float(sum(
            A[k] * mp.besselj(0, zeros[k] * s / a)
            * mp.sinh(zeros[k] * z / a) / mp.sinh(zeros[k] * L / a)
            for k in range(len(zeros))))
    return V


# ── spherical Bessel + the infinite spherical well ──────────────────
def spherical_bessel_j(l, x):
    """Spherical Bessel function j_l(x) (symbolic sympy)."""
    return sp.jn(l, x)


def spherical_well_zeros(l, n):
    """First n zeros of j_l = zeros of J_{l+1/2}; sets the infinite-spherical-well
    spectrum E_{nl} = hbar^2 beta_{nl}^2 / (2 m a^2). For l=0 they are n*pi."""
    if n < 1:
        raise ValueError("n must be >= 1")
    return [float(mp.besseljzero(l + sp.Rational(1, 2), k)) for k in range(1, n + 1)]


def spherical_well_energies(l, n, hbar=1.0, mass=1.0, a=1.0):
    """Energies of the first n states of given l in an infinite spherical well."""
    betas = spherical_well_zeros(l, n)
    return [hbar**2 * b**2 / (2 * mass * a**2) for b in betas]


# ── step-index optical fibre LP modes (the repo's physics) ──────────
def fiber_LP01(V):
    """Fundamental LP01 mode of a weakly-guiding step-index fibre, V-number V.

    Core field ~ J_0(u r/a), cladding ~ K_0(w r/a), with u^2 + w^2 = V^2 and the
    characteristic equation  u J_1(u)/J_0(u) = w K_1(w)/K_0(w).  Returns
    (u, w, b) where b = (w/V)^2 is the normalised propagation constant in (0,1).
    """
    if V <= 0:
        raise ValueError("V-number must be > 0")
    V = mp.mpf(V)

    def char(u):
        w = mp.sqrt(V**2 - u**2)
        return (u * mp.besselj(1, u) / mp.besselj(0, u)
                - w * mp.besselk(1, w) / mp.besselk(0, w))

    # LP01 has no cutoff; its u lies in (0, min(V, 2.4048)) below the first J_0 zero,
    # where char(u) changes sign -- bracket it and bisect for a guaranteed real root.
    lo = mp.mpf("1e-4")
    hi = mp.mpf(min(float(V), 2.4048)) - mp.mpf("1e-4")
    u = mp.findroot(char, (lo, hi), solver="bisect")
    w = mp.sqrt(V**2 - u**2)
    b = float((w / V)**2)
    return float(u), float(w), b


def fiber_mode_profile(u, w, a, r):
    """Radial LP01 field at radius r: J_0(u r/a) in the core (r<a), matched
    K_0(w r/a) in the cladding (r>=a)."""
    if r < a:
        return float(mp.besselj(0, u * r / a))
    scale = mp.besselj(0, u) / mp.besselk(0, w)     # continuity at r = a
    return float(scale * mp.besselk(0, w * r / a))


# ── the hanging chain: a continuous pendulum whose modes are J_0 ────
def hanging_chain_frequencies(L, g, n_modes):
    """Transverse normal-mode frequencies of a flexible chain hanging under
    gravity, fixed at the top and free at the bottom (Bernoulli, 1732).

    Tension at height x above the free end is mu*g*x, and the mode equation
    g (x u')' + omega^2 u = 0 is Bessel's equation of order 0. Regularity at the
    bottom keeps J_0; the fixed top u(L)=0 forces J_0(2 omega sqrt(L/g)) = 0, so

        omega_n = (alpha_{0,n} / 2) sqrt(g / L),   J_0(alpha_{0,n}) = 0.

    Returns (frequencies, zeros).
    """
    if L <= 0 or g <= 0:
        raise ValueError("L and g must be > 0")
    if n_modes < 1:
        raise ValueError("n_modes must be >= 1")
    zeros = bessel_zeros(0, n_modes)
    freqs = [(a / 2) * float(mp.sqrt(g / L)) for a in zeros]
    return freqs, zeros


def hanging_chain_modeshape(alpha_n, xfrac):
    """Mode shape u = J_0(alpha_n sqrt(x/L)) vs xfrac = x/L in [0, 1], with x the
    height above the free (bottom) end. xfrac=0 is the free end (max amplitude),
    xfrac=1 the fixed top (zero)."""
    import numpy as _np
    xf = _np.atleast_1d(_np.asarray(xfrac, dtype=float))
    return _np.array([float(mp.besselj(0, alpha_n * _np.sqrt(v))) for v in xf])


# ── FM/PM spectra: the sidebands are Bessel functions ───────────────
def fm_sideband_amplitudes(beta, n_max):
    """Sideband amplitudes of a frequency-/phase-modulated tone.

    cos(w_c t + beta sin(w_m t)) = sum_n J_n(beta) cos((w_c + n w_m) t), so the
    line at the carrier + n*(modulation freq) has amplitude J_n(beta). Returns
    {n: J_n(beta)} for n in [-n_max, n_max] (J_{-n} = (-1)^n J_n).
    """
    if beta < 0 or n_max < 0:
        raise ValueError("beta and n_max must be >= 0")
    return {n: float(mp.besselj(n, beta)) for n in range(-n_max, n_max + 1)}


def carrier_null_indices(n):
    """Modulation indices beta where the FM *carrier* vanishes -- the zeros of
    J_0 (2.405, 5.520, ...). At these beta all power is in the sidebands."""
    return bessel_zeros(0, n)
