"""Kronig-Penney model -- where atomic levels become energy BANDS (and gaps).

NAME
    kronig_penney -- 1-D band structure of an electron in a periodic potential.

SYNOPSIS
    f = kp_rhs(z, P); mask = allowed(z, P); K, E = dispersion(P)

DESCRIPTION
    Put an electron in a periodic comb of delta potentials V(x)=alpha*sum delta(x-ja)
    (period a). Bloch's theorem + the Schrodinger equation give a single condition
    linking the Bloch wavevector K to the energy E:

        cos(K a) = f(z) = cos(z) + P * sin(z)/z,
        z = sqrt(2 m E) a / hbar   (dimensionless energy),
        P = m alpha a / hbar^2     (dimensionless barrier strength).

    Because the left side is a cosine, a real K exists ONLY where |f(z)| <= 1. Those
    z-intervals are the ALLOWED bands; where |f(z)| > 1 there is no propagating state
    -- a forbidden BAND GAP. So a continuous free-electron spectrum, once the electron
    feels a periodic lattice, breaks into bands separated by gaps. That gap is the
    whole of semiconductor physics: fill a band and leave a gap to the next, and you
    have an insulator/semiconductor; leave a band half-full and you have a metal.

LIMITS
    P -> 0 : f = cos(z), |f|<=1 everywhere -> no gaps, the free electron (E ~ K^2).
    P large: bands shrink to narrow, nearly flat levels -> the tight-binding / atomic
             limit (Bohr's discrete levels, recovered from the other side).

NumPy only. Education -- the bridge from quantum atoms to transistors.
"""

import numpy as np


def kp_rhs(z, P):
    """The Kronig-Penney band function f(z) = cos(z) + P*sin(z)/z.

    z : dimensionless energy sqrt(2mE)a/hbar (array or scalar, z >= 0).
    P : dimensionless barrier strength m*alpha*a/hbar^2 (>= 0).
    The allowed energies are exactly where |f(z)| <= 1. sin(z)/z is taken as 1 at z=0."""
    z = np.asarray(z, float)
    return np.cos(z) + P * np.sinc(z / np.pi)        # np.sinc(z/pi) = sin(z)/z


def allowed(z, P):
    """Boolean mask: True where a real Bloch wavevector exists, i.e. |f(z)| <= 1.
    True = inside an allowed band; False = inside a forbidden gap."""
    return np.abs(kp_rhs(z, P)) <= 1.0


def allowed_fraction(P, z_max=30.0, n=20000):
    """Fraction of the energy axis (0..z_max) that lies in allowed bands. Falls from
    1 (free electron, P=0) toward 0 as the lattice gets stronger -- gaps widen."""
    z = np.linspace(1e-6, z_max, n)
    return float(np.mean(allowed(z, P)))


def band_edges(P, z_max=30.0, n=200000):
    """Energies (as z) where |f(z)| = 1 -- the edges between allowed bands and gaps.
    Found as sign changes of |f(z)| - 1. Returns a sorted array of edge z-values."""
    z = np.linspace(1e-6, z_max, n)
    g = np.abs(kp_rhs(z, P)) - 1.0
    idx = np.where(np.diff(np.sign(g)) != 0)[0]
    # linear-interpolate each crossing for a sharper edge
    edges = z[idx] - g[idx] * (z[idx + 1] - z[idx]) / (g[idx + 1] - g[idx])
    return edges


def dispersion(P, z_max=12.0, n=6000):
    """Band structure E(K). For each allowed z, the Bloch phase is K a = arccos(f(z))
    in [0, pi]. Returns (Ka, E_dimensionless) with E ~ z^2; masked (NaN) inside gaps,
    so plotting Ka vs E shows the bands separated by the forbidden gaps."""
    z = np.linspace(1e-6, z_max, n)
    f = kp_rhs(z, P)
    Ka = np.where(np.abs(f) <= 1.0, np.arccos(np.clip(f, -1, 1)), np.nan)
    E = z**2                                          # E proportional to z^2
    return Ka, E


if __name__ == "__main__":
    for P in (0.0, 1.0, 5.0):
        frac = allowed_fraction(P)
        edges = band_edges(P)[:4]
        print(f"P={P:4.1f}: allowed fraction = {frac:.3f}, first band edges z = {np.round(edges,3)}")
    # the first gap (just above z = pi) for P=2
    z = np.linspace(np.pi, np.pi + 1.5, 500)
    print("\nfirst forbidden gap (P=2) spans z in",
          f"[{z[~allowed(z,2.0)][0]:.3f}, {z[~allowed(z,2.0)][-1]:.3f}]  (|f|>1 there)")
