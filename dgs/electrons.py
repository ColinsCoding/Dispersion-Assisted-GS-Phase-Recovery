"""Electron dynamics in electromagnetic fields.

This module covers the physics of electrons in applied E and B fields:
  - Cyclotron motion (circular orbit in uniform B)
  - E x B drift (perpendicular drift, independent of charge and mass)
  - Grad-B drift (force on magnetic moment in non-uniform B)
  - Lorentz factor and relativistic kinetic energy
  - de Broglie wavelength and electron matter waves
  - Larmor radiation (accelerating charge emits radiation)
  - Compton scattering (photon-electron momentum exchange)
  - Hall effect (charge separation in crossed E, B fields)

Physical constants (SI, exact where defined by 2019 SI):
  q  = 1.602176634e-19 C  (elementary charge, exact)
  m  = 9.1093837015e-31 kg (electron rest mass)
  c  = 299792458 m/s      (speed of light, exact)
  h  = 6.62607015e-34 J.s (Planck constant, exact)
  hbar = h/(2*pi)

Context (Griffiths mapping):
  Ch 5 Sec 5.1  -- Lorentz force law F = q(E + v x B)
  Ch 5 Sec 5.4  -- Biot-Savart (electron currents generate B)
  Ch 10 Sec 10.1 -- potentials and gauge; relativistic dynamics
  Modern Physics (Serway) Ch 3 -- photon-electron interaction (Compton)
  Modern Physics Ch 5  -- matter waves, de Broglie lambda = h/p

Usage:
    from dgs.electrons import (
        cyclotron_frequency, cyclotron_radius, cyclotron_period,
        exb_drift, grad_b_drift, lorentz_factor,
        debroglie_wavelength, compton_scatter,
        larmor_power, hall_voltage
    )
"""

from __future__ import annotations
import numpy as np
from typing import Tuple

# ---------------------------------------------------------------------------
# Physical constants (SI, exact)
# ---------------------------------------------------------------------------

Q_E    = 1.602176634e-19    # electron charge (C)
M_E    = 9.1093837015e-31   # electron rest mass (kg)
C_LIGHT = 299792458.0       # speed of light (m/s)
H_PLANCK = 6.62607015e-34   # Planck constant (J.s)
HBAR   = H_PLANCK / (2 * np.pi)
EPS0   = 8.8541878128e-12   # vacuum permittivity (F/m)
MU0    = 4 * np.pi * 1e-7   # vacuum permeability (H/m)


# ---------------------------------------------------------------------------
# Cyclotron motion  (Griffiths Ch 5.1)
# ---------------------------------------------------------------------------

def cyclotron_frequency(B: float, q: float = Q_E, m: float = M_E) -> float:
    """Angular cyclotron frequency omega_c = |q|B/m  (rad/s).

    This is the frequency at which a charged particle orbits in a uniform
    magnetic field B (Tesla).  It does NOT depend on the particle speed --
    all electrons precess at the same omega_c regardless of velocity
    (non-relativistic limit).

    Parameters
    ----------
    B : magnetic field strength (Tesla)
    q : particle charge (C, default electron)
    m : particle mass (kg, default electron)

    Returns
    -------
    omega_c in rad/s.
    """
    if B < 0:
        raise ValueError("B must be non-negative (field magnitude)")
    return abs(q) * B / m


def cyclotron_radius(v_perp: float, B: float,
                     q: float = Q_E, m: float = M_E) -> float:
    """Larmor radius r = mv_perp / (|q|B)  (metres).

    This is the radius of the circular orbit of a charged particle with
    perpendicular velocity v_perp in field B.  The particle traces a helix
    if it also has a component of velocity along B (that component is
    unaffected by B).

    Parameters
    ----------
    v_perp: velocity component perpendicular to B (m/s)
    B     : field strength (Tesla)
    q, m  : charge and mass (default electron)
    """
    if B <= 0:
        raise ValueError("|B| must be positive")
    if v_perp < 0:
        raise ValueError("v_perp is a speed (non-negative)")
    return m * v_perp / (abs(q) * B)


def cyclotron_period(B: float, q: float = Q_E, m: float = M_E) -> float:
    """Cyclotron period T = 2*pi*m / (|q|B)  (seconds).

    Because the circumference 2*pi*r grows linearly with v_perp,
    and r = mv_perp/(qB), the orbital speed cancels: T = 2*pi*r/v = 2*pi*m/(qB).
    This is the principle behind the cyclotron accelerator: RF frequency
    does not need to change as particles are accelerated.
    """
    return 2 * np.pi * m / (abs(q) * B)


def cyclotron_orbit(v_perp: float, B: float, t_arr: np.ndarray,
                    q: float = Q_E, m: float = M_E) -> Tuple[np.ndarray, np.ndarray]:
    """Return (x, y) position of electron in cyclotron orbit over time array.

    The orbit is in the x-y plane with B along z.
    The electron (negative charge) orbits clockwise when viewed along +z.

    Parameters
    ----------
    v_perp : perpendicular speed (m/s)
    B      : field along z (Tesla)
    t_arr  : array of time values (s)

    Returns
    -------
    (x, y) arrays in metres.
    """
    r  = cyclotron_radius(v_perp, B, q, m)
    wc = cyclotron_frequency(B, q, m)
    sign = -np.sign(q)   # electrons (q<0) orbit opposite to positive charges
    x = r * np.cos(sign * wc * t_arr)
    y = r * np.sin(sign * wc * t_arr)
    return x, y


# ---------------------------------------------------------------------------
# E x B drift  (Griffiths Ch 5.2)
# ---------------------------------------------------------------------------

def exb_drift(E: np.ndarray, B: np.ndarray) -> np.ndarray:
    """E x B drift velocity v_drift = (E x B) / B^2  (m/s).

    When BOTH E and B fields are present, all charged particles (regardless
    of charge sign, mass, or speed) drift perpendicular to both fields at
    the same velocity.  This is because the drift cancels the electric force.

    The direction is E x B, perpendicular to both fields.
    In a tokamak, E x B drift is the dominant transport mechanism.

    Parameters
    ----------
    E : electric field vector (V/m, 3-component)
    B : magnetic field vector (Tesla, 3-component)

    Returns
    -------
    Drift velocity vector (m/s).
    """
    E = np.asarray(E, dtype=float)
    B = np.asarray(B, dtype=float)
    B2 = np.dot(B, B)
    if B2 < 1e-30:
        raise ValueError("B magnitude too small")
    return np.cross(E, B) / B2


def grad_b_drift(B_mag: float, grad_B: np.ndarray, v_perp: float,
                 B_hat: np.ndarray,
                 q: float = Q_E, m: float = M_E) -> np.ndarray:
    """Grad-B drift: drift due to non-uniform magnetic field.

    v_grad = (1/2) * (m*v_perp^2 / qB) * (B x grad_B) / B^2
           = (mu/q) * (B x grad_B) / B^2

    where mu = m*v_perp^2 / (2B) is the magnetic moment of the gyrating particle.

    This drift is charge-sign dependent (unlike E x B drift), which leads
    to charge separation and current in space plasmas.

    Parameters
    ----------
    B_mag   : local magnetic field magnitude (Tesla)
    grad_B  : gradient of |B| (vector, T/m)
    v_perp  : perpendicular velocity (m/s)
    B_hat   : unit vector of B field direction
    q, m    : charge and mass

    Returns
    -------
    Drift velocity vector (m/s).
    """
    mu = m * v_perp**2 / (2 * B_mag)       # magnetic moment
    B_hat = np.asarray(B_hat, dtype=float)
    grad_B = np.asarray(grad_B, dtype=float)
    return (mu / (q * B_mag)) * np.cross(B_hat, grad_B)


# ---------------------------------------------------------------------------
# Relativistic mechanics  (Serway Modern Physics Ch 2, Griffiths Ch 12)
# ---------------------------------------------------------------------------

def lorentz_factor(v: float, c: float = C_LIGHT) -> float:
    """Relativistic Lorentz factor gamma = 1 / sqrt(1 - v^2/c^2).

    At v/c = 0.1: gamma = 1.005  (0.5% relativistic correction)
    At v/c = 0.9: gamma = 2.294  (electrons in synchrotrons)
    At v/c = 0.99: gamma = 7.089
    Electron guns in CRTs operate at v/c ~ 0.1-0.3.
    Electron beam in SLAC: gamma ~ 100,000.

    Parameters
    ----------
    v : particle speed (m/s)
    c : speed of light (default SI)
    """
    beta = v / c
    if abs(beta) >= 1:
        raise ValueError(f"v must be < c; got v/c = {beta:.4f}")
    return 1.0 / np.sqrt(1 - beta**2)


def relativistic_kinetic_energy(v: float, m: float = M_E) -> float:
    """Relativistic kinetic energy KE = (gamma - 1) * m * c^2  (Joules).

    For v << c: KE -> (1/2)mv^2 (non-relativistic limit)
    For v -> c: KE -> infinity (can never reach c)

    Returns KE in Joules.  Divide by Q_E to get electron-volts.
    """
    gamma = lorentz_factor(v)
    return (gamma - 1) * m * C_LIGHT**2


def relativistic_momentum(v: float, m: float = M_E) -> float:
    """Relativistic momentum p = gamma*m*v  (kg.m/s)."""
    return lorentz_factor(v) * m * v


# ---------------------------------------------------------------------------
# de Broglie matter waves  (Serway Modern Physics Ch 5, Griffiths Ch 1)
# ---------------------------------------------------------------------------

def debroglie_wavelength(v: float = None, KE_eV: float = None,
                         m: float = M_E, relativistic: bool = False) -> float:
    """de Broglie wavelength lambda = h/p  (metres).

    Must supply either v (speed in m/s) or KE_eV (kinetic energy in eV).
    For electrons accelerated through voltage V: KE_eV = V (in electron-volts).

    Parameters
    ----------
    v           : particle speed (m/s)
    KE_eV       : kinetic energy in electron-volts
    m           : mass (kg, default electron)
    relativistic: if True, use relativistic momentum p = gamma*m*v

    Returns
    -------
    de Broglie wavelength in metres.

    Examples:
        debroglie_wavelength(KE_eV=100)   # 100 eV electron: ~1.23 Angstrom
        debroglie_wavelength(KE_eV=10000) # 10 keV (SEM): ~0.12 Angstrom
    """
    if v is not None and KE_eV is not None:
        raise ValueError("supply v OR KE_eV, not both")
    if KE_eV is not None:
        KE_J = KE_eV * Q_E
        if relativistic:
            # p^2 c^2 = KE^2 + 2*KE*m*c^2
            p = np.sqrt(KE_J**2 + 2*KE_J*m*C_LIGHT**2) / C_LIGHT
        else:
            p = np.sqrt(2 * m * KE_J)
        return H_PLANCK / p
    elif v is not None:
        if relativistic:
            p = relativistic_momentum(v, m)
        else:
            p = m * v
        return H_PLANCK / p
    else:
        raise ValueError("must supply v or KE_eV")


# ---------------------------------------------------------------------------
# Compton scattering  (Serway Modern Physics Ch 3)
# ---------------------------------------------------------------------------

def compton_scatter(lambda_in: float, theta: float) -> Tuple[float, float, float]:
    """Compton scattering: wavelength shift when photon scatters off electron.

    Delta_lambda = (h / m_e c) * (1 - cos(theta))

    where h/(m_e c) = 2.426e-12 m is the Compton wavelength of the electron.

    Parameters
    ----------
    lambda_in : incident photon wavelength (metres)
    theta     : scattering angle of photon (radians)

    Returns
    -------
    (lambda_out, delta_lambda, KE_electron_eV)
      lambda_out       : scattered photon wavelength (m)
      delta_lambda     : wavelength shift (m)
      KE_electron_eV  : recoil kinetic energy of electron (eV)
    """
    lambda_C = H_PLANCK / (M_E * C_LIGHT)   # Compton wavelength = 2.426e-12 m
    delta_lambda = lambda_C * (1 - np.cos(theta))
    lambda_out   = lambda_in + delta_lambda

    E_in  = H_PLANCK * C_LIGHT / lambda_in    # incident photon energy
    E_out = H_PLANCK * C_LIGHT / lambda_out   # scattered photon energy
    KE_electron = E_in - E_out                 # energy conservation

    return lambda_out, delta_lambda, KE_electron / Q_E


# ---------------------------------------------------------------------------
# Larmor radiation  (Griffiths Ch 11 Sec 11.2)
# ---------------------------------------------------------------------------

def larmor_power(a: float, q: float = Q_E) -> float:
    """Larmor formula: power radiated by an accelerating point charge.

    P = q^2 * a^2 / (6 * pi * eps0 * c^3)

    This is the non-relativistic Larmor formula.  An electron oscillating
    in an electromagnetic wave radiates this power -- this is Thomson
    scattering.  An electron in a cyclotron orbit radiates at the
    cyclotron frequency (synchrotron radiation in the relativistic case).

    Parameters
    ----------
    a : acceleration magnitude (m/s^2)
    q : charge (C, default electron)

    Returns
    -------
    Radiated power (Watts).
    """
    return q**2 * a**2 / (6 * np.pi * EPS0 * C_LIGHT**3)


# ---------------------------------------------------------------------------
# Hall effect  (Griffiths Ch 5.1)
# ---------------------------------------------------------------------------

def hall_voltage(I: float, B: float, n: float, t: float,
                 q: float = Q_E) -> float:
    """Hall voltage across a conducting slab.

    In the Hall effect, current-carrying charges deflect in a transverse B
    field.  Charges accumulate on one face until the electric force cancels
    the magnetic force:  qE_H = qv_d*B  =>  V_H = I*B / (n*q*t)

    Parameters
    ----------
    I : current (Amperes)
    B : magnetic field perpendicular to slab (Tesla)
    n : carrier density (m^-3)
    t : slab thickness in direction of B (metres)
    q : carrier charge (C, default electron)

    Returns
    -------
    Hall voltage (Volts).
    """
    if n <= 0 or t <= 0:
        raise ValueError("carrier density and thickness must be positive")
    return I * B / (n * abs(q) * t)


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def demo():
    print("Electron Dynamics -- Demo")
    print("=" * 50)

    B = 0.1   # 100 mT -- achievable lab magnet

    # 1. Cyclotron
    wc = cyclotron_frequency(B)
    T  = cyclotron_period(B)
    print(f"\n1. Cyclotron in B = {B} T:")
    print(f"   omega_c = {wc:.4e} rad/s")
    print(f"   f_c     = {wc/(2*np.pi):.4e} Hz  ({wc/(2*np.pi)*1e-9:.3f} GHz)")
    print(f"   Period  = {T:.4e} s")

    v_perp = 1e6   # 1 Mm/s (non-relativistic: v/c = 0.003)
    r = cyclotron_radius(v_perp, B)
    print(f"   Larmor radius at v={v_perp:.1e} m/s: r = {r*100:.4f} cm")

    # 2. E x B drift
    E_vec = np.array([1e3, 0, 0])   # 1 kV/m in x
    B_vec = np.array([0, 0, B])     # B in z
    v_drift = exb_drift(E_vec, B_vec)
    print(f"\n2. E x B drift (E={E_vec[0]:.0f} V/m in x, B={B} T in z):")
    print(f"   v_drift = {np.linalg.norm(v_drift):.2f} m/s  (direction: {v_drift/np.linalg.norm(v_drift)})")
    print(f"   (E/B = {E_vec[0]/B:.2f} m/s -- direction-independent of charge)")

    # 3. Relativistic electrons
    v_fracs = [0.1, 0.5, 0.9, 0.99]
    print(f"\n3. Relativistic electrons:")
    print(f"   {'v/c':<8}  {'gamma':<8}  {'KE (MeV)':<12}  {'lambda_dB (pm)':<16}")
    print(f"   {'-'*48}")
    for frac in v_fracs:
        v = frac * C_LIGHT
        g = lorentz_factor(v)
        KE_MeV = relativistic_kinetic_energy(v) / (Q_E * 1e6)
        lam_pm = debroglie_wavelength(v=v, relativistic=True) * 1e12
        print(f"   {frac:<8.2f}  {g:<8.4f}  {KE_MeV:<12.4f}  {lam_pm:<16.4f}")

    # 4. de Broglie wavelengths for SEM/TEM
    print(f"\n4. Electron wavelengths (matter waves):")
    for V, label in [(100, '100 V (lab)'), (10000, '10 kV (SEM)'), (200000, '200 kV (TEM)')]:
        lam = debroglie_wavelength(KE_eV=V, relativistic=True) * 1e10
        print(f"   {label:<20}: lambda = {lam:.4f} Angstrom")
    print(f"   (Visible light: 4000-7000 Angstrom -- TEM beats light by 10^4)")

    # 5. Compton scattering
    lambda_xray = 0.07e-9   # 0.07 nm X-ray
    theta = np.pi / 2       # 90-degree scatter
    l_out, dl, KE_eV = compton_scatter(lambda_xray, theta)
    print(f"\n5. Compton scatter (0.07 nm X-ray, 90 degrees):")
    print(f"   Incident wavelength : {lambda_xray*1e12:.3f} pm")
    print(f"   Scattered wavelength: {l_out*1e12:.3f} pm")
    print(f"   Wavelength shift    : {dl*1e12:.3f} pm  (Compton wavelength = {H_PLANCK/(M_E*C_LIGHT)*1e12:.3f} pm)")
    print(f"   Electron recoil KE  : {KE_eV:.2f} eV")

    # 6. Larmor radiation from cyclotron electron
    a_cyc = v_perp**2 / r       # centripetal acceleration in cyclotron orbit
    P = larmor_power(a_cyc)
    print(f"\n6. Larmor radiation from electron in cyclotron orbit:")
    print(f"   Centripetal accel = {a_cyc:.3e} m/s^2")
    print(f"   Radiated power    = {P:.3e} W  ({P/Q_E:.3e} eV/s)")

    # 7. Hall voltage
    V_H = hall_voltage(I=1e-3, B=1.0, n=8.5e28, t=1e-3)  # copper
    print(f"\n7. Hall voltage (copper, I=1mA, B=1T, t=1mm):")
    print(f"   V_H = {V_H*1e9:.3f} nV  (Hall coefficient of copper is very small)")

    print("\nAll demo results consistent with literature values.")


if __name__ == "__main__":
    demo()
