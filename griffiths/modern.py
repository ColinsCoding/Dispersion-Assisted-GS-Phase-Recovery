"""Modern physics -- Compton scattering and bremsstrahlung (the photon's two faces).

COMPTON. A photon scattering off a free electron loses energy, and its wavelength
shifts by an amount that depends only on the scattering angle:
    Delta_lambda = lambda' - lambda = lambda_C (1 - cos theta),   lambda_C = h/(m_e c),
the Compton wavelength (~2.426 pm). This is pure relativistic energy-momentum
conservation -- proof that light carries momentum p = h/lambda, i.e. that the photon
is a particle.

BREMSSTRAHLUNG ("braking radiation"). A fast electron decelerating in matter (an
X-ray tube) radiates a continuous spectrum with a sharp SHORT-wavelength cutoff:
    lambda_min = h c /(e V)   (Duane-Hunt),
where the entire kinetic energy eV becomes one photon. The cutoff is the photon
showing its energy quantum -- you cannot make a photon more energetic than the
electron that made it.

SI numbers (numpy). Education.
"""

import numpy as np

_E = 1.602176634e-19           # elementary charge [C]
_H = 6.62607015e-34            # Planck constant [J s]
_C = 299792458.0               # speed of light [m/s]
_ME = 9.1093837015e-31         # electron mass [kg]
_MEC2 = _ME * _C**2            # electron rest energy [J] ~ 8.187e-14 J = 511 keV


# ── Compton scattering ──────────────────────────────────────────────
def compton_wavelength():
    """Electron Compton wavelength lambda_C = h/(m_e c) ~ 2.426 pm -- the length scale
    of the wavelength shift, set by the electron's rest energy."""
    return _H / (_ME * _C)


def compton_shift(theta):
    """Wavelength shift of a photon scattered through angle theta:
    Delta_lambda = lambda_C (1 - cos theta). Zero forward, maximum 2 lambda_C at
    backscatter (theta = pi). Independent of the incident wavelength."""
    return compton_wavelength() * (1 - np.cos(theta))


def compton_scattered_wavelength(lambda_in, theta):
    """Scattered photon wavelength lambda' = lambda + lambda_C (1 - cos theta)."""
    return lambda_in + compton_shift(theta)


def compton_scattered_energy(E_in, theta):
    """Scattered photon energy E' = E / (1 + (E/m_e c^2)(1 - cos theta)) [same units as
    E_in]. From relativistic energy-momentum conservation. Minimum at backscatter
    (the Compton edge)."""
    return E_in / (1 + (E_in / _MEC2) * (1 - np.cos(theta)))


def electron_rest_energy():
    """m_e c^2 ~ 8.187e-14 J = 511 keV (the energy scale that makes Compton matter)."""
    return _MEC2


# ── Bremsstrahlung (the X-ray tube continuum) ───────────────────────
def bremsstrahlung_cutoff_wavelength(voltage):
    """Duane-Hunt short-wavelength cutoff lambda_min = h c /(e V): the whole electron
    kinetic energy eV turned into ONE photon. No radiation below lambda_min."""
    return _H * _C / (_E * voltage)


def bremsstrahlung_max_photon_energy(voltage):
    """Maximum bremsstrahlung photon energy = e V (the electron's kinetic energy) [J]."""
    return _E * voltage


def bremsstrahlung_spectrum(lam, voltage):
    """Kramers' law for the bremsstrahlung intensity vs wavelength:
    I(lambda) ~ (1/lambda^2)(lambda/lambda_min - 1) for lambda >= lambda_min, else 0.
    Reproduces the broad continuum and the sharp short-wavelength edge."""
    lam = np.asarray(lam, float)
    lmin = bremsstrahlung_cutoff_wavelength(voltage)
    return np.where(lam >= lmin, (lam / lmin - 1) / lam**2, 0.0)


if __name__ == "__main__":
    lamC = compton_wavelength()
    print(f"Compton wavelength lambda_C = {lamC*1e12:.4f} pm")
    for deg in (0, 90, 180):
        th = np.radians(deg)
        print(f"  theta={deg:3d}: shift = {compton_shift(th)*1e12:.4f} pm "
              f"= {compton_shift(th)/lamC:.2f} lambda_C")
    # a 17.5 keV photon (Mo K-alpha) backscattering:
    E = 17.5e3 * _E
    print(f"\n17.5 keV photon backscatter -> E' = {compton_scattered_energy(E, np.pi)/_E/1e3:.2f} keV")
    print(f"\nbremsstrahlung at 50 kV: lambda_min = {bremsstrahlung_cutoff_wavelength(50e3)*1e12:.2f} pm,"
          f" max photon = {bremsstrahlung_max_photon_energy(50e3)/_E/1e3:.0f} keV")
