"""Atomic physics -- the normal Zeeman effect and the Bohr magneton.

An external magnetic field B splits an atomic energy level into 2l+1 equally spaced
sublevels. The orbital magnetic moment mu = -(mu_B/hbar) L has an orientation energy
    U = -mu . B = m_l * mu_B * B ,   m_l = -l, ..., +l,
so the level fans out into evenly spaced lines. Classically this is the torque
tau = mu x B making the moment precess (Larmor); quantum-mechanically m_l is
quantized, so the splitting is discrete.

The NORMAL Zeeman effect: although a level has 2l+1 sublevels, the selection rule
Delta m_l in {-1, 0, +1} collapses every transition to just THREE observed spectral
lines (the Lorentz triplet), spaced by the Larmor frequency
    Delta_nu = mu_B B / h .
The Bohr magneton mu_B = e hbar /(2 m_e) is the natural unit of atomic magnetism.
SI numbers (numpy) + symbolic shifts (sympy). Education.
"""

import numpy as np
import sympy as sp

_E = 1.602176634e-19           # elementary charge [C]
_HBAR = 1.054571817e-34        # reduced Planck constant [J s]
_H = 2 * np.pi * _HBAR         # Planck constant [J s]
_ME = 9.1093837015e-31         # electron mass [kg]

MU_B = _E * _HBAR / (2 * _ME)  # Bohr magneton ~ 9.274e-24 J/T


def bohr_magneton():
    """Bohr magneton mu_B = e hbar /(2 m_e) ~ 9.274e-24 J/T -- one quantum of orbital
    magnetic moment, the natural scale of atomic magnetism."""
    return MU_B


def zeeman_energy_shift(m_l, B):
    """Energy shift of an orbital sublevel in field B: Delta_E = m_l mu_B B [J]."""
    return m_l * MU_B * B


def zeeman_sublevels(l, B):
    """The 2l+1 equally spaced energy shifts m_l mu_B B for m_l = -l..+l [J] -- the
    level fanning out symmetrically about the unshifted energy."""
    return np.array([zeeman_energy_shift(m, B) for m in range(-l, l + 1)])


def level_spacing(B):
    """Energy gap between adjacent Zeeman sublevels: mu_B B [J] (the same for all)."""
    return MU_B * B


def larmor_frequency(B):
    """Normal-Zeeman line spacing Delta_nu = mu_B B / h [Hz] (the Larmor frequency).
    The three observed lines sit at nu0 - Delta_nu, nu0, nu0 + Delta_nu."""
    return MU_B * B / _H


def normal_zeeman_lines(nu0, B):
    """The three frequencies of the normal Zeeman triplet about a line at nu0 [Hz]."""
    d = larmor_frequency(B)
    return np.array([nu0 - d, nu0, nu0 + d])


def zeeman_shift_symbolic(l):
    """Symbolic energy shifts m_l*mu_B*B for every sublevel m_l = -l..+l (for SymPy
    display). Returns a list of (m_l, expression) using symbols mu_B and B."""
    mu_B, B = sp.symbols("mu_B B", positive=True)
    return [(m, m * mu_B * B) for m in range(-l, l + 1)]


if __name__ == "__main__":
    print(f"Bohr magneton mu_B = {MU_B:.4e} J/T")
    B = 1.0
    print(f"\nat B = {B} T:")
    print(f"  level spacing mu_B B = {level_spacing(B):.4e} J = {level_spacing(B)/_E*1e6:.3f} ueV")
    print(f"  Larmor (line) spacing Delta_nu = mu_B B/h = {larmor_frequency(B)/1e9:.3f} GHz")
    print(f"  normal Zeeman triplet about 500 THz: "
          f"{np.round(normal_zeeman_lines(500e12, B)/1e12, 5)} THz")
    print(f"\n2l+1 sublevels for l=2 (shifts / mu_B B):",
          (zeeman_sublevels(2, B) / level_spacing(B)).astype(int))
