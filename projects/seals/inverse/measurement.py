"""
measurement.py -- the explicit forward-physics-to-detector boundary.

A real detector measures intensity, I = |E|^2. It does not measure phase.
This module makes that boundary an explicit function call rather than
something implicit in how a plot is drawn, and provides a MieFields helper
that reconstructs Mie's complex far fields from the validated
(I_p, I_s, T_p, T_s) outputs -- without duplicating the Bessel-function
internals in _seals_physics.mie().

Rayleigh-Debye-Gans (rayleigh_debye()) returns intensity only, in both the
original MATLAB and this port -- there is no complex field to reconstruct
for it, and this module does not invent one.
"""
from dataclasses import dataclass

import numpy as np
import torch

from . import _seals_physics as physics


def intensity_measurement(E):
    """
    The detector model: I = |E|^2.

    Works for numpy arrays (real code path) and torch tensors (differentiable
    code path used throughout phase_retrieval.py) via the same expression.
    """
    if isinstance(E, torch.Tensor):
        return E.abs() ** 2
    return np.abs(E) ** 2


@dataclass
class MieFields:
    """Complex far fields and everything derived from them, for one Mie call."""
    E_p: np.ndarray       # complex, p-polarization
    E_s: np.ndarray       # complex, s-polarization
    I_p: np.ndarray
    I_s: np.ndarray
    phase_p: np.ndarray   # = np.angle(E_p), radians
    phase_s: np.ndarray
    sigma_s: float
    an: np.ndarray
    bn: np.ndarray


def mie_complex_fields(npar, nmed, dia, lam, angles_rad, r) -> MieFields:
    """
    Call the validated Mie model and reconstruct its complex far fields.

    _seals_physics.mie() (== seals_stable.mie(), verified identical) already
    computes E_theta, E_phi internally, then discards them in favor of
    returning I_p=|E_theta|^2, I_s=|E_phi|^2, T_p=angle(E_theta), T_s=angle(E_phi)
    -- exactly matching mie-2.m's own T_p/T_s = angle(...) computation. Since
    E = sqrt(I) * exp(i*phase) reconstructs a complex number exactly from its
    modulus and argument, this recovers E_p/E_s without touching
    _seals_physics.py's internals or changing any validated numerical value.
    """
    sigma_s, I_p, I_s, an, bn, T_p, T_s = physics.mie(npar, nmed, dia, lam, angles_rad, r)
    E_p = np.sqrt(I_p) * np.exp(1j * T_p)
    E_s = np.sqrt(I_s) * np.exp(1j * T_s)
    return MieFields(E_p=E_p, E_s=E_s, I_p=I_p, I_s=I_s, phase_p=T_p, phase_s=T_s,
                      sigma_s=sigma_s, an=an, bn=bn)
