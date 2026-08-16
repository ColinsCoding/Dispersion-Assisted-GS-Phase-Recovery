"""
gs_multiplane.py -- N-plane classical GS and Mie-envelope amplitude
regularization, extending the 2-plane SEALS<->TD-GSA bridge in
seals_to_tdgsa.py per SEALS_TO_TDGSA_REPORT.md Section 6's two recommended
next steps:

  1. More measurement diversity: generalize the 2-plane (I1, D1, I2, D2) loop
     in dgs.gs_core.gs_iteration to N >= 2 dispersion planes -- the general
     fix for underdetermined phase retrieval.
  2. Hybrid physical-model regularization: blend the amplitude-constraint
     step toward the Mie-predicted envelope (from inverse_scattering.py's
     fitted diameter) wherever the native trace intensity is too weak for
     the ordinary |E|=sqrt(I) constraint to carry phase information -- the
     amplitude analog of gs_core's unit_amplitude=True flag, targeted at the
     specific low-SNR mechanism SEALS_TO_TDGSA_REPORT.md Section 5 diagnosed
     (Pearson r=-0.55 between phase error and log-amplitude), not a generic
     tweak.

Reuses dgs.gs_core.disperse/undisperse/apply_amplitude_constraint and its
input-validation helpers directly (the canonical, already-tested primitives)
rather than reimplementing dispersion or the core GS projection step -- only
the loop structure (N planes instead of 2) and the amplitude-prior blend are
new here.
"""
from __future__ import annotations

import sys
import pathlib

import numpy as np

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from dgs import gs_core  # noqa: E402


def _check_planes(Is, Ds):
    if len(Is) != len(Ds):
        raise ValueError(
            f"measurements ({len(Is)}) and dispersions ({len(Ds)}) must have the same length")
    if len(Is) < 2:
        raise ValueError(
            f"N-plane GS needs at least 2 measurement planes for diversity (got {len(Is)}) -- "
            "a single intensity trace cannot determine phase.")
    if len(set(Ds)) != len(Ds):
        raise ValueError(f"dispersions must all be distinct for measurement diversity, got {Ds}")
    for D in Ds:
        gs_core._check_dispersion(D)


def apply_prior_regularized_amplitude(E, amplitude_prior, I_native, floor_frac=0.01):
    """
    Blend |E| toward amplitude_prior wherever the native SEALS trace
    intensity I_native is below floor_frac of its peak, instead of leaving
    the amplitude-constraint step fully unconstrained there.

    This targets the specific mechanism SEALS_TO_TDGSA_REPORT.md Section 5
    diagnosed: apply_amplitude_constraint sets |E|=sqrt(I_measured), which
    carries almost no phase information once I_measured is near zero (any
    phase fits an ~0 amplitude equally well). Borrowing the physically-
    motivated Mie envelope there gives the near-zero-intensity tail of the
    trace something to converge toward, rather than floating freely.
    """
    amplitude_prior = np.asarray(amplitude_prior, dtype=float)
    I_native = np.asarray(I_native, dtype=float)
    if amplitude_prior.shape != I_native.shape:
        raise ValueError(
            f"amplitude_prior shape {amplitude_prior.shape} != I_native shape {I_native.shape}")
    floor = floor_frac * I_native.max()
    weak = I_native < floor
    amp = np.where(weak, amplitude_prior, np.abs(E))
    return amp * np.exp(1j * np.angle(E))


def gs_iteration_n(E, Is, Ds, unit_amplitude=False,
                    amplitude_prior=None, I_native=None, floor_frac=0.01):
    """
    One GS iteration across N dispersion planes: for each (I_j, D_j) pair,
    in order, disperse -> constrain amplitude to sqrt(I_j) -> undisperse.
    Generalizes dgs.gs_core.gs_iteration's 2-plane loop body to N planes,
    reusing its disperse/undisperse/apply_amplitude_constraint primitives
    directly (not reimplemented).

    If amplitude_prior is given (e.g. inverse_scattering.estimate_diameter(
    ...).predicted_fields's |E_p| envelope), blends toward it once per full
    iteration wherever I_native is weak -- see
    apply_prior_regularized_amplitude.
    """
    for I_j, D_j in zip(Is, Ds):
        E_d = gs_core.disperse(E, D_j)
        E_d = gs_core.apply_amplitude_constraint(E_d, I_j)
        E = gs_core.undisperse(E_d, D_j)
        if unit_amplitude:
            E = np.exp(1j * np.angle(E))

    if amplitude_prior is not None:
        E = apply_prior_regularized_amplitude(E, amplitude_prior, I_native, floor_frac)

    return E


def retrieve_phase_n_plane(Is, Ds, n_iter=150, unit_amplitude=False,
                            amplitude_prior=None, I_native=None, floor_frac=0.01):
    """
    N-plane analog of dgs.gs_core.retrieve_phase_with_history. Returns
    (phi, errors, E_history) in the same shape as the 2-plane version, so
    it is a drop-in generalization for the SEALS bridge's comparison code.

    Parameters
    ----------
    Is, Ds : lists of float arrays / dispersions, same length, length >= 2
    n_iter : int -- GS iterations
    unit_amplitude : bool -- see dgs.gs_core.gs_iteration
    amplitude_prior, I_native, floor_frac : see apply_prior_regularized_amplitude;
        amplitude_prior is None by default (off), matching plain N-plane GS.
    """
    _check_planes(Is, Ds)
    n_iter = gs_core._check_n_iter(n_iter)
    Is = [gs_core._check_intensities(I, f'Is[{j}]') for j, I in enumerate(Is)]
    if amplitude_prior is not None and I_native is None:
        raise ValueError(
            "amplitude_prior requires I_native (the native untransformed SEALS trace) "
            "to know where the signal is weak")

    N = min(len(I) for I in Is)
    Is = [I[:N] for I in Is]

    f1_init = np.sqrt(np.maximum(Is[0], 0)).astype(complex)
    E = gs_core.undisperse(f1_init, Ds[0])

    errors = []
    E_history = [E.copy()]
    for _ in range(n_iter):
        E = gs_iteration_n(E, Is, Ds, unit_amplitude=unit_amplitude,
                            amplitude_prior=amplitude_prior, I_native=I_native,
                            floor_frac=floor_frac)
        err = float(np.sqrt(np.mean(
            (np.abs(gs_core.disperse(E, Ds[-1])) ** 2 - Is[-1]) ** 2
        )))
        errors.append(err)
        E_history.append(E.copy())

    return np.angle(E), errors, np.array(E_history)
