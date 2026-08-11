"""
inverse_scattering.py -- SEALS-specific model-based inverse scattering.

MODEL-BASED INVERSE SCATTERING, NOT generic phase retrieval (see
phase_retrieval.py for that). Here, the unknown is a small number of known
PHYSICAL PARAMETERS (currently: particle diameter) rather than an arbitrary
unconstrained complex field -- a much more constrained, better-posed problem.
Because the Mie forward model predicts a complex field, estimating the
particle parameters from intensity measurements permits reconstruction of
the phase predicted by that physical model. This is more constrained than
recovering an arbitrary unknown complex field.

WHY DERIVATIVE-FREE, NOT AUTOGRAD: _seals_physics.mie() (verified identical
to the validated seals_stable.mie()) computes spherical Bessel functions via
scipy.special.spherical_jn/yn called on plain Python floats. It is not a
PyTorch computational graph and has no autograd support. Rather than faking
a gradient or silently swapping in a different, unvalidated differentiable
Mie implementation, this module fits the diameter with
scipy.optimize.minimize_scalar (a bounded, derivative-free 1-D search)
directly against the real, validated Mie physics. Full autograd through Mie
is not implemented here; it would require a separate differentiable
Bessel-function implementation, out of scope for this commit.
"""
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize_scalar

from . import measurement
from .measurement import MieFields


def synthesize_measurement(true_diameter, npar, nmed, lam, theta_rad, r,
                            noise_std=0.05, seed=42):
    """
    known diameter -> validated Mie model -> synthetic intensity -> add noise.

    5% multiplicative Gaussian noise, deterministic seed (reproducible tests).
    """
    fields = measurement.mie_complex_fields(npar, nmed, true_diameter, lam, theta_rad, r)
    I_true = fields.I_p + fields.I_s
    rng = np.random.RandomState(seed)
    noise = rng.normal(0.0, noise_std, size=I_true.shape)
    return np.clip(I_true * (1.0 + noise), 1e-30, None)


def log_intensity_loss(dia, npar, nmed, lam, theta_rad, r, I_meas, eps=1e-30):
    """
    L_d = (1/N) sum_i [log(I_pred,i + eps) - log(I_meas,i + eps)]^2

    Log intensity, not linear intensity: SEALS intensities span a large
    dynamic range (tens of dB, see the forward-model plots). An ordinary
    squared-error loss on linear intensity would be dominated entirely by
    the forward-scattering peak and effectively ignore the weaker side
    lobes, which carry most of the size information (each lobe's angular
    position and depth depends sensitively on diameter).
    """
    fields = measurement.mie_complex_fields(npar, nmed, dia, lam, theta_rad, r)
    I_pred = fields.I_p + fields.I_s
    return float(np.mean((np.log(I_pred + eps) - np.log(I_meas + eps)) ** 2))


@dataclass
class DiameterEstimate:
    diameter: float
    loss: float
    n_evals: int
    predicted_fields: MieFields   # complex field / phase AT the fitted diameter


def estimate_diameter(I_meas, npar, nmed, lam, theta_rad, r, bounds) -> DiameterEstimate:
    """
    Recover particle diameter from a measured intensity spectrum via a
    bounded, derivative-free 1-D search (scipy.optimize.minimize_scalar,
    method='bounded') against the validated NumPy/SciPy Mie model.

    Returns the fitted diameter AND the complex Mie field it predicts --
    the "connect inverse scattering to phase" step: because Mie predicts a
    complex field, once the physical parameter is fitted, its phase comes
    for free from the (already-validated) forward model. This is why the
    result is more constrained than generic phase retrieval: the phase here
    is not itself an optimization variable, it is read off a physical model
    that has already been validated against the intensity data.
    """
    def objective(dia):
        return log_intensity_loss(dia, npar, nmed, lam, theta_rad, r, I_meas)

    # scipy's default xatol=1e-5 for method='bounded' is in the SAME units as `dia`
    # (meters, SI throughout this package) -- for a micron-scale particle, the entire
    # search bracket (bounds[1]-bounds[0]) is typically only ~1e-6 m wide, i.e. NARROWER
    # than that default tolerance, so minimize_scalar would report "converged" after a
    # single evaluation without actually searching. xatol is set explicitly, well below
    # the bracket width, to avoid this silent-unit-mismatch trap.
    bracket_width = bounds[1] - bounds[0]
    xatol = min(1e-10, bracket_width * 1e-4)
    result = minimize_scalar(objective, bounds=bounds, method='bounded',
                              options={'xatol': xatol})
    fields = measurement.mie_complex_fields(npar, nmed, result.x, lam, theta_rad, r)
    return DiameterEstimate(diameter=float(result.x), loss=float(result.fun),
                             n_evals=int(result.nfev), predicted_fields=fields)
