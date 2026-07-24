"""retinal_scan_imaging.py -- retinal biometric/clinical imaging physics.

Three genuinely different pieces, kept honestly separate rather than
conflated into one narrative:

1. THE EYE AS A PARAXIAL SYSTEM (real, textbook optics).
   The "reduced eye" (Emsley's schematic eye): one refracting surface,
   modeled with dgs/paraxial_optics_abcd.py's spherical_interface_matrix +
   free_space_matrix -- the same ABCD formalism already in this repo, applied
   to a new system. Gives the diffraction-limited spot size on the retina.

2. ULTRAFAST LINE-SCAN, EXTENDING THE JALALI LAB'S STEAM (PROPOSED, not a
   documented existing retinal-imaging product). dgs/steam_imaging.py's
   time_stretch_pulse (H(f)=exp(i*pi*D*f^2), the exact operator this whole
   repo is built around) applied to a synthetic retinal-vessel reflectance
   line -- same honesty posture as dgs/sbir_portfolio.py's P9: a plausible
   combination of a real technique (STEAM) and a real need (fast retinal
   biometrics/vessel imaging), not a claim that this has been built or
   published.

3. TWO DIFFERENT PHASE-RETRIEVAL FAMILIES, NOT ONE (this is where "X-ray"
   comes in, and where it's easy to overclaim). dgs/gs_core.py's dispersion-
   DIVERSITY GS (two different H(f) dispersions) and the SUPPORT-constraint
   GS/Fienup algorithm this module adds (support_constraint_gs) are BOTH
   alternating-projection phase retrieval descended from the same 1972
   Gerchberg-Saxton paper, but they are not the same algorithm and this
   module does not claim they are. The support-constraint flavor (single
   diffraction-plane magnitude + a real-space support) is the one X-ray/
   electron coherent diffractive imaging (CDI) actually uses; the dispersion-
   diversity flavor is what this repo's dgs/gs_core.py uses for the optical-
   communications / STEAM problem. Verified citations:
     - R. W. Gerchberg, W. O. Saxton, "A practical algorithm for the
       determination of phase from image and diffraction plane pictures,"
       Optik 35, 237-246 (1972) -- the shared ancestor of both flavors.
     - J. R. Fienup, "Reconstruction of an object from the modulus of its
       Fourier transform," Opt. Lett. 3, 27-29 (1978) -- error-reduction /
       hybrid input-output, the support-constraint flavor this module adds.
     - J. Miao, P. Charalambous, J. Kirz, D. Sayre, "Extending the
       methodology of X-ray crystallography to allow imaging of micrometre-
       sized non-crystalline specimens," Nature 400, 342-344 (1999) -- the
       seminal X-ray CDI demonstration using that support-constraint flavor.

Photon-budget (shot-noise) numbers below are generic Poisson-statistics
scaling (SNR ~ sqrt(N)), not a specific laser-safety exposure limit -- ANSI
Z136.1 eye-safe exposure limits are wavelength- and duration-dependent and
would need to be looked up per configuration before this became a real
imaging-dose claim, not assumed here.
"""

from __future__ import annotations
import numpy as np
from typing import Dict

from dgs.paraxial_optics_abcd import (
    free_space_matrix, spherical_interface_matrix, compose_system,
)
from dgs.steam_imaging import time_stretch_pulse
from dgs.gs_core import retrieve_phase


# ── 1. The eye as a paraxial system ───────────────────────────────────────────

def reduced_eye_matrix(R_cornea_mm: float = 5.55, n_vitreous: float = 1.336,
                        axial_length_mm: float = 22.3):
    """ABCD matrix of Emsley's "reduced eye": one refracting surface
    (air -> vitreous, index n_vitreous) of radius R_cornea_mm, then free
    propagation axial_length_mm to the retina. Standard schematic-eye
    numbers (R=5.55mm, n=1.336, length=22.3mm) give an eye focal length of
    ~22.1mm, close to the textbook ~60 diopters total power -- checked
    below, not assumed."""
    if R_cornea_mm <= 0 or axial_length_mm <= 0:
        raise ValueError("R_cornea_mm and axial_length_mm must be positive")
    if n_vitreous <= 1.0:
        raise ValueError(f"n_vitreous={n_vitreous}: must exceed air's index 1.0")
    M_refract = spherical_interface_matrix(1.0, n_vitreous, R_cornea_mm)
    M_propagate = free_space_matrix(axial_length_mm)
    return compose_system(M_refract, M_propagate)


def eye_focal_length_mm(R_cornea_mm: float = 5.55, n_vitreous: float = 1.336) -> float:
    """f = n*R/(n-1) for a single refracting surface -- the image-space
    (posterior nodal) distance, ~22.1mm for the textbook numbers, close to
    the eye's real axial length."""
    return n_vitreous * R_cornea_mm / (n_vitreous - 1.0)


def eye_power_diopters(R_cornea_mm: float = 5.55, n_vitreous: float = 1.336) -> float:
    """Refractive power P=(n2-n1)/R in diopters (R in meters) -- ~60D for
    the textbook reduced-eye numbers, the commonly quoted total power of
    the human eye. NOT 1/eye_focal_length_mm(): that focal length is
    measured in the vitreous (index n_vitreous), and dividing it into 1
    silently assumes an image-space index of 1 (air), undercounting power
    by a factor of n_vitreous -- this function uses the direct refracting-
    surface power formula instead, avoiding that mistake."""
    return (n_vitreous - 1.0) / (R_cornea_mm / 1000.0)


def diffraction_limited_spot_radius_um(pupil_diameter_mm: float,
                                        wavelength_nm: float = 550.0,
                                        eye_focal_length_mm: float = 22.3) -> float:
    """Airy-disk radius on the retina: r = 1.22*lambda*f/D (small-angle,
    paraxial diffraction limit -- the same physics as any imaging aperture,
    applied to the eye's own pupil and focal length).

    Bounds: pupil_diameter_mm and wavelength_nm must be positive.
    """
    if pupil_diameter_mm <= 0:
        raise ValueError(f"pupil_diameter_mm={pupil_diameter_mm}: must be positive")
    if wavelength_nm <= 0:
        raise ValueError(f"wavelength_nm={wavelength_nm}: must be positive")
    lam_mm = wavelength_nm * 1e-6
    r_mm = 1.22 * lam_mm * eye_focal_length_mm / pupil_diameter_mm
    return r_mm * 1000.0  # mm -> um


# ── 2. Proposed ultrafast retinal line-scan (STEAM extension) ───────────────

def steam_retinal_linescan(reflectance_profile: np.ndarray, D_ps2: float,
                            lambda0_nm: float = 1550.0) -> Dict:
    """Time-stretch a synthetic retinal-vessel reflectance line through the
    SAME H(f)=exp(i*pi*D*f^2) operator as dgs/steam_imaging.py's flow-
    cytometry use case -- a proposed application (fast vessel/biometric
    line-scan), not a documented existing instrument. See module docstring.
    """
    reflectance_profile = np.asarray(reflectance_profile, dtype=complex)
    n = len(reflectance_profile)
    if n < 8:
        raise ValueError(f"n={n}: need at least 8 samples for a meaningful line-scan")
    f = np.fft.fftfreq(n)
    result = time_stretch_pulse(reflectance_profile, f, D_ps2, lambda0_nm=lambda0_nm)
    result["reflectance_profile"] = reflectance_profile
    return result


def synthetic_vessel_reflectance(n: int = 256, n_vessels: int = 4,
                                  rng_seed: int = 0) -> np.ndarray:
    """Toy retinal-vessel reflectance line: a baseline tissue reflectance
    with a few narrow, darker Gaussian dips (vessels absorb/scatter more
    than surrounding tissue) -- a stand-in test signal, not real fundus
    photography data."""
    if n < 8:
        raise ValueError(f"n={n}: need at least 8 samples")
    rng = np.random.default_rng(rng_seed)
    x = np.arange(n)
    profile = np.ones(n)
    centers = rng.uniform(0.1, 0.9, n_vessels) * n
    widths = rng.uniform(2.0, 6.0, n_vessels)
    depths = rng.uniform(0.3, 0.6, n_vessels)
    for c, w, d in zip(centers, widths, depths):
        profile -= d * np.exp(-0.5 * ((x - c) / w) ** 2)
    return np.sqrt(np.maximum(profile, 1e-6)).astype(complex)  # field amplitude


def synthetic_vessel_reflectance_with_depth(n: int = 256, n_vessels: int = 4,
                                             depth_amplitude_rad: float = 1.5,
                                             rng_seed: int = 0) -> np.ndarray:
    """Same reflectance-amplitude profile as synthetic_vessel_reflectance,
    but with a nonzero synthetic PHASE encoding smoothly-varying layer depth
    (e.g. nerve-fiber-layer thickness) -- a real target for
    retinal_depth_phase_recovery(), unlike a pure-real profile whose phase
    is trivially zero everywhere."""
    amplitude = np.abs(synthetic_vessel_reflectance(n, n_vessels, rng_seed))
    x = np.arange(n) / n
    rng = np.random.default_rng(rng_seed + 1000)
    k1, k2 = rng.uniform(1, 3, 2)
    depth_phase = depth_amplitude_rad * (np.sin(2 * np.pi * k1 * x) +
                                          0.5 * np.sin(2 * np.pi * k2 * x + 1.0))
    return (amplitude * np.exp(1j * depth_phase)).astype(complex)


# ── 3a. Dispersion-diversity GS applied to retinal depth (OCT-like) ─────────

def retinal_depth_phase_recovery(reflectance_profile: np.ndarray,
                                  D1: float = -5000.0, D2: float = -5750.0,
                                  n_iter: int = 50) -> Dict:
    """Apply dgs/gs_core.py's dispersion-diversity GS (the SAME kernel used
    throughout this repo for optical comms/STEAM) to two dispersed
    measurements of a retinal reflectance line, to recover a depth-coding
    phase -- structurally the same inverse problem as OCT-style depth
    sensing, PROPOSED here, not a claim this matches a real OCT system's
    physics (real OCT uses interferometric path-length matching, a
    different mechanism than dispersion-diversity GS).

    NOTE on the returned 'errors': with unit_amplitude=False, gs_iteration's
    LAST internal step already forces disperse(E,D2)'s amplitude to exactly
    sqrt(I2) (no unit-amplitude re-projection afterward to perturb it away,
    unlike the unit_amplitude=True case) -- so this diagnostic sits near
    machine epsilon from the first iteration and is not a meaningful
    convergence signal here. Judge convergence by the recovered phase
    itself (phi_est vs. phi_true, up to the unavoidable global-phase
    offset), not by 'errors'.
    """
    from dgs.gs_core import disperse
    E_true = np.asarray(reflectance_profile, dtype=complex)
    I1 = np.abs(disperse(E_true, D1)) ** 2
    I2 = np.abs(disperse(E_true, D2)) ** 2
    phi_est, errors = retrieve_phase(I1, I2, D1, D2, n_iter=n_iter, unit_amplitude=False)
    return {"phi_est": phi_est, "errors": errors, "I1": I1, "I2": I2,
            "phi_true": np.angle(E_true)}


# ── 3b. Support-constraint GS / Fienup -- the real X-ray CDI algorithm ──────

def support_constraint_gs(magnitude: np.ndarray, support_mask: np.ndarray,
                           n_iter: int = 100, rng_seed: int = 0) -> Dict:
    """Classic Gerchberg-Saxton/Fienup error-reduction phase retrieval:
    alternate between (a) enforcing the MEASURED Fourier-magnitude (the
    diffraction pattern) and (b) enforcing a real-space SUPPORT constraint
    (object is known to be zero outside some region) -- the actual
    algorithm family behind X-ray/electron coherent diffractive imaging
    (Miao et al. 1999), distinct from dgs/gs_core.py's dispersion-diversity
    variant (see module docstring).

    Parameters
    ----------
    magnitude     : measured Fourier-magnitude |F[object]| (the "diffraction
                    pattern"), real array, must be non-negative
    support_mask  : boolean array, True where the object is allowed to be
                    nonzero, same length as magnitude
    n_iter        : number of error-reduction iterations
    """
    magnitude = np.asarray(magnitude, dtype=float)
    support_mask = np.asarray(support_mask, dtype=bool)
    if magnitude.ndim != 1 or support_mask.shape != magnitude.shape:
        raise ValueError("magnitude and support_mask must be 1-D arrays of equal length")
    if np.any(magnitude < 0):
        raise ValueError("magnitude must be non-negative (it's |F[object]|)")
    if n_iter < 1:
        raise ValueError(f"n_iter={n_iter}: must be >= 1")

    rng = np.random.default_rng(rng_seed)
    n = len(magnitude)
    phase_guess = rng.uniform(-np.pi, np.pi, n)
    g = np.fft.ifft(magnitude * np.exp(1j * phase_guess))

    errors = []
    for _ in range(n_iter):
        # (a) Fourier-magnitude constraint
        G = np.fft.fft(g)
        G_constrained = magnitude * np.exp(1j * np.angle(G))
        g_prime = np.fft.ifft(G_constrained)
        # (b) real-space support constraint (error-reduction: zero outside support)
        g = np.where(support_mask, g_prime, 0.0)
        err = float(np.sqrt(np.mean((np.abs(np.fft.fft(g)) - magnitude) ** 2)))
        errors.append(err)

    return {"object_est": g, "errors": errors, "phase_est": np.angle(g)}


# ── Quantum-limited (shot-noise) photon budget ───────────────────────────────

def shot_noise_snr(n_photons) -> float:
    """SNR = sqrt(N) for a Poisson-statistics (shot-noise-limited) photon
    count -- the quantum floor on any intensity measurement, retinal or
    otherwise. Not a claim about any specific detector's actual noise
    (real detectors also have dark current, read noise, etc. on top)."""
    n_photons = np.asarray(n_photons, dtype=float)
    if np.any(n_photons < 0):
        raise ValueError("n_photons must be non-negative")
    return np.sqrt(n_photons)


def min_photons_for_snr(target_snr: float) -> float:
    """Inverse of shot_noise_snr: minimum photon count for a target
    shot-noise-limited SNR (N = SNR^2)."""
    if target_snr <= 0:
        raise ValueError(f"target_snr={target_snr}: must be positive")
    return target_snr ** 2


if __name__ == "__main__":
    print("=== 1. The eye as a paraxial system ===")
    M_eye = reduced_eye_matrix()
    f_eye = eye_focal_length_mm()
    P_eye = eye_power_diopters()
    print(f"Reduced-eye ABCD matrix:\n{M_eye}")
    print(f"Eye focal length (in vitreous): {f_eye:.2f} mm")
    print(f"Eye refractive power: {P_eye:.1f} D  (textbook human eye: ~60 D)")
    for pupil_mm in [2.0, 4.0, 8.0]:
        r = diffraction_limited_spot_radius_um(pupil_mm, eye_focal_length_mm=f_eye)
        print(f"  pupil={pupil_mm:.1f}mm  Airy spot radius on retina = {r:.2f} um "
              f"(foveal cone spacing is ~2-3 um -- for comparison only)")

    print("\n=== 2. Proposed ultrafast STEAM retinal line-scan ===")
    profile = synthetic_vessel_reflectance(n=256, n_vessels=4)
    result = steam_retinal_linescan(profile, D_ps2=5000.0)
    print(f"Line-scan: {len(profile)} samples, output intensity range "
          f"[{result['I_out'].min():.3f}, {result['I_out'].max():.3f}]")

    print("\n=== 3a. Dispersion-diversity GS: retinal depth phase (proposed) ===")
    depth_profile = synthetic_vessel_reflectance_with_depth(n=256)
    depth_result = retinal_depth_phase_recovery(depth_profile)
    off = np.angle(np.mean(np.exp(1j * (depth_result["phi_true"] - depth_result["phi_est"]))))
    aligned_err = np.angle(np.exp(1j * (depth_result["phi_est"] + off - depth_result["phi_true"])))
    rms_deg = float(np.degrees(np.sqrt(np.mean(aligned_err ** 2))))
    print(f"('errors' diagnostic is near machine-epsilon by construction here "
          f"-- see docstring; not a convergence signal for unit_amplitude=False)")
    print(f"Recovered depth-phase RMS error (after global-offset alignment): {rms_deg:.2f} deg")

    print("\n=== 3b. Support-constraint GS (the real X-ray CDI algorithm) ===")
    n = 128
    support = np.zeros(n, dtype=bool)
    support[40:88] = True
    obj_true = np.zeros(n, dtype=complex)
    obj_true[50:80] = np.exp(1j * np.linspace(0, 2, 30))
    mag = np.abs(np.fft.fft(obj_true))
    cdi_result = support_constraint_gs(mag, support, n_iter=200)
    print(f"Final RMS magnitude error: {cdi_result['errors'][-1]:.2e} "
          f"(from {cdi_result['errors'][0]:.2e})")

    print("\n=== Quantum-limited photon budget ===")
    for n_ph in [100, 10_000, 1_000_000]:
        print(f"  N={n_ph:>9,}  SNR = {shot_noise_snr(n_ph):.1f}")
    print(f"  photons needed for SNR=100: {min_photons_for_snr(100.0):,.0f}")
