"""Test dgs/retinal_scan_imaging.py's three pieces: the reduced-eye ABCD
optics, the proposed STEAM retinal line-scan, and the two distinct
phase-retrieval families (dispersion-diversity GS vs. support-constraint
GS/Fienup, the latter being the actual X-ray CDI algorithm)."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
from dgs.retinal_scan_imaging import (
    reduced_eye_matrix, eye_focal_length_mm, eye_power_diopters,
    diffraction_limited_spot_radius_um,
    synthetic_vessel_reflectance, synthetic_vessel_reflectance_with_depth,
    steam_retinal_linescan, retinal_depth_phase_recovery,
    support_constraint_gs, shot_noise_snr, min_photons_for_snr,
)

# 1. Reduced-eye power should match the textbook ~60 D human-eye value
P = eye_power_diopters()
assert abs(P - 60.5) < 1.0, f"expected ~60.5 D for textbook reduced-eye numbers, got {P:.2f}"

# 2. Reduced-eye matrix must be unimodular in the raw-angle convention used
#    for free_space_matrix/thin_lens (spherical_interface breaks det=1 by
#    design -- see paraxial_optics_abcd.py -- so check det=n1/n2 instead)
M = reduced_eye_matrix()
det = np.linalg.det(M)
assert abs(det - 1.0 / 1.336) < 1e-9, (
    f"reduced_eye_matrix should have det=1/n_vitreous (refraction step only), got {det}")

# 3. Airy spot radius must shrink as pupil grows (diffraction limit, D in denominator)
r_small_pupil = diffraction_limited_spot_radius_um(2.0)
r_large_pupil = diffraction_limited_spot_radius_um(8.0)
assert r_large_pupil < r_small_pupil, "larger pupil must give a smaller diffraction-limited spot"

# 4. Bounds: nonpositive pupil/wavelength must raise
for bad_kwargs in [dict(pupil_diameter_mm=0.0), dict(pupil_diameter_mm=2.0, wavelength_nm=-1.0)]:
    try:
        diffraction_limited_spot_radius_um(**bad_kwargs)
        raise AssertionError(f"expected ValueError for {bad_kwargs}")
    except ValueError:
        pass

# 5. STEAM retinal line-scan: output intensity must be finite and non-negative
profile = synthetic_vessel_reflectance(n=256, n_vessels=4)
result = steam_retinal_linescan(profile, D_ps2=5000.0)
assert np.all(np.isfinite(result["I_out"])), "line-scan output must be finite"
assert np.all(result["I_out"] >= 0), "intensity must be non-negative"

# 6. Depth-encoded profile must actually carry nonzero phase (unlike the
#    plain reflectance profile, whose phase is trivially zero everywhere)
depth_profile = synthetic_vessel_reflectance_with_depth(n=256)
assert np.max(np.abs(np.angle(depth_profile))) > 0.5, (
    "synthetic_vessel_reflectance_with_depth should carry real phase structure")
assert np.allclose(np.angle(profile), 0.0), (
    "plain synthetic_vessel_reflectance should have trivially zero phase (sanity check "
    "on the fixture the depth-recovery test below is contrasted against)")

# 7. Dispersion-diversity GS recovers the depth phase to within a modest
#    RMS error after global-offset alignment (this is a harder,
#    varying-amplitude problem -- not expecting QPSK-level accuracy)
depth_result = retinal_depth_phase_recovery(depth_profile, D1=-5000.0, D2=-5750.0, n_iter=50)
off = np.angle(np.mean(np.exp(1j * (depth_result["phi_true"] - depth_result["phi_est"]))))
aligned_err = np.angle(np.exp(1j * (depth_result["phi_est"] + off - depth_result["phi_true"])))
rms_deg = float(np.degrees(np.sqrt(np.mean(aligned_err ** 2))))
assert rms_deg < 90.0, f"expected clearly-better-than-random depth-phase recovery, got RMS={rms_deg:.1f} deg"

# 8. Support-constraint GS (the real X-ray-CDI-style algorithm) must
#    substantially reduce the Fourier-magnitude error from its random start
n = 128
support = np.zeros(n, dtype=bool)
support[40:88] = True
obj_true = np.zeros(n, dtype=complex)
obj_true[50:80] = np.exp(1j * np.linspace(0, 2, 30))
mag = np.abs(np.fft.fft(obj_true))
cdi_result = support_constraint_gs(mag, support, n_iter=200)
assert cdi_result["errors"][-1] < cdi_result["errors"][0] * 0.1, (
    "support-constraint GS should reduce the magnitude error by >90% over 200 iterations")

# 9. support_constraint_gs bounds: mismatched shapes and negative magnitude must raise
try:
    support_constraint_gs(mag, support[:-1])
    raise AssertionError("expected ValueError for mismatched shapes")
except ValueError:
    pass
try:
    support_constraint_gs(-mag, support)
    raise AssertionError("expected ValueError for negative magnitude")
except ValueError:
    pass

# 10. Shot-noise SNR/photon-budget: sqrt scaling and its exact inverse
assert abs(shot_noise_snr(10_000.0) - 100.0) < 1e-9
assert abs(min_photons_for_snr(100.0) - 10_000.0) < 1e-9
try:
    min_photons_for_snr(0.0)
    raise AssertionError("expected ValueError for target_snr<=0")
except ValueError:
    pass

print("all dgs.retinal_scan_imaging tests passed")
