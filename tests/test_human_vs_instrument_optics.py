"""Test dgs/human_vs_instrument_optics.py: the differential-area/solid-angle
derivation (Feynman/Griffiths), the ported Mie angular-scattering machinery,
the eye-vs-instrument optical metrics, and the flux/dynamic-range/temporal
comparisons built on top of them."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
import sympy as sp
from dgs.human_vs_instrument_optics import (
    derive_spherical_area_element_symbolic, verify_full_sphere_solid_angle,
    optical_system_metrics, HUMAN_EYE, JALALI_INSTRUMENT, _focal_length_from_NA,
    collected_scattering_flux, compare_eye_vs_instrument_collection,
    dynamic_range_bits, temporal_resolution_comparison,
)

# 1. Differential area element must be exactly r^2*sin(theta)
r, theta = sp.symbols("r theta", positive=True)
assert sp.simplify(derive_spherical_area_element_symbolic() - r**2 * sp.sin(theta)) == 0, \
    "spherical area element must simplify to exactly r^2*sin(theta)"

# 2. Full-sphere solid angle must be 4*pi to numerical-integration precision
assert abs(verify_full_sphere_solid_angle() - 4 * np.pi) < 1e-4, \
    "integrated solid angle over the full sphere must equal 4*pi steradians"

# 3. optical_system_metrics bounds validation
for bad_kwargs in [{"aperture_diameter_mm": 0.0}, {"aperture_diameter_mm": -1.0},
                    {"focal_length_mm": 0.0}, {"wavelength_nm": -5.0}]:
    kwargs = {"aperture_diameter_mm": 4.0, "focal_length_mm": 17.0, "wavelength_nm": 555.0}
    kwargs.update(bad_kwargs)
    try:
        optical_system_metrics(**kwargs)
        raise AssertionError(f"optical_system_metrics({bad_kwargs}) should have raised ValueError")
    except ValueError:
        pass

# 4. _focal_length_from_NA must reproduce the target NA exactly through the
#    forward (non-paraxial) optical_system_metrics formula
D, target_NA = 5.8, 0.70
f = _focal_length_from_NA(D, target_NA)
m = optical_system_metrics(aperture_diameter_mm=D, focal_length_mm=f, wavelength_nm=1590.0)
assert abs(m["NA"] - target_NA) < 1e-9, (
    f"_focal_length_from_NA inversion failed: got NA={m['NA']}, expected {target_NA}")

# 5. Sanity checks on the two named presets
eye_m = optical_system_metrics(**HUMAN_EYE)
inst_m = optical_system_metrics(**JALALI_INSTRUMENT)
assert abs(inst_m["NA"] - 0.70) < 1e-9, "JALALI_INSTRUMENT must reproduce SEALS's NA=0.70 exactly"
assert eye_m["acceptance_solid_angle_sr"] < inst_m["acceptance_solid_angle_sr"], \
    "the instrument's NA=0.70 aperture must subtend a larger solid angle than the eye's pupil"

# 6. collected_scattering_flux bounds validation
try:
    collected_scattering_flux(1.39, 1.0, 9940e-6, 555.0, theta_max_rad=0.0)
    raise AssertionError("collected_scattering_flux(theta_max_rad=0) should have raised ValueError")
except ValueError:
    pass
try:
    collected_scattering_flux(1.39, 1.0, 9940e-6, 555.0, theta_max_rad=np.pi)
    raise AssertionError("collected_scattering_flux(theta_max_rad=pi) should have raised ValueError")
except ValueError:
    pass

# 7. Flux integral must be positive and convergent (stable across sample counts)
f_coarse = collected_scattering_flux(1.39, 1.0, 9940e-6, 555.0, 0.117, n_theta=500)
f_fine = collected_scattering_flux(1.39, 1.0, 9940e-6, 555.0, 0.117, n_theta=8000)
assert f_coarse > 0 and f_fine > 0, "collected flux must be positive"
assert abs(f_fine - f_coarse) / f_fine < 1e-3, (
    f"flux integral not converged: n=500 gives {f_coarse}, n=8000 gives {f_fine}")

# 8. Full eye-vs-instrument comparison runs end to end and returns positive fluxes
c = compare_eye_vs_instrument_collection()
assert c["eye"]["collected_flux"] > 0 and c["instrument"]["collected_flux"] > 0
assert c["flux_ratio_instrument_over_eye"] > 0

# 9. dynamic_range_bits bounds + a known value (2^10=1024)
assert abs(dynamic_range_bits(1024) - 10.0) < 1e-9
try:
    dynamic_range_bits(1.0)
    raise AssertionError("dynamic_range_bits(1.0) should have raised ValueError")
except ValueError:
    pass
try:
    dynamic_range_bits(0.5)
    raise AssertionError("dynamic_range_bits(0.5) should have raised ValueError")
except ValueError:
    pass

# 10. temporal_resolution_comparison: instrument must be faster (smaller
#     resolution time) than the eye's flicker-fusion frame time
t = temporal_resolution_comparison()
assert t["instrument_resolution_s"] < t["eye_frame_time_s"]
assert t["speedup_factor"] > 1.0
try:
    temporal_resolution_comparison(osc_bw_ghz=0.0)
    raise AssertionError("temporal_resolution_comparison(osc_bw_ghz=0) should have raised ValueError")
except ValueError:
    pass

print("all dgs.human_vs_instrument_optics tests passed")
