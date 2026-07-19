"""Test rice grain moisture-diffusion physics (Fick's 2nd law sphere
solution), starch gelatinization thresholds, and the rice-vs-silica-gel
desiccant comparison -- checked against physically sensible bounds and
internal consistency, not fabricated precision."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
from dgs import rice_grain_physics as rgp

D = 5e-10
r = 1.5e-3

# 1. moisture_ratio_sphere_diffusion: MR starts near 1 at t=0, decays
#    monotonically, and approaches 0 for long times
MR_t0 = rgp.moisture_ratio_sphere_diffusion(D, r, 0.0)
MR_5min = rgp.moisture_ratio_sphere_diffusion(D, r, 5 * 60)
MR_20min = rgp.moisture_ratio_sphere_diffusion(D, r, 20 * 60)
MR_long = rgp.moisture_ratio_sphere_diffusion(D, r, 100 * 3600)
assert 0.9 < MR_t0 <= 1.0
assert MR_5min > MR_20min > MR_long
assert MR_long < 0.001

# 2. faster diffusivity -> faster equilibration (lower MR at the same time)
MR_slow = rgp.moisture_ratio_sphere_diffusion(1e-11, r, 600)
MR_fast = rgp.moisture_ratio_sphere_diffusion(1e-9, r, 600)
assert MR_fast < MR_slow

# 3. larger grain -> slower equilibration (higher MR at the same time,
#    same diffusivity) -- diffusion into a bigger object takes longer
MR_small_grain = rgp.moisture_ratio_sphere_diffusion(D, 0.5e-3, 600)
MR_big_grain = rgp.moisture_ratio_sphere_diffusion(D, 3e-3, 600)
assert MR_big_grain > MR_small_grain

# 4. diffusion_time_scale_s: scales as r^2 and inversely with D
tau1 = rgp.diffusion_time_scale_s(1e-3, 1e-10)
tau2 = rgp.diffusion_time_scale_s(2e-3, 1e-10)
tau3 = rgp.diffusion_time_scale_s(1e-3, 2e-10)
assert abs(tau2 / tau1 - 4.0) < 1e-9   # r doubled -> tau x4 (r^2 scaling)
assert abs(tau3 / tau1 - 0.5) < 1e-9   # D doubled -> tau halved

# 5. is_starch_gelatinized: correctly thresholds against each variety's
#    real onset temperature, and cold soaking never gelatinizes anything
for variety in rgp.GELATINIZATION_RANGES_C:
    onset, _ = rgp.GELATINIZATION_RANGES_C[variety]
    assert rgp.is_starch_gelatinized(onset + 5, variety)
    assert not rgp.is_starch_gelatinized(onset - 5, variety)
    assert not rgp.is_starch_gelatinized(20.0, variety)   # room temperature soak

# 6. waxy glutinous rice gelatinizes at a LOWER temperature than long-grain
#    indica -- a real, known rice-science fact this module should reproduce
waxy_onset = rgp.GELATINIZATION_RANGES_C["waxy_glutinous"][0]
indica_onset = rgp.GELATINIZATION_RANGES_C["long_grain_indica"][0]
assert waxy_onset < indica_onset

# 7. rice_vs_silica_gel_desiccant_comparison: rice's diffusion time scale
#    is MANY orders of magnitude longer than silica gel's (geometry-driven)
result = rgp.rice_vs_silica_gel_desiccant_comparison()
assert result["ratio"] > 1e6   # at least a million times slower
assert result["tau_rice_hours"] > 0
assert result["tau_silica_seconds"] > 0

# 8. input validation
for bad_call in [
    lambda: rgp.moisture_ratio_sphere_diffusion(-1.0, r, 600),
    lambda: rgp.moisture_ratio_sphere_diffusion(D, -1.0, 600),
    lambda: rgp.moisture_ratio_sphere_diffusion(D, r, -1.0),
    lambda: rgp.moisture_ratio_sphere_diffusion(D, r, 600, n_terms=0),
    lambda: rgp.diffusion_time_scale_s(-1.0, D),
    lambda: rgp.diffusion_time_scale_s(r, -1.0),
    lambda: rgp.is_starch_gelatinized(70.0, variety="not_a_real_variety"),
    lambda: rgp.rice_vs_silica_gel_desiccant_comparison(rice_radius_m=-1.0),
    lambda: rgp.rice_vs_silica_gel_desiccant_comparison(rice_D_m2_s=-1.0),
]:
    try:
        bad_call()
        assert False, "should have raised ValueError"
    except ValueError:
        pass

print("all dgs.rice_grain_physics tests passed")
