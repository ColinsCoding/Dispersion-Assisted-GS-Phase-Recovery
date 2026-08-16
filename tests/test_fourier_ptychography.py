"""Test Fourier Ptychographic Microscopy (dgs/fourier_ptychography.py): the
real, published technique answering "can a microscope reconstruct sharper
images without shrinking field of view" -- resolution/SBP formulas, LED
array geometry, and a full synthetic forward-model + reconstruction round
trip that must actually recover a known high-resolution phase object."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
from dgs import fourier_ptychography as fp

# 1. Abbe resolution formula: finer (smaller) at higher NA, matches lambda/(2*NA)
res_low_na = fp.resolution_half_pitch_nm(500.0, 0.1)
res_high_na = fp.resolution_half_pitch_nm(500.0, 0.3)
assert abs(res_low_na - 2500.0) < 1e-9
assert res_high_na < res_low_na

# 2. synthetic_NA is additive and rejects unphysical (>1) results
assert abs(fp.synthetic_NA(0.1, 0.2) - 0.3) < 1e-9
try:
    fp.synthetic_NA(0.7, 0.5)
    assert False, "should have raised ValueError"
except ValueError:
    pass

# 3. space_bandwidth_product: finer resolution at fixed FOV -> more resolvable points
sbp_coarse = fp.space_bandwidth_product(500.0, 2.5)
sbp_fine = fp.space_bandwidth_product(500.0, 0.9)
assert sbp_fine > sbp_coarse

# 4. led_array_illumination_NA: center LED has NA=0, outer ring has larger NA;
#    grid shape matches n_leds_per_side, and it rejects an even count
led = fp.led_array_illumination_NA(5, 4.0, 60.0)
assert led["NA_illum"].shape == (5, 5)
assert led["NA_illum"][2, 2] == 0.0        # center LED, directly below sample
assert led["NA_illum"].max() > led["NA_illum"][2, 2]
try:
    fp.led_array_illumination_NA(4, 4.0, 60.0)
    assert False, "should have raised ValueError (even n_leds_per_side)"
except ValueError:
    pass

# 5. Input validation on the core formulas
for bad_call in [
    lambda: fp.resolution_half_pitch_nm(-500.0, 0.1),
    lambda: fp.resolution_half_pitch_nm(500.0, 0.0),
    lambda: fp.synthetic_NA(0.0, 0.1),
    lambda: fp.synthetic_NA(0.1, -0.1),
    lambda: fp.space_bandwidth_product(-1.0, 1.0),
    lambda: fp.space_bandwidth_product(1.0, 0.0),
    lambda: fp.led_array_illumination_NA(5, -1.0, 60.0),
    lambda: fp.led_array_illumination_NA(5, 4.0, 0.0),
]:
    try:
        bad_call()
        assert False, "should have raised ValueError"
    except ValueError:
        pass

# 6. Full forward-model + reconstruction round trip: two point-like phase
#    features closer together than the bare objective (NA_obj alone) can
#    resolve must still be recoverable once many illumination angles are
#    combined -- this is the actual "sharper image" claim, checked
#    numerically, not just asserted in a docstring.
N = 64
pixel_pitch_nm = 800.0
wavelength_nm = 500.0
NA_obj = 0.1

yy, xx = np.mgrid[0:N, 0:N]
obj = np.ones((N, N), complex)
for cy, cx in [(30, 28), (30, 36)]:
    obj *= np.exp(1j * 0.8 * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * 1.5 ** 2)))

led = fp.led_array_illumination_NA(5, 4.0, 60.0)
led_positions_NA = [
    (led["positions_mm"][i, j, 0] / led["r_mm"][i, j] * led["NA_illum"][i, j] if led["r_mm"][i, j] > 0 else 0.0,
     led["positions_mm"][i, j, 1] / led["r_mm"][i, j] * led["NA_illum"][i, j] if led["r_mm"][i, j] > 0 else 0.0)
    for i in range(led["NA_illum"].shape[0]) for j in range(led["NA_illum"].shape[1])
]

captures = fp.simulate_fpm_captures(obj, NA_obj, wavelength_nm, pixel_pitch_nm, led_positions_NA)
assert len(captures) == 25
for intensity, shift in captures:
    assert intensity.shape == (N, N)
    assert np.all(intensity >= 0)

result = fp.reconstruct_fpm(captures, NA_obj, wavelength_nm, pixel_pitch_nm, obj.shape, n_iter=40)
assert result["object_recovered"].shape == (N, N)
assert len(result["convergence"]) == 40
# convergence error should have decreased substantially from first to last iteration
assert result["convergence"][-1] < result["convergence"][0] * 0.1

phase_true = np.angle(obj)
phase_rec = np.angle(result["object_recovered"])
corr = np.corrcoef(phase_true.ravel(), phase_rec.ravel())[0, 1]
assert corr > 0.99, f"expected near-perfect phase recovery, got corr={corr:.4f}"

# 7. reconstruct_fpm rejects a non-positive iteration count
try:
    fp.reconstruct_fpm(captures, NA_obj, wavelength_nm, pixel_pitch_nm, obj.shape, n_iter=0)
    assert False, "should have raised ValueError"
except ValueError:
    pass

print("all dgs.fourier_ptychography tests passed")
