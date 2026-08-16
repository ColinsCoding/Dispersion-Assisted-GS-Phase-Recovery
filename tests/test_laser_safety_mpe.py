"""Test dgs/laser_safety_mpe.py: the wavelength correction factor, the
thermal-regime MPE power law, NOHD scaling (checked in both physical
directions, not assumed), the illustrative classification check, and the
wavelength-range guard that correctly rejects dgs.retinal_scan_imaging's
own default wavelength (1550nm) as outside this module's documented
scope, rather than silently answering incorrectly."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from dgs.laser_safety_mpe import (
    wavelength_correction_CA, mpe_thermal_regime, mpe_cw_long_duration,
    nohd, verify_nohd_scaling, exceeds_class1_illustrative,
    check_retinal_scan_exposure_illustrative, MODULE_SAFETY_DISCLAIMER,
)

# 1. wavelength_correction_CA: exactly 1.0 across the visible band,
#    strictly increasing into the near-IR
assert wavelength_correction_CA(400) == 1.0
assert wavelength_correction_CA(550) == 1.0
assert wavelength_correction_CA(700) == 1.0
assert wavelength_correction_CA(850) > 1.0
assert wavelength_correction_CA(1050) > wavelength_correction_CA(850)

for bad in (300, 1100, 1550):
    try:
        wavelength_correction_CA(bad)
        raise AssertionError(f"expected ValueError for wavelength={bad}nm")
    except ValueError:
        pass

print("dgs.laser_safety_mpe: wavelength correction factor checks passed")

# 2. mpe_thermal_regime: known value at t=1s, wavelength=633nm (C_A=1):
#    MPE = 1.8*1*1^0.75 mJ/cm^2 = 1.8e-3 J/cm^2, exactly
mpe_1s = mpe_thermal_regime(1.0, 633.0)
assert abs(mpe_1s - 1.8e-3) < 1e-9

# MPE must increase monotonically with exposure time (more time allowed
# -> higher cumulative energy permitted, per the t^0.75 power law)
mpe_short = mpe_thermal_regime(0.001, 633.0)
mpe_long = mpe_thermal_regime(1.0, 633.0)
assert mpe_long > mpe_short

# MPE must increase with wavelength in the NIR band (C_A > 1 there)
mpe_visible = mpe_thermal_regime(1.0, 633.0)
mpe_nir = mpe_thermal_regime(1.0, 1050.0)
assert mpe_nir > mpe_visible

for bad_t in (1e-6, 15.0, -1.0):
    try:
        mpe_thermal_regime(bad_t, 633.0)
        raise AssertionError(f"expected ValueError for exposure_s={bad_t}")
    except ValueError:
        pass

print("dgs.laser_safety_mpe: thermal-regime MPE checks passed")

# 3. mpe_cw_long_duration: correctly rejects exposures under 10s (that's
#    the thermal regime's job, not this function's)
mpe_cw = mpe_cw_long_duration(633.0)
assert abs(mpe_cw - 2.5e-3) < 1e-9   # baseline value at C_A=1

try:
    mpe_cw_long_duration(633.0, exposure_s=1.0)
    raise AssertionError("expected ValueError for exposure_s < 10")
except ValueError:
    pass

print("dgs.laser_safety_mpe: CW long-duration MPE checks passed")

# 4. nohd / verify_nohd_scaling: both physical directions checked, not
#    just one number returned and trusted
check = verify_nohd_scaling()
assert check["power_increases_NOHD"] is True
assert check["tighter_divergence_increases_NOHD"] is True
assert check["baseline_NOHD_m"] > 0

for bad in [dict(power_W=-1.0, mpe_W_cm2=1e-3, divergence_rad=1e-3),
            dict(power_W=1e-3, mpe_W_cm2=-1.0, divergence_rad=1e-3),
            dict(power_W=1e-3, mpe_W_cm2=1e-3, divergence_rad=-1.0)]:
    try:
        nohd(**bad)
        raise AssertionError(f"expected ValueError for {bad}")
    except ValueError:
        pass

print("dgs.laser_safety_mpe: NOHD scaling checks passed")

# 5. exceeds_class1_illustrative: a high-irradiance beam must exceed MPE,
#    a very-low-power beam must NOT -- both outcomes checked
high_power = exceeds_class1_illustrative(power_W=1e-3, beam_area_cm2=0.01, wavelength_nm=633.0)
assert high_power["exceeds_mpe"] is True
assert high_power["margin_factor"] > 1.0

low_power = exceeds_class1_illustrative(power_W=1e-9, beam_area_cm2=0.01, wavelength_nm=633.0)
assert low_power["exceeds_mpe"] is False
assert low_power["margin_factor"] < 1.0

print("dgs.laser_safety_mpe: classification checks passed")

# 6. check_retinal_scan_exposure_illustrative: the actual flagged-gap
#    resolution -- works at an in-scope wavelength (850nm default)
result = check_retinal_scan_exposure_illustrative(power_W=1e-6, beam_diameter_um=20.0)
assert result["beam_area_cm2"] > 0
assert result["disclaimer"] == MODULE_SAFETY_DISCLAIMER
assert isinstance(result["exceeds_mpe"], bool)

# and CORRECTLY REJECTS dgs.retinal_scan_imaging's own default wavelength
# (1550nm) rather than silently answering outside this model's scope --
# the specific honesty property this module's development caught
try:
    check_retinal_scan_exposure_illustrative(power_W=1e-6, beam_diameter_um=20.0,
                                             wavelength_nm=1550.0)
    raise AssertionError("expected ValueError for wavelength=1550nm (outside this model's scope)")
except ValueError:
    pass

print("dgs.laser_safety_mpe: retinal-scan flagged-gap checks passed "
      "(including correctly rejecting the out-of-scope 1550nm default)")
print("all dgs.laser_safety_mpe tests passed")
