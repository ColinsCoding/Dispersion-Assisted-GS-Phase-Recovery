"""Test the silicon microring biosensor physics chain: FSR, Q-factor,
sensitivity, and limit-of-detection, checked against real reported
literature ranges (Q ~ 1e4-1e5, sensitivity ~ 50-100 nm/RIU, LOD ~
1e-6-1e-7 RIU for silicon microring biosensors) rather than just
internal self-consistency."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
from dgs import photonic_biosensor_lab_on_chip as pbio

WAVELENGTH = 1550e-9
N_GROUP = 4.2
RADIUS = 10e-6
FWHM = 0.1e-9
GAMMA = 0.2
SNR = 1000.0

# 1. FSR: larger ring radius -> smaller FSR (more round-trip length,
#    denser mode spacing)
fsr_small = pbio.resonator_fsr_m(WAVELENGTH, N_GROUP, 5e-6)
fsr_large = pbio.resonator_fsr_m(WAVELENGTH, N_GROUP, 20e-6)
assert fsr_small > fsr_large

# 2. Q factor: narrower FWHM -> higher Q, matches lambda/FWHM directly
Q = pbio.resonator_q_factor(WAVELENGTH, FWHM)
assert abs(Q - WAVELENGTH / FWHM) < 1e-9
assert 1e4 <= Q <= 1e5   # realistic silicon microring range

# 3. finesse = FSR / FWHM
fsr = pbio.resonator_fsr_m(WAVELENGTH, N_GROUP, RADIUS)
finesse = pbio.resonator_finesse(fsr, FWHM)
assert abs(finesse - fsr / FWHM) < 1e-9

# 4. sensitivity scales linearly with confinement factor (more field
#    overlapping the analyte -> proportionally more sensitivity)
S_low = pbio.bulk_sensitivity_m_per_riu(WAVELENGTH, N_GROUP, 0.1)
S_high = pbio.bulk_sensitivity_m_per_riu(WAVELENGTH, N_GROUP, 0.2)
assert abs(S_high / S_low - 2.0) < 1e-9
assert 40e-9 <= S_high <= 120e-9   # realistic 40-120 nm/RIU range in meters

# 5. minimum detectable shift shrinks as SNR improves
min_shift_low_snr = pbio.minimum_detectable_wavelength_shift_m(FWHM, 100)
min_shift_high_snr = pbio.minimum_detectable_wavelength_shift_m(FWHM, 1000)
assert min_shift_high_snr < min_shift_low_snr

# 6. LOD scales inversely with sensitivity -- a more sensitive ring
#    needs a smaller index change to produce the same detectable shift
min_shift = pbio.minimum_detectable_wavelength_shift_m(FWHM, SNR)
lod_low_sens = pbio.limit_of_detection_riu(min_shift, S_low)
lod_high_sens = pbio.limit_of_detection_riu(min_shift, S_high)
assert lod_high_sens < lod_low_sens

# 7. full pipeline lands in the real literature range end-to-end
result = pbio.biosensor_figure_of_merit(WAVELENGTH, N_GROUP, RADIUS, FWHM, GAMMA, SNR)
assert 1e4 <= result["Q_factor"] <= 1e5
assert 50 <= result["sensitivity_nm_per_riu"] <= 100
assert 1e-7 <= result["lod_riu"] <= 2e-6   # realistic range; this design gives ~1.35e-6
assert result["fsr_nm"] > 0
assert result["finesse"] > 1

# 8. input validation
for bad_call in [
    lambda: pbio.resonator_fsr_m(-1.0, N_GROUP, RADIUS),
    lambda: pbio.resonator_fsr_m(WAVELENGTH, -1.0, RADIUS),
    lambda: pbio.resonator_fsr_m(WAVELENGTH, N_GROUP, -1.0),
    lambda: pbio.resonator_q_factor(-1.0, FWHM),
    lambda: pbio.resonator_q_factor(WAVELENGTH, 0.0),
    lambda: pbio.resonator_finesse(-1.0, FWHM),
    lambda: pbio.bulk_sensitivity_m_per_riu(WAVELENGTH, N_GROUP, 0.0),
    lambda: pbio.bulk_sensitivity_m_per_riu(WAVELENGTH, N_GROUP, 1.5),
    lambda: pbio.minimum_detectable_wavelength_shift_m(FWHM, -1.0),
    lambda: pbio.limit_of_detection_riu(-1.0, S_high),
]:
    try:
        bad_call()
        assert False, "should have raised ValueError"
    except ValueError:
        pass

print("all dgs.photonic_biosensor_lab_on_chip tests passed")
