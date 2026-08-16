"""Test dgs/steam_imaging.py's oscilloscope_bandwidth_requirement and
spectral_resolving_power -- new this session, derived from the actual
Solli, Gupta & Jalali (APL 2009) dispersive-FT spectroscopy setup. Does
not re-test the file's pre-existing content (adc_clock_requirements,
ULTRAFAST_PHENOMENA, etc.), which had no test coverage before this file
and is out of scope here."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
from dgs.steam_imaging import oscilloscope_bandwidth_requirement, spectral_resolving_power

# 1. Reproduces the exact hand-derived numbers for the paper's real setup
#    (lambda0=1563nm, D1~695 ps/nm, 5 GHz linewidth): delta_t~28.3ps,
#    BW floor~12.4GHz, recommended~37.1GHz at the default 3x margin --
#    close to (slightly under) the paper's own real 40 GHz choice.
r = oscilloscope_bandwidth_requirement(wavelength_nm=1563.0, D_ps_per_nm=695.0, linewidth_ghz=5.0)
assert abs(r["delta_t_ps"] - 28.3) < 0.5
assert abs(r["bw_min_ghz"] - 12.4) < 0.5
assert abs(r["bw_recommended_ghz"] - 37.1) < 0.5
assert abs(r["sample_rate_gsps"] - 111.2) < 2.0

# 2. Sign of D_ps_per_nm shouldn't matter (paper's D values are quoted
#    negative, e.g. D1=-695 ps/nm -- the physical stretch magnitude is the
#    same either way)
r_pos = oscilloscope_bandwidth_requirement(wavelength_nm=1563.0, D_ps_per_nm=695.0, linewidth_ghz=5.0)
r_neg = oscilloscope_bandwidth_requirement(wavelength_nm=1563.0, D_ps_per_nm=-695.0, linewidth_ghz=5.0)
assert abs(r_pos["bw_min_ghz"] - r_neg["bw_min_ghz"]) < 1e-9

# 3. Monotonicity checks: more dispersion (bigger |D|) stretches features
#    MORE, needing LESS bandwidth to resolve; a narrower target linewidth
#    needs MORE bandwidth
r_more_D = oscilloscope_bandwidth_requirement(wavelength_nm=1563.0, D_ps_per_nm=1400.0, linewidth_ghz=5.0)
assert r_more_D["bw_min_ghz"] < r["bw_min_ghz"]

r_narrower_line = oscilloscope_bandwidth_requirement(wavelength_nm=1563.0, D_ps_per_nm=695.0, linewidth_ghz=1.0)
assert r_narrower_line["bw_min_ghz"] > r["bw_min_ghz"]

# 4. margin_factor and oversample_factor scale linearly as documented
r_2x_margin = oscilloscope_bandwidth_requirement(wavelength_nm=1563.0, D_ps_per_nm=695.0,
                                                  linewidth_ghz=5.0, margin_factor=6.0)
assert abs(r_2x_margin["bw_recommended_ghz"] / r["bw_recommended_ghz"] - 2.0) < 1e-6

# 5. Input validation
for bad_call in [
    lambda: oscilloscope_bandwidth_requirement(-1, 695.0, 5.0),
    lambda: oscilloscope_bandwidth_requirement(1563.0, 0.0, 5.0),
    lambda: oscilloscope_bandwidth_requirement(1563.0, 695.0, -5.0),
    lambda: oscilloscope_bandwidth_requirement(1563.0, 695.0, 5.0, margin_factor=0.0),
    lambda: oscilloscope_bandwidth_requirement(1563.0, 695.0, 5.0, oversample_factor=-1.0),
]:
    try:
        bad_call()
        assert False, "should have raised ValueError"
    except ValueError:
        pass

# 6. spectral_resolving_power: reproduces the hand-derived R~5.33e6 for the
#    paper's 36 MHz mode-locked source at 1563nm
res = spectral_resolving_power(wavelength_nm=1563.0, rep_rate_hz=36e6)
assert abs(res["resolving_power"] - 5.33e6) / 5.33e6 < 0.01
assert abs(res["delta_lambda_nm"] - 2.93e-4) / 2.93e-4 < 0.02

# 7. Higher rep rate -> coarser (larger delta_nu) -> LOWER resolving power
res_faster_rep = spectral_resolving_power(wavelength_nm=1563.0, rep_rate_hz=360e6)
assert res_faster_rep["resolving_power"] < res["resolving_power"]

# 8. resolving_power = nu / delta_nu, checked directly against speed of light
C_LIGHT = 2.998e8
nu_expected = C_LIGHT / (1563.0 * 1e-9)
assert abs(res["resolving_power"] - nu_expected / 36e6) / (nu_expected / 36e6) < 1e-6

# 9. Input validation
for bad_call in [
    lambda: spectral_resolving_power(-1, 36e6),
    lambda: spectral_resolving_power(1563.0, 0.0),
    lambda: spectral_resolving_power(1563.0, -36e6),
]:
    try:
        bad_call()
        assert False, "should have raised ValueError"
    except ValueError:
        pass

print("all dgs.steam_imaging bandwidth/resolving-power tests passed")
