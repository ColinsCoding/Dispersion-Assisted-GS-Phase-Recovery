"""Test dgs/harsh_environment_quantum_sensing.py: thermal detuning vs. the
ring's own linewidth, the radiation microring-vs-recirculating-loop size-
scale contrast (both reusing dgs.optical_loops's functions directly), and
vibration-induced phase jitter vs. linewidth."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from dgs.optical_loops import ring_finesse, loop_threshold_gain_dB
from dgs.harsh_environment_quantum_sensing import (
    ring_FSR_wavelength, thermal_resonance_shift, verify_thermal_detuning_vs_linewidth,
    microring_finesse_under_radiation, recirculating_loop_threshold_gain_under_radiation,
    vibration_phase_jitter, verify_vibration_jitter_vs_linewidth,
)

# 1. ring_FSR_wavelength / thermal_resonance_shift: basic sanity, monotonic scaling
fsr = ring_FSR_wavelength(radius_m=10e-6, n_group=4.2)
assert fsr > 0

shift_1K = thermal_resonance_shift(1.0)
shift_10K = thermal_resonance_shift(10.0)
assert abs(shift_10K - 10 * shift_1K) / shift_10K < 1e-9   # linear in delta_T, exactly

for bad in [dict(radius_m=-1.0, n_group=4.2), dict(radius_m=10e-6, n_group=-1.0)]:
    try:
        ring_FSR_wavelength(**bad)
        raise AssertionError(f"expected ValueError for {bad}")
    except ValueError:
        pass

print("dgs.harsh_environment_quantum_sensing: thermal shift basics passed")

# 2. verify_thermal_detuning_vs_linewidth: a small dT should NOT exceed the
#    linewidth, a large dT SHOULD -- both outcomes checked, not just one
small = verify_thermal_detuning_vs_linewidth(delta_T_K=0.1)
assert small["exceeds_linewidth"] is False
large = verify_thermal_detuning_vs_linewidth(delta_T_K=20.0)
assert large["exceeds_linewidth"] is True
# fraction_of_linewidth should increase monotonically with dT
assert large["fraction_of_linewidth"] > small["fraction_of_linewidth"]

print("dgs.harsh_environment_quantum_sensing: thermal-vs-linewidth checks passed")

# 3. microring_finesse_under_radiation: reuses dgs.optical_loops.ring_finesse
#    directly -- at dose=0, must exactly match the baseline ring_finesse call
zero_dose = microring_finesse_under_radiation(dose_krad=0.0)
assert abs(zero_dose["finesse"] - ring_finesse(0.9, 0.98)) < 1e-9
assert abs(zero_dose["finesse_relative_change"]) < 1e-9

# finesse must strictly decrease (or stay equal) with increasing dose --
# radiation never IMPROVES a resonator
doses = [0, 10, 100, 500, 2000]
finesses = [microring_finesse_under_radiation(d)["finesse"] for d in doses]
assert all(finesses[i] >= finesses[i + 1] for i in range(len(finesses) - 1))

try:
    microring_finesse_under_radiation(dose_krad=-1.0)
    raise AssertionError("expected ValueError for negative dose")
except ValueError:
    pass

print("dgs.harsh_environment_quantum_sensing: microring radiation checks passed")

# 4. recirculating_loop_threshold_gain_under_radiation: reuses
#    dgs.optical_loops.loop_threshold_gain_dB directly; threshold gain
#    must strictly INCREASE with dose (more loss -> more gain needed)
zero_dose_loop = recirculating_loop_threshold_gain_under_radiation(dose_krad=0.0)
assert abs(zero_dose_loop["threshold_gain_dB"] -
           loop_threshold_gain_dB(0.2, 5.0, 1.0)) < 1e-9
assert abs(zero_dose_loop["additional_gain_needed_dB"]) < 1e-9

gains = [recirculating_loop_threshold_gain_under_radiation(d)["threshold_gain_dB"]
         for d in (0, 1, 5, 10, 20)]
assert all(gains[i] < gains[i + 1] for i in range(len(gains) - 1))

# the SAME dose should require a much bigger RELATIVE gain increase for
# the km-scale loop than the microring's finesse change -- the actual
# size-scale contrast claim, checked directly rather than just eyeballed
loop_at_10krad = recirculating_loop_threshold_gain_under_radiation(dose_krad=10.0)
microring_at_10krad = microring_finesse_under_radiation(dose_krad=10.0)
assert loop_at_10krad["additional_gain_needed_dB"] > 100.0   # dramatic, real degradation
assert abs(microring_at_10krad["finesse_relative_change"]) < 1e-3   # negligible

print("dgs.harsh_environment_quantum_sensing: recirculating-loop radiation checks passed")
print("dgs.harsh_environment_quantum_sensing: microring-vs-loop size-scale contrast confirmed")

# 5. vibration_phase_jitter / verify_vibration_jitter_vs_linewidth
jitter_1nm = vibration_phase_jitter(1e-9)
jitter_10nm = vibration_phase_jitter(10e-9)
assert abs(jitter_10nm - 10 * jitter_1nm) / jitter_10nm < 1e-9   # linear in displacement

small_vib = verify_vibration_jitter_vs_linewidth(displacement_amplitude_m=0.1e-9)
assert small_vib["exceeds_linewidth"] is False
large_vib = verify_vibration_jitter_vs_linewidth(displacement_amplitude_m=100e-9)
assert large_vib["exceeds_linewidth"] is True

try:
    vibration_phase_jitter(displacement_amplitude_m=-1.0)
    raise AssertionError("expected ValueError for negative displacement")
except ValueError:
    pass

print("dgs.harsh_environment_quantum_sensing: vibration-vs-linewidth checks passed")
print("all dgs.harsh_environment_quantum_sensing tests passed")
