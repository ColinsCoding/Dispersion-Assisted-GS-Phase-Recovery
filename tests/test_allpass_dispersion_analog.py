"""Test the electronic all-pass-filter analog of optical dispersion: unit
magnitude at every frequency (pure phase), energy conservation (Parseval)
despite time-domain reshaping, two-arm circuit design with real phase
diversity, and V1(t)/V2(t) generation matching dgs.gs_core's
make_measurements structure but for a bench-buildable circuit."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
from dgs import allpass_dispersion_analog as apd

R, C = 15915.0, 10e-9

# 0. SPICE-style check: H(f) derived from the op-amp's actual node
#    equations (KCL + virtual short) must match the assumed formula exactly
_, circuit_matches = apd.derive_transfer_function_from_circuit_equations()
assert circuit_matches is True

# 1. |H(f)| = 1 for every frequency (pure phase, the defining all-pass property)
freqs = np.array([1.0, 10.0, 100.0, 1000.0, 10000.0])
H = apd.allpass_transfer_function(freqs, R, C)
assert np.allclose(np.abs(H), 1.0, atol=1e-12)

# 2. phase is 0 at DC, approaches -pi as f -> infinity (standard all-pass behavior)
assert abs(apd.allpass_phase(np.array([1e-6]), R, C)[0]) < 1e-3
assert apd.allpass_phase(np.array([1e9]), R, C)[0] < -3.0   # approaching -pi

# 3. two DIFFERENT time constants give DIFFERENT phase at the same frequency
#    (the electronic analog of D1 != D2 providing measurement diversity)
R2 = R / 10.0
phi1 = apd.allpass_phase(np.array([1000.0]), R, C)[0]
phi2 = apd.allpass_phase(np.array([1000.0]), R2, C)[0]
assert abs(phi1 - phi2) > 0.1

# 4. filtering preserves TOTAL ENERGY (Parseval's theorem, since |H(f)|=1
#    for every frequency) even though it's a nontrivial, non-identity filter
rng = np.random.default_rng(0)
n, fs = 2000, 200e3
t = np.arange(n) / fs
dt = 1.0 / fs
phi_true = np.cumsum(rng.normal(0, 0.05, n))
Vin = np.cos(2 * np.pi * 1000 * t + phi_true)

V1 = np.real(apd.apply_allpass_filter(Vin, dt, R, C))
energy_in, energy_out = np.sum(Vin**2), np.sum(V1**2)
assert abs(energy_in - energy_out) / energy_in < 1e-6

# 5. but it's NOT the identity filter -- the signal is genuinely reshaped
#    (peak amplitude changes, confirming real phase-only dispersion, not a no-op)
assert np.max(np.abs(V1 - Vin)) > 0.1
assert abs(np.max(np.abs(V1)) - np.max(np.abs(Vin))) > 0.1

# 6. design_two_arm_circuit picks component values that actually achieve
#    the requested phase diversity target
design = apd.design_two_arm_circuit(f_signal_bandwidth_hz=1000.0, phase_diversity_target_rad=1.0)
assert design["meets_target"] == True
assert design["achieved_phase_diversity_rad"] >= 1.0
assert design["R1"] != design["R2"]
assert design["R2"] < design["R1"]   # by construction, a decade lower

# a much larger target should eventually fail to be met with this simple
# decade-spacing heuristic -- confirms meets_target isn't vacuously always True
design_impossible = apd.design_two_arm_circuit(f_signal_bandwidth_hz=1000.0,
                                                phase_diversity_target_rad=100.0)
assert design_impossible["meets_target"] == False

# 7. make_electronic_measurements produces two DIFFERENT traces from one input
V1_full, V2_full = apd.make_electronic_measurements(Vin, dt, design["R1"], design["C1"],
                                                      design["R2"], design["C2"])
assert V1_full.shape == Vin.shape == V2_full.shape
assert np.max(np.abs(V1_full - V2_full)) > 0.1   # genuinely different arms

# 8. input validation
for bad_call in [
    lambda: apd.allpass_transfer_function(freqs, -1.0, C),
    lambda: apd.allpass_transfer_function(freqs, R, -1.0),
    lambda: apd.apply_allpass_filter(Vin, -1.0, R, C),
    lambda: apd.design_two_arm_circuit(-1.0, 1.0),
    lambda: apd.design_two_arm_circuit(1000.0, -1.0),
]:
    try:
        bad_call()
        assert False, "should have raised ValueError"
    except ValueError:
        pass

print("all dgs.allpass_dispersion_analog tests passed")
