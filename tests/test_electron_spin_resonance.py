"""Test electron spin resonance physics: gyromagnetic ratio, Zeeman
splitting, Larmor/resonance condition, space quantization, and the
tooth-enamel EPR dosimetry dose-response chain -- checked against real
reference values (free-electron gyromagnetic ratio ~28.025 GHz/T,
X-band EPR field ~0.34 T), not just internal self-consistency."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
from dgs import electron_spin_resonance as esr

# 1. gyromagnetic ratio matches the real free-electron value (~28.025 GHz/T)
gamma_ghz = esr.gyromagnetic_ratio_hz_per_tesla() / 1e9
assert abs(gamma_ghz - 28.025) < 0.05

# 2. Zeeman splitting scales linearly with field
dE_1 = esr.zeeman_splitting_joules(1.0)
dE_2 = esr.zeeman_splitting_joules(2.0)
assert abs(dE_2 / dE_1 - 2.0) < 1e-9

# 3. larmor_frequency_hz and field_for_resonance_tesla are exact inverses
f = esr.larmor_frequency_hz(0.339)
B_recovered = esr.field_for_resonance_tesla(f)
assert abs(B_recovered - 0.339) < 1e-6

# 4. X-band EPR: 9.5 GHz should require a field near the real ~0.34 T value
B_xband = esr.field_for_resonance_tesla(9.5e9)
assert abs(B_xband - 0.34) < 0.02

# 5. spin_quantum_numbers: spin-1/2 gives exactly 2 states, {-0.5, +0.5}
m_s = esr.spin_quantum_numbers(0.5)
assert len(m_s) == 2
assert abs(m_s[0] - (-0.5)) < 1e-9 and abs(m_s[1] - 0.5) < 1e-9

# 6. spin_quantum_numbers generalizes correctly: spin-1 gives 3 states
m_s_spin1 = esr.spin_quantum_numbers(1.0)
assert len(m_s_spin1) == 3
assert abs(m_s_spin1[0] - (-1.0)) < 1e-9
assert abs(m_s_spin1[-1] - 1.0) < 1e-9

# 7. enamel_epr_signal: linear dose-response, zero dose -> just the background
sig_zero = esr.enamel_epr_signal(0.0, 50.0, background_au=2.0)
assert abs(sig_zero - 2.0) < 1e-9
sig_1gy = esr.enamel_epr_signal(1.0, 50.0, background_au=2.0)
assert abs(sig_1gy - 52.0) < 1e-9

# 8. minimum_detectable_dose_gray: higher sensitivity -> lower (better) LOD
lod_low_sens = esr.minimum_detectable_dose_gray(5.0, 10.0)
lod_high_sens = esr.minimum_detectable_dose_gray(5.0, 50.0)
assert lod_high_sens < lod_low_sens
assert abs(esr.minimum_detectable_dose_gray(5.0, 50.0) - 0.1) < 1e-9

# 9. input validation
for bad_call in [
    lambda: esr.gyromagnetic_ratio_hz_per_tesla(g_factor=-1.0),
    lambda: esr.zeeman_splitting_joules(-1.0),
    lambda: esr.zeeman_splitting_joules(1.0, g_factor=-1.0),
    lambda: esr.larmor_frequency_hz(-1.0),
    lambda: esr.field_for_resonance_tesla(-1.0),
    lambda: esr.spin_quantum_numbers(0.0),
    lambda: esr.spin_quantum_numbers(-0.5),
    lambda: esr.spin_quantum_numbers(0.3),
    lambda: esr.enamel_epr_signal(-1.0, 50.0),
    lambda: esr.enamel_epr_signal(1.0, -1.0),
    lambda: esr.enamel_epr_signal(1.0, 50.0, background_au=-1.0),
    lambda: esr.minimum_detectable_dose_gray(-1.0, 50.0),
    lambda: esr.minimum_detectable_dose_gray(5.0, -1.0),
]:
    try:
        bad_call()
        assert False, "should have raised ValueError"
    except ValueError:
        pass

print("all dgs.electron_spin_resonance tests passed")
