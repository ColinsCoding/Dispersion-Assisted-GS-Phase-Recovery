"""Test dgs.power_electronics: rectifier DC/RMS and ripple factors (half-wave ~1.21,
full-wave ~0.48, checked against a numerical average of the rectified wave), and the
DC-DC converter laws (buck steps down, boost steps up, buck-boost inverts) with the
inductor/output ripple scalings."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
from dgs import power_electronics as pe

Vp = 170.0

# 1. rectifier closed forms
hw = pe.halfwave_rectifier(Vp)
assert np.isclose(hw["v_dc"], Vp / np.pi) and np.isclose(hw["v_rms"], Vp / 2)
fw = pe.fullwave_rectifier(Vp)
assert np.isclose(fw["v_dc"], 2 * Vp / np.pi) and np.isclose(fw["v_rms"], Vp / np.sqrt(2))

# 2. ripple factors: half-wave ~1.21, full-wave ~0.48
assert np.isclose(hw["ripple_factor"], 1.211, atol=1e-3)
assert np.isclose(fw["ripple_factor"], 0.483, atol=1e-3)
assert np.isclose(pe.ripple_factor(Vp / 2, Vp / np.pi), np.sqrt((np.pi / 2) ** 2 - 1))

# 3. full-wave: twice the DC and much less ripple than half-wave
assert np.isclose(fw["v_dc"], 2 * hw["v_dc"])
assert fw["ripple_factor"] < hw["ripple_factor"]

# 4. numeric average of the rectified wave matches the closed forms
dc_f, rms_f = pe.rectifier_numeric(Vp, full_wave=True)
assert np.isclose(dc_f, 2 * Vp / np.pi, rtol=1e-3) and np.isclose(rms_f, Vp / np.sqrt(2), rtol=1e-3)
dc_h, rms_h = pe.rectifier_numeric(Vp, full_wave=False)
assert np.isclose(dc_h, Vp / np.pi, rtol=1e-3) and np.isclose(rms_h, Vp / 2, rtol=1e-3)

# 5. converter laws: buck < Vin, boost > Vin, buck-boost inverted
assert np.isclose(pe.buck_output(12, 0.5), 6.0)
assert np.isclose(pe.boost_output(12, 0.5), 24.0)
assert np.isclose(pe.buck_boost_output(12, 0.5), -12.0)
assert pe.buck_output(12, 0.3) < 12 and pe.boost_output(12, 0.3) > 12
# monotonic in D
assert pe.buck_output(12, 0.7) > pe.buck_output(12, 0.3)
assert pe.boost_output(12, 0.7) > pe.boost_output(12, 0.3)

# 6. duty <-> output round trip
assert np.isclose(pe.duty_for_buck(12, 5), 5 / 12)
assert np.isclose(pe.buck_output(12, pe.duty_for_buck(12, 5)), 5.0)
assert np.isclose(pe.duty_for_boost(12, 20), 1 - 12 / 20)
assert np.isclose(pe.boost_output(12, pe.duty_for_boost(12, 20)), 20.0)

# 7. inductor ripple: formulas and scalings
dI = pe.inductor_ripple_current(12, 0.5, L=100e-6, f_sw=500e3, topology="buck")
assert np.isclose(dI, 12 * (1 - 0.5) * 0.5 / (100e-6 * 500e3))
assert np.isclose(pe.inductor_ripple_current(12, 0.5, 200e-6, 500e3), dI / 2)   # ~1/L
assert np.isclose(pe.inductor_ripple_current(12, 0.5, 100e-6, 1e6), dI / 2)     # ~1/f_sw
assert np.isclose(pe.inductor_ripple_current(12, 0.5, 100e-6, 500e3, "boost"),
                  12 * 0.5 / (100e-6 * 500e3))

# 8. output ripple = dI/(8 C f_sw), and efficiency
assert np.isclose(pe.output_voltage_ripple(0.05, C=47e-6, f_sw=500e3),
                  0.05 / (8 * 47e-6 * 500e3))
assert np.isclose(pe.efficiency(9.0, 10.0), 0.9)

# 9. kwarg bounds
for bad in (lambda: pe.buck_output(12, 0.0),
            lambda: pe.boost_output(12, 1.0),
            lambda: pe.duty_for_buck(12, 15),          # buck can't step up
            lambda: pe.duty_for_boost(12, 5),          # boost can't step down
            lambda: pe.inductor_ripple_current(12, 0.5, 0, 500e3),
            lambda: pe.halfwave_rectifier(0),
            lambda: pe.efficiency(1, 0)):
    try:
        bad(); assert False
    except ValueError:
        pass

print("test_power_electronics: all checks passed")
