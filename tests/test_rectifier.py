"""Test rectifiers: half-wave=ReLU, full-wave=|x|, average outputs, RC smoothing."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
from dgs import rectifier as rc

x = np.linspace(-5, 5, 1001)

# 1. half-wave rectifier IS ReLU = max(x, 0)
assert np.allclose(rc.half_wave_rectify(x), np.maximum(x, 0.0))
# 2. full-wave rectifier IS |x|
assert np.allclose(rc.full_wave_rectify(x), np.abs(x))

# 3. Shockley diode: I(0)=0, large positive forward, ~ -Is reverse (one-way valve)
assert abs(rc.diode_iv(0.0)) < 1e-15
assert rc.diode_iv(0.6) > 1e-3                      # forward conducts strongly (~12 mA)
assert rc.diode_iv(0.6) > 1e6 * abs(rc.diode_iv(-0.6))   # forward >> reverse (a valve)
assert -1e-12 - 1e-15 < rc.diode_iv(-1.0) < 0       # reverse: tiny -Is

# 4. average (DC) output of a rectified sinusoid: full=(2/pi)Vp, half=Vp/pi
t = np.linspace(0, 0.04, 8000); vin = 5.0 * np.sin(2*np.pi*50*t)   # exact 2 periods
assert abs(rc.full_wave_rectify(vin).mean() - rc.average_output(5, "full")) < 1e-2
assert abs(rc.half_wave_rectify(vin).mean() - rc.average_output(5, "half")) < 1e-2
# full-wave delivers exactly twice the DC of half-wave
assert abs(rc.average_output(5, "full") - 2*rc.average_output(5, "half")) < 1e-12

# 5. RC smoothing: in steady state the cap holds near the peak (ripple << full swing)
fw = rc.full_wave_rectify(vin)
sm = rc.rc_smooth(fw, t, R=1000, C=100e-6)          # tau = 100 ms >> 20 ms period
tail = sm[t > 0.03]
assert (tail.max() - tail.min()) < 0.6              # small steady ripple
assert tail.mean() > 4.0                            # sits near the 5 V peak
# more smoothing (bigger C) -> less ripple
sm2 = rc.rc_smooth(fw, t, R=1000, C=10e-6)          # tau = 10 ms -> more ripple
assert (sm2[t > 0.03].max() - sm2[t > 0.03].min()) > (tail.max() - tail.min())

# 6. ripple formula: full-wave ripples half as much as half-wave (double frequency),
#    and ripple falls as 1/C
assert np.isclose(rc.ripple_voltage(0.05, 100e-6, 50, full_wave=True),
                  0.5 * rc.ripple_voltage(0.05, 100e-6, 50, full_wave=False))
assert rc.ripple_voltage(0.05, 200e-6, 50) < rc.ripple_voltage(0.05, 100e-6, 50)

print(f"TEST PASS  (half-wave=ReLU=max(x,0); full-wave=|x|; diode one-way; "
      f"avg full {rc.average_output(5,'full'):.2f}V = 2x half {rc.average_output(5,'half'):.2f}V; "
      f"RC smoothing ripple {tail.max()-tail.min():.2f}V near 5V peak)")
