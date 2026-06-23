"""Test the Z-transform: shifts, geometric series, discrete derivative/integral, stability."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
from dgs import z_transform as zt

# 1. Z-transform basics: delta[n] -> 1; a shift delta[n-k] -> z^{-k}
assert np.isclose(zt.z_transform([1], 3.0), 1.0)               # delta is 1 for all z
assert np.isclose(zt.z_transform([0, 0, 1], 2.0), 2.0**-2)     # delta[n-2] -> z^-2

# 2. geometric sequence a^n -> 1/(1 - a z^{-1}) for |z| > |a|
a = 0.7; N = 200; x = a ** np.arange(N)
z = 2.0 + 0j
assert np.isclose(zt.z_transform(x, z), 1.0 / (1 - a / z), atol=1e-6)

# 3. discrete DERIVATIVE (backward difference) H(z) = 1 - z^{-1}: kills DC, high-pass
Hdc = zt.filter_response(*zt.DIFFERENCE, 1.0)                  # z=1 is DC (omega=0)
assert abs(Hdc) < 1e-12                                        # a difference removes DC
omega = np.linspace(0.01, np.pi, 200)
mag = np.abs(zt.frequency_response(*zt.DIFFERENCE, omega))
assert np.allclose(mag, 2 * np.abs(np.sin(omega / 2)), atol=1e-9)   # |1 - e^{-jw}| = 2|sin(w/2)|
assert mag[-1] > mag[0]                                        # high-pass: grows with frequency

# 4. discrete INTEGRAL (accumulator) H(z) = 1/(1 - z^{-1}): pole at z=1 (marginally stable)
poles, _ = zt.poles_zeros(*zt.ACCUMULATOR)
assert np.allclose(poles, [1.0])                              # the integrator pole sits on the unit circle
assert not zt.is_stable(zt.ACCUMULATOR[1])                    # marginal -> not strictly stable

# 5. the DISCRETE Fundamental Theorem: difference * accumulator = 1 (they undo each other)
for z in (1.4 + 0.2j, 0.6 - 0.5j, 3.0):
    Hd = zt.filter_response(*zt.DIFFERENCE, z)
    Ha = zt.filter_response(*zt.ACCUMULATOR, z)
    assert np.isclose(Hd * Ha, 1.0)

# 6. stability: poles strictly inside the unit circle
assert zt.is_stable([1, -0.9]) and not zt.is_stable([1, -1.1])
assert zt.is_stable([1, 0.5, 0.06])                           # roots 0.2, 0.3 inside

# 7. time domain: accumulate(difference(x)) = x (the FTC, as running ops)
x = np.array([0.0, 1.0, 4.0, 9.0, 16.0, 25.0])               # n^2
d = zt.apply_filter(*zt.DIFFERENCE, x)                        # discrete derivative -> 1,3,5,7,9...
assert np.allclose(d, [0, 1, 3, 5, 7, 9])                     # successive odd differences of n^2
recon = zt.apply_filter(*zt.ACCUMULATOR, d)                   # integrate it back
assert np.allclose(recon, x)                                  # accumulate(difference(x)) == x

print(f"TEST PASS  (delta shifts -> z^-k; geometric -> 1/(1-az^-1); difference kills DC, "
      f"high-pass 2|sin(w/2)|; accumulator pole at z=1; difference*accumulator=1 (discrete "
      f"FTC); poles-in-unit-circle stability; accumulate(diff(n^2))=n^2)")
