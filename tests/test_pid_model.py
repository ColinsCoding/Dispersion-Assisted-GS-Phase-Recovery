"""Test the PID math model: transfer function + closed-loop stability."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
import sympy as sp
from dgs import pid as pc

s = sp.Symbol("s")
kp, ki, kd = sp.symbols("K_p K_i K_d", positive=True)

# 1. transfer function C(s) = Kp + Ki/s + Kd s = (Kd s^2 + Kp s + Ki)/s
C = pc.transfer_function(kp, ki, kd, s)
assert sp.simplify(C - (kd * s**2 + kp * s + ki) / s) == 0

# 2. closed-loop characteristic poly with first-order plant P = 1/(s+1):
#    1 + C/(s+1) -> numerator  s^2 + (1+Kp) s + Ki
char = pc.characteristic_polynomial(kp, ki, kd, 1, s + 1, s)
expected = s**2 + (1 + kp) * s + ki + kd * s**2     # with Kd: (1+Kd)s^2 + (1+Kp)s + Ki
assert sp.simplify(char - sp.expand(expected)) == 0

# 3. stability (numeric): a well-chosen PI gain is stable; a negative-feedback-gone-wrong isn't
#    P=1/(s+1), PI: char = s^2 + (1+Kp)s + Ki  -> stable iff 1+Kp>0 and Ki>0
assert pc.is_stable([1, 1 + 2.0, 0.6])              # Kp=2, Ki=0.6 -> stable
assert not pc.is_stable([1, 1 + 2.0, -0.6])         # Ki<0 -> a root crosses into RHP
assert not pc.is_stable([1, -3.0, 2.0])             # negative s^1 coeff -> unstable

# 4. marginal/clearly stable hand cases
assert pc.is_stable([1, 2, 1])                      # (s+1)^2 -> roots -1,-1
assert not pc.is_stable([1, 0, 1])                  # s^2+1 -> roots +-i on the axis

print(f"TEST PASS  (C(s)=(Kd s^2+Kp s+Ki)/s; char poly correct; stability test works)")
