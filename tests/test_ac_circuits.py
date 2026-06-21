"""Test AC power + op-amps: power factor, complex power, analog add, FTC in hardware."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
from dgs import ac_circuits as ac

t = np.linspace(0, 1, 4000)            # one period of a 1 Hz signal
w = 2 * np.pi

# 1. RMS of a sinusoid amplitude A is A/sqrt(2)
assert abs(ac.rms(3.0 * np.cos(w * t)) - 3.0 / np.sqrt(2)) < 1e-3

# 2. average power = Vrms Irms cos(phi)  (numeric integral matches the formula)
Vm, Im, phi = 10.0, 2.0, np.pi / 3
v = Vm * np.cos(w * t)
i = Im * np.cos(w * t - phi)
P_num = ac.average_power(v, i, t)
P_formula = ac.rms(v) * ac.rms(i) * ac.power_factor(0, phi)
assert abs(P_num - P_formula) < 1e-2, (P_num, P_formula)
assert abs(P_num - 0.5 * Vm * Im * np.cos(phi)) < 1e-2     # = 1/2 Vm Im cos phi = 5 W

# 3. purely reactive load (phi = 90 deg) does NO net work
v90 = Vm * np.cos(w * t)
i90 = Im * np.cos(w * t - np.pi / 2)
assert abs(ac.average_power(v90, i90, t)) < 1e-2           # ~0 W
assert abs(ac.power_factor(0, np.pi / 2)) < 1e-12

# 4. complex power: |S|^2 = P^2 + Q^2 (apparent^2 = active^2 + reactive^2)
S = ac.complex_power(ac.rms(v), ac.rms(i), phi)
assert abs(abs(S) ** 2 - (S.real ** 2 + S.imag ** 2)) < 1e-9
assert S.real > 0 and S.imag > 0                            # inductive: P>0, Q>0

# 5. impedances: R real, L is +j, C is -j
assert ac.impedance_R(50) == 50
assert abs(ac.impedance_L(1e-3, w).imag - w * 1e-3) < 1e-12 and ac.impedance_L(1e-3, w).real == 0
assert ac.impedance_C(1e-6, w).imag < 0                     # capacitor reactance negative

# 6. op-amp gains
assert ac.inverting_gain(10e3, 1e3) == -10.0
assert ac.noninverting_gain(10e3, 1e3) == 11.0

# 7. summing amp = analog addition: 1 V + 2 V -> 3 V (magnitude), "1 2 = 3"
out = ac.summing_amplifier([1.0, 2.0], [1e3, 1e3], 1e3)
assert abs(out + 3.0) < 1e-9 and abs(abs(out) - 3.0) < 1e-9

# 8. FTC in hardware: differentiator undoes integrator (constants -1/RC and -RC cancel)
vin = np.sin(w * t)
R, C = 1e3, 1e-3
recovered = ac.opamp_differentiator(ac.opamp_integrator(vin, t, R, C), t, R, C)
assert np.max(np.abs(recovered[5:-5] - vin[5:-5])) < 1e-2  # gets sin back

print(f"TEST PASS  (rms=A/sqrt2; avg P={P_num:.2f}W=Vrms Irms cos phi; reactive=0W; "
      f"|S|^2=P^2+Q^2; Z_R/L/C signs; gains; summing amp 1+2=3; "
      f"differentiator o integrator = identity (FTC))")
