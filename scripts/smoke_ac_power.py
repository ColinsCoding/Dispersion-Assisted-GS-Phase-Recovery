"""Smoke-test AC steady state: phasor impedance and complex power."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
import circuits as ct

w = 2 * np.pi * 60     # 60 Hz mains

# 1. element impedances
assert ct.impedance_resistor(10) == 10 + 0j
assert np.isclose(ct.impedance_inductor(w, 0.1), 1j * w * 0.1)
assert np.isclose(ct.impedance_capacitor(w, 1e-4), 1 / (1j * w * 1e-4))

# 2. series/parallel combination
assert np.isclose(ct.series_impedance(10, 20, 1j * 5), 30 + 5j)
assert np.isclose(ct.parallel_impedance(10, 10), 5 + 0j)

# 3. a pure RESISTOR: all active power, Q=0, power factor 1
V = 120 + 0j
Zr = ct.impedance_resistor(10)
sr = ct.complex_power(V, V / Zr)
assert abs(sr["P"] - 1440) < 1e-9 and abs(sr["Q"]) < 1e-9 and abs(sr["pf"] - 1) < 1e-12

# 4. a pure INDUCTOR: no active power, Q > 0 (reactive), pf = 0
si = ct.complex_power(V, V / ct.impedance_inductor(w, 0.1))
assert abs(si["P"]) < 1e-9 and si["Q"] > 0 and abs(si["pf"]) < 1e-9
# a pure CAPACITOR: Q < 0 (opposite sign)
sc = ct.complex_power(V, V / ct.impedance_capacitor(w, 1e-4))
assert abs(sc["P"]) < 1e-9 and sc["Q"] < 0

# 5. an R-L load: 0 < pf < 1, both P and Q positive; |S|^2 = P^2 + Q^2
Z = ct.series_impedance(ct.impedance_resistor(10), ct.impedance_inductor(w, 0.02))
s = ct.complex_power(V, V / Z)
assert s["P"] > 0 and s["Q"] > 0 and 0 < s["pf"] < 1
assert abs(s["apparent"]**2 - (s["P"]**2 + s["Q"]**2)) < 1e-6
# power factor equals R/|Z| for a series load
assert abs(s["pf"] - 10 / abs(Z)) < 1e-9

# 6. series RLC resonance: at omega0 the reactances cancel -> impedance is real
L, C = 1e-3, 1e-6
w0 = ct.resonant_frequency(L, C)
Zres = ct.series_impedance(ct.impedance_resistor(5),
                           ct.impedance_inductor(w0, L), ct.impedance_capacitor(w0, C))
assert abs(Zres.imag) < 1e-6 and abs(Zres.real - 5) < 1e-9    # purely resistive at resonance

# 7. validation
for bad in (lambda: ct.impedance_capacitor(0, 1e-6), lambda: ct.resonant_frequency(0, 1e-6)):
    try:
        bad()
    except ValueError:
        pass
    else:
        raise AssertionError("should reject bad value")

print(f"SMOKE PASS  (R: P={sr['P']:.0f}W pf=1; L: Q={si['Q']:.0f}VAR pf=0; "
      f"RL pf={s['pf']:.3f}; resonance f0={w0/(2*np.pi):.0f}Hz -> Z real)")
