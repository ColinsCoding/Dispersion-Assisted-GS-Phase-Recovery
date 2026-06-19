"""Smoke-test circuit calculus: capacitor integrates, RC/RL/RLC responses, bandwidth."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
import sympy as sp
import circuits as ct

t = sp.Symbol("t", real=True)
C, L, I0 = sp.symbols("C L I_0", positive=True)

# 1. the capacitor integrates current: constant I -> ramp I t / C
assert sp.simplify(ct.capacitor_voltage(I0, t, C) - I0 * t / C) == 0
# integrating a sinusoid: V = (I0/(C w))(1 - cos w t)
w = sp.Symbol("omega", positive=True)
vc = ct.capacitor_voltage(I0 * sp.sin(w * t), t, C)
assert sp.simplify(vc - I0 * (1 - sp.cos(w * t)) / (C * w)) == 0

# 2. the inductor differentiates current: V_L = L di/dt
assert sp.simplify(ct.inductor_voltage(I0 * sp.sin(w * t), t, L) - L * I0 * w * sp.cos(w * t)) == 0

# 3. symbolic RC step solution is V_in (1 - e^{-t/RC})
sol = ct.solve_rc_symbolic()
tt = sp.Symbol("t", positive=True)
R_, C_, Vin_ = sp.symbols("R C V_in", positive=True)
assert sp.simplify(sol.rhs - Vin_ * (1 - sp.exp(-tt / (R_ * C_)))) == 0

# 4. numeric RC: at t=tau the capacitor is at 63.2% of final
R, Cap = 1e3, 1e-9
tau = R * Cap
assert abs(ct.rc_step(tau, R, Cap, 1.0) - (1 - 1 / np.e)) < 1e-12
assert abs(ct.rc_step(5 * tau, R, Cap, 1.0) - 1.0) < 0.01        # ~settled after 5 tau

# 5. RC bandwidth f_3dB = 1/(2 pi R C); 10x faster (smaller RC) -> 10x bandwidth
assert abs(ct.rc_bandwidth(R, Cap) - 1 / (2 * np.pi * tau)) < 1e-6
assert abs(ct.rc_bandwidth(R, Cap / 10) / ct.rc_bandwidth(R, Cap) - 10) < 1e-9

# 6. RL step: at t=L/R the current is at 63.2% of V/R
R2, L2 = 100.0, 1e-3
assert abs(ct.rl_step(L2 / R2, R2, L2, 1.0) - (1 / R2) * (1 - 1 / np.e)) < 1e-12

# 7. RLC damping classification
_, z_under, r_under = ct.rlc_damping(10, 1e-3, 1e-6)        # small R -> rings
_, z_crit, r_crit = ct.rlc_damping(2 * np.sqrt(1e-3 / 1e-6), 1e-3, 1e-6)  # R=2 sqrt(L/C)
_, z_over, r_over = ct.rlc_damping(1e4, 1e-3, 1e-6)         # big R -> sluggish
assert r_under == "underdamped" and z_under < 1
assert r_crit == "critically damped" and abs(z_crit - 1) < 1e-9
assert r_over == "overdamped" and z_over > 1

# 8. validation
for bad in (lambda: ct.rc_step(1, 0, 1), lambda: ct.rc_bandwidth(1, -1), lambda: ct.rlc_damping(1, 0, 1)):
    try:
        bad()
    except ValueError:
        pass
    else:
        raise AssertionError("should reject bad component value")

print(f"SMOKE PASS  (capacitor integrates I->It/C; RC tau=63.2%; "
      f"f_3dB={ct.rc_bandwidth(R,Cap)/1e3:.1f} kHz; RLC zeta={z_under:.3f}/{z_crit:.0f}/{z_over:.1f})")
