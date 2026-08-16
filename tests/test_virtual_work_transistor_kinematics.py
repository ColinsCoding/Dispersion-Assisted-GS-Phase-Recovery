"""Test dgs.virtual_work_transistor_kinematics: the symbolic Euler-Lagrange
derivation must match the closed-form velocity used numerically, and the
three physically unrelated systems (falling mass with drag, RL circuit,
transistor Miller pole) must produce genuinely identical normalized step
responses -- not just similar-looking exponentials."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
import sympy as sp
from dgs.virtual_work_transistor_kinematics import (
    virtual_work_falling_mass_sympy, falling_mass_velocity, rl_circuit_current,
    transistor_dominant_pole, transistor_step_response, isomorphism_table,
    verify_same_functional_form,
)

# 1. the symbolic v(t) solution matches the closed-form numeric function
deriv = virtual_work_falling_mass_sympy()
t_s, m_s, b_s, g_s = sp.symbols("t m b g", positive=True)
v_expr = deriv["v_of_t"].rhs
v_lambdified = sp.lambdify((t_s, m_s, b_s, g_s), v_expr, "numpy")
t_test = np.linspace(0, 10, 50)
m_test, b_test, g_test = 3.0, 1.2, 9.81
symbolic_v = v_lambdified(t_test, m_test, b_test, g_test)
numeric_v = falling_mass_velocity(t_test, m_test, b_test, g_test)
assert np.allclose(symbolic_v, numeric_v, atol=1e-10), "sympy dsolve() answer must match the closed-form numeric function"

# 2. terminal velocity and time constant match the docstring's closed forms
assert abs(deriv["tau_m"].subs({m_s: m_test, b_s: b_test}) - m_test / b_test) < 1e-12
assert abs(deriv["v_terminal"].subs({m_s: m_test, b_s: b_test, g_s: g_test}) - m_test * g_test / b_test) < 1e-12

# 3. falling_mass_velocity: monotonically increasing, approaches (never exceeds) v_terminal
v_terminal = m_test * g_test / b_test
v_curve = falling_mass_velocity(np.linspace(0, 50, 500), m_test, b_test, g_test)
assert np.all(np.diff(v_curve) >= 0), "velocity should be monotonically increasing"
assert np.all(v_curve <= v_terminal + 1e-9), "velocity should never exceed terminal velocity"
assert abs(v_curve[-1] - v_terminal) / v_terminal < 1e-6, "should have converged to v_terminal by t=50*tau"

# 4. RL circuit current: same shape, I_final = V_s/R
R, L, V_s = 100.0, 5e-6, 5.0
I_final = V_s / R
tau_rl = L / R
I_curve = rl_circuit_current(np.linspace(0, 10 * tau_rl, 200), R, L, V_s)
assert abs(I_curve[-1] - I_final) / I_final < 1e-3

# 5. transistor Miller pole: C_eq must exceed C_pi + C_mu (Miller multiplication is > 1x)
gm, r_pi, R_L, C_pi, C_mu = 0.05, 1500.0, 2000.0, 3e-12, 0.5e-12
pole = transistor_dominant_pole(gm, r_pi, R_L, C_pi, C_mu)
assert pole["C_eq"] > C_pi + C_mu, "Miller multiplication must increase the effective capacitance"
assert pole["miller_factor"] == 1 + gm * R_L
assert abs(pole["f_3dB_Hz"] - 1 / (2 * np.pi * pole["tau"])) < 1e-9

# 6. transistor step response approaches V_s
V_step = 2.0
resp = transistor_step_response(np.linspace(0, 10 * pole["tau"], 200), gm, r_pi, R_L, C_pi, C_mu, V_step)
assert np.all(np.diff(resp) >= -1e-12)
assert abs(resp[-1] - V_step) / V_step < 1e-3

# 7. the isomorphism table has one row per shared role, all three domains named in each
table = isomorphism_table()
assert len(table) >= 5
for row in table:
    assert "falling mass" in row and "RL circuit" in row and "transistor" in row

# 8. the central claim: all three normalized step responses are the SAME curve
check = verify_same_functional_form()
for name, err in check["max_errors"].items():
    assert err < 1e-9, f"{name} normalized curve should match 1-e^(-t/tau) to near machine precision, got err={err:.3e}"

# 9. input validation
try:
    falling_mass_velocity([1.0], m=-1.0, b=1.0)
    assert False, "should reject negative mass"
except ValueError:
    pass
try:
    transistor_dominant_pole(gm=0.05, r_pi=-1.0, R_L=100.0, C_pi=1e-12, C_mu=1e-12)
    assert False, "should reject negative r_pi"
except ValueError:
    pass

print("all dgs.virtual_work_transistor_kinematics tests passed")
