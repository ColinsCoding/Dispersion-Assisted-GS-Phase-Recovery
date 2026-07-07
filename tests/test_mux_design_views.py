"""Test dgs.mux_design_views: the 2:1 mux in three views. Behavioral truth
table + symbolic Boolean equation, structural gate netlist matching it,
transistor cost (transmission-gate = 6T), and the timing view where the gate
output is the input CONVOLVED with the RC impulse response (matching the
analytic step response), with propagation and critical-path delays. SymPy+NumPy."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
import sympy as sp
from itertools import product
from dgs import mux_design_views as mx
from dgs import digital_logic as dl

# 1. BEHAVIORAL: the function selects A (S=0) or B (S=1)
assert mx.mux2_behavioral(a=1, b=0, s=0) == 1     # S=0 -> A
assert mx.mux2_behavioral(a=1, b=0, s=1) == 0     # S=1 -> B
tt = mx.mux2_truth_table()
assert len(tt) == 8
for a, b, s, y in tt:
    assert y == (b if s else a)

# 2. the Boolean equation Y = S'.A + S.B is that same function, symbolically
eq = mx.mux2_boolean_equation()
assert eq["matches_table"]
A, B, S = sp.symbols("A B S")
expected = (A & ~S) | (B & S)
assert sp.simplify_logic(sp.Equivalent(eq["expr"], expected)) == True

# 3. STRUCTURAL: gate netlist reproduces the behavioral table exactly
assert mx.structural_matches_behavioral()
for a, b, s in product((0, 1), repeat=3):
    assert mx.mux2_structural(a, b, s) == mx.mux2_behavioral(a, b, s)

# 4. transistor cost: AND-OR-INV vs the compact transmission-gate mux
cost = mx.mux2_transistor_cost()
assert cost["and_or_invert"] == dl.cmos_cost("NOT") + 2*dl.cmos_cost("AND2") + dl.cmos_cost("OR2")
assert cost["and_or_invert"] == 20
assert cost["transmission_gate"] == 6           # 2 TGs (4T) + 1 inverter (2T)
assert cost["transmission_gate"] < cost["and_or_invert"]

# 5. SIGNAL & TIMING: output = input convolved with the RC impulse response,
#    and that equals the analytic step response
tau = 20e-12
t = np.linspace(0, 300e-12, 6000)
y = mx.gate_output_via_convolution(np.ones_like(t), t, tau)
assert np.max(np.abs(y - mx.rc_step_response(t, tau))) < 1e-2
# the impulse response is a unit-area (unit DC gain) causal exponential
assert np.isclose(np.trapezoid(mx.rc_impulse_response(t, tau), t), 1.0, atol=1e-3)
assert np.all(mx.rc_impulse_response(-t[1:], tau) == 0.0)     # causal
# step response hits 1-1/e at t=tau, and t_pd = tau*ln2 (50% crossing)
assert np.isclose(mx.rc_step_response(tau, tau), 1 - 1/np.e, rtol=1e-9)
assert np.isclose(mx.propagation_delay(tau), tau*np.log(2))
# verify t_pd really is the 50% point of the step response
assert np.isclose(mx.rc_step_response(mx.propagation_delay(tau), tau), 0.5, atol=1e-9)

# 6. critical path = sum of stage delays (INV -> AND -> OR)
taus = [tau, tau, tau]
assert np.isclose(mx.critical_path_delay(taus), 3 * tau * np.log(2))

# 7. cascading stages = convolving impulse responses: the chain is slower than
#    any single stage (its 50% delay exceeds one stage's t_pd)
yc = mx.cascade_step_response(t, taus)
t50 = t[np.argmax(yc >= 0.5)]
assert t50 > mx.propagation_delay(tau)
assert yc[-1] > 0.99                              # settles to logic 1

# 8. kwarg bounds
for bad in (lambda: mx.mux2_behavioral(2, 0, 0),
            lambda: mx.rc_impulse_response(t, 0),
            lambda: mx.rc_step_response(t, -1),
            lambda: mx.propagation_delay(0),
            lambda: mx.critical_path_delay([]),
            lambda: mx.critical_path_delay([1e-12, -1]),
            lambda: mx.gate_output_via_convolution(np.ones(5), np.array([0,1,4,9,16]), tau)):
    try:
        bad()
        assert False, "expected ValueError"
    except ValueError:
        pass

print("test_mux_design_views: all checks passed")
