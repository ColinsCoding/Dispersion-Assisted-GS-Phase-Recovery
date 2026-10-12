"""Test the mini-SPICE: MNA DC solve (linear algebra) + RLC transient (ODE)."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
from dgs import spice

# 1. voltage divider: 10 V across two equal 1k -> middle node at 5 V
net = [("V", 1, 0, 10.0), ("R", 1, 2, 1000.0), ("R", 2, 0, 1000.0)]
v = spice.dc_nodal_analysis(net, 2)
assert abs(v[0] - 10.0) < 1e-9 and abs(v[1] - 5.0) < 1e-9

# 2. unequal divider: 12 V, 1k over (1k+2k) -> v2 = 12 * 2k/3k = 8 V at the tap above 2k
net2 = [("V", 1, 0, 12.0), ("R", 1, 2, 1000.0), ("R", 2, 0, 2000.0)]
v2 = spice.dc_nodal_analysis(net2, 2)
assert abs(v2[1] - 8.0) < 1e-9, v2

# 3. current source into a resistor: I*R sets the node voltage (Ohm's law)
net3 = [("I", 0, 1, 2e-3), ("R", 1, 0, 1000.0)]      # 2 mA into 1k -> 2 V
v3 = spice.dc_nodal_analysis(net3, 1)
assert abs(v3[0] - 2.0) < 1e-9, v3

# 4. two resistors in parallel via MNA == product/sum hand calc
#    1k || 1k = 500; 10 mA through it -> 5 V
net4 = [("I", 0, 1, 10e-3), ("R", 1, 0, 1000.0), ("R", 1, 0, 1000.0)]
v4 = spice.dc_nodal_analysis(net4, 1)
assert abs(v4[0] - 5.0) < 1e-9, v4

# 5. damping classification from the characteristic roots
L, C = 1e-3, 1e-6
Rc = spice.critical_resistance(L, C)                 # = 2 sqrt(L/C)
assert spice.rlc_damping(10.0, L, C)[0] == "under"   # small R -> rings
assert spice.rlc_damping(Rc, L, C)[0] == "critical"
assert spice.rlc_damping(500.0, L, C)[0] == "over"   # big R -> no oscillation

# 6. underdamped RLC step: capacitor voltage OVERSHOOTS 1 V then settles to it
t = np.linspace(0, 5e-3, 20000)
vc, il = spice.rlc_step_response(10.0, L, C, t, V=1.0)
assert vc.max() > 1.05                                 # overshoot above the 1 V step
assert abs(vc[-1] - 1.0) < 0.05                        # settles to the source voltage
assert abs(il[-1]) < 1e-3                              # steady state: no current through C

# 7. overdamped RLC step: monotonic, NO overshoot (R=200 > R_crit, settles in-window)
assert spice.rlc_damping(200.0, L, C)[0] == "over"
vc_od, _ = spice.rlc_step_response(200.0, L, C, t, V=1.0)
assert vc_od.max() <= 1.0 + 1e-3                        # never exceeds the step
assert abs(vc_od[-1] - 1.0) < 0.05                      # settled to the source voltage
assert np.all(np.diff(vc_od) >= -1e-6)                  # monotonic rise (no ringing)

# 8. resonant frequency f0 = 1/(2 pi sqrt(LC))
assert abs(spice.resonant_frequency(L, C) - 1.0 / (2 * np.pi * np.sqrt(L * C))) < 1e-6

# 9. s_domain_nodal_analysis on a purely resistive circuit must reduce EXACTLY to
# dc_nodal_analysis (same math, generalized solver) -- checked on two independent nets
v_s = spice.s_domain_nodal_analysis(net, 2, s=1.0 + 0j)
assert np.allclose(v, v_s.real, atol=1e-9) and np.allclose(v_s.imag, 0.0, atol=1e-12)
v2_s = spice.s_domain_nodal_analysis(net2, 2, s=5.0 + 0j)
assert np.allclose(v2, v2_s.real, atol=1e-9)

# 10. RC low-pass: H(s)=1/(1+sRC) exactly, at several frequencies including the corner
R_rc, C_rc = 1000.0, 1e-6
net_rc = [("R", 1, 2, R_rc), ("C", 2, 0, C_rc)]
omegas = np.array([100.0, 1000.0, 1.0 / (R_rc * C_rc), 1e4, 1e5])
s_vals = 1j * omegas
H = spice.transfer_function_from_circuit(net_rc, n_nodes=2, s_values=s_vals, out_node=2, in_node=1)
H_exact = 1.0 / (1 + s_vals * R_rc * C_rc)
assert np.allclose(H, H_exact, rtol=1e-9), (H, H_exact)
# at the corner frequency omega=1/RC, |H| must be exactly 1/sqrt(2) (the -3dB point)
corner_idx = 2
assert abs(abs(H[corner_idx]) - 1 / np.sqrt(2)) < 1e-9

# 11. dependent source (VCVS): a non-inverting op-amp amplifier, ideal op-amp modeled
# as a VCVS with large open-loop gain A. As A -> infinity, closed-loop gain -> the
# textbook closed form 1 + Rf/Rin, NOT asserted -- actually driven to convergence
Rf, Rin = 9000.0, 1000.0
expected_gain = 1 + Rf / Rin   # = 10.0
gains = []
for A in (1e2, 1e4, 1e6, 1e8):
    net_amp = [
        ("V", 1, 0, 1.0),
        ("E", 3, 0, 1, 2, A),   # ideal op-amp: V(node3) = A*(V(node1)-V(node2))
        ("R", 3, 2, Rf),
        ("R", 2, 0, Rin),
    ]
    v_amp = spice.s_domain_nodal_analysis(net_amp, n_nodes=3, s=1.0 + 0j)
    gains.append((v_amp[2] / v_amp[0]).real)
assert gains[-1] > gains[0]                              # monotonically approaching the limit
assert abs(gains[-1] - expected_gain) < 1e-4              # essentially at the ideal-op-amp limit
assert abs(gains[0] - expected_gain) > 0.5                # finite gain is measurably NOT ideal yet

# 12. s=0 with a reactive element must raise, not silently divide by zero or return garbage
try:
    spice.s_domain_nodal_analysis([("C", 1, 0, 1e-6)], 1, s=0)
    assert False
except ValueError:
    pass

print(f"TEST PASS  (MNA: divider 5V, 8V, Ohm's law 2V, parallel 5V; damping "
      f"under/critical/over; underdamped overshoots {vc.max():.2f}>1, overdamped "
      f"monotonic; f0={spice.resonant_frequency(L,C):.0f} Hz; s-domain matches DC; "
      f"RC low-pass matches 1/(1+sRC) exactly; VCVS op-amp converges to gain="
      f"{expected_gain} as A->inf, reaching {gains[-1]:.6f})")
