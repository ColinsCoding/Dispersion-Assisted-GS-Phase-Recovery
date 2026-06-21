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

print(f"TEST PASS  (MNA: divider 5V, 8V, Ohm's law 2V, parallel 5V; damping "
      f"under/critical/over; underdamped overshoots {vc.max():.2f}>1, overdamped "
      f"monotonic; f0={spice.resonant_frequency(L,C):.0f} Hz)")
