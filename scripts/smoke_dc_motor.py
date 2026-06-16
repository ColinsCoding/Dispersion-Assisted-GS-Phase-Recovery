"""Smoke-test dc_motor against the analytic motor relations."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
import dc_motor as dm

# a small PMDC motor (SI): Ke=Kt=0.05, R=1, L=1e-3, J=1e-4, b=1e-5
V, R, L, Ke, Kt, J, b = 12.0, 1.0, 1e-3, 0.05, 0.05, 1e-4, 1e-5

# no-load speed ~ V/Ke; stall torque = Kt V / R
w0 = dm.no_load_speed(V, R, Ke, Kt, b)
print(f"no-load speed = {w0:.1f} rad/s  (V/Ke = {V/Ke:.1f}, slightly less due to friction)")
print(f"stall torque  = {dm.stall_torque(V, R, Kt):.3f} N*m  (= Kt V/R = {Kt*V/R:.3f})")
assert w0 < V/Ke and w0 > 0.9*V/Ke

# torque-speed line: stall at omega=0, zero torque at no-load speed
w, tau = dm.torque_speed_curve(V, R, Ke, Kt)
print(f"\ntorque-speed: tau(0) = {tau[0]:.3f} (stall), tau(end) = {tau[-1]:.3e} (~0 at no-load)")
assert abs(tau[0] - Kt*V/R) < 1e-9 and abs(tau[-1]) < 1e-6

# transient: starts at rest, settles to the steady state
out = dm.simulate(V, R, L, Ke, Kt, J, b, tau_load=0.0, dt=1e-5, t_end=0.3)
ss = dm.steady_state(V, R, Ke, Kt, b)
print(f"\ntransient final speed = {out['w'][-1]:.1f} rad/s  (steady-state {ss['speed']:.1f})")
print(f"inrush current peak   = {out['i'].max():.2f} A  (stall current V/R = {V/R:.1f})")
assert abs(out['w'][-1] - ss['speed']) / ss['speed'] < 0.02

# loaded operating point: applying a load slows it and draws more current
ss_load = dm.steady_state(V, R, Ke, Kt, b, tau_load=0.1)
print(f"\nwith 0.1 N*m load: speed {ss_load['speed']:.1f} rad/s (slower), "
      f"current {ss_load['current']:.2f} A (higher)")
assert ss_load['speed'] < ss['speed'] and ss_load['current'] > ss['current']

# efficiency at the loaded point
eta, Pin, Pout = dm.efficiency(V, ss_load['current'], ss_load['speed'], ss_load['torque'])
print(f"loaded efficiency = {float(eta)*100:.1f}%  (copper loss i^2 R = {ss_load['current']**2*R:.2f} W)")

for bad in [lambda: dm.steady_state(V, 0, Ke, Kt),
            lambda: dm.simulate(V, R, 0, Ke, Kt, J, b)]:
    try:
        bad()
    except ValueError as e:
        print("err ok:", e)
print("SMOKE PASS")
