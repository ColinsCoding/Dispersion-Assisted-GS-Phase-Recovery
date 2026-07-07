"""Test dgs.zeeman_classical (Problem 3.1): the induced E field E=(r/2)dB/dt,
the speed change Delta_v=erB/2m_e, the radius-independent Larmor shift
Delta_omega=eB/2m_e (~8.8e10 rad/s at 1 T, fractional ~2.3e-5 at 500 nm), the
symmetric triplet, the quantum bridge Delta_E=hbar*Delta_omega=mu_B*B, and the
SymPy derivation of parts (a) and (b)."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
import sympy as sp
from dgs import zeeman_classical as zc

# (a) induced E field E=(r/2)dB/dt, and Faraday self-consistency:
#     E * (2 pi r) must equal the flux rate pi r^2 dB/dt
r, dBdt = 1e-10, 3.0
E = zc.induced_electric_field(r, dBdt)
assert np.isclose(E, r / 2 * dBdt)
assert np.isclose(E * 2 * np.pi * r, np.pi * r**2 * dBdt)

# (b) speed change Delta_v = e r B / (2 m_e), linear in r and B
assert np.isclose(zc.delta_v(1e-10, 1.0),
                  zc.E_CHARGE * 1e-10 / (2 * zc.M_ELECTRON))
assert np.isclose(zc.delta_v(2e-10, 1.0), 2 * zc.delta_v(1e-10, 1.0))     # ~ r
assert np.isclose(zc.delta_v(1e-10, 3.0), 3 * zc.delta_v(1e-10, 1.0))     # ~ B
# ramping from a nonzero initial field uses the difference
assert np.isclose(zc.delta_v(1e-10, 5.0, 2.0), zc.delta_v(1e-10, 3.0))

# (c) Larmor shift Delta_omega = e B / (2 m_e): the key numeric, and it is
#     INDEPENDENT of the orbit radius (Delta_v/r cancels r)
dw = zc.larmor_delta_omega(1.0)
assert np.isclose(dw, zc.E_CHARGE / (2 * zc.M_ELECTRON))
assert np.isclose(dw, 8.794e10, rtol=1e-3)
for r_test in (1e-11, 1e-10, 7e-10):
    assert np.isclose(zc.delta_v(r_test, 1.0) / r_test, dw)     # r cancels
assert np.isclose(zc.larmor_delta_omega(2.0), 2 * dw)          # linear in B

# omega_0 and the fractional shift for a 500 nm line (~2.3e-5)
assert np.isclose(zc.angular_frequency(500), 2*np.pi*zc.C_LIGHT/500e-9)
assert np.isclose(zc.fractional_shift(1.0, 500), 2.334e-5, rtol=1e-3)
assert zc.fractional_shift(1.0, 500) < 1e-4                    # small but real

# (d) the triplet is symmetric about omega_0 with spacing Delta_omega
w0 = zc.angular_frequency(500)
lo, mid, hi = zc.zeeman_triplet(w0, 1.0)
assert mid == w0
assert np.isclose(hi - mid, dw) and np.isclose(mid - lo, dw)
assert np.isclose((lo + hi) / 2, w0)                          # centered

# the quantum bridge: Delta_E = hbar*Delta_omega = mu_B * B
assert np.isclose(zc.energy_splitting(1.0), zc.BOHR_MAGNETON, rtol=1e-4)
assert np.isclose(zc.energy_splitting(1.0), zc.HBAR * dw)
assert np.isclose(zc.energy_splitting(2.5), 2.5 * zc.BOHR_MAGNETON, rtol=1e-4)

# the SymPy derivation reproduces the hand algebra
t, rr, e, m = sp.symbols("t r e m", positive=True)
Bf = sp.symbols("B_f", positive=True)
B = sp.Function("B")
sym = zc.zeeman_symbolic()
assert sp.simplify(sym["E_field"] - (-rr/2 * sp.diff(B(t), t))) == 0   # E = -r/2 dB/dt
assert sp.simplify(sym["delta_v"] - e*rr*Bf/(2*m)) == 0                # Delta_v = e r B/2m

# kwarg bounds
for bad in (lambda: zc.induced_electric_field(0, 1.0),
            lambda: zc.delta_v(-1, 1.0),
            lambda: zc.angular_frequency(0)):
    try:
        bad(); assert False
    except ValueError:
        pass

print("test_zeeman_classical: all checks passed")
