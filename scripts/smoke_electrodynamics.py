"""Smoke-test griffiths.electrodynamics: the physics + the bridge to disperse()."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
import sympy as sp
from griffiths import electrodynamics as ed
import dispersion_gs_prototype as dg

mu, eps, omega, k = ed.mu, ed.eps, ed.omega, ed.k

# 1. plane wave satisfies the wave equation -> dispersion relation k^2 = mu eps w^2
disp, k_w, n = ed.plane_wave_dispersion()
assert disp.rhs == mu * eps * omega**2
assert sp.simplify(k_w**2 - mu * eps * omega**2) == 0

# 2. 1-D wave equation comes out as d2E/dz2 = mu eps d2E/dt2
wave, steps = ed.wave_equation_1d()
E = sp.Function("E")(ed.z, ed.t)
assert sp.simplify(wave.lhs - sp.diff(E, ed.z, 2)) == 0
assert sp.simplify(wave.rhs - mu * eps * sp.diff(E, ed.t, 2)) == 0

# 3. vacuum wave impedance ~ 377 ohm (numeric check with SI constants)
Z0 = ed.wave_impedance(medium=False)
val = float(Z0.subs({ed.mu0: 4e-7 * np.pi, ed.eps0: 8.8541878128e-12}))
assert abs(val - 376.73) < 0.5, val

# 4. Ohm's-law tensor: isotropic default gives J = sigma E componentwise
J, Sig = ed.ohms_law_tensor()
s = sp.Symbol("sigma")
Ex, Ey, Ez = sp.symbols("E_x E_y E_z")
assert J == sp.Matrix([s * Ex, s * Ey, s * Ez])
# anisotropic: off-diagonal sigma mixes components
J2, _ = ed.ohms_law_tensor([[1, 2, 0], [0, 1, 0], [0, 0, 3]])
assert J2[0] == Ex + 2 * Ey

# 5. decibels: power 10*log10, amplitude 20*log10, factor-2 in power = 3.01 dB
assert abs(ed.to_decibels(2.0) - 3.0103) < 1e-3
assert abs(ed.to_decibels(10.0, kind="amplitude") - 20.0) < 1e-9
assert ed.to_decibels(0.0) == -np.inf

# 6. THE BRIDGE: gvd_transfer == repo's disperse() with D = 2 pi beta2 L
N = 1024
t, x, A, phi = dg.make_field(N=N, seed=3)
beta2, L = 1.7, 0.9
D = ed.dispersion_param_D(beta2, L)
f = np.fft.fftfreq(N)
H_phys = ed.gvd_transfer(f, beta2, L)
H_repo = np.exp(1j * np.pi * D * f**2)
assert np.allclose(H_phys, H_repo), "GVD operator must equal exp(i pi D f^2)"
# and applying it equals dg.disperse(x, D)
xd_phys = np.fft.ifft(np.fft.fft(x) * H_phys)
xd_repo = dg.disperse(x, D)
assert np.allclose(xd_phys, xd_repo), "Maxwell-GVD propagation must equal disperse()"

print("SMOKE PASS  (D = 2*pi*beta2*L =", round(D, 4), ")")
