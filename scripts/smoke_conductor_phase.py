"""Smoke-test conductor E-B phase lag, B/E ratio, wavelength (Griffiths 9.129-9.137, Prob 9.20)."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
from griffiths import electrodynamics as ed

eps0, mu0 = ed._EPS0, ed._MU0
sigma_cu = 5.96e7
c = 1.0 / np.sqrt(mu0 * eps0)

# 1. GOOD conductor: B lags E by ~45 degrees (Problem 9.20c)
phi = ed.conductor_E_B_phase_lag(2 * np.pi * 1e6, sigma_cu)
assert abs(phi - np.pi / 4) < 1e-3, np.degrees(phi)        # 45 deg

# 2. POOR conductor (sigma << omega*eps): B nearly in phase with E (phi -> 0)
phi_poor = ed.conductor_E_B_phase_lag(2 * np.pi * 1e14, 1e-4)   # tiny sigma, optical omega
assert phi_poor < 0.01, phi_poor

# 3. B/E amplitude ratio: for a (near) non-conductor -> sqrt(mu0 eps0) = 1/c
ratio_vac = ed.conductor_B_E_amplitude_ratio(2 * np.pi * 1e14, 1e-12)
assert abs(ratio_vac - 1 / c) / (1 / c) < 1e-3
# in a good conductor the ratio is enormously larger (B >> E/c)
assert ed.conductor_B_E_amplitude_ratio(2 * np.pi * 1e6, sigma_cu) > 1000 / c

# 4. Problem 9.20(b): in a good conductor, skin depth = lambda / (2 pi)
w = 2 * np.pi * 1e6
lam = ed.conductor_wavelength(w, sigma_cu)
d = ed.skin_depth(w, sigma_cu)
assert abs(d - lam / (2 * np.pi)) / d < 1e-3, (d, lam / (2 * np.pi))

# 5. phase velocity v = omega/k, and is far below c in a conductor (slow wave)
v = ed.conductor_phase_velocity(w, sigma_cu)
assert v < 0.01 * c                                        # heavily slowed
assert abs(v - w / ed.conductor_wavenumber(w, sigma_cu).real) < 1e-6

# 6. consistency: K = |k~|, phi = atan(kappa/k) reconstruct k~ = K e^{i phi}
kt = ed.conductor_wavenumber(w, sigma_cu)
K = ed.conductor_B_E_amplitude_ratio(w, sigma_cu) * w      # K = ratio * omega
assert abs(K * np.exp(1j * ed.conductor_E_B_phase_lag(w, sigma_cu)) - kt) < 1e-3 * abs(kt)

print(f"SMOKE PASS  (good conductor: B lags E by {np.degrees(phi):.1f} deg; "
      f"skin depth = lambda/2pi ({d*1e6:.1f} um); v={v:.2e} m/s << c)")
