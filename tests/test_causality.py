"""Test causality (Kramers-Kronig) and conservation (continuity equation)."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
from dgs import causality as ca

# 1. Hilbert transform basics: H[cos] = sin (interior, away from FFT wrap)
t = np.linspace(0, 20 * np.pi, 20000)
H = ca.hilbert_transform(np.cos(t))
s = slice(4000, 16000)
assert np.corrcoef(H[s], np.sin(t)[s])[0, 1] > 0.999

# 2. KRAMERS-KRONIG: causality reconstructs dispersion from absorption (and back)
w = np.linspace(-60, 60, 12000)
chi = ca.lorentz_susceptibility(w, omega0=10.0, gamma=1.5)
Re_kk = ca.kramers_kronig_real(chi.imag)
Im_kk = ca.kramers_kronig_imag(chi.real)
core = slice(3000, 9000)
assert np.max(np.abs(chi.real[core] - Re_kk[core])) < 0.01 * np.max(np.abs(chi.real[core])) + 1e-3
assert np.max(np.abs(chi.imag[core] - Im_kk[core])) < 0.01 * np.max(np.abs(chi.imag[core])) + 1e-3
# and the absorption line peaks at the resonance omega0 (positive-frequency side)
pos = w > 0
assert abs(w[pos][np.argmax(chi.imag[pos])] - 10.0) < 0.5

# 3. CONSERVATION: a drifting charge packet satisfies continuity (residual ~ 0)
x = np.linspace(-10, 10, 400)
tt = np.linspace(0, 4, 300)
rho, J = ca.drifting_packet(x, tt, v=1.5)
res = ca.continuity_residual(rho, J, x, tt)
assert np.max(np.abs(res[5:-5, 5:-5])) < 1e-2          # charge conserved

# 4. a NON-conserving current breaks continuity (charge appears from nowhere)
rho_bad = rho.copy()
J_zero = np.zeros_like(J)                                # charge moves but no current
res_bad = ca.continuity_residual(rho_bad, J_zero, x, tt)
assert np.max(np.abs(res_bad[5:-5, 5:-5])) > 0.1        # large residual -> unphysical

# 5. the Lorentz model is causal: |chi| is finite everywhere and -> 0 at high freq
assert np.all(np.isfinite(np.abs(chi)))
assert np.abs(chi[-1]) < np.abs(chi[np.argmin(np.abs(w - 10.0))])   # decays off resonance

print(f"TEST PASS  (H[cos]=sin; Kramers-Kronig Re<->Im to <1%; absorption peaks at "
      f"omega0=10; continuity residual~0 for drift, large for non-conserving J)")
