"""Test EM-wave dispersion: Lorentz index, group<phase velocity, pulse spreading."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
from dgs import em_dispersion as ed

w0, g, wp = 1.0, 0.05, 0.4

# 1. Lorentz index: n>1 below resonance (normal), absorption Im(n) peaks near omega0
assert ed.lorentz_index(0.5, w0, g, wp).real > 1.0
w = np.linspace(0.2, 3.0, 4000)
absorb = ed.lorentz_index(w, w0, g, wp).imag
assert abs(w[np.argmax(absorb)] - w0) < 0.05                  # absorption line at the resonance

# 2. group velocity < phase velocity in normal dispersion (below resonance)
ws = np.linspace(0.3, 0.8, 400)                               # below resonance, dn/dw > 0
n = ed.lorentz_index(ws, w0, g, wp).real
vg = ed.group_velocity(ws, n); vp = ed.phase_velocity(ws, n)
mid = slice(50, -50)
assert np.all(vg[mid] < vp[mid])                             # envelope slower than the crests
assert np.all(vp[mid] < ed.C)                                 # phase speed below c here (n>1)

# 3. GVD beta_2 is finite and nonzero away from resonance
b2 = ed.gvd_beta2(ws, n)
assert np.all(np.isfinite(b2)) and np.any(np.abs(b2) > 0)

# 4. a transform-limited Gaussian SPREADS under dispersion, more for larger beta2*L
t = np.linspace(-60, 60, 8192); pulse = np.exp(-t**2 / (2 * 2.0**2))
w0_width = ed.pulse_width(t, pulse)
widths = [ed.pulse_width(t, ed.disperse_pulse(pulse, t, b, 1.0)) for b in (0, 20, 80)]
assert np.isclose(widths[0], w0_width)                       # no dispersion -> unchanged
assert widths[0] < widths[1] < widths[2]                     # spreads with beta2*L

# 5. dispersion is UNITARY (energy conserved): |H|=1, so Parseval holds
out = ed.disperse_pulse(pulse, t, 80, 1.0)
assert np.isclose(np.sum(np.abs(out)**2), np.sum(np.abs(pulse)**2), rtol=1e-9)

# 6. THE receiver move: disperse(+L) then (-L) recovers the pulse to machine precision
back = ed.disperse_pulse(ed.disperse_pulse(pulse, t, 80, 1.0), t, -80, 1.0)
assert np.max(np.abs(back - pulse)) < 1e-12                   # exact undisperse (the GS step)

# 7. pulse_width is the INTENSITY rms: a tau=2 amplitude Gaussian -> width tau/sqrt(2)=1.41
assert abs(w0_width - 2.0 / np.sqrt(2)) < 0.05

print(f"TEST PASS  (Lorentz n>1 below resonance, absorption at omega0; v_g<v_p<c; "
      f"GVD finite; Gaussian spreads {widths[0]:.1f}->{widths[2]:.1f}; dispersion unitary; "
      f"disperse +/-L recovers to <1e-12)")
