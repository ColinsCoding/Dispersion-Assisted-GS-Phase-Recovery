"""Test the uncertainty principle: time-bandwidth product, chirp, energy-time."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
from dgs import uncertainty as un

t = np.linspace(-50, 50, 8192)
g = un.gaussian_pulse(t, 2.0)

# 1. a transform-limited Gaussian SATURATES the bound: Dt * Dw = 1/2
assert abs(un.time_bandwidth_product(t, g) - 0.5) < 0.02

# 2. chirp leaves the intensity (Dt) unchanged but widens the spectrum (Dw)
gc = un.gaussian_pulse(t, 2.0, chirp=0.3)
assert abs(un.rms_width(t, np.abs(g)**2) - un.rms_width(t, np.abs(gc)**2)) < 1e-6
assert un.time_bandwidth_product(t, gc) > un.time_bandwidth_product(t, g) + 0.2

# 3. shorter pulse -> proportionally broader spectrum (Dw ~ 1/tau)
def dw(tau):
    w, F = un.spectrum(t, un.gaussian_pulse(t, tau))
    return un.rms_width(w, np.abs(F)**2)
assert abs(dw(1.0) / dw(2.0) - 2.0) < 0.05            # halving tau doubles Dw
# and the product stays at the 0.5 floor at every tau
for tau in (1.0, 2.0, 4.0):
    assert abs(un.time_bandwidth_product(t, un.gaussian_pulse(t, tau)) - 0.5) < 0.02

# 4. a non-Gaussian pulse (hard edges) EXCEEDS the Gaussian minimum
assert un.time_bandwidth_product(t, (np.abs(t) < 2.0).astype(float)) > 0.5

# 5. energy-time: the Gaussian gives Delta_E * Delta_t = hbar/2
assert abs(un.energy_time_product(t, g) - un.HBAR / 2) < 0.02 * un.HBAR

# 6. rms_width sanity: the rms of a Gaussian density of sigma is sigma
xx = np.linspace(-40, 40, 8000); sigma = 3.0
assert abs(un.rms_width(xx, np.exp(-xx**2 / (2 * sigma**2))) - sigma) < 0.01

print(f"TEST PASS  (Gaussian Dt*Dw=0.5 floor at all tau; chirp raises it "
      f"({un.time_bandwidth_product(t,gc):.2f}) with Dt fixed; Dw~1/tau; "
      f"rect>0.5; energy-time = hbar/2; rms ok)")
