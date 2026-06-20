"""Smoke-test the GPU Dirac-delta visualization (torch, py-3.12)."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from dgs import delta_cuda as dc

dev = dc._device()
print("device:", dev)

# 1. Fourier delta: REAL part = sin(Kx)/(pi x); IMAG part ~ 0 (cancels)
x = np.linspace(-3, 3, 1601)
K = 60.0
d = dc.fourier_delta(x, K, n_k=6000)
real = d.real.numpy(); imag = d.imag.numpy()
analytic = (K / np.pi) * np.sinc(K * x / np.pi)           # = sin(Kx)/(pi x), clean at x=0
# abs error ~3e-3 vs a peak of K/pi~19 (Riemann-sum truncation): <0.02%, physically exact
assert np.allclose(real, analytic, atol=2e-2), np.abs(real - analytic).max()
assert np.abs(imag).max() < 1e-9, np.abs(imag).max()      # imaginary cancels

# 2. peak at x=0 grows like K/pi (taller, narrower -> approaches delta).
# the discrete sum gives K/pi * n_k/(n_k-1) (endpoints), so compare relatively.
for K in (20.0, 50.0, 100.0):
    d0 = dc.fourier_delta(np.array([0.0]), K).real.item()
    assert abs(d0 / (K / np.pi) - 1) < 2e-3, (K, d0, K / np.pi)

# 3. nascent deltas integrate to 1 (unit area) and sharpen as eps -> 0
xx = np.linspace(-50, 50, 200001)
for eps in (0.5, 0.1):
    g = dc.gaussian_delta(xx, eps).numpy()
    lo = dc.lorentzian_delta(xx, eps).numpy()
    assert abs(np.trapezoid(g, xx) - 1) < 1e-3, eps
    assert abs(np.trapezoid(lo, xx) - 1) < 2e-2, eps        # Lorentzian has fat tails
assert dc.gaussian_delta(np.array([0.0]), 0.1).item() > dc.gaussian_delta(np.array([0.0]), 0.5).item()

# 4. device round-trips to CPU as a real/complex tensor
assert dc.fourier_delta(x, 30.0).is_cpu
assert torch.is_complex(dc.fourier_delta(x, 30.0))

# 5. validation
for bad in (lambda: dc.fourier_delta(x, -1.0),
            lambda: dc.gaussian_delta(x, 0.0),
            lambda: dc.lorentzian_delta(x, -1.0)):
    try:
        bad()
    except ValueError:
        pass
    else:
        raise AssertionError("should reject bad parameter")

# 6. the figure renders
out = dc.render(out="figures/_smoke_delta.png")
assert pathlib.Path(out).exists()
pathlib.Path(out).unlink()

print(f"SMOKE PASS  (real matches sin(Kx)/(pi*x); max|Im|={np.abs(imag).max():.1e}; device={dev})")
