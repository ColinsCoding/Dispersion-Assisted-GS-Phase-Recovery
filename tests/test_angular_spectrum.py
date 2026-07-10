"""Test dgs.angular_spectrum: the transfer function and its evanescent cutoff, energy-conserving
and reversible propagation, exact Gaussian-beam spreading, Talbot self-imaging, and 4f filtering."""
import sys, pathlib, math
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
from dgs import angular_spectrum as asm

lam = 0.5e-6

# 1. transfer function: unit modulus for propagating waves, decaying for evanescent
fx = np.linspace(-3/lam, 3/lam, 2001)
H = asm.free_space_transfer_function(fx, z=1e-4, lam=lam)
prop = np.abs(lam * fx) < 1
assert np.allclose(np.abs(H[prop]), 1.0, atol=1e-9)             # propagating: |H| = 1
assert np.all(np.abs(H[~prop]) < 1.0 - 1e-9)                    # evanescent: decays
# at z = 0, H is identically 1
assert np.allclose(asm.free_space_transfer_function(fx, 0.0, lam), 1.0)

# 2. propagation is reversible for a band-limited field (z then -z recovers it)
x = asm.make_grid(4096, 0.1e-6)
E0 = asm.gaussian_beam(x, 20e-6)
there = asm.propagate(E0, x[1]-x[0], 2e-3, lam)
back = asm.propagate(there, x[1]-x[0], -2e-3, lam)
assert np.allclose(back, E0, atol=1e-9)

# 3. energy is conserved (|H|=1 on the propagating band, Parseval)
for z in (0.5e-3, 1e-3, 3e-3):
    Ez = asm.propagate(E0, x[1]-x[0], z, lam)
    assert math.isclose(np.sum(np.abs(Ez)**2), np.sum(np.abs(E0)**2), rel_tol=1e-6)

# 4. Gaussian-beam spreading matches w(z) = w0 sqrt(1 + (z/zR)^2) exactly
w0 = 20e-6
zR = asm.rayleigh_range(w0, lam)
assert math.isclose(zR, np.pi*w0**2/lam, rel_tol=1e-12)
assert math.isclose(asm.beam_width(E0, x), w0, rel_tol=1e-3)    # waist at z=0
for zr in (0.5, 1.0, 2.0):
    Ez = asm.propagate(E0, x[1]-x[0], zr*zR, lam)
    assert math.isclose(asm.beam_width(Ez, x), w0*np.sqrt(1+zr**2), rel_tol=5e-3)
# at the Rayleigh range the beam is exactly sqrt(2) wider
EzR = asm.propagate(E0, x[1]-x[0], zR, lam)
assert math.isclose(asm.beam_width(EzR, x), math.sqrt(2)*w0, rel_tol=5e-3)

# 5. Talbot self-imaging: a grating revives at z_T, and better there than at z_T/2
d = 20e-6
zT = asm.talbot_distance(d, lam)
assert math.isclose(zT, 2*d**2/lam, rel_tol=1e-12)
xg = asm.make_grid(4000, 0.05e-6)                              # exactly 10 periods
U0 = asm.ronchi_grating(xg, d)
I0 = np.abs(U0)**2
IT = np.abs(asm.propagate(U0, xg[1]-xg[0], zT, lam))**2
IhalfT = np.abs(asm.propagate(U0, xg[1]-xg[0], zT/2, lam))**2
corr_T = np.corrcoef(IT, I0)[0, 1]
corr_half = np.corrcoef(IhalfT, I0)[0, 1]
assert corr_T > 0.85, f"Talbot self-image correlation {corr_T}"
assert corr_T > corr_half                                      # self-image cleaner than half-Talbot
# half-Talbot is the input shifted by half a period
shift = int(round((d/2) / (xg[1]-xg[0])))
assert np.corrcoef(IhalfT, np.roll(I0, shift))[0, 1] > 0.85

# 6. evanescent cutoff: sub-wavelength structure cannot propagate (detail is lost)
xf = asm.make_grid(2048, 0.02e-6)
fine = asm.ronchi_grating(xf, 0.3*lam)                         # period < wavelength
out = asm.propagate(fine, xf[1]-xf[0], 5e-6, lam)
# the modulation depth collapses toward a uniform field
assert np.std(np.abs(out)**2) < 0.15 * np.std(np.abs(fine)**2)

# 7. 4f spatial filtering: low-pass smooths a step, high-pass isolates the edge
xs = asm.make_grid(1024, 1e-6)
step = (xs > 0).astype(complex)
lp = asm.spatial_filter_4f(step, xs[1]-xs[0], 2e4, "lowpass")
hp = asm.spatial_filter_4f(step, xs[1]-xs[0], 2e4, "highpass")
# low-pass keeps the average level and softens the edge
assert abs(np.mean(lp.real) - 0.5) < 0.05
assert np.max(np.abs(np.diff(lp.real))) < 0.2                  # edge slope softened
# high-pass removes the DC level; its steepest variation sits at the edge (x = 0)
assert abs(np.mean(hp.real)) < 0.05
edge_idx = np.argmax(np.abs(np.diff(hp.real)))
assert abs(xs[edge_idx]) < 5e-6

# 8. kwarg bounds
for bad in (lambda: asm.propagate(E0, 0, 1e-3, lam),
            lambda: asm.propagate(E0, 1e-7, 1e-3, 0),
            lambda: asm.rayleigh_range(0, lam),
            lambda: asm.talbot_distance(d, 0),
            lambda: asm.gaussian_beam(x, -1)):
    try:
        bad(); assert False
    except ValueError:
        pass

print("test_angular_spectrum: all checks passed")
