"""Angular spectrum propagation: Fourier optics as multiply-by-a-transfer-function.

A light field is a superposition of plane waves -- its ANGULAR SPECTRUM is just its spatial
Fourier transform, A(f_x) = FT{U(x)}. Each plane-wave component travels a distance z by picking up
a phase exp(i k_z z) with k_z = (2 pi/lambda) sqrt(1 - (lambda f_x)^2). So free-space propagation is
one line of Fourier optics:

    U(x, z) = IFT{ FT{U(x,0)} * H(f_x) },   H(f_x) = exp( i (2 pi/lambda) z sqrt(1 - (lambda f_x)^2) ).

This is EXACT (no paraxial approximation) and contains everything: near-field, far-field, and the
cutoff at |lambda f_x| = 1 where k_z turns imaginary and the wave becomes EVANESCENT -- decaying,
not propagating. That cutoff is the diffraction limit: features finer than ~lambda can't radiate to
the far field, which is why a lens can't resolve below ~lambda/2.

Two exact predictions fall straight out and are checked here:
  * GAUSSIAN BEAM spreading: a waist w0 grows as w(z) = w0 sqrt(1 + (z/z_R)^2), z_R = pi w0^2/lambda.
  * TALBOT self-imaging: a periodic grating of period d reproduces itself at z_T = 2 d^2/lambda,
    and appears half-period-shifted at z_T/2 -- diffraction with no lens at all.

The same FT-in / filter / FT-out idea is optical COMPUTING: a lens performs the Fourier transform,
a stop in the Fourier plane filters spatial frequencies (spatial_filter_4f), and a second lens
inverts it -- the 4f processor. This is the spatial-domain twin of the dispersive Fourier transform
(dgs.dispersive_fourier / time_stretch_dft) this repo runs in TIME. NumPy; SI units. py-3.13.
"""

import numpy as np


def free_space_transfer_function(fx, z, lam):
    """Angular-spectrum transfer function H(f_x) = exp(i (2 pi/lambda) z sqrt(1-(lambda f_x)^2)).
    |H| = 1 for propagating components (|lambda f_x| < 1) and < 1 (evanescent decay) beyond."""
    if lam <= 0:
        raise ValueError("wavelength must be > 0")
    k = 2 * np.pi / lam
    root = np.sqrt((1 - (lam * np.asarray(fx, float)) ** 2).astype(complex))
    arg = 1j * k * z * root
    # band-limited ASM: evanescent components may only DECAY, never grow (|H| <= 1),
    # so back-propagation stays numerically stable instead of overflowing.
    decay = np.minimum(np.real(arg), 0.0)
    return np.exp(decay) * np.exp(1j * np.imag(arg))


def propagate(field, dx, z, lam):
    """Propagate a 1-D complex field a distance z by the angular spectrum method (exact, valid
    near and far field). dx is the sample spacing."""
    if dx <= 0 or lam <= 0:
        raise ValueError("dx and lam must be > 0")
    field = np.asarray(field, complex)
    fx = np.fft.fftfreq(len(field), d=dx)
    H = free_space_transfer_function(fx, z, lam)
    return np.fft.ifft(np.fft.fft(field) * H)


def make_grid(n, dx):
    """Centered coordinate grid of n points spaced dx."""
    return (np.arange(n) - n // 2) * dx


def gaussian_beam(x, w0):
    """Gaussian field of waist w0: E(x) = exp(-x^2/w0^2) (intensity 1/e^2 radius w0)."""
    if w0 <= 0:
        raise ValueError("w0 must be > 0")
    return np.exp(-np.asarray(x, float) ** 2 / w0 ** 2).astype(complex)


def rayleigh_range(w0, lam):
    """Rayleigh range z_R = pi w0^2 / lambda -- where the beam has spread to sqrt(2) * w0."""
    if w0 <= 0 or lam <= 0:
        raise ValueError("w0 and lam must be > 0")
    return np.pi * w0 ** 2 / lam


def beam_width(field, x):
    """1/e^2 intensity radius w = 2 * RMS width of |field|^2 (for a Gaussian, RMS = w/2)."""
    I = np.abs(field) ** 2
    xbar = np.sum(x * I) / np.sum(I)
    var = np.sum((x - xbar) ** 2 * I) / np.sum(I)
    return 2.0 * np.sqrt(var)


def talbot_distance(period, lam):
    """Talbot distance z_T = 2 d^2 / lambda: a periodic field self-images here (and appears
    shifted by half a period at z_T/2), with no lens."""
    if period <= 0 or lam <= 0:
        raise ValueError("period and lam must be > 0")
    return 2 * period ** 2 / lam


def ronchi_grating(x, period, duty=0.5):
    """Binary (0/1) amplitude grating of the given period and duty cycle."""
    phase = np.mod(np.asarray(x, float), period) / period
    return (phase < duty).astype(complex)


def spatial_filter_4f(field, dx, cutoff_fx, mode="lowpass"):
    """A 4f optical processor: Fourier-transform the field (lens 1), keep or block spatial
    frequencies in the Fourier plane, and inverse-transform (lens 2). 'lowpass' passes
    |f_x| <= cutoff (blurs), 'highpass' passes |f_x| > cutoff (edge-enhances)."""
    if dx <= 0 or cutoff_fx < 0:
        raise ValueError("dx > 0 and cutoff_fx >= 0")
    field = np.asarray(field, complex)
    fx = np.fft.fftfreq(len(field), d=dx)
    F = np.fft.fft(field)
    passband = np.abs(fx) <= cutoff_fx if mode == "lowpass" else np.abs(fx) > cutoff_fx
    return np.fft.ifft(F * passband)


if __name__ == "__main__":
    lam = 0.5e-6
    print("=== Gaussian beam spreading (waist w0 = 20 um) ===")
    w0 = 20e-6
    zR = rayleigh_range(w0, lam)
    x = make_grid(8192, 0.1e-6)
    E0 = gaussian_beam(x, w0)
    print(f"  Rayleigh range z_R = {zR*1e3:.3f} mm")
    print("   z/z_R   w(z) measured   w0*sqrt(1+(z/zR)^2)")
    for zr in (0.0, 1.0, 2.0):
        Ez = propagate(E0, x[1] - x[0], zr * zR, lam)
        print(f"    {zr:.1f}     {beam_width(Ez, x)*1e6:7.2f} um     "
              f"{w0*np.sqrt(1+zr**2)*1e6:7.2f} um")

    print("\n=== Talbot self-imaging (grating period d = 20 um) ===")
    d = 20e-6
    zT = talbot_distance(d, lam)
    xg = make_grid(4000, 0.05e-6)                       # 10 periods
    U0 = ronchi_grating(xg, d)
    IT = np.abs(propagate(U0, xg[1] - xg[0], zT, lam)) ** 2
    I0 = np.abs(U0) ** 2
    corr = np.corrcoef(IT, I0)[0, 1]
    print(f"  Talbot distance z_T = {zT*1e3:.3f} mm")
    print(f"  correlation of intensity at z_T with the input grating: {corr:.3f} (self-image)")

    print("\n=== 4f spatial filtering (a step edge) ===")
    xs = make_grid(1024, 1e-6)
    step = (xs > 0).astype(complex)
    lp = spatial_filter_4f(step, xs[1] - xs[0], 2e4, "lowpass")
    hp = spatial_filter_4f(step, xs[1] - xs[0], 2e4, "highpass")
    print(f"  low-pass smooths the edge (max slope {np.max(np.abs(np.diff(lp.real))):.3f} "
          f"< step 1.0);  high-pass peaks at the edge (|value| max {np.max(np.abs(hp)):.3f})")
