"""Phase-Stretch Transform (PST): the physics-inspired edge-detection kernel
behind Jalali's PhyCV library. It's the SAME operation as dgs.gs_core.disperse
and dgs.em_dispersion.disperse_pulse -- multiply a signal's spectrum by a
phase-only kernel, then transform back -- just 2D (an image, not a pulse) and
with a bounded, nonlinear (arctan) phase profile instead of the unbounded
quadratic H(f)=exp(i*pi*D*f^2) used elsewhere in this repo.

WHY ARCTAN INSTEAD OF QUADRATIC:
The quadratic dispersion phase phi(f) = pi*D*f^2 GROWS without bound as
f -> infinity -- fine for a 1D pulse where you control the bandwidth, bad
for a 2D image's raw pixel-noise frequencies, which would get arbitrarily
large phase kicks and just amplify noise. PST's kernel phi(fr) = S*atan(W*fr)
saturates: high spatial frequencies (real edges AND noise) all get pushed
toward the same phase (+-S*pi/2), so edges get consistently enhanced without
the quadratic kernel's runaway noise amplification.

WATER / ATMOSPHERE PHOTONICS: turbid water and atmospheric turbulence both
blur/scatter high-spatial-frequency image content (edges) -- exactly the
frequency band this kernel selectively phase-shifts. That's the real
motivation for PhyCV's use in underwater and through-turbulence imaging,
not a coincidence of naming.

NumPy only (no scipy dependency on py-3.13). Education.
"""

import numpy as np


def pst_kernel(fr, S, W):
    """Radially symmetric, BOUNDED nonlinear phase kernel phi(fr) = S*arctan(W*fr),
    fr = normalized radial spatial frequency (sqrt(fx^2+fy^2), typically in [0,1]
    after normalizing by the Nyquist frequency). S sets overall phase strength
    (edge contrast), W sets how quickly the kernel saturates with frequency
    (higher W -> even low/mid frequencies get phase-shifted strongly)."""
    if S == 0:
        raise ValueError("S (phase strength) must be nonzero to have any effect")
    if W <= 0:
        raise ValueError(f"W (warp strength) must be positive, got {W}")
    return S * np.arctan(W * fr)


def gaussian_lpf(fr, sigma):
    """Gaussian low-pass filter in the frequency domain, applied BEFORE the
    phase kernel to keep pixel-level noise (highest frequencies) from
    dominating the phase-stretched output -- same role as limiting a
    dispersive pulse's input bandwidth elsewhere in this repo."""
    if sigma <= 0:
        raise ValueError(f"sigma must be positive, got {sigma}")
    return np.exp(-(fr ** 2) / (2 * sigma ** 2))


def _radial_frequency_grid(shape):
    """Normalized radial spatial frequency grid (0 at DC, 1 at the highest
    representable frequency on the smaller axis) for an image of the given
    (ny, nx) shape."""
    ny, nx = shape
    fy = np.fft.fftfreq(ny)
    fx = np.fft.fftfreq(nx)
    FX, FY = np.meshgrid(fx, fy)
    fr = np.sqrt(FX ** 2 + FY ** 2)
    return fr / fr.max()


def apply_pst(image, S=1.0, W=15.0, sigma_lpf=0.3):
    """Full PST pipeline: Gaussian LPF -> phase-stretch kernel -> inverse FFT
    -> take the phase AND the amplitude. Returns (phase, amplitude).

    IMPORTANT: the phase alone is only meaningful where the amplitude is
    non-negligible -- angle() of a near-zero complex number is numerically
    ill-conditioned (dominated by rounding noise, not signal), which is
    exactly what happens over a flat, featureless region with no spatial-
    frequency content for the kernel to act on. Real PST/PhyCV
    implementations gate on amplitude (or apply morphology) for this same
    reason; use amplitude to build a validity mask before thresholding phase
    for edges, not the phase alone."""
    image = np.asarray(image, dtype=float)
    fr = _radial_frequency_grid(image.shape)

    I_f = np.fft.fft2(image)
    I_f_filtered = I_f * gaussian_lpf(fr, sigma_lpf)

    phase_kernel = pst_kernel(fr, S, W)
    I_f_stretched = I_f_filtered * np.exp(1j * phase_kernel)

    I_stretched = np.fft.ifft2(I_f_stretched)
    return np.angle(I_stretched), np.abs(I_stretched)


def edge_response_strength(phase_image, region_mask, amplitude=None, amp_floor=0.1):
    """RMS of the stretched-phase image over a given boolean mask. If
    `amplitude` is given, pixels with amplitude below `amp_floor` (relative
    to the image's own max amplitude) are EXCLUDED -- their phase is noise,
    not signal, and including them would make a flat/featureless region look
    spuriously 'high response' (verified: this is a real effect, not a
    hypothetical edge case)."""
    if amplitude is not None:
        valid = amplitude[region_mask] >= amp_floor * amplitude.max()
        region = phase_image[region_mask][valid]
        if region.size == 0:
            raise ValueError("no pixels in region_mask pass the amplitude floor "
                              "-- region has essentially no signal content")
    else:
        region = phase_image[region_mask]
        if region.size == 0:
            raise ValueError("region_mask selects no pixels")
    return float(np.sqrt(np.mean(region ** 2)))


if __name__ == "__main__":
    # synthetic test image: a bright square on a dark background --
    # unambiguous edges at the square's boundary, flat/uniform elsewhere
    N = 128
    image = np.zeros((N, N))
    image[40:88, 40:88] = 1.0

    phase_out, amp_out = apply_pst(image, S=2.0, W=20.0, sigma_lpf=0.4)

    # edge mask: a thin ring right at the square's boundary
    edge_mask = np.zeros((N, N), dtype=bool)
    edge_mask[38:42, 38:90] = True   # top edge band
    edge_mask[86:90, 38:90] = True   # bottom edge band

    # smooth interior masks: well inside the square, and well inside the background
    interior_mask = np.zeros((N, N), dtype=bool)
    interior_mask[55:75, 55:75] = True
    background_mask = np.zeros((N, N), dtype=bool)
    background_mask[5:20, 5:20] = True

    print("amplitude check (why the amplitude floor matters):")
    print(f"  edge amplitude mean:       {amp_out[edge_mask].mean():.4f}")
    print(f"  interior amplitude mean:   {amp_out[interior_mask].mean():.4f}")
    print(f"  background amplitude mean: {amp_out[background_mask].mean():.4f}  "
          f"(near zero -- its phase would be pure noise if not excluded)")

    edge_strength = edge_response_strength(phase_out, edge_mask, amp_out)
    interior_strength = edge_response_strength(phase_out, interior_mask, amp_out)
    try:
        background_strength = edge_response_strength(phase_out, background_mask, amp_out)
    except ValueError as e:
        background_strength = None
        print(f"\nbackground region: {e}")

    print(f"\nPST phase response RMS at edge (amplitude-gated):     {edge_strength:.4f}")
    print(f"PST phase response RMS in flat interior (amplitude-gated): {interior_strength:.4f}")
    print(f"\nedge/interior ratio: {edge_strength/interior_strength:.2f}x "
          f"(should be > 1: PST responds more strongly at edges than in smooth regions)")
