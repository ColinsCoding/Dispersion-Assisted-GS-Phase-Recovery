"""Fourier Ptychographic Microscopy (FPM): a REAL, published technique for
reconstructing a sharper (higher-resolution) microscope image WITHOUT
shrinking the field of view -- the direct answer to "can a microscope
reconstruct sharper images?"

THE FOV/RESOLUTION TRADEOFF THIS BREAKS:
  A conventional microscope objective has one fixed numerical aperture NA.
  Higher NA -> finer resolution (Abbe limit: resolution = lambda/(2*NA))
  but higher-NA objectives are physically designed with a SMALLER field of
  view -- you trade sharpness for how much you can see at once. This is the
  space-bandwidth product (SBP) limit of a single lens.

HOW FPM BREAKS IT (real algorithm, not proposed-but-unverified like
dgs/steam_3d_depth_encoding.py):
  Illuminate the sample from many different angles (an LED array below the
  sample). Each oblique illumination angle shifts a DIFFERENT patch of the
  sample's spatial-frequency spectrum into the objective's fixed NA_obj
  passband -- exactly like synthetic-aperture radar shifts what an antenna
  "sees". Capture one low-resolution intensity image per angle (using the
  SAME low-NA, large-FOV objective every time -- no scanning, no moving
  parts). An iterative phase-retrieval algorithm then stitches these
  overlapping, angle-shifted sub-apertures together in Fourier space,
  synthesizing an effective NA_synthetic = NA_obj + NA_illum_max that can be
  several times the objective's own NA -- while every capture kept the
  objective's original, large field of view. Resolution goes up; FOV does
  not go down.

RELATION TO THIS REPO'S EXISTING GS ENGINE (dgs/gs_core.py):
  Both are alternating-projections phase retrieval: intensity is measured,
  phase is lost, and the algorithm alternates between enforcing measured
  amplitudes and propagating via a known linear operator until it converges
  on a self-consistent complex field (dgs/gs_core.py's
  apply_amplitude_constraint is the exact same operation used here in
  update_subaperture). The two differ in what plane is stitched:
    - dgs/gs_core.py (Solli, Gupta, Jalali, APL 2009): TWO measurements at
      different TEMPORAL dispersions D1, D2, same full spectrum each time.
    - This module (Zheng, Horstmeyer, Yang, "Wide-field, high-resolution
      Fourier ptychographic microscopy," Nature Photonics 7, 739-745,
      2013): MANY measurements at different illumination ANGLES, each
      seeing a different spatial-frequency PATCH through a fixed aperture.
  Different papers, different physical setups -- same alternating-
  projections lineage (Gerchberg-Saxton 1972 / Fienup 1982).

SIMPLIFICATION stated honestly: captures are simulated on the SAME pixel
grid as the high-resolution object (no explicit downsampling to a coarser
low-NA camera grid). This is a common simulation shortcut in FPM teaching
demos -- it assumes the low-res camera oversamples relative to its own
NA-limited resolution. Real hardware would additionally resample each
capture to a coarser native pixel pitch before reconstruction.
"""
import numpy as np


# ── Resolution / space-bandwidth-product formulas ─────────────────────────────

def resolution_half_pitch_nm(wavelength_nm, NA):
    """Abbe diffraction limit: finest resolvable half-pitch = lambda / (2*NA).
    Standard result (Abbe 1873); NOT specific to FPM -- this is what ANY
    conventional objective of that NA can resolve on its own."""
    if wavelength_nm <= 0:
        raise ValueError("wavelength_nm must be positive")
    if NA <= 0:
        raise ValueError("NA must be positive")
    return wavelength_nm / (2.0 * NA)


def synthetic_NA(NA_obj, NA_illum_max):
    """Effective numerical aperture FPM synthesizes: the objective's own
    NA_obj plus the maximum illumination-angle NA_illum_max reachable by
    the LED array (sin of the steepest illumination angle used). This sum
    is the real FPM result (Zheng et al. 2013 eq. 1-2) -- resolution scales
    with NA_obj + NA_illum_max, not NA_obj alone."""
    if NA_obj <= 0 or NA_illum_max < 0:
        raise ValueError("NA_obj must be positive, NA_illum_max must be non-negative")
    NA_syn = NA_obj + NA_illum_max
    if NA_syn > 1.0:
        raise ValueError(
            f"synthetic NA {NA_syn:.3f} exceeds 1.0 -- unphysical for the "
            "unmodified formula (would require an immersion medium n>1 "
            "to be physically realizable; not handled by this function)")
    return NA_syn


def space_bandwidth_product(fov_um, resolution_um):
    """SBP = (FOV / resolution)^2 -- total number of independently
    resolvable points in a 2D image. This is the quantity a conventional
    single-NA objective trades off (higher NA -> finer resolution but
    smaller FOV -> similar SBP); FPM's point is to grow SBP itself by
    keeping FOV fixed (large, low-NA objective) while shrinking resolution
    (via synthesized high NA)."""
    if fov_um <= 0 or resolution_um <= 0:
        raise ValueError("fov_um and resolution_um must be positive")
    return (fov_um / resolution_um) ** 2


def led_array_illumination_NA(n_leds_per_side, led_spacing_mm, array_height_mm):
    """Real LED-array FPM hardware geometry: a matrix of LEDs sits a known
    height below the sample; illumination NA from an LED at radial distance
    r from the center is NA_illum = sin(atan(r / height)) -- basic
    trigonometry, not FPM-specific, but this is exactly how NA_illum_max in
    synthetic_NA() is set by real hardware design choices (LED spacing and
    array height trade off angular coverage vs. angular resolution).

    Returns
    -------
    dict with 'positions_mm' (Ny, Nx, 2) grid of (x, y) LED offsets, and
    'NA_illum' (Ny, Nx) array of the corresponding illumination NA per LED.
    """
    if n_leds_per_side < 1 or n_leds_per_side % 2 == 0:
        raise ValueError("n_leds_per_side must be a positive odd integer (centered LED)")
    if led_spacing_mm <= 0 or array_height_mm <= 0:
        raise ValueError("led_spacing_mm and array_height_mm must be positive")

    half = n_leds_per_side // 2
    idx = np.arange(-half, half + 1)
    x_mm, y_mm = np.meshgrid(idx * led_spacing_mm, idx * led_spacing_mm)
    r_mm = np.sqrt(x_mm**2 + y_mm**2)
    NA_illum = np.sin(np.arctan2(r_mm, array_height_mm))

    positions_mm = np.stack([x_mm, y_mm], axis=-1)
    return {"positions_mm": positions_mm, "NA_illum": NA_illum, "r_mm": r_mm}


# ── Forward model + reconstruction ────────────────────────────────────────────

def _circular_pupil_mask(shape, radius_px, center_px):
    """Binary circular low-pass mask of given pixel radius, centered at
    center_px = (cy, cx) -- represents the objective's NA_obj passband
    shifted to where a given illumination angle puts it in Fourier space."""
    ny, nx = shape
    yy, xx = np.mgrid[0:ny, 0:nx]
    cy, cx = center_px
    r2 = (yy - cy) ** 2 + (xx - cx) ** 2
    return r2 <= radius_px ** 2


def _na_to_pixel_shift(NA_illum_x, NA_illum_y, wavelength_nm, pixel_pitch_nm, shape):
    """Convert an illumination NA (sin of angle) to a Fourier-space pixel
    shift: spatial frequency shift = NA / lambda (cycles per unit length),
    converted to FFT bin units via the real-space pixel pitch and array
    size (standard DFT frequency-to-pixel relation, ν_bin = ν_physical *
    N * pixel_pitch)."""
    ny, nx = shape
    shift_x = NA_illum_x / wavelength_nm * nx * pixel_pitch_nm
    shift_y = NA_illum_y / wavelength_nm * ny * pixel_pitch_nm
    return shift_y, shift_x


def simulate_fpm_captures(object_complex, NA_obj, wavelength_nm, pixel_pitch_nm,
                          led_positions_NA):
    """Forward model: given a ground-truth high-resolution complex object
    and a set of illumination angles (as NA_illum_x, NA_illum_y pairs),
    generate the low-resolution intensity capture each angle would produce
    through a fixed NA_obj objective.

    Parameters
    ----------
    object_complex   : complex 2D array (Ny, Nx) -- ground-truth high-res field
    NA_obj            : float -- objective's own numerical aperture
    wavelength_nm     : float
    pixel_pitch_nm    : float -- real-space sample pitch of object_complex
    led_positions_NA  : list of (NA_illum_x, NA_illum_y) tuples

    Returns
    -------
    list of (capture_intensity, pixel_shift) -- one per illumination angle;
    pixel_shift is the (dy, dx) Fourier-space shift used, needed again at
    reconstruction time.
    """
    obj = np.asarray(object_complex, complex)
    if obj.ndim != 2:
        raise ValueError("object_complex must be a 2D array")
    if NA_obj <= 0:
        raise ValueError("NA_obj must be positive")

    shape = obj.shape
    O_hat_true = np.fft.fftshift(np.fft.fft2(obj))
    ny, nx = shape
    radius_px = NA_obj / wavelength_nm * min(ny, nx) * pixel_pitch_nm
    center0 = (ny // 2, nx // 2)

    captures = []
    for NA_x, NA_y in led_positions_NA:
        dy, dx = _na_to_pixel_shift(NA_x, NA_y, wavelength_nm, pixel_pitch_nm, shape)
        center = (center0[0] + dy, center0[1] + dx)
        mask = _circular_pupil_mask(shape, radius_px, center)
        sub_spectrum = O_hat_true * mask
        field = np.fft.ifft2(np.fft.ifftshift(sub_spectrum))
        intensity = np.abs(field) ** 2
        captures.append((intensity, (dy, dx)))
    return captures


def reconstruct_fpm(captures, NA_obj, wavelength_nm, pixel_pitch_nm, shape, n_iter=10):
    """Iterative FPM reconstruction: alternating-projections phase
    retrieval that stitches the angle-shifted low-res captures into one
    high-resolution complex spectrum, exactly the amplitude-constraint
    projection used in dgs/gs_core.py's apply_amplitude_constraint, applied
    per sub-aperture instead of per temporal-dispersion plane.

    Returns
    -------
    dict with 'object_recovered' (complex 2D array, high-res reconstruction)
    and 'convergence' (list of per-iteration mean sub-aperture amplitude error).
    """
    if n_iter < 1:
        raise ValueError("n_iter must be >= 1")
    ny, nx = shape
    radius_px = NA_obj / wavelength_nm * min(ny, nx) * pixel_pitch_nm
    center0 = (ny // 2, nx // 2)

    # Initial guess: zero everywhere in Fourier space. Each sub-aperture
    # update below only ever writes inside its own mask, so any spectrum
    # location never covered by an LED's passband correctly stays zero
    # (no signal claimed beyond the synthesized aperture) instead of
    # carrying over spurious full-spectrum content from a naive image guess.
    O_hat = np.zeros(shape, complex)

    convergence = []
    for _ in range(n_iter):
        errs = []
        for intensity_meas, (dy, dx) in captures:
            center = (center0[0] + dy, center0[1] + dx)
            mask = _circular_pupil_mask(shape, radius_px, center)

            sub_spectrum = O_hat * mask
            field_est = np.fft.ifft2(np.fft.ifftshift(sub_spectrum))

            amp_target = np.sqrt(np.maximum(intensity_meas, 0.0))
            err = float(np.sqrt(np.mean((np.abs(field_est) - amp_target) ** 2)))
            errs.append(err)

            field_updated = amp_target * np.exp(1j * np.angle(field_est))
            sub_spectrum_updated = np.fft.fftshift(np.fft.fft2(field_updated))

            O_hat = np.where(mask, sub_spectrum_updated, O_hat)
        convergence.append(float(np.mean(errs)))

    object_recovered = np.fft.ifft2(np.fft.ifftshift(O_hat))
    return {"object_recovered": object_recovered, "convergence": convergence}


if __name__ == "__main__":
    print("=== Fourier Ptychographic Microscopy: sharper images, same FOV ===\n")

    wavelength_nm = 500.0
    NA_obj = 0.1
    res_obj_nm = resolution_half_pitch_nm(wavelength_nm, NA_obj)
    print(f"Objective alone: NA={NA_obj}, resolution = {res_obj_nm:.0f} nm half-pitch")

    led = led_array_illumination_NA(n_leds_per_side=5, led_spacing_mm=4.0, array_height_mm=60.0)
    NA_illum_max = float(led["NA_illum"].max())
    NA_syn = synthetic_NA(NA_obj, NA_illum_max)
    res_syn_nm = resolution_half_pitch_nm(wavelength_nm, NA_syn)
    print(f"LED array max illumination NA: {NA_illum_max:.3f}")
    print(f"Synthesized NA: {NA_syn:.3f} -> resolution = {res_syn_nm:.0f} nm half-pitch "
          f"({res_obj_nm/res_syn_nm:.1f}x sharper, SAME field of view)\n")

    fov_um = 500.0
    sbp_obj = space_bandwidth_product(fov_um, res_obj_nm / 1000.0)
    sbp_syn = space_bandwidth_product(fov_um, res_syn_nm / 1000.0)
    print(f"Space-bandwidth product: {sbp_obj:.2e} (objective alone) -> "
          f"{sbp_syn:.2e} (FPM synthesized), {sbp_syn/sbp_obj:.1f}x more resolvable points\n")

    # Small synthetic reconstruction demo. pixel_pitch_nm here is chosen large
    # enough that the objective's pupil mask spans several pixels in this
    # small N=64 grid -- with too few pixels per pupil, the FFT-based
    # reconstruction is dominated by discretization error rather than the
    # actual FPM physics.
    N = 64
    pixel_pitch_nm = 800.0
    yy, xx = np.mgrid[0:N, 0:N]
    obj = np.ones((N, N), complex)
    # two point-like phase features closer together than the objective alone can resolve
    for cy, cx in [(30, 28), (30, 36)]:
        obj *= np.exp(1j * 0.8 * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * 1.5 ** 2)))

    led_positions_NA = [
        (led["positions_mm"][i, j, 0] / led["r_mm"][i, j] * led["NA_illum"][i, j] if led["r_mm"][i, j] > 0 else 0.0,
         led["positions_mm"][i, j, 1] / led["r_mm"][i, j] * led["NA_illum"][i, j] if led["r_mm"][i, j] > 0 else 0.0)
        for i in range(led["NA_illum"].shape[0]) for j in range(led["NA_illum"].shape[1])
    ]

    n_iter = 40
    captures = simulate_fpm_captures(obj, NA_obj, wavelength_nm, pixel_pitch_nm, led_positions_NA)
    result = reconstruct_fpm(captures, NA_obj, wavelength_nm, pixel_pitch_nm, obj.shape, n_iter=n_iter)

    phase_true = np.angle(obj)
    phase_rec = np.angle(result["object_recovered"])
    corr = np.corrcoef(phase_true.ravel(), phase_rec.ravel())[0, 1]
    print(f"Synthetic reconstruction (two close phase features, N={N}x{N}, "
          f"{len(captures)} illumination angles, {n_iter} iterations):")
    print(f"  phase correlation to ground truth: {corr:.4f}")
    print(f"  final sub-aperture amplitude error: {result['convergence'][-1]:.4f}")
