"""PST as a VECTOR field, applied to a synthetic medical (retinal fundus)
image -- the Jalali Lab's "spatial computing" framing of PST made literal.

dgs.pst.pst() returns a SCALAR phase-edge map, angle(radians) per pixel --
useful for "is this an edge," but it throws away edge ORIENTATION. PST is
built on the same "multiply by a frequency-dependent phase, apply, look at
the phase" move as this repo's dispersion receiver (see dgs.pst's own
docstring); here that scalar phase map is differentiated spatially to
recover a genuine 2-D VECTOR field (magnitude + direction) -- the same
"phase field -> vector field via a spatial derivative" move used elsewhere
in optics (e.g. wavefront slope from a phase map), applied to PST's output
instead of a lens's.

Wrap-safe: PST's phase output is an ANGLE (periodic mod 2*pi), so a naive
finite difference of adjacent pixel values is wrong at a wrap boundary
(e.g. +3.13 next to -3.13 looks like a huge jump, not the tiny true
difference). The gradient here uses the standard wrap-safe form
    d/dx phi  ~=  angle(exp(i*phi[x+1]) / exp(i*phi[x-1])) / 2
i.e. differences are taken in the complex exponential and unwrapped via
`np.angle`, not on the raw radian values.

Applied to `synthetic_fundus_image` -- a synthetic retinal-fundus-style
test image (bright optic disk, dark branching vessels) standing in for the
"medical imaging" use case dgs.pst's own docstring already names as one of
the Jalali Lab's real PST applications -- with a numeric check that the
vector field's MAGNITUDE really is concentrated on the vessels, not
assumed.
"""

import numpy as np

from dgs.pst import pst as pst_phase_map


def synthetic_fundus_image(size: int = 128, n_vessels: int = 6, seed: int = 0) -> dict:
    """A synthetic retinal-fundus-style test image: a bright circular
    "optic disk" on a mid-gray background, with `n_vessels` dark branching
    vessel-like segments (short connected line chains, Gaussian-blurred to
    look vascular rather than drafted). NOT real fundus photography --
    a stand-in test pattern, same honesty posture as
    dgs.retinal_scan_imaging.synthetic_vessel_reflectance. Returns the
    image and a boolean vessel_mask (True on/near a vessel centerline) for
    verify_vector_field_highlights_vessels to check against."""
    if size < 32:
        raise ValueError(f"size={size}: need at least 32 for a meaningful fundus image")
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size]
    image = np.full((size, size), 0.5)

    disk_center = (size * 0.5, size * 0.45)
    disk_r = size * 0.12
    r_disk = np.sqrt((xx - disk_center[1])**2 + (yy - disk_center[0])**2)
    image += 0.35 * np.exp(-(r_disk / disk_r)**2)

    vessel_mask = np.zeros((size, size), dtype=bool)
    for _ in range(n_vessels):
        n_segments = rng.integers(4, 8)
        pos = np.array(disk_center) + rng.uniform(-5, 5, 2)
        angle = rng.uniform(0, 2 * np.pi)
        width = rng.uniform(1.5, 3.0)
        for _ in range(n_segments):
            step = rng.uniform(size * 0.06, size * 0.12)
            angle += rng.uniform(-0.5, 0.5)
            new_pos = pos + step * np.array([np.sin(angle), np.cos(angle)])
            t = np.linspace(0, 1, 50)
            seg_y = pos[0] + t * (new_pos[0] - pos[0])
            seg_x = pos[1] + t * (new_pos[1] - pos[1])
            for sy, sx in zip(seg_y, seg_x):
                dist = np.sqrt((xx - sx)**2 + (yy - sy)**2)
                image -= 0.25 * np.exp(-(dist / width)**2)
                vessel_mask |= dist < width
            pos = new_pos

    return {"image": np.clip(image, 0.0, 1.0), "vessel_mask": vessel_mask}


def _wrap_safe_gradient_1d(phase: np.ndarray, axis: int) -> np.ndarray:
    """d(phase)/d(axis) via central differences in the complex exponential
    (wrap-safe): angle(exp(i*phi_fwd) / exp(i*phi_bwd)) / 2, edges use a
    one-sided version. Avoids the false large-jump a naive np.diff on raw
    radians would give at a +pi/-pi wrap boundary."""
    z = np.exp(1j * phase)
    fwd = np.roll(z, -1, axis=axis)
    bwd = np.roll(z, 1, axis=axis)
    central = np.angle(fwd / bwd) / 2.0
    # fix the wrap-around endpoints (np.roll wraps the array itself, which
    # is physically wrong at the image border) with one-sided differences
    central = np.moveaxis(central, axis, 0)
    fwd_ax = np.moveaxis(np.angle(np.exp(1j * phase) / np.roll(np.exp(1j * phase), 1, axis=axis)), axis, 0)
    bwd_ax = np.moveaxis(np.angle(np.roll(np.exp(1j * phase), -1, axis=axis) / np.exp(1j * phase)), axis, 0)
    central[0] = fwd_ax[0]
    central[-1] = bwd_ax[-1]
    return np.moveaxis(central, 0, axis)


def pst_vector_field(image: np.ndarray, warp: float = 15.0, strength: float = 0.48,
                      sigma: float = 0.12) -> dict:
    """PST's scalar phase-edge map (dgs.pst.pst), differentiated spatially
    (wrap-safe) into a genuine 2-D vector field: grad_y, grad_x (row/col
    components), magnitude, and direction (radians). This is the "spatial
    computing" framing made literal -- PST doesn't just flag edges, its
    phase output's spatial GRADIENT is itself a vector field an analog
    optical system could read out directly (e.g. via a shear
    interferometer), the same way a wavefront-sensor reads a lens's phase
    gradient as a slope map."""
    phase = pst_phase_map(image, warp=warp, strength=strength, sigma=sigma)
    grad_y = _wrap_safe_gradient_1d(phase, axis=0)
    grad_x = _wrap_safe_gradient_1d(phase, axis=1)
    magnitude = np.sqrt(grad_x**2 + grad_y**2)
    direction = np.arctan2(grad_y, grad_x)
    return {"phase": phase, "grad_y": grad_y, "grad_x": grad_x,
            "magnitude": magnitude, "direction": direction}


def verify_vector_field_highlights_vessels(size: int = 128, n_vessels: int = 6, seed: int = 0,
                                            dilate_px: int = 2) -> dict:
    """CHECKED, not assumed: the vector field's MAGNITUDE, averaged over
    vessel pixels (dilated by `dilate_px` to tolerate PST's own edge-
    localization ring-out, the same slack dgs.pst's own disk-rim test
    uses), must be substantially larger than the magnitude averaged over
    non-vessel background pixels -- direct numeric evidence that this
    vector field is doing something useful on a medical-imaging-style
    test image, not just producing numbers."""
    data = synthetic_fundus_image(size=size, n_vessels=n_vessels, seed=seed)
    image, vessel_mask = data["image"], data["vessel_mask"]

    dilated = vessel_mask.copy()
    for _ in range(dilate_px):
        dilated = (dilated | np.roll(dilated, 1, 0) | np.roll(dilated, -1, 0) |
                   np.roll(dilated, 1, 1) | np.roll(dilated, -1, 1))

    field = pst_vector_field(image)
    mag = field["magnitude"]
    vessel_mean = float(mag[dilated].mean())
    background_mean = float(mag[~dilated].mean())
    ratio = vessel_mean / max(background_mean, 1e-12)
    return {"vessel_mean_magnitude": vessel_mean, "background_mean_magnitude": background_mean,
            "ratio": ratio, "highlights_vessels": bool(ratio > 2.0)}


if __name__ == "__main__":
    print("=== PST as a vector field, applied to a synthetic fundus image ===")
    check = verify_vector_field_highlights_vessels()
    print(f"  vessel-region mean |grad phase|     = {check['vessel_mean_magnitude']:.4f}")
    print(f"  background-region mean |grad phase| = {check['background_mean_magnitude']:.4f}")
    print(f"  ratio = {check['ratio']:.2f}x  (highlights vessels: {check['highlights_vessels']})")

    print("\nSame PST kernel dgs.pst already uses for civilian edge detection; the")
    print("spatial gradient of its phase output turns 'is this an edge' into 'which")
    print("way does the edge point,' a genuine vector field rather than a scalar map.")
