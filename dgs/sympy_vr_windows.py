""""Live SymPy windows in VR" -- floating panels of symbolic math,
rendered as a genuine stereoscopic (left/right eye) scene, reusing
dgs.spatial_computing's existing perspective-projection and stereo-depth
machinery rather than reimplementing it.

APPROACH: each "window" is a billboard rectangle (always facing the
viewer, like a real VR app panel -- e.g. Vision Pro/Quest UI windows ARE
rendered this way, not fully 3D-warped) placed at a 3D depth. Its four
corners are projected through TWO cameras offset by the interpupillary
baseline (dgs.spatial_computing.project_points, called once per eye with
the points shifted by -baseline/2 and +baseline/2) to get the left-eye
and right-eye screen positions. The horizontal pixel shift between those
two projections is real stereo disparity -- and it's checked against
dgs.spatial_computing.stereo_depth's own Z=B*f/d formula: feeding the
rendered disparity back through that formula should recover the
window's true depth, confirming the rendering pipeline and the existing
depth-sensing math agree with each other.

py-3.13 compatible (sympy + matplotlib only, no torch needed here).
"""

import numpy as np
import sympy as sp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dgs.spatial_computing import perspective_matrix, project_points, stereo_depth

DEFAULT_BASELINE_M = 0.064   # human-eye-like interpupillary distance


def focal_px_from_fov(fov_deg, image_w, image_h):
    """The pixel-space focal length a pinhole-camera formula (like
    stereo_depth's Z=B*f/d) actually needs, DERIVED from the same
    fov_deg/image_w/image_h used to build the OpenGL-style
    perspective_matrix -- rather than an independently chosen constant.
    An earlier version of this module used an arbitrary DEFAULT_FOCAL_PX
    that didn't match the projection's real implied focal length,
    producing a ~92% systematic depth-recovery error; this derivation
    fixes that by construction."""
    if fov_deg <= 0 or fov_deg >= 180:
        raise ValueError("fov_deg must be in (0, 180)")
    if image_w <= 0 or image_h <= 0:
        raise ValueError("image_w and image_h must be positive")
    f = 1.0 / np.tan(np.deg2rad(fov_deg) / 2)
    aspect = image_w / image_h
    return (image_w / 2) * f / aspect


def render_sympy_texture(expr, fontsize=28, dpi=100):
    """Render a SymPy expression to an RGBA image array via matplotlib's
    mathtext renderer (no external LaTeX install needed). matplotlib's
    mathtext only implements a SUBSET of LaTeX -- constructs like
    Integral's \\limits are real LaTeX matplotlib can't parse without a
    full system LaTeX install. Falls back to a plain-text render of the
    expression rather than crashing on those cases."""
    fig = plt.figure(figsize=(2.5, 1.2), dpi=dpi)
    fig.patch.set_alpha(0.0)
    try:
        fig.text(0.5, 0.5, f"${sp.latex(expr)}$", fontsize=fontsize,
                  ha="center", va="center", color="white")
        fig.canvas.draw()
    except ValueError:
        plt.close(fig)
        fig = plt.figure(figsize=(2.5, 1.2), dpi=dpi)
        fig.patch.set_alpha(0.0)
        fig.text(0.5, 0.5, str(expr), fontsize=fontsize * 0.7,
                  ha="center", va="center", color="white")
        fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())
    plt.close(fig)
    return buf


def make_vr_window(expr, position_3d, width_m=0.4, height_m=0.25):
    """A billboard rectangle centered at position_3d, facing the viewer
    (lying in a plane of constant depth Z, extent in X/Y)."""
    if width_m <= 0 or height_m <= 0:
        raise ValueError("width_m and height_m must be positive")
    x, y, z = position_3d
    if z <= 0:
        raise ValueError("position_3d's z (depth) must be positive (in front of the viewer)")
    corners_3d = np.array([
        [x - width_m / 2, y - height_m / 2, z],
        [x + width_m / 2, y - height_m / 2, z],
        [x + width_m / 2, y + height_m / 2, z],
        [x - width_m / 2, y + height_m / 2, z],
    ])
    texture = render_sympy_texture(expr)
    return {"expr": expr, "position_3d": np.array(position_3d, float),
            "corners_3d": corners_3d, "texture": texture}


def project_window_stereo(window, baseline_m=DEFAULT_BASELINE_M,
                            image_w=640, image_h=480, fov_deg=60, near=0.05, far=50):
    """Project a window's 4 corners through LEFT and RIGHT eye cameras
    (offset by -+baseline/2 along x), reusing dgs.spatial_computing's
    perspective_matrix + project_points for the actual projection math."""
    if baseline_m <= 0:
        raise ValueError("baseline_m must be positive")
    P = perspective_matrix(fov_deg, aspect=image_w / image_h, near=near, far=far)

    corners = window["corners_3d"]
    left_pts = corners - np.array([-baseline_m / 2, 0, 0])   # shift world into left-eye frame
    right_pts = corners - np.array([baseline_m / 2, 0, 0])

    # dgs.spatial_computing.perspective_matrix is OpenGL-style: the camera
    # looks down -Z. This module's public API uses a positive "depth"
    # (intuitive: distance in front of the viewer), so negate z here to
    # match the projection matrix's actual convention -- using +z directly
    # was a real bug caught during testing (systematic ~290% depth error).
    left_pts_cam = left_pts.copy(); left_pts_cam[:, 2] = -left_pts_cam[:, 2]
    right_pts_cam = right_pts.copy(); right_pts_cam[:, 2] = -right_pts_cam[:, 2]

    left_screen = project_points(left_pts_cam, P)["screen"]
    right_screen = project_points(right_pts_cam, P)["screen"]

    def to_pixels(screen_ndc):
        px = (screen_ndc[:, 0] * 0.5 + 0.5) * image_w
        py = (1.0 - (screen_ndc[:, 1] * 0.5 + 0.5)) * image_h
        return np.column_stack([px, py])

    return {"left_px": to_pixels(left_screen), "right_px": to_pixels(right_screen)}


def verify_disparity_matches_depth(window, baseline_m=DEFAULT_BASELINE_M,
                                     image_w=640, image_h=480, fov_deg=60, **proj_kwargs):
    """Render the window's stereo pair, measure the horizontal pixel
    disparity between the left/right projected centers, and feed that
    disparity back through dgs.spatial_computing.stereo_depth's Z=B*f/d
    formula -- confirming it recovers the window's TRUE depth (a real
    cross-check between the rendering pipeline and the existing
    depth-sensing physics, not just an internal assertion). focal_px is
    DERIVED from the same fov_deg/image_w/image_h used for projection
    (via focal_px_from_fov), not an independently chosen constant -- an
    earlier version used an unrelated fixed focal_px and got a
    systematic ~92% depth-recovery error from that mismatch."""
    proj = project_window_stereo(window, baseline_m, image_w, image_h, fov_deg, **proj_kwargs)
    left_center_x = proj["left_px"][:, 0].mean()
    right_center_x = proj["right_px"][:, 0].mean()
    disparity_px = left_center_x - right_center_x   # left eye sees it shifted right
    focal_px = focal_px_from_fov(fov_deg, image_w, image_h)
    depth_result = stereo_depth(disparity_px, baseline_m, focal_px)
    true_depth = window["position_3d"][2]
    return {
        "disparity_px": disparity_px,
        "recovered_depth_m": depth_result["depth_m"],
        "true_depth_m": true_depth,
        "relative_error": abs(depth_result["depth_m"] - true_depth) / true_depth,
    }


def compose_stereo_image(windows, baseline_m=DEFAULT_BASELINE_M,
                          image_w=640, image_h=480, **proj_kwargs):
    """Render a real side-by-side stereo pair: every window's texture
    placed at its projected screen-space rectangle, for both eyes."""
    if not windows:
        raise ValueError("windows must be a non-empty list")
    left_img = np.zeros((image_h, image_w, 4))
    right_img = np.zeros((image_h, image_w, 4))

    for window in windows:
        proj = project_window_stereo(window, baseline_m, image_w, image_h, **proj_kwargs)
        for img, key in [(left_img, "left_px"), (right_img, "right_px")]:
            px = proj[key]
            x0, y0 = int(px[:, 0].min()), int(px[:, 1].min())
            x1, y1 = int(px[:, 0].max()), int(px[:, 1].max())
            x0c, x1c = max(0, x0), min(image_w, x1)
            y0c, y1c = max(0, y0), min(image_h, y1)
            if x1c <= x0c or y1c <= y0c:
                continue   # window projected fully off-screen for this eye
            tex_resized_h, tex_resized_w = y1c - y0c, x1c - x0c
            tex = window["texture"]
            # nearest-neighbor resample of the rendered texture into the screen rectangle
            ys = np.linspace(0, tex.shape[0] - 1, tex_resized_h).astype(int)
            xs = np.linspace(0, tex.shape[1] - 1, tex_resized_w).astype(int)
            resampled = tex[np.ix_(ys, xs)] / 255.0
            alpha = resampled[..., 3:4]
            img[y0c:y1c, x0c:x1c] = alpha * resampled + (1 - alpha) * img[y0c:y1c, x0c:x1c]

    return left_img, right_img


if __name__ == "__main__":
    print("=== Live SymPy windows in VR: stereo projection + depth verification ===\n")

    x, D, f = sp.symbols('x D f')
    windows = [
        make_vr_window(sp.exp(sp.I * sp.pi * D * f**2), (-0.3, 0.1, 1.0)),
        make_vr_window(sp.Integral(sp.exp(-x**2), (x, -sp.oo, sp.oo)), (0.3, -0.1, 2.0)),
        make_vr_window(sp.Eq(sp.Symbol('E'), sp.Symbol('h') * sp.Symbol('c') / sp.Symbol('lambda')), (0.0, 0.2, 3.5)),
    ]

    print("Depth verification (rendered disparity fed back through Z=B*f/d):\n")
    for w in windows:
        result = verify_disparity_matches_depth(w)
        print(f"  window at true depth {result['true_depth_m']:.2f} m: "
              f"disparity={result['disparity_px']:.2f} px, "
              f"recovered depth={result['recovered_depth_m']:.3f} m, "
              f"relative error={result['relative_error']*100:.2f}%")

    left_img, right_img = compose_stereo_image(windows)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), facecolor="black")
    for ax, img, label in zip(axes, [left_img, right_img], ["LEFT eye", "RIGHT eye"]):
        ax.imshow(img)
        ax.set_title(label, color="white")
        ax.axis("off")
    fig.tight_layout()
    fig.savefig("sympy_vr_windows_stereo.png", dpi=120, facecolor="black")
    plt.close(fig)
    print("\nstereo pair saved to sympy_vr_windows_stereo.png")
