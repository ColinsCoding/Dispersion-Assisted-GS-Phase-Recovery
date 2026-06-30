"""Spatial computing: 3D transforms, quaternions, homogeneous coordinates,
perspective projection, stereo depth — math for Apple Vision Pro / AR / VR.

Precalculus entry point: a fraction IS a perspective projection.
  screen_x / screen_w = world_x / world_z  (same a/b = c/d from algebra)

Connection to phase retrieval:
  GS algorithm recovers phase (depth) from intensity (2D projection).
  Stereo imaging recovers depth from two 2D projections.
  Both invert a projection — same mathematical problem, different physics.
"""
import numpy as np
import sympy as sp


# ── Elementary: fractions as ratios in 3D ────────────────────────────────────

def perspective_divide(point_3d, focal_length=1.0):
    """Project 3D point (X, Y, Z) onto image plane at z=focal_length.

    The fundamental operation: x_screen = f * X / Z
    This IS a fraction. Precalculus prerequisite for all of computer graphics.
    """
    X, Y, Z = point_3d
    if abs(Z) < 1e-12:
        raise ValueError("Z=0: point is at the camera — undefined projection")
    x_s = focal_length * X / Z
    y_s = focal_length * Y / Z
    return {
        "x_screen": x_s,
        "y_screen": y_s,
        "scale": focal_length / Z,
        "depth": Z,
        "note": "x_screen = f*X/Z  -- a fraction, same as a/b=c/d",
    }


def homogeneous_coords(point_3d):
    """Convert (X,Y,Z) -> (X,Y,Z,1) homogeneous coordinates.

    Homogeneous coords let you represent projection as a matrix multiply.
    Translation (which can't be linear) becomes linear in 4D.
    """
    X, Y, Z = point_3d
    return np.array([X, Y, Z, 1.0])


def dehomogenize(h):
    """(X,Y,Z,W) -> (X/W, Y/W, Z/W) -- the perspective divide as division."""
    return h[:3] / h[3]


# ── Rotation matrices ─────────────────────────────────────────────────────────

def rot_x(theta_rad):
    """Rotation matrix around X axis."""
    c, s = np.cos(theta_rad), np.sin(theta_rad)
    return np.array([[1, 0,  0, 0],
                     [0, c, -s, 0],
                     [0, s,  c, 0],
                     [0, 0,  0, 1]], float)

def rot_y(theta_rad):
    """Rotation matrix around Y axis."""
    c, s = np.cos(theta_rad), np.sin(theta_rad)
    return np.array([[ c, 0, s, 0],
                     [ 0, 1, 0, 0],
                     [-s, 0, c, 0],
                     [ 0, 0, 0, 1]], float)

def rot_z(theta_rad):
    """Rotation matrix around Z axis."""
    c, s = np.cos(theta_rad), np.sin(theta_rad)
    return np.array([[c, -s, 0, 0],
                     [s,  c, 0, 0],
                     [0,  0, 1, 0],
                     [0,  0, 0, 1]], float)

def translation_matrix(tx, ty, tz):
    """4x4 homogeneous translation matrix."""
    T = np.eye(4)
    T[:3, 3] = [tx, ty, tz]
    return T

def scale_matrix(sx, sy, sz):
    """4x4 homogeneous scale matrix."""
    return np.diag([sx, sy, sz, 1.0])


# ── Quaternions ───────────────────────────────────────────────────────────────

def quaternion_from_axis_angle(axis, theta_rad):
    """q = (cos(theta/2), sin(theta/2)*axis) -- unit quaternion for rotation.

    Quaternions avoid gimbal lock. Apple Vision Pro uses them for head tracking.
    """
    axis = np.asarray(axis, float)
    axis = axis / np.linalg.norm(axis)
    w = np.cos(theta_rad / 2)
    xyz = np.sin(theta_rad / 2) * axis
    q = np.array([w, xyz[0], xyz[1], xyz[2]])
    return q / np.linalg.norm(q)   # ensure unit


def quaternion_multiply(q1, q2):
    """Hamilton product q1 * q2."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])


def quaternion_to_matrix(q):
    """Convert unit quaternion to 3x3 rotation matrix (then embed in 4x4)."""
    w, x, y, z = q / np.linalg.norm(q)
    R = np.array([
        [1-2*(y*y+z*z),   2*(x*y-w*z),   2*(x*z+w*y)],
        [  2*(x*y+w*z), 1-2*(x*x+z*z),   2*(y*z-w*x)],
        [  2*(x*z-w*y),   2*(y*z+w*x), 1-2*(x*x+y*y)],
    ])
    M = np.eye(4)
    M[:3, :3] = R
    return M


def quaternion_slerp(q0, q1, t):
    """Spherical linear interpolation — smooth rotation for AR animation."""
    q0 = q0 / np.linalg.norm(q0)
    q1 = q1 / np.linalg.norm(q1)
    dot = np.clip(np.dot(q0, q1), -1.0, 1.0)
    if dot < 0:
        q1 = -q1; dot = -dot
    if dot > 0.9995:
        return (q0 + t * (q1 - q0)) / np.linalg.norm(q0 + t*(q1-q0))
    theta0 = np.arccos(dot)
    theta  = theta0 * t
    sin0   = np.sin(theta0)
    return (np.sin(theta0 - theta)/sin0 * q0 +
            np.sin(theta)/sin0 * q1)


# ── Perspective projection matrix (camera) ────────────────────────────────────

def perspective_matrix(fov_deg, aspect, near, far):
    """OpenGL-style perspective projection matrix.

    Maps view frustum to normalized device coordinates [-1,1]^3.
    Apple Vision Pro renders to two such frustums (one per eye).
    """
    f = 1.0 / np.tan(np.deg2rad(fov_deg) / 2)
    return np.array([
        [f/aspect, 0,                           0,  0],
        [0,        f,                           0,  0],
        [0,        0,   (far+near)/(near-far), 2*far*near/(near-far)],
        [0,        0,                          -1,  0],
    ])


def project_points(points_3d, P_matrix):
    """Project array of 3D points (N,3) through 4x4 projection matrix."""
    pts = np.asarray(points_3d, float)
    h   = np.hstack([pts, np.ones((len(pts), 1))])   # homogeneous
    p   = (P_matrix @ h.T).T                          # (N,4)
    screen = p[:, :2] / p[:, 3:4]                    # perspective divide
    return {"screen": screen, "clip": p, "depth": p[:, 2] / p[:, 3]}


# ── Stereo depth (spatial computing parallax) ─────────────────────────────────

def stereo_depth(disparity_px, baseline_m, focal_px):
    """Z = B * f / d  -- from two camera images.

    Same equation as satellite stereo. Apple Vision Pro uses this with
    its two front cameras to sense depth for hand tracking.
    """
    d = np.asarray(disparity_px, float)
    Z = baseline_m * focal_px / np.where(d == 0, np.inf, d)
    return {
        "depth_m": Z,
        "disparity_px": d,
        "baseline_m": baseline_m,
        "focal_px": focal_px,
        "note": "Z = B*f/d  identical to satellite stereo (same fraction)",
    }


def depth_uncertainty(Z, baseline_m, focal_px, sigma_d=0.5):
    """dZ = Z^2 * sigma_d / (B * f) -- error propagation on depth."""
    return Z**2 * sigma_d / (baseline_m * focal_px)


# ── GS phase retrieval connection ─────────────────────────────────────────────

def phase_retrieval_vs_stereo_analogy():
    """Show the mathematical analogy between GS and stereo imaging."""
    return {
        "Stereo imaging": {
            "known":   "Two 2D intensity images I_L, I_R",
            "unknown": "3D depth Z(x,y)",
            "method":  "Disparity d = match features; Z = Bf/d",
            "equation": "Z = B*f/d",
        },
        "GS phase retrieval": {
            "known":   "Two intensity spectra |E_in|^2, |E_out|^2",
            "unknown": "Phase phi(omega) of E_out",
            "method":  "Alternate projections in object/Fourier domain",
            "equation": "phi = angle(IFFT(sqrt(I_out) * exp(i*phi_estimate)))",
        },
        "Remote sensing": {
            "known":   "Multispectral intensity I(lambda)",
            "unknown": "Land cover class, material composition",
            "method":  "Feature importance -> decision tree",
            "equation": "class = argmax P(class | NDVI, NIR, Red, ...)",
        },
        "common_math": "All three invert a projection: recover hidden dimensions from observed intensities",
    }


# ── SymPy equations ───────────────────────────────────────────────────────────

def spatial_computing_sympy_5():
    """Five symbolic spatial computing equations."""
    X, Y, Z_s, f_s = sp.symbols('X Y Z f', real=True)
    w, x, y, z_q   = sp.symbols('w x y z', real=True)
    theta           = sp.Symbol('theta', real=True)
    B_s, d_s        = sp.symbols('B d', positive=True)

    return {
        "Perspective_divide":
            sp.Eq(sp.Symbol('x_s'), f_s * X / Z_s),
        "Quaternion_rotation":
            sp.Eq(sp.Symbol('q'),
                  sp.cos(theta/2) + sp.sin(theta/2)*sp.Symbol('n_hat')),
        "Stereo_depth":
            sp.Eq(sp.Symbol('Z'), B_s * f_s / d_s),
        "Depth_uncertainty":
            sp.Eq(sp.Symbol('dZ'),
                  sp.Symbol('Z')**2 * sp.Symbol('sigma_d') / (B_s * f_s)),
        "Homogeneous_projection":
            sp.Eq(sp.Matrix([sp.Symbol('x_s'), sp.Symbol('y_s'), 1]),
                  (1/Z_s) * sp.Matrix([f_s*X, f_s*Y, Z_s])),
    }


if __name__ == "__main__":
    print("=== Perspective divide: point (3, 2, 5), f=2 ===")
    p = perspective_divide([3, 2, 5], focal_length=2.0)
    print(f"  screen = ({p['x_screen']:.4f}, {p['y_screen']:.4f})")
    print(f"  {p['note']}")

    print("\n=== Quaternion: 90 deg around Z axis ===")
    q = quaternion_from_axis_angle([0, 0, 1], np.pi/2)
    print(f"  q = {q.round(4)}")
    M = quaternion_to_matrix(q)
    pt = M[:3, :3] @ np.array([1, 0, 0])
    print(f"  Rotate (1,0,0) -> {pt.round(4)}  (expect 0,1,0)")

    print("\n=== Stereo depth: baseline=6cm, focal=800px ===")
    disp = np.array([5, 10, 20, 40])
    sd = stereo_depth(disp, 0.06, 800)
    for d, Z in zip(disp, sd["depth_m"]):
        print(f"  disparity={d:2d} px -> depth={Z:.2f} m")

    print("\n=== Phase retrieval vs stereo analogy ===")
    analogy = phase_retrieval_vs_stereo_analogy()
    for system, info in analogy.items():
        if isinstance(info, dict):
            print(f"  {system}: unknown={info['unknown']}")

    print("\n=== SymPy 5 ===")
    for k, eq in spatial_computing_sympy_5().items():
        print(f"  {k}: {eq}")
