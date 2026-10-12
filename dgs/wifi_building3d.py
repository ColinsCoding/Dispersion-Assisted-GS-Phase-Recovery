"""WiFi coverage in a 3-D building: statistics and animation.

PROGRAMMING STYLE (chosen on purpose): vectorized array programming -- pure
functions, no per-cell loops. Every receiver point and every wall is handled at
once by NumPy broadcasting; the same style is used in the MATLAB twin
(matlab/.. learn/matlab/wifi_building3d.m). The loop-based 2-D model in
dgs/wifi_floorplan.py is the reference it must agree with.

MODEL (first-order, all losses are labelled assumptions):
  received[dBm] = P_t - FSPL(3-D distance) - sum(wall losses) - n_slabs * L_slab
  * walls: the floor plan of dgs.wifi_floorplan, extruded vertically on every
    floor; crossings counted on the 2-D projection of the ray (standard
    multi-wall approximation);
  * slabs: one concrete floor slab per storey crossed, L_slab (assumed 18 dB @
    2.4 GHz, 25 dB @ 5 GHz);
  * no reflections / multipath / stairwell leakage.
"""
import numpy as np

from dgs.wifi_floorplan import WALL_LOSS_DB, default_floorplan

C = 299792458.0
SLAB_LOSS_DB = (18.0, 25.0)       # assumption: concrete floor slab at (2.4, 5) GHz
FLOOR_HEIGHT = 3.5                 # m


def walls_as_arrays(walls, band=0):
    """(A, B, loss): A, B are (M, 2) endpoint arrays, loss is (M,) dB."""
    A = np.array([w[0] for w in walls], float)
    B = np.array([w[1] for w in walls], float)
    L = np.array([WALL_LOSS_DB[w[2]][band] for w in walls], float)
    return A, B, L


def _cross(o, a, b):
    """z-component of (a - o) x (b - o), broadcast over leading dims."""
    return (a[..., 0] - o[..., 0]) * (b[..., 1] - o[..., 1]) - (a[..., 1] - o[..., 1]) * (b[..., 0] - o[..., 0])


def wall_loss_vectorized(router_xy, points_xy, A, B, L):
    """Total wall loss for every point at once.

    points_xy: (N, 2). Builds an (N, M) boolean crossing matrix with the same
    strict orientation test as the 2-D reference, then a matrix-vector product."""
    p = np.broadcast_to(np.asarray(router_xy, float), points_xy.shape)[:, None, :]   # (N,1,2)
    q = points_xy[:, None, :]                                                        # (N,1,2)
    a, b = A[None, :, :], B[None, :, :]                                              # (1,M,2)
    d1, d2 = np.sign(_cross(p, q, a)), np.sign(_cross(p, q, b))
    d3, d4 = np.sign(_cross(a, b, p)), np.sign(_cross(a, b, q))
    crosses = (d1 * d2 < 0) & (d3 * d4 < 0)                                          # (N,M)
    return crosses.astype(float) @ L                                                 # (N,)


def fspl_db(d, f_hz):
    return 20 * np.log10(4 * np.pi * np.maximum(d, 0.1) * f_hz / C)


def building_grid(W, H, n_floors=3, res=0.25, floor_height=FLOOR_HEIGHT, rx_height=1.0):
    """Receiver points (N, 3) at desk height on every floor, and their floor index."""
    xs = np.arange(res / 2, W, res)
    ys = np.arange(res / 2, H, res)
    X, Y = np.meshgrid(xs, ys)
    pts, fl = [], []
    for k in range(n_floors):
        pts.append(np.column_stack([X.ravel(), Y.ravel(), np.full(X.size, k * floor_height + rx_height)]))
        fl.append(np.full(X.size, k))
    return np.vstack(pts), np.concatenate(fl), xs, ys


def received_power_3d(router, points, floors, walls, f_hz=2.4e9, ptx_dbm=20.0,
                      floor_height=FLOOR_HEIGHT):
    """Vectorized received power (dBm) at every point in the building."""
    if f_hz <= 0 or floor_height <= 0:
        raise ValueError("frequency and floor height must be positive")
    band = 0 if f_hz < 4e9 else 1
    A, B, L = walls_as_arrays(walls, band)
    r = np.asarray(router, float)
    d = np.linalg.norm(points - r, axis=1)
    r_floor = int(r[2] // floor_height)
    slabs = np.abs(floors - r_floor)
    return ptx_dbm - fspl_db(d, f_hz) - wall_loss_vectorized(r[:2], points[:, :2], A, B, L) \
        - slabs * SLAB_LOSS_DB[band]


def floor_statistics(P, floors, threshold_dbm=-67.0):
    """Per-floor coverage %, median, 10th percentile (the 'worst desks'), mean in dBm."""
    rows = []
    for k in np.unique(floors):
        v = P[floors == k]
        rows.append({"floor": int(k), "coverage %": 100 * float(np.mean(v >= threshold_dbm)),
                     "median dBm": float(np.median(v)), "p10 dBm": float(np.percentile(v, 10)),
                     "mean dBm": float(np.mean(v))})
    return rows


def empirical_cdf(v):
    v = np.sort(np.asarray(v, float))
    return v, np.arange(1, v.size + 1) / v.size


def hallway_path(n_frames=24, floor=1, floor_height=FLOOR_HEIGHT, z_above=2.5):
    """Router positions walking the south corridor (y = 3 m) on a given floor."""
    xs = np.linspace(5.0, 13.5, n_frames)
    return np.column_stack([xs, np.full(n_frames, 3.0), np.full(n_frames, floor * floor_height + z_above)])


def export_walls_csv(path, walls):
    """x1,y1,x2,y2,loss24,loss5 -- the single source of truth shared with MATLAB."""
    rows = [(a[0], a[1], b[0], b[1], *WALL_LOSS_DB[m]) for a, b, m in walls]
    np.savetxt(path, np.array(rows), delimiter=",", fmt="%.6g")
    return path
