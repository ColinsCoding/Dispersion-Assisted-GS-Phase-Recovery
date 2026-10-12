"""WiFi coverage on a floor plan from existing laws, plus why thermal cameras see
in the 8-14 um band (Wien's law for skin).

RADIO (coverage planning, not tracking):
  Friis:  P_r / P_t = G_t G_r (lambda / 4 pi d)^2      (inverse-square spreading)
  in dB:  FSPL = 20 log10(4 pi d f / c)                 -> 6.02 dB per doubling of d
  multi-wall model (COST 231 style):  L = FSPL(d) + sum over walls crossed of L_wall
  Received power  P_r[dBm] = P_t[dBm] + G_t + G_r - L.
  Walls crossed = segment-segment intersections between the router->point ray and
  each wall (computational geometry, orientation tests).
  Wall losses are TYPICAL ORDER-OF-MAGNITUDE ASSUMPTIONS (they vary a lot with
  construction); measure your own walls for real planning. No reflections,
  diffraction or multipath -- this is a first-order planning model.

THERMAL:
  Wien: lambda_peak = b / T; skin (~306 K) peaks near 9.5 um, so long-wave IR
  cameras use 8-14 um, where the atmosphere is also transparent. Band radiance
  from dgs.thermal_radiation.planck_spectral_radiance.
"""
import numpy as np
import sympy as sp

from dgs.thermal_radiation import planck_spectral_radiance
from dgs.wien_wave_502 import wien_constant

C = 299792458.0

# dB loss per wall crossing at (2.4 GHz, 5 GHz) -- assumptions, typical ranges
WALL_LOSS_DB = {
    "drywall": (3.0, 4.0),
    "glass": (2.0, 3.0),
    "brick": (8.0, 12.0),
    "concrete": (12.0, 20.0),
    "door (wood)": (3.0, 4.0),
}


def default_floorplan():
    """A 20 m x 14 m office floor: concrete shell and stair/elevator core, brick
    restrooms, drywall offices and meeting rooms, glass meeting-room front.
    Each wall = ((x1, y1), (x2, y2), material). Gaps are doorways."""
    W, H = 20.0, 14.0
    walls = [
        ((0, 0), (W, 0), "concrete"), ((W, 0), (W, H), "concrete"),
        ((W, H), (0, H), "concrete"), ((0, H), (0, 0), "concrete"),
        # concrete core (stairs + elevator), centre of the floor, door on the south side
        ((8.5, 5.5), (8.5, 8.5), "concrete"), ((8.5, 8.5), (11.5, 8.5), "concrete"),
        ((11.5, 8.5), (11.5, 5.5), "concrete"), ((11.5, 5.5), (10.5, 5.5), "concrete"),
        ((9.5, 5.5), (8.5, 5.5), "concrete"),
        # brick restrooms, north-east
        ((15, 14), (15, 10), "brick"), ((15, 10), (17, 10), "brick"), ((18, 10), (20, 10), "brick"),
        # drywall private offices along the west wall
        ((4, 0), (4, 3), "drywall"), ((4, 4), (4, 9), "drywall"), ((4, 10), (4, 14), "drywall"),
        ((0, 4.5), (4, 4.5), "drywall"), ((0, 9.5), (4, 9.5), "drywall"),
        # glass-fronted meeting room, south-east
        ((14, 0), (14, 2.5), "drywall"), ((14, 3.5), (14, 5), "glass"), ((14, 5), (20, 5), "glass"),
    ]
    return W, H, walls


# ── laws ─────────────────────────────────────────────────────────────

def friis_symbolic():
    """Friis -> FSPL in dB, and the dB change per doubling of distance."""
    Gt, Gr, lam, d, f, c = sp.symbols("G_t G_r lambda d f c", positive=True)
    ratio = Gt * Gr * (lam / (4 * sp.pi * d)) ** 2
    fspl = 20 * sp.log(4 * sp.pi * d * f / c, 10)
    assert sp.simplify(sp.expand_log(fspl + 10 * sp.log(ratio.subs({Gt: 1, Gr: 1, lam: c / f}), 10),
                                     force=True)) == 0           # FSPL = -10 log10(Friis ratio)
    per_doubling = sp.simplify(sp.expand_log(fspl.subs(d, 2 * d) - fspl, force=True))
    return ratio, fspl, sp.nsimplify(per_doubling), float(per_doubling)


def fspl_db(d, f_hz):
    d = np.maximum(np.asarray(d, float), 0.1)          # avoid log(0) at the router
    if f_hz <= 0:
        raise ValueError("frequency must be positive")
    return 20 * np.log10(4 * np.pi * d * f_hz / C)


def _orient(a, b, c):
    return np.sign((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))


def segments_cross(p, q, a, b):
    """Proper intersection of segments pq and ab (orientation test)."""
    return (_orient(p, q, a) * _orient(p, q, b) < 0) and (_orient(a, b, p) * _orient(a, b, q) < 0)


def wall_loss_db(p, q, walls, band=0):
    return sum(WALL_LOSS_DB[m][band] for a, b, m in walls if segments_cross(p, q, a, b))


def coverage_map(router, walls, W, H, f_hz=2.4e9, ptx_dbm=20.0, gains_db=0.0, res=0.1):
    """Received power (dBm) on a grid of cell centres."""
    if res <= 0:
        raise ValueError("res must be positive")
    band = 0 if f_hz < 4e9 else 1
    xs = np.arange(res / 2, W, res)
    ys = np.arange(res / 2, H, res)
    P = np.empty((ys.size, xs.size))
    for j, y in enumerate(ys):
        for i, x in enumerate(xs):
            d = np.hypot(x - router[0], y - router[1])
            P[j, i] = ptx_dbm + gains_db - fspl_db(d, f_hz) - wall_loss_db(router, (x, y), walls, band)
    return xs, ys, P


def coverage_fraction(P, threshold_dbm=-67.0):
    """Fraction of the floor at or above the threshold (-67 dBm ~ common 'good' level)."""
    return float(np.mean(P >= threshold_dbm))


def distance_to_segment(p, a, b):
    p, a, b = (np.asarray(v, float) for v in (p, a, b))
    ab = b - a
    t = np.clip(np.dot(p - a, ab) / max(np.dot(ab, ab), 1e-12), 0.0, 1.0)
    return float(np.linalg.norm(p - (a + t * ab)))


def best_router_position(walls, W, H, f_hz=5e9, step=1.0, res=0.4, threshold_dbm=-67.0,
                         min_wall_clearance=0.3):
    """Grid search for the router spot maximising coverage. Candidate spots closer
    than min_wall_clearance to any wall are skipped (a router cannot sit in a wall)."""
    best = (None, -1.0)
    for x in np.arange(step / 2, W, step):
        for y in np.arange(step / 2, H, step):
            if min(distance_to_segment((x, y), a, b) for a, b, _ in walls) < min_wall_clearance:
                continue
            _, _, P = coverage_map((x, y), walls, W, H, f_hz, res=res)
            frac = coverage_fraction(P, threshold_dbm)
            if frac > best[1]:
                best = ((float(x), float(y)), frac)
    return best


# ── thermal ──────────────────────────────────────────────────────────

def wien_peak_um(T):
    if T <= 0:
        raise ValueError("temperature must be positive")
    return wien_constant() / T * 1e6


def band_radiance(T, lo_um=8.0, hi_um=14.0, emissivity=1.0, n=2000):
    """Radiance (W/m^2/sr) in [lo, hi] um: trapezoid over Planck (W/m^2/sr/nm)."""
    lam_nm = np.linspace(lo_um * 1e3, hi_um * 1e3, n)
    return emissivity * float(np.trapezoid(planck_spectral_radiance(lam_nm, T), lam_nm))
