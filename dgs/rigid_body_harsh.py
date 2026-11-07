"""A sensor mast in a harsh environment: matter and light loads, rigid-body equilibrium, cost per operation.

In plain words: an optical sensor (a camera or light detector) sits on top of a pole that is
pinned at the base and held up by one guy cable. In calm weather the only loads are WEIGHT
(matter). In a storm the wind pushes on the head and the pole (drag = 1/2 rho Cd A v^2), and ice
coats everything, adding weight AND making the parts fatter so the wind pushes harder. Sunlight
also pushes -- radiation pressure I/c -- but on the sensor head about 200,000 times weaker than
even a 1 m/s breeze (and ~2e7 times weaker than a 10 m/s wind): light carries momentum, just very
little of it.

Equilibrium (sum Fx = sum Fy = sum M = 0) gives the pin reactions and the cable tension
(dgs.rigid_body_core, pure Python). Because equilibrium is linear in the loads and drag is
quadratic in wind speed, tension follows T(v) = T0 + c v^2 exactly; a least-squares regression
recovers c from simulated (v, T) pairs, the same way you would from field data.

"Cost per operation": time N solves for several N and regress time = overhead + slope * N. The
slope is the cost per solve; dividing by the counted arithmetic operations per solve gives the
cost per operation. Done here for pure Python, for NumPy one-at-a-time (dgs.free_body_diagram),
and for NumPy batched; the same pure-Python code is packaged for a BBC micro:bit in
projects/microbit_statics/main.py, which measures and regresses on the device.

SI units. Load and strength numbers are typical textbook values, labeled as assumptions.
"""

import time

import numpy as np

from dgs import rigid_body_core as core
from dgs.free_body_diagram import solve_reactions

RHO_AIR = 1.225          # kg/m^3 (sea level, 15 C); colder storm air is denser
RHO_ICE = 900.0          # kg/m^3, glaze ice
G = 9.81
C_LIGHT = 299792458.0

MAST = dict(H=3.0, d=2.0, pole_D=0.06, pole_mass=12.0, cd_pole=1.2,
            head_mass=6.0, head_area=0.04, head_surface=0.25, cd_head=1.1)


def drag(v, area, cd, rho=RHO_AIR):
    """Aerodynamic drag 1/2 rho Cd A v^2 (N)."""
    if area < 0 or cd < 0:
        raise ValueError("area and cd must be non-negative")
    return 0.5 * rho * cd * area * np.asarray(v, float) ** 2


def radiation_force(irradiance, area, reflectivity=0.0):
    """Radiation pressure force (1 + R) I A / c (N): light's momentum flux."""
    if not 0 <= reflectivity <= 1:
        raise ValueError("reflectivity must be in [0, 1]")
    return (1 + reflectivity) * irradiance * area / C_LIGHT


def ice_mass_pole(H, D, t_ice):
    """Radial glaze ice of thickness t on a pole of diameter D and height H (kg)."""
    return RHO_ICE * np.pi * ((D / 2 + t_ice) ** 2 - (D / 2) ** 2) * H


def mast_loads(v=0.0, t_ice=0.0, irradiance=0.0, mast=MAST):
    """Known forces (Fx, Fy, x, y) on the mast. Wind blows in +x; the guy cable is on the upwind side."""
    m = mast
    H, D = m["H"], m["pole_D"]
    w_pole = (m["pole_mass"] + ice_mass_pole(H, D, t_ice)) * G
    w_head = (m["head_mass"] + RHO_ICE * m["head_surface"] * t_ice) * G
    head_side = np.sqrt(m["head_area"])
    F_head = float(drag(v, (head_side + 2 * t_ice) ** 2, m["cd_head"]))
    F_pole = float(drag(v, (D + 2 * t_ice) * H, m["cd_pole"]))
    F_light = float(radiation_force(irradiance, m["head_area"]))
    return [(0.0, -w_pole, 0.0, H / 2), (0.0, -w_head, 0.0, H),
            (F_pole, 0.0, 0.0, H / 2), (F_head + F_light, 0.0, 0.0, H)]


def solve_mast(v=0.0, t_ice=0.0, irradiance=0.0, mast=MAST):
    """[Ax, Ay, T] via the pure-Python core."""
    return core.guyed_mast_reactions(mast["H"], mast["d"], mast_loads(v, t_ice, irradiance, mast))


def solve_mast_numpy(v=0.0, t_ice=0.0, irradiance=0.0, mast=MAST):
    """Same problem through dgs.free_body_diagram.solve_reactions (NumPy) as an independent check."""
    H, d = mast["H"], mast["d"]
    L = np.hypot(d, H)
    known = [((Fx, Fy), (x, y)) for Fx, Fy, x, y in mast_loads(v, t_ice, irradiance, mast)]
    reactions = [((1.0, 0.0), (0.0, 0.0)), ((0.0, 1.0), (0.0, 0.0)), ((-d / L, -H / L), (0.0, H))]
    mags, _ = solve_reactions(known, reactions)
    return mags


def equilibrium_residual(v=0.0, t_ice=0.0, irradiance=0.0, mast=MAST):
    """Max |sum F| or |sum M| with the solved reactions added back (should be ~0)."""
    H, d = mast["H"], mast["d"]
    L = np.hypot(d, H)
    Ax, Ay, T = solve_mast(v, t_ice, irradiance, mast)
    allf = mast_loads(v, t_ice, irradiance, mast) + [(Ax, 0, 0, 0), (0, Ay, 0, 0), (-T * d / L, -T * H / L, 0, H)]
    fx = sum(f[0] for f in allf)
    fy = sum(f[1] for f in allf)
    mz = sum(f[2] * f[1] - f[3] * f[0] for f in allf)
    return max(abs(fx), abs(fy), abs(mz))


def regress_quadratic(v, T):
    """Least-squares fit T = T0 + c v^2; returns (T0, c, r^2)."""
    v, T = np.asarray(v, float), np.asarray(T, float)
    X = np.column_stack([np.ones_like(v), v ** 2])
    coef, *_ = np.linalg.lstsq(X, T, rcond=None)
    resid = T - X @ coef
    r2 = 1 - resid.var() / T.var()
    return float(coef[0]), float(coef[1]), float(r2)


def ops_per_solve():
    """Arithmetic operations (+ - * /) counted inside one guyed-mast solve (4 loads)."""
    core.OPS[0] = 0
    solve_mast(20.0, 0.01)
    return core.OPS[0]


def time_batches(fn, Ns, repeats=3):
    """Best-of-repeats wall time to run fn(N) for each N."""
    out = []
    for N in Ns:
        best = np.inf
        for _ in range(repeats):
            t0 = time.perf_counter()
            fn(N)
            best = min(best, time.perf_counter() - t0)
        out.append(best)
    return np.array(out)


def regress_line(N, t):
    """time = overhead + slope * N by least squares; returns (overhead_s, slope_s, r^2)."""
    N, t = np.asarray(N, float), np.asarray(t, float)
    slope, intercept = np.polyfit(N, t, 1)
    r2 = 1 - np.var(t - (intercept + slope * N)) / np.var(t)
    return float(intercept), float(slope), float(r2)


def cost_per_operation(Ns_scalar=(2000, 4000, 8000, 16000), Ns_batch=(10000, 50000, 200000, 800000)):
    """Measure cost per solve for three implementations on THIS machine and regress.
    Returns a list of dicts. Timing is machine- and load-dependent."""
    loads = mast_loads(30.0, 0.01)
    H, d = MAST["H"], MAST["d"]
    L = np.hypot(d, H)
    reac_np = [((1.0, 0.0), (0.0, 0.0)), ((0.0, 1.0), (0.0, 0.0)), ((-d / L, -H / L), (0.0, H))]
    known_np = [((Fx, Fy), (x, y)) for Fx, Fy, x, y in loads]

    def pure(N):
        for _ in range(N):
            core.guyed_mast_reactions(H, d, loads)

    def numpy_one(N):
        for _ in range(N):
            solve_reactions(known_np, reac_np)

    A0, b0 = core.moment_rows(loads, [(1.0, 0.0, 0.0, 0.0), (0.0, 1.0, 0.0, 0.0), (-d / L, -H / L, 0.0, H)])
    A0, b0 = np.array(A0), np.array(b0)

    def numpy_batch(N):
        np.linalg.solve(np.broadcast_to(A0, (N, 3, 3)), np.broadcast_to(b0, (N, 3))[..., None])

    nops = ops_per_solve()
    rows = []
    for name, fn, Ns in [("pure Python (micro:bit-portable)", pure, Ns_scalar),
                         ("NumPy, one solve per call", numpy_one, Ns_scalar),
                         ("NumPy, batched solve", numpy_batch, Ns_batch)]:
        t = time_batches(fn, Ns)
        ov, sl, r2 = regress_line(Ns, t)
        rows.append(dict(implementation=name, Ns=list(Ns), times_s=t.tolist(), overhead_s=ov,
                         s_per_solve=sl, r2=r2, ops_per_solve=nops, s_per_op=sl / nops))
    return rows
