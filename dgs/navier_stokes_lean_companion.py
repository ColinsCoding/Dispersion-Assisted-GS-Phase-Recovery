"""SymPy companion to the Lean 4 Navier-Stokes statements in openai/NavierStokesAndEuler.

The Lean file ``ComparatorChallenges/NavierStokes.lean`` defines what a
"global smooth solution" means and states Clay alternatives (C) and (D). This
module rebuilds each of those definitions in SymPy so they can be read and
*checked on examples*:

    Lean                                      SymPy here
    ----------------------------------------  ---------------------------------
    divergence v x := (fderiv v x).trace      divergence(v) = trace(jacobian(v))
    divergence_add / _smul / _const / _zero   divergence_lemmas()
    IsOnePeriodic f                           is_one_periodic(expr)
    (fderiv v x) (v x)   [= (v.grad) v]       convective(v) = J(v) v
    navier_stokes field of the structure      ns_residual(v, p, f, nu)

What SymPy does here is *check* (an identity simplifies to zero, or a
concrete flow satisfies the equation). What Lean does is *prove* (a kernel
checks every step for every function in the class). A SymPy check on
Taylor-Green or ABC flow says nothing about the breakdown theorems, which are
existence statements about some other, carefully built initial data.

The Lean sources are read from a sibling clone at ``../NavierStokesAndEuler``
(or the path in the environment variable ``NSE_REPO``). If the clone is
missing, the functions that read Lean return ``None`` and everything else
still works.
"""
import os
import pathlib
import re

import numpy as np
import sympy as sp

x, y, z, t = sp.symbols("x y z t", real=True)
nu = sp.symbols("nu", positive=True)
lam = sp.symbols("lambda", positive=True)
X = (x, y, z)

CHALLENGE_FILE = pathlib.Path("ComparatorChallenges") / "NavierStokes.lean"
SOLUTION_FILE = pathlib.Path("NavierStokes") / "ComparatorSolution.lean"


# ── reading the Lean repository ──────────────────────────────────────

def lean_repo(root=None):
    """Path to the NavierStokesAndEuler clone, or None if it isn't there."""
    if root is None:
        root = os.environ.get("NSE_REPO")
    if root is None:
        root = pathlib.Path(__file__).resolve().parents[2] / "NavierStokesAndEuler"
    root = pathlib.Path(root)
    return root if (root / CHALLENGE_FILE).is_file() else None


def lean_statement(name, root=None, path=CHALLENGE_FILE):
    """Source text of the Lean declaration `name` (def/structure/theorem), with
    its doc comment, or None if the repo or the declaration is missing."""
    repo = lean_repo(root)
    if repo is None:
        return None
    src = (repo / path).read_text(encoding="utf-8")
    m = re.search(r"(/--(?:(?!-/).)*-/\s*)?(?:theorem|def|structure|lemma)\s+"
                  + re.escape(name) + r"\b", src, re.S)
    if m is None:
        return None
    start = m.start()
    nxt = re.compile(r"\n(?=/--|theorem |def |structure |lemma |end |noncomputable |@\[)")
    stop = nxt.search(src, m.end())
    return src[start:stop.start() if stop else len(src)].rstrip()


def repo_inventory(root=None):
    """Counts that describe the clone: Lean files and lines in the proof trees,
    and `sorry` occurrences in the proof trees vs the challenge file."""
    repo = lean_repo(root)
    if repo is None:
        return None
    word = re.compile(r"\bsorry\b")
    out = {}
    for tree in ("NavierStokes", "Euler"):
        files = list((repo / tree).rglob("*.lean"))
        text = [f.read_text(encoding="utf-8") for f in files]
        out[tree] = {"lean_files": len(files),
                     "lines": sum(s.count("\n") for s in text),
                     "sorry": sum(len(word.findall(s)) for s in text)}
    chal = (repo / CHALLENGE_FILE).read_text(encoding="utf-8")
    # the challenge file's doc comment mentions the word once; count tactic uses only
    out["challenge_sorry_tactics"] = len(re.findall(r"^\s*sorry\s*$", chal, re.M))
    out["toolchain"] = (repo / "lean-toolchain").read_text().strip()
    return out


# ── vector calculus, written the way Lean writes it ──────────────────

def _vec(v):
    v = sp.Matrix(v)
    if v.shape != (3, 1):
        raise ValueError("velocity must have 3 components")
    return v


def jacobian(v):
    """J_ij = d v_i / d x_j: the matrix of fderiv R v x."""
    return _vec(v).jacobian(X)


def divergence(v):
    """Lean: divergence v x := (fderiv R v x).trace R (R^n)."""
    return sp.simplify(jacobian(v).trace())


def convective(v):
    """Lean writes the nonlinear term as (fderiv R (v . t) x) (v x t), i.e. the
    Jacobian applied to the velocity: J v. Componentwise this is (v.grad) v."""
    v = _vec(v)
    return jacobian(v) * v


def laplacian(v):
    v = _vec(v)
    return v.applyfunc(lambda c: sum(sp.diff(c, s, 2) for s in X))


def gradient(p):
    return sp.Matrix([sp.diff(p, s) for s in X])


def curl(v):
    v = _vec(v)
    return sp.Matrix([sp.diff(v[2], y) - sp.diff(v[1], z),
                      sp.diff(v[0], z) - sp.diff(v[2], x),
                      sp.diff(v[1], x) - sp.diff(v[0], y)])


def ns_residual(v, p, f, viscosity=nu):
    """The `navier_stokes` field of the Lean structure, moved to one side:
        dv/dt + (fderiv v)(v) - nu Lap v + grad p - f.
    Zero vector <=> (v, p, f) satisfies Navier-Stokes pointwise."""
    v, f = _vec(v), _vec(f)
    r = v.diff(t) + convective(v) - viscosity * laplacian(v) + gradient(p) - f
    return sp.simplify(r)


def is_one_periodic(expr):
    """Lean: IsOnePeriodic f := forall x i, f (x + e_i) = f x. Checked
    symbolically for each coordinate direction and each component."""
    comps = list(sp.Matrix([expr])) if not isinstance(expr, sp.MatrixBase) else list(expr)
    return all(sp.simplify(sp.expand_trig(c.subs(s, s + 1) - c)) == 0
               for c in comps for s in X)


# ── the four divergence lemmas, for generic smooth fields ────────────

def _generic_field(name):
    return sp.Matrix([sp.Function(f"{name}{i}")(x, y, z) for i in (1, 2, 3)])


def divergence_lemmas():
    """Each Lean lemma as an expression that must simplify to 0, for arbitrary
    differentiable components (SymPy's undefined Functions)."""
    v, w = _generic_field("v"), _generic_field("w")
    c = sp.symbols("c", real=True)
    k = sp.Matrix(sp.symbols("k1 k2 k3", real=True))
    return {
        "divergence_zero":  divergence(sp.zeros(3, 1)),
        "divergence_const": divergence(k),
        "divergence_add":   sp.simplify(divergence(v + w) - divergence(v) - divergence(w)),
        "divergence_smul":  sp.simplify(divergence(c * v) - c * divergence(v)),
    }


def trace_is_coordinate_free(theta=None):
    """divergence is a trace, and a trace is invariant under change of basis:
    tr(Q J Q^T) = tr(J). Checked for a rotation about z by angle theta and a
    generic Jacobian."""
    th = sp.symbols("theta", real=True) if theta is None else theta
    Q = sp.Matrix([[sp.cos(th), -sp.sin(th), 0], [sp.sin(th), sp.cos(th), 0], [0, 0, 1]])
    J = sp.Matrix(3, 3, lambda i, j: sp.Symbol(f"J{i+1}{j+1}"))
    return sp.simplify((Q * J * Q.T).trace() - J.trace())


def convective_matches_textbook(v=None):
    """(fderiv v)(v) equals the textbook (v . grad) v, component by component."""
    v = _generic_field("v") if v is None else _vec(v)
    textbook = sp.Matrix([sum(v[j] * sp.diff(v[i], X[j]) for j in range(3)) for i in range(3)])
    return sp.simplify(convective(v) - textbook)


# ── example flows ────────────────────────────────────────────────────

TWO_PI = 2 * sp.pi


def taylor_green(amplitude=1, k=TWO_PI):
    """Decaying Taylor-Green vortex, an exact unforced NS solution. With
    k = 2 pi it is 1-periodic, matching Lean's R^3 / Z^3 convention."""
    A = sp.sympify(amplitude)
    decay = sp.exp(-2 * nu * k**2 * t)
    v = sp.Matrix([A * sp.sin(k * x) * sp.cos(k * y),
                   -A * sp.cos(k * x) * sp.sin(k * y), 0]) * decay
    p = A**2 / 4 * (sp.cos(2 * k * x) + sp.cos(2 * k * y)) * decay**2
    return v, p, sp.zeros(3, 1)


def abc_flow(A=1, B=1, C=1, k=TWO_PI):
    """Arnold-Beltrami-Childress flow: curl v = k v. Steady NS solution when
    driven by f = nu k^2 v and p = -|v|^2 / 2 (the Bernoulli pressure)."""
    A, B, C = map(sp.sympify, (A, B, C))
    v = sp.Matrix([A * sp.sin(k * z) + C * sp.cos(k * y),
                   B * sp.sin(k * x) + A * sp.cos(k * z),
                   C * sp.sin(k * y) + B * sp.cos(k * x)])
    p = -sp.expand(v.dot(v)) / 2
    f = nu * k**2 * v
    return v, p, f


def gaussian_swirl(width=1):
    """Divergence-free, Schwartz-class initial data on R^3: the curl of
    (0, 0, exp(-r^2/w^2)). Satisfies InitialVelocityConditionDecay for all K."""
    w = sp.sympify(width)
    psi = sp.exp(-(x**2 + y**2 + z**2) / w**2)
    return curl(sp.Matrix([0, 0, psi]))


def algebraic_swirl():
    """Divergence-free but only algebraically decaying: |u| ~ 1/r^3. Fails the
    Lean decay condition once K > 3."""
    psi = 1 / (1 + x**2 + y**2 + z**2)
    return curl(sp.Matrix([0, 0, psi]))


def decay_profile(v, K, radii=None, direction=(1, 1, 0)):
    """max over a ray of |v(x)| (1 + |x|)^K. The Lean condition
    `forall K, exists C, |v x| <= C / (1+|x|)^K` holds iff this stays bounded
    as the ray goes to infinity, for every K."""
    radii = np.logspace(-1, 3, 400) if radii is None else np.asarray(radii, float)
    d = np.asarray(direction, float)
    d = d / np.linalg.norm(d)
    fn = sp.lambdify(X, list(_vec(v)), "numpy")
    pts = radii[:, None] * d
    vals = np.array([np.broadcast_to(np.asarray(c, float), radii.shape)
                     for c in fn(pts[:, 0], pts[:, 1], pts[:, 2])])
    return radii, np.linalg.norm(vals, axis=0) * (1 + radii) ** K


# ── energy identity and scaling ─────────────────────────────────────

def _box_integral(expr):
    """Integral over one period cell [0,1]^3 of the torus."""
    return sp.simplify(sp.integrate(expr, (x, 0, 1), (y, 0, 1), (z, 0, 1)))


def energy_balance(v, f=None):
    """For a periodic flow, return (dE/dt, -nu int|grad v|^2 + int f.v) where
    E = 1/2 int |v|^2. Equality is the energy identity; the pressure and the
    convective term integrate to zero because v is divergence-free and periodic."""
    v = _vec(v)
    f = sp.zeros(3, 1) if f is None else _vec(f)
    E = _box_integral(v.dot(v) / 2)
    dissipation = _box_integral(sum(sp.diff(v[i], s)**2 for i in range(3) for s in X))
    work = _box_integral(f.dot(v))
    return sp.simplify(sp.diff(E, t)), sp.simplify(-nu * dissipation + work)


def scaled_solution(v, p, f):
    """Navier-Stokes scaling: if (v, p, f) solves NS, so does
        v_l = l v(l x, l^2 t),  p_l = l^2 p(l x, l^2 t),  f_l = l^3 f(l x, l^2 t)."""
    sub = {x: lam * x, y: lam * y, z: lam * z, t: lam**2 * t}
    sc = lambda e: e.xreplace(sub)
    return lam * _vec(v).applyfunc(sc), lam**2 * sc(p), lam**3 * _vec(f).applyfunc(sc)


def energy_scaling_exponent(dim=3):
    """Exponent a in  int |v_l|^2 dx = l^a int |v|^2 dx  on R^dim.
    v_l = l v(l x) and the change of variables dx = l^-dim dx' give a = 2 - dim.
    a < 0 in 3D: the energy shrinks at small scales, so the energy bound gives
    no control there (the equation is 'supercritical')."""
    d = sp.Integer(dim)
    # |v_l|^2 contributes l^2; dx -> l^-d dx'
    return sp.simplify(sp.log(lam**2 * lam**(-d)) / sp.log(lam))


def vortex_stretching_2d():
    """Vorticity equation stretching term (omega . grad) v for a generic planar
    flow v = (u(x,y,t), w(x,y,t), 0). It vanishes identically -- the reason 2D
    Navier-Stokes has global smooth solutions and the open problem is in 3D."""
    u = sp.Function("u")(x, y, t)
    w = sp.Function("w")(x, y, t)
    v = sp.Matrix([u, w, 0])
    om = curl(v)
    return sp.simplify(jacobian(v) * om)


# ── numerical: synthetic cell imaging and tracking in a known flow ───
#
# Everything below is synthetic. Bright Gaussian "cells" ride a 2D
# Taylor-Green flow on the unit torus (an exact NS solution, so the true
# velocity is known everywhere). We render noisy frames, detect the cells,
# link detections into tracks, and compare the tracked velocity with the
# true field. Then the Lean `div_free` condition is used as a constraint:
# projecting a noisy velocity field onto divergence-free fields removes
# noise without touching the signal.

def tg2d_velocity(px, py, time, viscosity=0.01, k=2 * np.pi, amplitude=1.0):
    """Numeric 2D Taylor-Green velocity (the x, y components of taylor_green)."""
    d = amplitude * np.exp(-2 * viscosity * k**2 * time)
    return (d * np.sin(k * px) * np.cos(k * py),
            -d * np.cos(k * px) * np.sin(k * py))


def advect(pos, t0, dt, n_steps, viscosity=0.01):
    """RK4 particle paths in the Taylor-Green flow, wrapped to [0,1)^2.
    pos: (N, 2). Returns (n_steps+1, N, 2)."""
    if n_steps < 1 or dt <= 0:
        raise ValueError("need n_steps >= 1 and dt > 0")
    f = lambda q, s: np.stack(tg2d_velocity(q[:, 0], q[:, 1], s, viscosity), axis=1)
    out = [np.mod(np.asarray(pos, float), 1.0)]
    q, s = out[0].copy(), t0
    for _ in range(n_steps):
        k1 = f(q, s)
        k2 = f(q + dt / 2 * k1, s + dt / 2)
        k3 = f(q + dt / 2 * k2, s + dt / 2)
        k4 = f(q + dt * k3, s + dt)
        q = q + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        s += dt
        out.append(np.mod(q, 1.0))
    return np.array(out)


def render_frame(pos, n_pix=128, sigma_pix=1.5, noise=0.05, rng=None):
    """Periodic image (rows = y, cols = x) of Gaussian blobs at `pos` plus
    white Gaussian noise."""
    rng = np.random.default_rng(0) if rng is None else rng
    g = (np.arange(n_pix) + 0.5) / n_pix
    img = np.zeros((n_pix, n_pix))
    s = sigma_pix / n_pix
    for cx, cy in pos:
        dx = (g - cx + 0.5) % 1.0 - 0.5        # minimum-image distance
        dy = (g - cy + 0.5) % 1.0 - 0.5
        img += np.exp(-dy[:, None]**2 / (2 * s * s)) * np.exp(-dx[None, :]**2 / (2 * s * s))
    return img + noise * rng.standard_normal(img.shape)


def detect_cells(img, threshold=0.5, max_area=None):
    """Intensity-weighted centroids (x, y) in [0,1) of connected bright
    regions. Periodic wrap: label a 2x2-tiled image and keep the copy of each
    blob whose centroid lands in the central tile. Blobs with more than
    `max_area` pixels are dropped: they are usually two touching cells, and
    their single centroid would swap track identities."""
    from scipy import ndimage
    n = img.shape[0]
    tiled = np.tile(img, (2, 2))
    lab, nlab = ndimage.label(tiled > threshold)
    if nlab == 0:
        return np.zeros((0, 2))
    idx = range(1, nlab + 1)
    w = np.clip(tiled, 0, None)
    cy, cx = np.array(ndimage.center_of_mass(w, lab, idx)).T
    area = np.asarray(ndimage.sum(np.ones_like(tiled), lab, idx))
    lo, hi = n / 2 - 0.5, 3 * n / 2 - 0.5
    keep = (cx >= lo) & (cx < hi) & (cy >= lo) & (cy < hi)
    if max_area is not None:
        keep &= area <= max_area
    return np.mod(np.stack([(cx[keep] + 0.5) / n, (cy[keep] + 0.5) / n], axis=1), 1.0)


def link_nearest(prev, curr, max_disp=0.02):
    """Index into `curr` for each row of `prev` (or -1), by minimum-image
    nearest neighbour within max_disp."""
    if len(curr) == 0:
        return -np.ones(len(prev), int)
    d = (curr[None, :, :] - prev[:, None, :] + 0.5) % 1.0 - 0.5
    r = np.linalg.norm(d, axis=2)
    j = np.argmin(r, axis=1)
    return np.where(r[np.arange(len(prev)), j] <= max_disp, j, -1)


def track_and_compare(n_cells=40, n_frames=30, dt=0.01, viscosity=0.01, noise=0.05,
                      n_pix=128, seed=1, max_area=None, max_disp=0.02, stride=1):
    """Full synthetic pipeline: advect -> render -> detect -> link -> velocity.
    Returns true paths, tracks, tracked and true velocities at the same
    midpoints, the RMS and median velocity error relative to the RMS speed,
    and the fraction of tracks that ever jump (error > 1, an identity swap).
    `stride` > 1 differences positions `stride` frames apart: centroid noise
    is divided by `stride`, at the cost of chord-vs-tangent (curvature) error."""
    rng = np.random.default_rng(seed)
    start = rng.random((n_cells, 2))
    paths = advect(start, 0.0, dt, n_frames - 1, viscosity)
    dets = [detect_cells(render_frame(p, n_pix, noise=noise, rng=rng), max_area=max_area)
            for p in paths]
    tracks = [dets[0]]
    alive = np.ones(len(dets[0]), bool)
    for k in range(1, n_frames):
        j = link_nearest(tracks[-1], dets[k], max_disp)
        alive &= j >= 0
        tracks.append(np.where((j >= 0)[:, None], dets[k][np.maximum(j, 0)], tracks[-1]))
    tr = np.array(tracks)[:, alive]
    disp = (tr[stride:] - tr[:-stride] + 0.5) % 1.0 - 0.5     # displacement over `stride` frames
    v_meas = disp / (stride * dt)
    mid = np.mod(tr[:-stride] + disp / 2, 1.0)
    tm = (np.arange(n_frames - stride) + stride / 2) * dt
    ut, vt = tg2d_velocity(mid[..., 0], mid[..., 1], tm[:, None], viscosity)
    v_true = np.stack([ut, vt], axis=-1)
    e = np.linalg.norm(v_meas - v_true, axis=-1)
    rms = np.sqrt(np.mean(np.sum(v_true**2, -1)))
    return {"paths": paths, "tracks": tr, "v_meas": v_meas, "v_true": v_true,
            "n_tracked": int(alive.sum()), "n_detected0": len(dets[0]), "n_cells": n_cells,
            "rel_rms_error": float(np.sqrt(np.mean(e**2)) / rms),
            "rel_median_error": float(np.median(e) / rms),
            "swap_fraction": float(np.mean(e.max(axis=0) > 1.0)) if e.size else 0.0,
            "pixel_size": 1.0 / n_pix}


def _wavenumbers(n, drop_nyquist=False):
    k = 2 * np.pi * np.fft.fftfreq(n, d=1.0 / n)
    if drop_nyquist and n % 2 == 0:
        k[n // 2] = 0.0               # the Nyquist mode has no well-defined real derivative
    return np.meshgrid(k, k)          # KX varies along columns (x), KY along rows (y)


def spectral_divergence(u, v):
    """Divergence of a periodic grid field on [0,1)^2 via FFT (Nyquist mode
    dropped, the standard convention for odd-order spectral derivatives)."""
    KX, KY = _wavenumbers(u.shape[0], drop_nyquist=True)
    return np.real(np.fft.ifft2(1j * KX * np.fft.fft2(u) + 1j * KY * np.fft.fft2(v)))


def leray_project(u, v):
    """Project a periodic 2D field onto divergence-free fields: in Fourier
    space P(k) = I - k k^T / |k|^2. The mean flow (k = 0) is kept."""
    KX, KY = _wavenumbers(u.shape[0], drop_nyquist=True)
    K2 = KX**2 + KY**2
    K2[K2 == 0] = 1.0                 # k = 0 (and dropped Nyquist) modes: k.U = 0 there anyway
    U, V = np.fft.fft2(u), np.fft.fft2(v)
    dot = (KX * U + KY * V) / K2
    return np.real(np.fft.ifft2(U - KX * dot)), np.real(np.fft.ifft2(V - KY * dot))


def projection_denoise_demo(n=64, sigma=0.2, seed=0, viscosity=0.01, time=0.0):
    """Noisy PIV-like grid velocity = Taylor-Green + white noise, then the
    Leray projection. The signal is already divergence-free, so it passes
    unchanged; white noise loses its gradient part, one of the two
    polarisations at each wavevector, so the expected error ratio in 2D is
    sqrt(1/2) ~ 0.707."""
    rng = np.random.default_rng(seed)
    g = (np.arange(n) + 0.5) / n
    PX, PY = np.meshgrid(g, g)
    u0, v0 = tg2d_velocity(PX, PY, time, viscosity)
    un = u0 + sigma * rng.standard_normal(u0.shape)
    vn = v0 + sigma * rng.standard_normal(v0.shape)
    up, vp = leray_project(un, vn)
    e = lambda a, b: np.sqrt(np.mean((a - u0)**2 + (b - v0)**2))
    rms_div = lambda a, b: float(np.sqrt(np.mean(spectral_divergence(a, b)**2)))
    return {"err_noisy": float(e(un, vn)), "err_projected": float(e(up, vp)),
            "ratio": float(e(up, vp) / e(un, vn)),
            "div_true_rms": rms_div(u0, v0),
            "div_noisy_rms": rms_div(un, vn), "div_projected_rms": rms_div(up, vp),
            "signal_change": float(np.max(np.abs(np.array(leray_project(u0, v0)) - np.array([u0, v0])))),
            "fields": (PX, PY, u0, v0, un, vn, up, vp)}


def channel_reynolds(speed_m_s=1.0, width_m=50e-6, kinematic_viscosity=1.0e-6):
    """Re = U W / nu for a microfluidic imaging channel (water by default).
    Re << 1 means the convective term is negligible: Stokes flow."""
    if speed_m_s <= 0 or width_m <= 0 or kinematic_viscosity <= 0:
        raise ValueError("speed, width and viscosity must be positive")
    return speed_m_s * width_m / kinematic_viscosity



# ── checking the built Lean proofs (needs elan + a completed `lake build`) ──

DEFINITIONS_FILE = pathlib.Path("NavierStokes") / "ComparatorDefinitions.lean"


def lean_build_dir(root=None):
    """Directory holding a COMPLETED `lake build` of the repo (it needs .lake/).
    Order: argument, env NSE_BUILD, ~/lean/NavierStokesAndEuler (NTFS -- a build
    on an exFAT drive fills it with small files), then the sibling clone."""
    cands = [root, os.environ.get("NSE_BUILD"),
             pathlib.Path.home() / "lean" / "NavierStokesAndEuler", lean_repo()]
    for c in cands:
        if c and (pathlib.Path(c) / ".lake").is_dir() and (pathlib.Path(c) / SOLUTION_FILE).is_file():
            return pathlib.Path(c)
    return None


def _lake_exe():
    import shutil
    exe = shutil.which("lake")
    if exe:
        return exe
    home = pathlib.Path.home() / ".elan" / "bin" / ("lake.exe" if os.name == "nt" else "lake")
    return str(home) if home.is_file() else None


def lean_axioms(root=None, timeout=900, threads=4):
    """Re-elaborate NavierStokes/ComparatorSolution.lean against the built cache
    and parse its `#print axioms` output. Returns {theorem: [axioms]} or None if
    Lean or a completed build is unavailable. Takes ~20 s with the cache built."""
    import subprocess
    build, lake = lean_build_dir(root), _lake_exe()
    if build is None or lake is None:
        return None
    env = dict(os.environ, LEAN_NUM_THREADS=str(threads),
               PATH=str(pathlib.Path(lake).parent) + os.pathsep + os.environ.get("PATH", ""))
    out = subprocess.run([lake, "env", "lean", str(SOLUTION_FILE)], cwd=build, env=env,
                         capture_output=True, text=True, timeout=timeout)
    text = out.stdout + out.stderr
    found = re.findall(r"'([^']+)' depends on axioms: \[([^\]]*)\]", text)
    return {name: [a.strip() for a in ax.split(",") if a.strip()] for name, ax in found}


STANDARD_AXIOMS = {"propext", "Classical.choice", "Quot.sound"}


def _definition_lines(path):
    s = pathlib.Path(path).read_text(encoding="utf-8")
    s = re.sub(r"/-.*?-/", "", s, flags=re.S)
    s = re.sub(r"--.*", "", s)
    i, j = s.find("def divergence"), s.find("theorem navier_stokes_breakdown_R3")
    s = s[i:j if j > 0 else len(s)]
    skip = ("namespace", "end ", "open", "variable", "local notation", "import")
    return [ln.strip() for ln in s.splitlines() if ln.strip() and not ln.strip().startswith(skip)]


def definitions_match(root=None):
    """Compare the challenge file's definitions with the solution's
    ComparatorDefinitions.lean, ignoring comments, namespaces and notation lines.
    Returns (n_lines_challenge, n_lines_solution, differing_lines) or None."""
    import difflib
    repo = lean_repo(root) or lean_build_dir(root)
    if repo is None or not (pathlib.Path(repo) / DEFINITIONS_FILE).is_file():
        return None
    a = _definition_lines(pathlib.Path(repo) / CHALLENGE_FILE)
    b = _definition_lines(pathlib.Path(repo) / DEFINITIONS_FILE)
    diff = [d for d in difflib.unified_diff(a, b, lineterm="", n=0) if not d.startswith(("---", "+++", "@@"))]
    return len(a), len(b), diff

if __name__ == "__main__":
    print({k: v for k, v in divergence_lemmas().items()})
    print("TG residual", list(ns_residual(*taylor_green())))
    print("ABC residual", list(ns_residual(*abc_flow())))
    print("energy TG", energy_balance(taylor_green()[0]))
    print("inventory", repo_inventory())
