"""NumPy side of the 3D battle: an independent reference renderer, and 3D visualizations.

    python python/droid3d_np.py check  out/frame.bin            # re-render a dumped C++ frame in NumPy, compare
    python python/droid3d_np.py arena  out/droid3d_run.csv out/arena3d.gif
    python python/droid3d_np.py camera out/frames out/droid3d_run.csv out/camera.gif

The renderer here is written separately from the C++ (float64, every pixel at once with array operations)
using the same equations: terrain heightfield h(x, y), ray-marched then bisected; analytic sphere hits;
capsule = cylinder + two end spheres; Lambert + ambient shading. Agreement between the two is the
cross-check that both implement the same world.
"""
import configparser
import csv
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent.parent
OI = dict(orange="#E69F00", sky="#56B4E9", green="#009E73", yellow="#F0E442", blue="#0072B2", red="#D55E00", purple="#CC79A7", black="#000000")
T = dict(a1=0.9, k1=0.35, a2=0.5, k2=0.21, a3=0.35, kx3=0.55, ky3=0.3)       # same as d3::Terrain defaults


def load_defs(path=HERE / "assets" / "characters.def"):
    cp = configparser.ConfigParser(comment_prefixes=("#", ";"), delimiters=("=",))
    cp.optionxform = str
    cp.read(path, encoding="utf-8")
    return cp


def height(x, y):
    return (T["a1"] * np.sin(T["k1"] * x) * np.cos(0.7 * T["k1"] * y) + T["a2"] * np.sin(T["k2"] * (x + y))
            + T["a3"] * np.cos(T["kx3"] * x - T["ky3"] * y))


def terrain_normal(x, y):
    hx = (T["a1"] * T["k1"] * np.cos(T["k1"] * x) * np.cos(0.7 * T["k1"] * y) + T["a2"] * T["k2"] * np.cos(T["k2"] * (x + y))
          - T["a3"] * T["kx3"] * np.sin(T["kx3"] * x - T["ky3"] * y))
    hy = (-T["a1"] * 0.7 * T["k1"] * np.sin(T["k1"] * x) * np.sin(0.7 * T["k1"] * y) + T["a2"] * T["k2"] * np.cos(T["k2"] * (x + y))
          + T["a3"] * T["ky3"] * np.sin(T["kx3"] * x - T["ky3"] * y))
    n = np.stack([-hx, -hy, np.ones_like(hx)], -1)
    return n / np.linalg.norm(n, axis=-1, keepdims=True)


def unit(v):
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def camera_rays(cam, pan, tilt, fov, n):
    f = np.array([np.cos(tilt) * np.cos(pan), np.cos(tilt) * np.sin(pan), np.sin(tilt)])
    r = unit(np.cross(f, [0, 0, 1.0]))
    u = np.cross(r, f)
    s = np.tan(0.5 * fov)
    i = np.arange(n)
    px = (2 * (i + 0.5) / n - 1) * s
    py = (1 - 2 * (i + 0.5) / n) * s
    PX, PY = np.meshgrid(px, py)                       # rows = j (y down), columns = i
    return unit(f + PX[..., None] * r + PY[..., None] * u)


def vdot(a, b):
    return (a * b).sum(-1)


def hit_sphere(ro, rd, c, rad):
    oc = ro - c
    b = vdot(rd, oc)
    h = b * b - (vdot(oc, oc) - rad * rad)
    t = -b - np.sqrt(np.maximum(h, 0))
    return np.where((h >= 0) & (t > 1e-4), t, np.inf)


def hit_capsule(ro, rd, pa, pb, rad):
    ba, oa = pb - pa, ro - pa
    baba, bard, baoa = ba @ ba, vdot(rd, ba), vdot(oa, ba)
    rdoa, oaoa = vdot(rd, oa), vdot(oa, oa)
    a = baba - bard ** 2
    b = baba * rdoa - baoa * bard
    c = baba * oaoa - baoa ** 2 - rad ** 2 * baba
    h = b * b - a * c
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (-b - np.sqrt(np.maximum(h, 0))) / a
    y = baoa + t * bard
    body = np.where((h >= 0) & (a > 1e-9 * baba) & (y > 0) & (y < baba) & (t > 1e-4), t, np.inf)
    return np.minimum(body, np.minimum(hit_sphere(ro, rd, pa, rad), hit_sphere(ro, rd, pb, rad)))


def capsule_normal(p, pa, pb, rad):
    ba = pb - pa
    k = np.clip(vdot(p - pa, ba) / (ba @ ba), 0, 1)
    return (p - (pa + k[..., None] * ba)) / rad


def hit_rect(ro, rd, P):
    den = vdot(rd, P["n"])
    with np.errstate(divide="ignore", invalid="ignore"):
        t = vdot(P["c"] - ro, P["n"]) / den
    q = ro + np.where(np.isfinite(t), t, 0)[..., None] * rd - P["c"]
    ok = (np.abs(den) >= 1e-7) & (t > 1e-4) & (np.abs(vdot(q, P["u"])) <= P["hu"]) & (np.abs(vdot(q, P["v"])) <= P["hv"])
    return np.where(ok, t, np.inf)


def hit_terrain(ro, rd, tmax=80.0):
    """Vectorized over rays (per-ray origins allowed): step t += 0.05 + 0.02 t, then 14 bisections."""
    shape = rd.shape[:-1]
    ro = np.broadcast_to(ro, rd.shape)
    out = np.full(shape, np.inf)
    p0 = ro + 0.05 * rd
    active = (p0[..., 2] - height(p0[..., 0], p0[..., 1])) >= 0            # starting under ground: no hit
    t = 0.05
    while t < tmax and active.any():
        t_new = t + 0.05 + 0.02 * t
        p = ro + t_new * rd
        under = active & (p[..., 2] - height(p[..., 0], p[..., 1]) < 0)
        if under.any():
            lo, hi = np.full(shape, t), np.full(shape, t_new)
            for _ in range(14):
                m = 0.5 * (lo + hi)
                q = ro + m[..., None] * rd
                below = q[..., 2] - height(q[..., 0], q[..., 1]) < 0
                hi = np.where(below, m, hi)
                lo = np.where(below, lo, m)
            out = np.where(under, 0.5 * (lo + hi), out)
            active &= ~under
        t = t_new
    return out


SKY, GROUND, BODY, HEAD, SWORD, PANEL, POST = range(7)


def trace(sc, ro, rd, tmax=80.0):
    """Nearest hit for every ray: (t, id, panel index). Same object list and order as the C++ trace()."""
    shape = rd.shape[:-1]
    t = np.minimum(hit_terrain(ro, rd, tmax), tmax)
    ids = np.where(t < tmax, GROUND, SKY)
    kidx = np.full(shape, -1)

    def take(tc, idv, k=-1):
        nonlocal t, ids, kidx
        better = tc < t
        t = np.where(better, tc, t)
        ids = np.where(better, idv, ids)
        kidx = np.where(better, k, kidx)

    take(hit_sphere(ro, rd, sc["body"][:3], sc["body"][3]), BODY)
    take(hit_sphere(ro, rd, sc["head"][:3], sc["head"][3]), HEAD)
    take(hit_capsule(ro, rd, sc["sword"][:3], sc["sword"][3:6], sc["sword"][6]), SWORD)
    for k, P in enumerate(sc["panels"]):
        take(hit_rect(ro, rd, P), PANEL, k)
        take(hit_capsule(ro, rd, P["base"], P["c"] - P["hv"] * P["v"], 0.04), POST, k)
    return t, ids, kidx


HORIZON = np.array([0.72, 0.80, 0.90])


def sky(sc, rd):
    up = np.clip(rd[..., 2], 0, 1)
    col = HORIZON + np.sqrt(up)[..., None] * (np.array([0.20, 0.42, 0.82]) - HORIZON)
    mu = np.clip(vdot(rd, sc["sun"]), 0, 1)
    col = col + (0.18 * mu ** 4)[..., None] * np.array([0.55, 0.50, 0.42]) + (0.35 * mu ** 24)[..., None] * np.array([1.0, 0.97, 0.90])
    col = np.where((mu > 0.9993)[..., None], np.array([1.0, 0.99, 0.95]), col)
    below = np.clip(-8 * rd[..., 2], 0, 1)[..., None]
    col = np.where((rd[..., 2] < 0)[..., None], col + below * (0.92 * HORIZON - col), col)
    return np.clip(col, 0, 1)


def render(sc):
    n, cam = sc["n"], sc["cam"]
    rd = camera_rays(cam, sc["pan"], sc["tilt"], sc["fov"], n)
    t, ids, kidx = trace(sc, cam, rd)
    hit = ids != SKY
    p = cam + np.where(hit, t, 0.0)[..., None] * rd
    nrm = np.zeros_like(p)
    nrm[..., 2] = 1.0
    alb = np.zeros_like(p)
    spec_k = np.zeros(p.shape[:-1])
    shin = np.ones(p.shape[:-1])
    g = ids == GROUND
    nrm = np.where(g[..., None], terrain_normal(p[..., 0], p[..., 1]), nrm)
    kk = np.clip(0.5 + 0.4 * p[..., 2], 0, 1)
    alb = np.where(g[..., None], np.stack([0.28 + 0.17 * kk, 0.46 - 0.08 * kk, 0.20 + 0.05 * kk], -1), alb)
    for idv, key, colr in ((BODY, "body", [0.95, 0.52, 0.10]), (HEAD, "head", [0.90, 0.72, 0.56])):
        m = ids == idv
        nrm = np.where(m[..., None], unit(p - sc[key][:3]), nrm)
        alb = np.where(m[..., None], np.array(colr), alb)
    m = ids == SWORD
    nrm = np.where(m[..., None], capsule_normal(p, sc["sword"][:3], sc["sword"][3:6], sc["sword"][6]), nrm)
    alb = np.where(m[..., None], np.array([1.0, 0.92, 0.30]), alb)
    for k, P in enumerate(sc["panels"]):
        m = (ids == POST) & (kidx == k)
        top = P["c"] - P["hv"] * P["v"]
        nrm = np.where(m[..., None], capsule_normal(p, P["base"], top, 0.04), nrm)
        alb = np.where(m[..., None], np.array([0.62, 0.64, 0.66]), alb)
        spec_k = np.where(m, 0.2, spec_k)
        shin = np.where(m, 20.0, shin)
        m = (ids == PANEL) & (kidx == k)
        if m.any():
            a = vdot(p - P["c"], P["u"]) / P["hu"]
            b = vdot(p - P["c"], P["v"]) / P["hv"]
            ga, gb = (a + 1) * 5, (b + 1) * 3
            line = ((ga - np.floor(ga)) < 0.06) | ((gb - np.floor(gb)) < 0.08)
            frame = (np.abs(a) > 0.96) | (np.abs(b) > 0.94)
            c_pan = np.where(frame[..., None], [0.70, 0.72, 0.74], np.where(line[..., None], [0.55, 0.58, 0.62], [0.05, 0.10, 0.26]))
            face = np.where((vdot(rd, P["n"]) < 0)[..., None], P["n"], -P["n"])
            nrm = np.where(m[..., None], face, nrm)
            alb = np.where(m[..., None], c_pan, alb)
            spec_k = np.where(m, np.where(frame, 0.3, 0.9), spec_k)
            shin = np.where(m, 140.0, shin)
    sun = sc["sun"]
    ts, ids_s, _ = trace(sc, p + 2e-3 * nrm, np.broadcast_to(sun, p.shape))      # shadow rays
    lit = (ids_s == SKY).astype(float)
    lam = np.clip(vdot(nrm, sun), 0, 1) * lit
    nz = nrm[..., 2]
    ambient = (0.30 + 0.20 * nz)[..., None] * np.array([0.66, 0.70, 0.78]) + (0.10 - 0.08 * nz)[..., None] * np.array([0.46, 0.50, 0.36])
    col = alb * (ambient + (0.95 * lam)[..., None] * np.array([1.0, 0.95, 0.86]))
    hv = unit(sun - rd)
    spec = np.clip(vdot(nrm, hv), 0, 1) ** shin
    col = col + ((spec_k * spec * lit) > 0)[..., None] * (spec_k * spec)[..., None] * np.array([1.0, 0.97, 0.92])
    haze = 1 - np.exp(-np.where(hit, t, 0) / 120.0)
    col = col + haze[..., None] * (HORIZON - col)
    col = np.clip(col, 0, 1)
    return np.where(hit[..., None], col, sky(sc, rd))


def read_scene(path):
    kv = dict(line.split("=", 1) for line in pathlib.Path(path).read_text().split() if "=" in line)
    vec = lambda s: np.array([float(v) for v in s.split(",")])
    panels = []
    for k in range(int(kv.get("panels", 0))):
        v = vec(kv[f"panel{k}"])
        panels.append(dict(c=v[0:3], n=v[3:6], u=v[6:9], v=v[9:12], hu=v[12], hv=v[13], base=v[14:17]))
    return dict(n=int(kv["n"]), cam=vec(kv["cam"]), pan=float(kv["pan"]), tilt=float(kv["tilt"]), fov=float(kv["fov"]),
                sun=vec(kv["sun"]), body=vec(kv["body"]), head=vec(kv["head"]), sword=vec(kv["sword"]), panels=panels)


def check(dump):
    sc = read_scene(dump + ".scene.txt")
    cpp = np.fromfile(dump, dtype=np.float32).reshape(sc["n"], sc["n"], 3)
    ref = render(sc)
    d = np.abs(cpp - ref).max(-1)
    frac = (d > 0.02).mean()
    print(f"NumPy (float64) vs C++/CUDA (float32), {sc['n']}x{sc['n']}: median |diff| {np.median(d):.2e}, "
          f"pixels differing by > 0.02: {100 * frac:.2f}% (silhouette edges), max {d.max():.3f}")
    return frac


def solar_field(defs, sun):
    """Panel corner polygons from [solar] (same layout rules as g3::build_solar_field), for plotting."""
    s = defs["solar"]
    rows, cols = int(s["rows"]), int(s["cols"])
    tilt = np.radians(float(s["latitude_deg"]))
    hu, hv, post = 0.5 * float(s["panel_w_m"]), 0.5 * float(s["panel_h_m"]), float(s["post_height_m"])
    hs = unit(np.array([sun[0], sun[1], 0.0]))
    z = np.array([0, 0, 1.0])
    n = unit(np.sin(tilt) * hs + np.cos(tilt) * z)
    u = unit(np.cross(z, n))
    v = np.cross(n, u)
    polys, posts = [], []
    for r in range(rows):
        for c in range(cols):
            g = np.array([float(s["field_x_m"]), float(s["field_y_m"]), 0.0]) + (r - 0.5 * (rows - 1)) * float(s["row_spacing_m"]) * hs \
                + (c - 0.5 * (cols - 1)) * float(s["col_spacing_m"]) * u
            lower = np.array([g[0], g[1], height(g[0], g[1]) + post])
            ctr = lower + hv * v
            polys.append([ctr + a * hu * u + b * hv * v for a, b in ((-1, -1), (1, -1), (1, 1), (-1, 1))])
            posts.append((np.array([g[0], g[1], height(g[0], g[1])]), lower))
    return polys, posts


def arena(csv_path, gif_path, every=6):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import animation
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    defs = load_defs()
    sun = unit(np.array([float(defs["world"][k]) for k in ("sun_x", "sun_y", "sun_z")]))
    polys, posts = solar_field(defs, sun)
    rows = list(csv.DictReader(open(csv_path, newline="", encoding="utf-8")))[::every]
    g = np.linspace(-14, 14, 70)
    X, Y = np.meshgrid(g, g)
    Z = height(X, Y)
    cam_z = height(0, 0) + 2.0
    fig = plt.figure(figsize=(7.5, 6), dpi=80)
    ax = fig.add_subplot(projection="3d")

    def frame(i):
        ax.clear()
        ax.plot_surface(X, Y, Z, color="#7fae6a", alpha=0.45, linewidth=0, rstride=2, cstride=2, shade=True)
        for base, top in posts:
            ax.plot([base[0], top[0]], [base[1], top[1]], [base[2], top[2]], "-", color="#8a8d92", lw=1.2)
        ax.add_collection3d(Poly3DCollection(polys, facecolor="#1a2f6b", edgecolor="#c9ccd2", linewidths=0.6, alpha=0.95))
        ax.quiver(-12, -12, 4, 3 * sun[0], 3 * sun[1], 3 * sun[2], color=OI["orange"], lw=2)   # toward the sun
        ax.plot([0, 0], [0, 0], [height(0, 0), cam_z], "-", color=OI["black"], lw=3)
        ax.scatter([0], [0], [cam_z], marker="D", s=60, color=OI["black"])
        hist = rows[max(0, i - 25):i + 1]
        ax.plot([float(r["hero_x"]) for r in hist], [float(r["hero_y"]) for r in hist],
                [float(r["hero_z"]) + 0.95 for r in hist], "-", color=OI["orange"], lw=1.5)
        r = rows[i]
        hx, hy, hz = float(r["hero_x"]), float(r["hero_y"]), float(r["hero_z"]) + 0.95
        ax.scatter([hx], [hy], [hz], s=90, color=OI["orange"], edgecolor=OI["black"])
        pan, tilt = np.radians(float(r["pan_deg"])), np.radians(float(r["tilt_deg"]))
        L = np.hypot(np.hypot(hx, hy), hz - cam_z)
        hit = int(r["laser_hit"])
        if int(r["firing"]):
            col = OI["red"] if hit in (2, 3) else (OI["yellow"] if hit == 4 else OI["green"])
            ax.plot([0, L * np.cos(tilt) * np.cos(pan)], [0, L * np.cos(tilt) * np.sin(pan)], [cam_z, cam_z + L * np.sin(tilt)], "-", color=col, lw=2.5)
        ax.set(xlim=(-14, 14), ylim=(-14, 14), zlim=(-2, 5), xlabel="x (m)", ylabel="y (m)", zlabel="z (m)")
        ax.view_init(elev=28, azim=-60 + 0.4 * i)
        ax.set_title(f"t = {float(r['t']):.1f} s   {defs['guardian']['name_en']} HP {r['guardian_hp']}   "
                     f"{defs['hero']['name_en'].split(',')[0]} HP {r['hero_hp']}   solar farm {float(r['pv_kw']):.2f} kW", fontsize=9)
        return []

    anim = animation.FuncAnimation(fig, frame, frames=len(rows), blit=False)
    anim.save(gif_path, writer=animation.PillowWriter(fps=10))
    print(f"wrote {gif_path}: {len(rows)} frames of the 3D arena. Grass-green terrain surface, the solar farm (dark-blue "
          "panels with grey frames, tilted at the site latitude toward the sun), an orange arrow pointing at the sun, the droid on "
          "its mast (black diamond), the swordsman (orange dot) with his recent path; the laser is drawn while firing: red = body "
          "hit, yellow = parried by the foam sword, green = miss. The title shows HP and the farm's output; the view slowly orbits.")


def camera(frames_dir, csv_path, gif_path):
    from PIL import Image, ImageDraw
    rows = {int(r["frame"]): r for r in csv.DictReader(open(csv_path, newline="", encoding="utf-8"))}
    files = sorted(pathlib.Path(frames_dir).glob("frame_*.ppm"))
    imgs = []
    for fpath in files:
        f = int(fpath.stem.split("_")[1])
        im = Image.open(fpath).convert("RGB")
        if im.width < 256:                                   # upscale small frames; keep HD frames native
            im = im.resize((256, 256), Image.NEAREST)
        W = im.width
        c, a = W // 2, max(10, W // 25)
        d = ImageDraw.Draw(im)
        d.line([(c, c - a), (c, c + a)], fill=(255, 255, 255))
        d.line([(c - a, c), (c + a, c)], fill=(255, 255, 255))
        r = rows.get(f)
        if r:
            txt = f"t={float(r['t']):.1f}s err={float(r['true_err_deg']):.2f}deg PV {float(r['pv_kw']):.2f}kW"
            if int(r["firing"]):
                txt += {2: " HIT", 3: " HIT", 4: " PARRY", 1: " HILL"}.get(int(r["laser_hit"]), "")
                d.ellipse([c - a // 2, c - a // 2, c + a // 2, c + a // 2], outline=(255, 40, 40), width=max(1, W // 256))
            d.text((4, 4), txt, fill=(255, 255, 255))
        imgs.append(im)
    imgs[0].save(gif_path, save_all=True, append_images=imgs[1:], duration=100, loop=0)
    print(f"wrote {gif_path}: {len(imgs)} frames of the droid's own camera view (ray-traced by C++/CUDA), crosshair at "
          "the image centre, red ring when the laser fires, status text from the CSV.")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "check":
        sys.exit(0 if check(sys.argv[2]) < 0.02 else 1)
    elif cmd == "arena":
        arena(sys.argv[2], sys.argv[3])
    elif cmd == "camera":
        camera(sys.argv[2], sys.argv[3], sys.argv[4])
    else:
        print(__doc__)
        sys.exit(2)
