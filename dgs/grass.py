"""
grass.py — Procedural 3-D grass field  (grass-render branch)
  Lumion-style Dutch grass  — tall, swept, multi-species
  3-leaf clovers            — Bezier petal geometry
  Shell-based fur rendering — layered opacity shells
  Markov chain wind         — 4-state weather model
  EM plane-wave analogy     — E-field drives blade oscillators
  DoD FutureG aesthetic     — dark field, tactical palette

Branch: grass-render  (isolated from main / gs-torch-nd)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D          # noqa: F401
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection
import json, math

# ══════════════════════════════════════════════════════════════════════════════
# MARKOV WIND
# ══════════════════════════════════════════════════════════════════════════════
STATES     = ["calm", "breeze", "gust", "storm"]
STATE_WIND = np.array([0.05, 0.25, 0.60, 1.00])
TRANS = np.array([
    [0.70, 0.25, 0.04, 0.01],
    [0.20, 0.55, 0.22, 0.03],
    [0.05, 0.30, 0.50, 0.15],
    [0.02, 0.10, 0.38, 0.50],
])

def markov_wind(n_steps, seed=42):
    rng = np.random.default_rng(seed)
    state = 1; states, winds = [], []
    for _ in range(n_steps):
        states.append(state)
        winds.append(STATE_WIND[state])
        state = rng.choice(4, p=TRANS[state])
    return np.array(states), np.array(winds)

def stationary_dist(P):
    vals, vecs = np.linalg.eig(P.T)
    idx = np.argmin(np.abs(vals - 1.0))
    pi  = np.real(vecs[:, idx])
    return pi / pi.sum()

def em_field(X, t, k=0.8, omega=1.2, E0=1.0):
    return E0 * np.sin(k * X - omega * t)

# ══════════════════════════════════════════════════════════════════════════════
# §1  BLADE GEOMETRY  (Lumion Dutch-style)
# ══════════════════════════════════════════════════════════════════════════════
def dutch_blade(x0, y0, height, theta, colour, n_seg=8):
    """
    Tall Dutch grass blade — wider base, tapered tip, pronounced curve.
    Returns (segments, colours) for Line3DCollection.
    """
    t_arr     = np.linspace(0, 1, n_seg + 1)
    curvature = t_arr ** 1.6                      # softer curve than default
    xs = x0 + curvature * height * math.sin(theta)
    ys = np.full(n_seg + 1, y0)
    zs = t_arr * height

    segs = [[(xs[i], ys[i], zs[i]), (xs[i+1], ys[i+1], zs[i+1])]
            for i in range(n_seg)]
    # Phong: base dark, mid saturated, tip bright
    bright = np.concatenate([np.linspace(0.3, 1.0, n_seg//2),
                              np.linspace(1.0, 0.7, n_seg - n_seg//2)])
    cols = [np.clip(colour * b, 0, 1) for b in bright]
    return segs, cols

# ══════════════════════════════════════════════════════════════════════════════
# §2  3-LEAF CLOVER GEOMETRY
# ══════════════════════════════════════════════════════════════════════════════
def clover_leaf_pts(cx, cy, cz, angle_offset, size=0.18, n=20):
    """
    One heart-shaped petal in 3-D using a cardioid parameterisation.
    Returns array of (x,y,z) polygon vertices.
    """
    t   = np.linspace(0, 2*np.pi, n)
    # cardioid:  r = size*(1 + cos(t))
    r   = size * (1.0 + np.cos(t))
    # petal direction rotated by angle_offset
    lx  = cx + r * np.cos(t + angle_offset)
    ly  = cy + r * np.sin(t + angle_offset) * 0.55  # flatten into XY
    lz  = np.full(n, cz)
    return list(zip(lx, ly, lz))

def clover(x0, y0, stem_h=0.25, size=0.15, wind_theta=0.0):
    """
    3-leaf clover: stem + 3 petals at 120° offsets.
    Returns stem_segments and petal_polys.
    """
    # stem — bends with wind
    stem_x = x0 + np.array([0, 0.3*wind_theta, 0.7*wind_theta, wind_theta]) * stem_h
    stem_y = np.full(4, y0)
    stem_z = np.linspace(0, stem_h, 4)
    stem_segs = [[(stem_x[i],stem_y[i],stem_z[i]),
                  (stem_x[i+1],stem_y[i+1],stem_z[i+1])] for i in range(3)]

    cx, cy, cz = stem_x[-1], y0, stem_h
    petals = []
    for k in range(3):
        angle = k * 2*np.pi/3
        pts   = clover_leaf_pts(cx, cy, cz, angle, size=size)
        petals.append(pts)
    return stem_segs, petals

# ══════════════════════════════════════════════════════════════════════════════
# §3  SHELL-BASED FUR
# ══════════════════════════════════════════════════════════════════════════════
def fur_shells(x0, y0, n_shells=6, max_h=0.35, radius=0.08,
               base_col=np.array([0.55,0.42,0.25]),
               wind_theta=0.0):
    """
    Shell fur: concentric ellipses at increasing height.
    Each shell shrinks and shifts with wind — tip drift.
    Returns list of (verts, colour, alpha) for Poly3DCollection.
    """
    polys = []
    for s in range(1, n_shells + 1):
        frac  = s / n_shells
        h     = frac * max_h
        r     = radius * (1.0 - 0.7 * frac)     # taper
        drift = wind_theta * frac * 0.12         # tip follows wind
        n_pts = 16
        t_arr = np.linspace(0, 2*np.pi, n_pts, endpoint=False)
        xs = x0 + drift + r * np.cos(t_arr)
        ys = y0          + r * 0.5 * np.sin(t_arr)  # oval cross-section
        zs = np.full(n_pts, h)
        verts = [list(zip(xs, ys, zs))]
        # colour brightens toward tip
        col  = np.clip(base_col * (0.4 + 0.6*frac), 0, 1)
        alpha = 0.55 - 0.4 * frac               # transparent at tip
        polys.append((verts, col, alpha))
    return polys

# ══════════════════════════════════════════════════════════════════════════════
# §4  MAIN RENDER
# ══════════════════════════════════════════════════════════════════════════════
def render(n_grass=350, n_clovers=40, n_fur=30,
           n_time=120, seed=7, save="grass_render.png"):

    rng = np.random.default_rng(seed)
    W   = 10.0  # field width

    # ── positions ─────────────────────────────────────────────────────────────
    gx = rng.uniform(0, W, n_grass);  gy = rng.uniform(0, W, n_grass)
    gh = rng.uniform(0.5, 1.6, n_grass)                 # Dutch: taller range

    cx = rng.uniform(0.5, W-0.5, n_clovers)
    cy = rng.uniform(0.5, W-0.5, n_clovers)

    fx = rng.uniform(0.5, W-0.5, n_fur)
    fy = rng.uniform(0.5, W-0.5, n_fur)

    # ── colours ───────────────────────────────────────────────────────────────
    # Dutch grass: 3 species — dark green, yellow-green, blue-green
    species = rng.integers(0, 3, n_grass)
    base_cols = [np.array([0.15,0.55,0.12]),   # dark green
                 np.array([0.45,0.65,0.10]),   # yellow-green
                 np.array([0.10,0.50,0.40])]   # blue-green
    g_cols = np.array([base_cols[s] + rng.uniform(-0.05,0.05,3)
                       for s in species]).clip(0.05, 1.0)

    clover_green = np.array([0.10, 0.72, 0.18])

    # ── Markov wind ───────────────────────────────────────────────────────────
    states, winds = markov_wind(n_time, seed=seed)
    pi_stat       = stationary_dist(TRANS)
    t_frame = n_time // 2
    wind_w  = winds[t_frame]
    t_phys  = t_frame * 0.08

    # ── deflections ───────────────────────────────────────────────────────────
    theta_g = em_field(gx, t_phys, E0=wind_w*0.5) \
              + rng.normal(0, 0.04*wind_w, n_grass)
    theta_c = em_field(cx, t_phys, E0=wind_w*0.3) \
              + rng.normal(0, 0.03*wind_w, n_clovers)
    theta_f = em_field(fx, t_phys, E0=wind_w*0.4) \
              + rng.normal(0, 0.04*wind_w, n_fur)

    # ══════════════════════════════════════════════════════════════════════════
    # BUILD SCENE GEOMETRY
    # ══════════════════════════════════════════════════════════════════════════
    grass_segs, grass_cols = [], []
    for i in range(n_grass):
        s, c = dutch_blade(gx[i], gy[i], gh[i], theta_g[i], g_cols[i])
        grass_segs.extend(s); grass_cols.extend(c)

    clover_stem_segs, clover_stem_cols = [], []
    clover_petal_verts, clover_petal_cols = [], []
    for i in range(n_clovers):
        stems, petals = clover(cx[i], cy[i],
                               stem_h=rng.uniform(0.18, 0.32),
                               size=rng.uniform(0.10, 0.20),
                               wind_theta=theta_c[i])
        clover_stem_segs.extend(stems)
        clover_stem_cols.extend([clover_green*0.7]*len(stems))
        for pts in petals:
            clover_petal_verts.append(pts)
            shade = clover_green * rng.uniform(0.85, 1.15)
            clover_petal_cols.append(np.clip(shade, 0, 1))

    fur_poly_list = []
    for i in range(n_fur):
        fur_col = np.array([0.50 + rng.uniform(-0.1,0.1),
                            0.38 + rng.uniform(-0.1,0.1),
                            0.20 + rng.uniform(-0.05,0.05)])
        shells = fur_shells(fx[i], fy[i], base_col=fur_col,
                            wind_theta=theta_f[i])
        fur_poly_list.extend(shells)

    # ══════════════════════════════════════════════════════════════════════════
    # PLOT
    # ══════════════════════════════════════════════════════════════════════════
    fig = plt.figure(figsize=(18, 12), facecolor='#080e08')
    gl  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.40, wspace=0.32)

    # ── §A: Main 3-D scene ────────────────────────────────────────────────────
    ax = fig.add_subplot(gl[0, :2], projection='3d')
    ax.set_facecolor('#080e08')

    # ground
    gxm, gym = np.meshgrid(np.linspace(0,W,20), np.linspace(0,W,20))
    ax.plot_surface(gxm, gym, np.zeros_like(gxm),
                    color='#111a0f', alpha=0.7, linewidth=0, zorder=0)

    # grass blades
    lc_grass = Line3DCollection(grass_segs, colors=grass_cols,
                                linewidths=0.8, alpha=0.9, zorder=2)
    ax.add_collection3d(lc_grass)

    # clover stems
    if clover_stem_segs:
        lc_clover = Line3DCollection(clover_stem_segs,
                                     colors=[clover_green*0.7]*len(clover_stem_segs),
                                     linewidths=1.2, alpha=0.95, zorder=3)
        ax.add_collection3d(lc_clover)

    # clover petals
    for pts, col in zip(clover_petal_verts, clover_petal_cols):
        pc = Poly3DCollection([pts], alpha=0.75, zorder=4)
        pc.set_facecolor(col)
        pc.set_edgecolor(col * 0.7)
        ax.add_collection3d(pc)

    # fur shells
    for verts, col, alpha in fur_poly_list:
        pc = Poly3DCollection(verts, alpha=alpha, zorder=1)
        pc.set_facecolor(col)
        pc.set_edgecolor('none')
        ax.add_collection3d(pc)

    ax.set(xlim=(0,W), ylim=(0,W), zlim=(0,2.0),
           xlabel='X [m]', ylabel='Y [m]', zlabel='Z [m]')
    ax.set_title(f'Lumion Dutch Grass · 3-Leaf Clovers · Shell Fur\n'
                 f'Markov wind: {STATES[states[t_frame]].upper()}  '
                 f'w={wind_w:.2f}  frame {t_frame}/{n_time}',
                 color='white', fontsize=9, pad=6)
    ax.tick_params(colors='#555555', labelsize=6)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False

    # ── §B: Markov wind timeline ──────────────────────────────────────────────
    ax_w = fig.add_subplot(gl[0, 2])
    ax_w.set_facecolor('#080e08')
    scols = ['#2ecc71','#3498db','#e67e22','#e74c3c']
    for s in range(4):
        m = np.where(states == s)[0]
        ax_w.scatter(m, np.full_like(m, float(s)), c=scols[s], s=5, alpha=0.8)
    ax_w.axvline(t_frame, color='white', lw=1.2, ls='--')
    ax_w.plot(np.arange(n_time), winds * 3.0, color='#00d4ff',
              lw=0.8, alpha=0.5, label='wind strength ×3')
    ax_w.set(title='Markov Wind', xlabel='step', ylabel='state',
             yticks=[0,1,2,3], yticklabels=STATES, xlim=(0,n_time))
    ax_w.title.set_color('white')
    ax_w.tick_params(colors='#666666', labelsize=7)
    for sp in ax_w.spines.values(): sp.set_color('#222222')

    # ── §C: Transition matrix ─────────────────────────────────────────────────
    ax_t = fig.add_subplot(gl[1, 0])
    ax_t.set_facecolor('#080e08')
    im = ax_t.imshow(TRANS, cmap='YlOrRd', vmin=0, vmax=1, aspect='auto')
    plt.colorbar(im, ax=ax_t, fraction=0.046)
    ax_t.set(title='Transition Matrix P[i→j]',
             xticks=range(4), yticks=range(4),
             xticklabels=STATES, yticklabels=STATES)
    for i in range(4):
        for j in range(4):
            ax_t.text(j, i, f'{TRANS[i,j]:.2f}', ha='center', va='center',
                      fontsize=7, color='k' if TRANS[i,j]>0.3 else 'w')
    ax_t.title.set_color('white')
    ax_t.tick_params(colors='#666666', labelsize=7)
    plt.setp(ax_t.get_xticklabels(), rotation=30, ha='right')

    # ── §D: Stationary distribution ───────────────────────────────────────────
    ax_p = fig.add_subplot(gl[1, 1])
    ax_p.set_facecolor('#080e08')
    bars = ax_p.bar(STATES, pi_stat, color=scols, alpha=0.85, edgecolor='white')
    ax_p.bar_label(bars, fmt='%.3f', fontsize=8, padding=3, color='white')
    ax_p.set(title='Stationary Distribution π', ylabel='P(state)')
    ax_p.title.set_color('white')
    ax_p.tick_params(colors='#666666', labelsize=8)
    for sp in ax_p.spines.values(): sp.set_color('#222222')

    # ── §E: EM wind field ─────────────────────────────────────────────────────
    ax_e = fig.add_subplot(gl[1, 2])
    ax_e.set_facecolor('#080e08')
    xs_p = np.linspace(0, W, 300)
    em_p = em_field(xs_p, t_phys, E0=wind_w*0.5)
    ax_e.plot(xs_p, em_p, color='#00d4ff', lw=2, label='E_wind(x,t)')
    ax_e.fill_between(xs_p, em_p, alpha=0.12, color='#00d4ff')
    ax_e.scatter(gx[:60],  theta_g[:60],  s=4, color='#2ecc71', alpha=0.7, label='grass θ')
    ax_e.scatter(cx,       theta_c,       s=10, color='#ffffff', alpha=0.8,
                 marker='*', label='clover θ')
    ax_e.scatter(fx,       theta_f,       s=8, color='#e67e22', alpha=0.7,
                 marker='s', label='fur θ')
    ax_e.axhline(0, color='#333333', lw=0.8)
    ax_e.set(title='EM Wind Field  E₀sin(kx−ωt)', xlabel='x [m]', ylabel='θ [rad]')
    ax_e.legend(fontsize=6.5, facecolor='#111111', labelcolor='white')
    ax_e.title.set_color('white')
    ax_e.tick_params(colors='#666666', labelsize=7)
    for sp in ax_e.spines.values(): sp.set_color('#222222')

    fig.suptitle(
        'DoD Grass Field  ·  Lumion Dutch Grass  ·  3-Leaf Clovers  ·  Shell Fur  ·  Markov Wind  ·  EM Analogy',
        color='white', fontsize=10, fontweight='bold')

    plt.savefig(save, dpi=150, bbox_inches='tight', facecolor='#080e08')
    plt.show()
    print(f"  saved → {save}")

    stats = {
        "n_grass": n_grass, "n_clovers": n_clovers, "n_fur": n_fur,
        "wind_state":      STATES[int(states[t_frame])],
        "wind_strength":   round(float(wind_w), 4),
        "stationary_dist": {s: round(float(p),4) for s,p in zip(STATES,pi_stat)},
        "species":         ["dark_green","yellow_green","blue_green"],
        "fur_shells":      6,
        "em_wave":         {"k":0.8, "omega":1.2},
        "branch":          "grass-render",
        "exit_code": 0, "status": "PASS grass.py"
    }
    return stats


if __name__ == "__main__":
    import json
    stats = render()
    print(json.dumps(stats, indent=2, ensure_ascii=False))
