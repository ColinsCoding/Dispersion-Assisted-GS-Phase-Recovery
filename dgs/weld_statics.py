"""Torque on a weld: the eccentrically loaded fillet-weld group (ABET statics).

A bracket welded to a column and loaded off to the side is the canonical
"statics + mechanics of materials" connection problem. The load P doesn't act
through the weld's centroid, so the weld carries TWO things at once, exactly the
two equilibrium conditions sum_F = 0 and sum_M = 0:

  * DIRECT SHEAR from sum_F: the whole load divided over the weld,
        f' = P / L_w            [force per unit length of weld]
  * TORSIONAL SHEAR from sum_M: the moment T = P*e (e = eccentricity from the
    weld centroid) twists the weld group, giving at a point a distance r from
    the centroid
        f'' = T * r / J         [force per unit length], perpendicular to r,
    where J is the group's polar moment about its centroid.

These two vectors ADD at every point; the resultant is largest at the weld point
farthest from the centroid in the direction the two reinforce -- the CRITICAL
point that sizes the weld. Dividing the peak force-per-length by the weld throat
(0.707 * leg for a 45-degree fillet) gives the actual shear stress, and the
allowable stress over that is the factor of safety.

This uses Blodgett's "treat the weld as a line" method: compute section
properties of the weld as a 1-D line (units of length, length^3), so J is found
once and reused. Everything is a line integral, verified here against the
closed form for a two-weld group. Ties to the equilibrium core of the old
statics work (sum_F, sum_M) -- the weld is simply what reacts them. NumPy only.
"""

import numpy as np


def _sample(segments, n=600):
    """Sample weld segments as a 1-D line: return (points Nx2, dL weights).
    Each segment is ((x1,y1),(x2,y2)); trapezoid weights make the line
    integrals (length, centroid, polar moment) accurate."""
    if not segments:
        raise ValueError("need at least one weld segment")
    if n < 20:
        raise ValueError("n must be >= 20 for an accurate line integral")
    pts, wts = [], []
    for (x1, y1), (x2, y2) in segments:
        L = float(np.hypot(x2 - x1, y2 - y1))
        if L == 0:
            raise ValueError("weld segment has zero length")
        ts = np.linspace(0.0, 1.0, n)
        p = np.column_stack([x1 + ts * (x2 - x1), y1 + ts * (y2 - y1)])
        w = np.full(n, L / (n - 1))
        w[0] *= 0.5
        w[-1] *= 0.5
        pts.append(p)
        wts.append(w)
    return np.vstack(pts), np.concatenate(wts)


def weld_properties(segments, n=600):
    """Section properties of the weld group treated as a line, about its own
    centroid: total length L_w, centroid (cx, cy), second moments Ix, Iy, and
    the polar moment J = Ix + Iy (all per unit throat -- units length, length^3).
    J is the number the torsional-shear term divides by."""
    p, dL = _sample(segments, n)
    Lw = float(dL.sum())
    c = (p * dL[:, None]).sum(axis=0) / Lw
    d = p - c
    Ix = float((d[:, 1] ** 2 * dL).sum())
    Iy = float((d[:, 0] ** 2 * dL).sum())
    return {"L_w": Lw, "centroid": (float(c[0]), float(c[1])),
            "Ix": Ix, "Iy": Iy, "J": Ix + Iy}


def fillet_throat(leg, angle_deg=45.0):
    """Effective throat of a fillet weld: t = leg * sin(angle). For the usual
    equal-leg 45-degree fillet this is the famous 0.707 * leg."""
    if leg <= 0:
        raise ValueError("leg must be positive")
    if not 0 < angle_deg < 180:
        raise ValueError("angle_deg must be in (0, 180)")
    return leg * np.sin(np.radians(angle_deg))


def eccentric_weld_stress(segments, P, load_point, throat, n=600):
    """Full analysis of an in-plane eccentrically loaded weld group.

    segments   : list of ((x1,y1),(x2,y2)) weld lines
    P          : (Px, Py) applied load vector [force]
    load_point : (x, y) where P is applied [length]
    throat     : weld throat thickness [length] (see fillet_throat)

    Returns a dict with the weld centroid, L_w and J, the moment T = P*e about
    the centroid, the uniform direct-shear force-per-length, and -- at the
    CRITICAL point -- the resultant force-per-length and the actual shear
    stress f_resultant / throat. Direct and torsional shear are combined as
    vectors at every weld point, and the maximum is reported."""
    P = np.asarray(P, float)
    if P.shape != (2,):
        raise ValueError("P must be a 2-vector (Px, Py)")
    if throat <= 0:
        raise ValueError("throat must be positive")
    props = weld_properties(segments, n)
    Lw, J = props["L_w"], props["J"]
    c = np.array(props["centroid"])
    # moment of P about the weld centroid: T = r_load x P  (z-component)
    r_load = np.asarray(load_point, float) - c
    T = float(r_load[0] * P[1] - r_load[1] * P[0])

    # direct shear: uniform, opposite-free vector P spread over the weld
    f_direct = P / Lw                                  # force per unit length

    # torsional shear at every weld point: f'' = (T/J) * (z_hat x r)
    p, _ = _sample(segments, n)
    r = p - c
    f_tors = (T / J) * np.column_stack([-r[:, 1], r[:, 0]])
    f_res = f_direct[None, :] + f_tors                 # vector sum per point
    mag = np.hypot(f_res[:, 0], f_res[:, 1])
    k = int(np.argmax(mag))

    f_peak = float(mag[k])
    return {
        "centroid": props["centroid"], "L_w": Lw, "J": J, "T": T,
        "direct_shear_per_length": float(np.hypot(*f_direct)),
        "peak_force_per_length": f_peak,
        "critical_point": (float(p[k, 0]), float(p[k, 1])),
        "shear_stress": f_peak / throat,               # force / area
    }


def factor_of_safety(shear_stress, allowable_stress):
    """FoS = allowable / actual. Below 1 means the weld is overstressed."""
    if shear_stress <= 0 or allowable_stress <= 0:
        raise ValueError("stresses must be positive")
    return allowable_stress / shear_stress


def weld_reaction(P, load_point, centroid):
    """The equilibrium tie-in: for the bracket to be static, the weld must
    supply a reaction force -P and a reaction moment -T (T = P*e about the
    weld centroid). Returns (reaction_force_xy, reaction_moment) -- the sum_F=0
    and sum_M=0 the weld group is there to satisfy."""
    P = np.asarray(P, float)
    r = np.asarray(load_point, float) - np.asarray(centroid, float)
    T = float(r[0] * P[1] - r[1] * P[0])
    return (-P[0], -P[1]), -T


if __name__ == "__main__":
    # two vertical fillet welds, each 100 mm long, 50 mm apart (a bracket)
    L, w = 100.0, 50.0
    segs = [((-w/2, -L/2), (-w/2, L/2)), ((w/2, -L/2), (w/2, L/2))]
    props = weld_properties(segs)
    J_closed = L**3/6 + L*w**2/2                  # closed form for this group
    print(f"L_w = {props['L_w']:.1f} mm  (expected {2*L:.0f})")
    print(f"J   = {props['J']:.1f} mm^3  vs closed form {J_closed:.1f}")

    # 10 kN downward, applied 100 mm to the right of the centroid
    leg = 6.0
    t = fillet_throat(leg)
    res = eccentric_weld_stress(segs, P=(0.0, -10000.0), load_point=(100.0, 0.0),
                                throat=t)
    print(f"\nthroat (6 mm fillet) = {t:.3f} mm")
    print(f"moment T = {res['T']:.0f} N.mm,  direct shear = "
          f"{res['direct_shear_per_length']:.1f} N/mm")
    print(f"critical point {tuple(round(x,1) for x in res['critical_point'])} "
          f"-> resultant {res['peak_force_per_length']:.1f} N/mm")
    print(f"shear stress = {res['shear_stress']:.1f} MPa")
    fos = factor_of_safety(res['shear_stress'], allowable_stress=96.0)  # ~E70xx
    print(f"factor of safety vs 96 MPa allowable = {fos:.2f}")
