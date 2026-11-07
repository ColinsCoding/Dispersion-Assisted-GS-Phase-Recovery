"""Jaws of life: a geometry engine for a hydraulic rescue spreader and cutter.

In plain words
--------------
When a car is crushed in a crash, firefighters use "jaws of life": hydraulic tools that pry a
door open (the *spreader*) or slice through a roof pillar (the *cutter*) so the paramedics in
the ambulance can reach the patient. Both tools are the same machine underneath:

* A small pump pushes oil at very high pressure (about 700 bar, i.e. 700 times atmospheric
  pressure) into a cylinder. Pressure times piston area is a big straight-line push, ~200 kN.
* That push drives a *crosshead* (a pin on the end of the piston rod). Two short *links* run from
  the crosshead to the *ears* of two arms. Each arm turns about one shared *pivot pin*.
* Each arm is a bent lever (a bell crank). The short ear sits behind the pivot; the long jaw
  sticks out in front. A short lever arm driving a long one trades force for travel: the tips
  move a lot (hundreds of millimetres) for a modest piston stroke, so the tip force is smaller
  than the piston force. That ratio, tip force over piston force, is the *mechanical advantage*.

This module works out, from geometry alone:

1. **Kinematics.** How far the tips open for a given piston extension, both ways round.
2. **Forces.** Tip force versus opening from *virtual work* (work in = work out, no friction), and
   the same number again from a free-body force balance on the crosshead and the arm, as a check.
3. **The cutter.** Blade torque, force at a point on the blade (torque / distance, so cut close to
   the pivot), the shear force needed to cut a steel tube or a car pillar, and when a round bar is
   squeezed out of the blades (when tan(half-angle) exceeds the friction coefficient).
4. **A small 2-D game engine.** Rigid-body transforms for every link, circle-vs-polygon contact
   between the jaw tips and a "door" panel, a crude elastic-plastic panel that dents and then pops
   its latch, half-plane clipping that crushes a pillar between the blades, a pump that sets the
   piston speed, a relief valve that makes the tool stall, and a list of animation frames. It is
   what a video game (an ambulance-and-rescue level in a GTA-style game) would run each tick.

The math
--------
Coordinates: pivot O at the origin, tool axis +x toward the tips. The upper arm is turned by the
half-opening angle ``theta``. Its tip is at ``L (cos theta, sin theta)``. Its ear is behind the pivot
on the *opposite* side (crossed links), at

    E = -r (cos psi, sin psi),     psi = theta + delta,

where ``r`` is the ear radius and ``delta`` the fixed ear angle. The crosshead C = (-c, 0) is on the
axis, joined to E by a link of length ``l``. The triangle O-E-C is a slider-crank, so by the law of
cosines  l^2 = c^2 + r^2 - 2 c r cos psi,  and

    c(theta)       = r cos psi + sqrt(l^2 - r^2 sin^2 psi)          (forward kinematics)
    cos psi        = (c^2 + r^2 - l^2) / (2 c r)                     (inverse kinematics)
    x_p(theta)     = c(0) - c(theta)                                  (piston extension)
    w(theta)       = 2 L sin theta                                    (tip gap)

Virtual work with piston force F_p = p A doing work against a separating force F_s between the
tips: F_p dx_p = F_s dw, so

    MA(theta) = F_s / F_p = (dx_p/dtheta) / (dw/dtheta) = r c sin psi / (2 L cos theta sqrt(l^2 - r^2 sin^2 psi))

The force balance gives the same thing: link compression N = F_p l / (2 sqrt(...)), arm torque
tau = N r c sin psi / l, and the tip balance tau = F_s L cos theta.

Cutter: each blade gets torque T = F_p (dx_p/dtheta) / 2 (two blades share the piston's work).
A force at distance d from the pivot, normal to the blade, is T/d. A round bar of diameter D is
first touched by blades at half-angle theta when d sin theta = D/2. Shearing a full cross-section
of area A needs about F = tau_u A, with ultimate shear strength tau_u ~ 0.6 x tensile strength.

Assumptions and honesty
-----------------------
* Every dimension and rating here (700 bar, 60 mm bore, 400 mm arms, 1 L/min pump, boron-steel
  pillars at 1500 MPa) is **illustrative and written from memory**, not from a datasheet or the
  NFPA 1936 / EN 13204 standards. The ratios and trends are the point, not the exact kN.
* Straight arms, point tips, no friction in the pins, rigid links. Real jaws are curved and have
  serrated tips; real tools lose a few percent to seal friction.
* The cutter uses the same bell-crank model for blade torque. Real cutters close on the full-bore
  push stroke with their own link layout; the force *magnitude* is computed the same way.
* Full-section shear is an *upper bound*: blades cutting progressively, like scissors, need less.
* Metal strengths in METALS (310-1500 MPa) and the 30 s tool-swap time are typical values from
  memory; the extrication clock counts only hydraulic motion plus that fixed handling time.
* The panel and pillar models are game physics (spring, yield cap, latch pop, clip-and-crush),
  meant to look and scale right, not finite-element crash mechanics.

Reference (from memory, unverified): J. E. Shigley and J. J. Uicker, *Theory of Machines and
Mechanisms*, McGraw-Hill (slider-crank kinematics and virtual work).
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math

import numpy as np

BAR = 1e5  # Pa


def _positive(name, value):
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a finite positive number, got {value!r}")
    return float(value)


def _nonneg(name, value):
    if not np.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and >= 0, got {value!r}")
    return float(value)


# ---------------------------------------------------------------------------------------------
# 1-2. Linkage kinematics and forces
# ---------------------------------------------------------------------------------------------

@dataclass
class Linkage:
    """Bell-crank jaw linkage shared by the spreader and the cutter (SI units, angles in rad)."""

    tip_length: float = 0.40          # L, pivot to tip [m]
    ear_radius: float = 0.12          # r, pivot to ear pin [m]
    link_length: float = 0.19         # l, ear pin to crosshead pin [m]
    ear_angle: float = math.radians(25.0)   # delta, ear offset when the tips are closed
    bore: float = 0.060               # piston diameter [m]
    pressure_max: float = 700 * BAR   # relief-valve pressure [Pa]
    theta_max: float = math.radians(60.0)   # mechanical stop on the half-opening

    def __post_init__(self):
        for n in ("tip_length", "ear_radius", "link_length", "bore", "pressure_max", "theta_max"):
            _positive(n, getattr(self, n))
        _nonneg("ear_angle", self.ear_angle)
        if self.link_length <= self.ear_radius:
            raise ValueError("link_length must exceed ear_radius, otherwise the slider-crank "
                             "can lock or fail to assemble")
        if self.theta_max + self.ear_angle > math.pi / 2 + 1e-12:
            raise ValueError("theta_max + ear_angle must be <= pi/2 (working range where the "
                             "piston extension rises monotonically with opening)")

    # -- geometry ---------------------------------------------------------------------------
    @property
    def piston_area(self):
        return math.pi * self.bore ** 2 / 4

    def piston_force(self, pressure=None):
        p = self.pressure_max if pressure is None else _nonneg("pressure", pressure)
        return p * self.piston_area

    def _theta(self, theta):
        th = np.asarray(theta, dtype=float)
        if np.any(~np.isfinite(th)) or np.any(th < -1e-12) or np.any(th > self.theta_max + 1e-12):
            raise ValueError(f"theta must lie in [0, theta_max={self.theta_max:.4f}] rad")
        return th

    def _root(self, psi):
        return np.sqrt(self.link_length ** 2 - (self.ear_radius * np.sin(psi)) ** 2)

    def crosshead_distance(self, theta):
        """c(theta): distance from the pivot back to the crosshead pin."""
        psi = self._theta(theta) + self.ear_angle
        return self.ear_radius * np.cos(psi) + self._root(psi)

    def piston_extension(self, theta):
        """x_p(theta) = c(0) - c(theta): how far the rod has moved toward the pivot."""
        return self.crosshead_distance(0.0) - self.crosshead_distance(theta)

    @property
    def stroke(self):
        return float(self.piston_extension(self.theta_max))

    def theta_from_extension(self, x):
        """Inverse kinematics (closed form, law of cosines)."""
        x = np.asarray(x, dtype=float)
        if np.any(~np.isfinite(x)) or np.any(x < -1e-12) or np.any(x > self.stroke + 1e-12):
            raise ValueError(f"piston extension must lie in [0, stroke={self.stroke:.4f}] m")
        c = self.crosshead_distance(0.0) - x
        r, l = self.ear_radius, self.link_length
        cos_psi = np.clip((c ** 2 + r ** 2 - l ** 2) / (2 * c * r), -1.0, 1.0)
        return np.clip(np.arccos(cos_psi) - self.ear_angle, 0.0, self.theta_max)

    def tip_gap(self, theta):
        return 2 * self.tip_length * np.sin(self._theta(theta))

    def tip_point(self, theta, upper=True):
        th = float(self._theta(theta))
        s = 1.0 if upper else -1.0
        return np.array([self.tip_length * math.cos(th), s * self.tip_length * math.sin(th)])

    def ear_point(self, theta, upper=True):
        psi = float(self._theta(theta)) + self.ear_angle
        s = 1.0 if upper else -1.0
        return np.array([-self.ear_radius * math.cos(psi), -s * self.ear_radius * math.sin(psi)])

    # -- derivatives and forces --------------------------------------------------------------
    def dxp_dtheta(self, theta):
        psi = self._theta(theta) + self.ear_angle
        r = self.ear_radius
        return r * np.sin(psi) * (1 + r * np.cos(psi) / self._root(psi))

    def mechanical_advantage(self, theta):
        """F_s / F_p from virtual work: (dx_p/dtheta) / (dw/dtheta)."""
        th = self._theta(theta)
        return self.dxp_dtheta(th) / (2 * self.tip_length * np.cos(th))

    def tip_force(self, theta, pressure=None):
        """Separating force between the tips (along the opening direction) [N]."""
        return self.piston_force(pressure) * self.mechanical_advantage(theta)

    def arm_torque(self, theta, pressure=None):
        """Torque on each arm (or blade) about the pivot [N m]: T = F_p (dx_p/dtheta) / 2."""
        return self.piston_force(pressure) * self.dxp_dtheta(theta) / 2

    def force_balance(self, theta, pressure=None):
        """Free-body check, independent of virtual work.

        Crosshead: F_p = 2 N u_x (u = unit vector crosshead -> ear).  Arm: torque of N u at the
        ear about O balanced by the load at the tip.  Returns link force N, arm torque, the
        separating force F_s and the force normal to the arm F_n = torque / L.
        """
        th = float(self._theta(theta))
        Fp = self.piston_force(pressure)
        E = self.ear_point(th)
        C = np.array([-float(self.crosshead_distance(th)), 0.0])
        u = (E - C) / np.linalg.norm(E - C)
        N = Fp / (2 * u[0])
        F_on_ear = N * u
        torque = E[0] * F_on_ear[1] - E[1] * F_on_ear[0]
        F_s = torque / (self.tip_length * math.cos(th))
        return {"link_force": N, "torque": torque, "separating_force": F_s,
                "normal_force": torque / self.tip_length,
                "link_length_check": float(np.linalg.norm(E - C))}

    # -- rigid-body poses for drawing -----------------------------------------------------------
    def poses(self, theta):
        """Transforms and world-space polygons of every body at half-opening theta."""
        th = float(self._theta(theta))
        bodies = {}
        for name, s in (("upper_arm", 1.0), ("lower_arm", -1.0)):
            T = Transform2D(angle=s * th, mirror_y=(s < 0))
            bodies[name] = T.apply(arm_outline(self))
        C = np.array([-float(self.crosshead_distance(th)), 0.0])
        for name, up in (("upper_link", True), ("lower_link", False)):
            bodies[name] = np.vstack([C, self.ear_point(th, upper=up)])
        rod_len = self.crosshead_distance(0.0) + 0.05
        h = self.bore / 2 + 0.012
        bodies["cylinder"] = np.array([[-rod_len - 0.30, -h], [-rod_len, -h], [-rod_len, h],
                                       [-rod_len - 0.30, h]])
        bodies["rod"] = np.vstack([[-rod_len, 0.0], C])
        bodies["crosshead"] = C
        bodies["pivot"] = np.array([0.0, 0.0])
        return bodies


def arm_outline(lk: Linkage, n=12):
    """Polygon of one arm in its own frame: ear at -r(cos d, sin d), tip at (L, 0).

    The working edge (jaw face / cutting edge) is the local x axis from the pivot to the tip, so
    the drawn edge is exactly the line used for contact and clipping.
    """
    L, r, d = lk.tip_length, lk.ear_radius, lk.ear_angle
    ear = -r * np.array([math.cos(d), math.sin(d)])
    hub = 0.11 * L
    xs = np.linspace(0, L, n)
    top = np.column_stack([xs, hub * (1 - xs / L) ** 0.8 + 0.015 * L])
    bot = np.column_stack([xs[::-1], np.zeros(n)])
    tail = np.array([ear + [0.0, -0.04 * L], ear + [-0.04 * L, 0.0]])
    return np.vstack([tail, top, bot, [[0.0, -hub]]])


# ---------------------------------------------------------------------------------------------
# 3. Cutter and material
# ---------------------------------------------------------------------------------------------

def blade_force(lk: Linkage, theta, distance, pressure=None):
    """Force normal to each blade at a point `distance` from the pivot [N]: T / d."""
    d = _positive("distance", distance)
    if d > lk.tip_length:
        raise ValueError("distance is beyond the end of the blade")
    return lk.arm_torque(theta, pressure) / d


def contact_distance(diameter, theta):
    """Distance from the pivot at which blades at half-angle theta first touch a round bar."""
    D = _positive("diameter", diameter)
    th = _positive("theta", theta)
    if th >= math.pi / 2:
        raise ValueError("theta must be below pi/2")
    return D / (2 * math.sin(th))


def shear_area_tube(outer_diameter, wall):
    D, t = _positive("outer_diameter", outer_diameter), _positive("wall", wall)
    if 2 * t > D:
        raise ValueError("wall thickness cannot exceed the radius")
    return math.pi * (D ** 2 - (D - 2 * t) ** 2) / 4


def shear_area_box(width, height, wall):
    """Rectangular hollow section (a car pillar idealized as a box tube)."""
    b, h, t = (_positive(n, v) for n, v in (("width", width), ("height", height), ("wall", wall)))
    if 2 * t > min(b, h):
        raise ValueError("wall thickness cannot exceed half the smaller side")
    return b * h - (b - 2 * t) * (h - 2 * t)


def cut_force(area, uts, shear_ratio=0.6):
    """Full-section shear estimate F = (shear_ratio * UTS) * area [N] (an upper bound)."""
    a, s = _positive("area", area), _positive("uts", uts)
    k = _positive("shear_ratio", shear_ratio)
    if k > 1:
        raise ValueError("shear_ratio is tau_u / UTS and should be <= 1")
    return k * s * a


def slides_out(theta, mu):
    """True if a round bar is squeezed out of straight blades: tan(theta) > mu."""
    return math.tan(_nonneg("theta", theta)) > _nonneg("mu", mu)


# ---------------------------------------------------------------------------------------------
# 4. The 2-D geometry engine
# ---------------------------------------------------------------------------------------------

@dataclass
class Transform2D:
    """Homogeneous 2-D rigid transform: optional mirror in y, then rotate, then translate."""

    angle: float = 0.0
    tx: float = 0.0
    ty: float = 0.0
    mirror_y: bool = False

    @property
    def matrix(self):
        c, s = math.cos(self.angle), math.sin(self.angle)
        M = np.array([[c, -s, self.tx], [s, c, self.ty], [0.0, 0.0, 1.0]])
        if self.mirror_y:
            M = M @ np.diag([1.0, -1.0, 1.0])
        return M

    def apply(self, pts):
        pts = np.atleast_2d(np.asarray(pts, dtype=float))
        h = np.column_stack([pts, np.ones(len(pts))])
        return (h @ self.matrix.T)[:, :2]

    def __matmul__(self, other):
        """Compose: (A @ B).apply(p) == A.apply(B.apply(p))."""
        M = self.matrix @ other.matrix
        mirror = np.linalg.det(M[:2, :2]) < 0
        R = M[:2, :2] @ (np.diag([1.0, -1.0]) if mirror else np.eye(2))
        return Transform2D(math.atan2(R[1, 0], R[0, 0]), M[0, 2], M[1, 2], bool(mirror))

    def inverse(self):
        Mi = np.linalg.inv(self.matrix)
        mirror = np.linalg.det(Mi[:2, :2]) < 0
        R = Mi[:2, :2] @ (np.diag([1.0, -1.0]) if mirror else np.eye(2))
        return Transform2D(math.atan2(R[1, 0], R[0, 0]), Mi[0, 2], Mi[1, 2], bool(mirror))


def polygon_area(poly):
    """Signed shoelace area (positive for counter-clockwise vertices)."""
    p = np.asarray(poly, dtype=float)
    x, y = p[:, 0], p[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def point_in_polygon(pt, poly):
    """Even-odd ray casting."""
    x, y = pt
    p = np.asarray(poly, dtype=float)
    inside = False
    for (x1, y1), (x2, y2) in zip(p, np.roll(p, -1, axis=0)):
        if (y1 > y) != (y2 > y) and x < x1 + (y - y1) * (x2 - x1) / (y2 - y1):
            inside = not inside
    return inside


def closest_point_on_polygon(pt, poly):
    pt = np.asarray(pt, dtype=float)
    p = np.asarray(poly, dtype=float)
    best, best_d = None, np.inf
    for a, b in zip(p, np.roll(p, -1, axis=0)):
        ab = b - a
        t = np.clip(np.dot(pt - a, ab) / max(np.dot(ab, ab), 1e-300), 0.0, 1.0)
        q = a + t * ab
        d = np.linalg.norm(pt - q)
        if d < best_d:
            best, best_d = q, d
    return best, best_d


def circle_polygon_contact(center, radius, poly):
    """Penetration depth of a circle into a polygon and the push-out normal (from polygon to circle).

    depth > 0 means overlap. The normal points the way the polygon surface must move to clear
    the circle (into the polygon if the centre is outside it).
    """
    _positive("radius", radius)
    c = np.asarray(center, dtype=float)
    q, d = closest_point_on_polygon(c, poly)
    inside = point_in_polygon(c, poly)
    if d < 1e-15:
        return radius, np.array([0.0, 0.0])
    n = (q - c) / d
    if inside:
        return radius + d, -n
    return radius - d, n


def clip_polygon_halfplane(poly, point, normal):
    """Sutherland-Hodgman: keep the part where (p - point) . normal <= 0."""
    p = np.asarray(poly, dtype=float)
    a0, n = np.asarray(point, dtype=float), np.asarray(normal, dtype=float)
    out = []
    for a, b in zip(p, np.roll(p, -1, axis=0)):
        da, db = np.dot(a - a0, n), np.dot(b - a0, n)
        if da <= 0:
            out.append(a)
        if da * db < 0:
            out.append(a + (b - a) * da / (da - db))
    return np.array(out) if out else np.zeros((0, 2))


def circle_polygon(center, radius, n=48, inner=None):
    """Polygon approximating a circle (or a tube cross-section outline if inner is given)."""
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    outer = np.column_stack([center[0] + radius * np.cos(t), center[1] + radius * np.sin(t)])
    return outer if inner is None else (outer, circle_polygon(center, inner, n))


@dataclass
class Panel:
    """Elastic-plastic sheet-metal panel with a latch that pops (game physics, illustrative)."""

    stiffness: float = 1.2e6        # N/m before yield
    yield_force: float = 22e3       # N plateau while the sheet folds
    pop_displacement: float = 0.045 # m of dent at which the latch / hinge lets go
    residual_force: float = 2e3     # N to keep swinging the freed door

    def __post_init__(self):
        for n in ("stiffness", "yield_force", "pop_displacement"):
            _positive(n, getattr(self, n))
        _nonneg("residual_force", self.residual_force)

    def resistance(self, dent, popped=False):
        dent = _nonneg("dent", dent)
        if popped or dent >= self.pop_displacement:
            return self.residual_force if dent > 0 else 0.0
        return min(self.stiffness * dent, self.yield_force)


@dataclass
class Frame:
    t: float
    extension: float
    theta: float
    gap: float
    pressure: float
    force_available: float
    force_required: float
    bodies: dict
    targets: dict
    event: str = ""
    extra: dict = field(default_factory=dict)


def _panels(gap, x0, x1, thick, n):
    xs = np.linspace(x0, x1, n)
    upper = np.vstack([np.column_stack([xs, np.full(n, gap / 2)]), [[x1, gap / 2 + thick],
                                                                       [x0, gap / 2 + thick]]])
    return upper, upper * np.array([1.0, -1.0])


def simulate_spreader(lk: Linkage | None = None, panel: Panel | None = None, gap=0.04,
                      tip_radius=0.012, pump_flow_lpm=1.0, dt=0.25, t_end=60.0,
                      dent_width=0.06, panel_thickness=0.08):
    """Game loop: pump drives the piston, tips dent two panels, latch pops or relief valve stalls.

    The panels are flat-faced strips above and below the axis with inner faces at +/- gap/2
    (the door edge and the pillar it is pried from; both treated as deformable for symmetry).
    """
    lk = lk or Linkage()
    panel = panel or Panel()
    _positive("gap", gap); _positive("tip_radius", tip_radius); _positive("dt", dt)
    _positive("pump_flow_lpm", pump_flow_lpm); _positive("t_end", t_end)
    if gap >= 2 * lk.tip_length * math.sin(lk.theta_max):
        raise ValueError("gap is wider than the fully open tips")
    v = pump_flow_lpm / 1000 / 60 / lk.piston_area      # piston speed [m/s]
    xt = lk.tip_length
    upper0, lower0 = _panels(gap, xt - 0.25, xt + 0.25, panel_thickness, 41)
    face = slice(0, 41)
    x, t, popped, stalled = 0.0, 0.0, False, False
    frames = []
    while t <= t_end + 1e-9:
        th = float(lk.theta_from_extension(x))
        tip = lk.tip_point(th)
        if popped:   # the freed door rides on the tip (flat-face closed form)
            dent = max(tip[1] + tip_radius - gap / 2, 0.0)
        else:
            dent = max(circle_polygon_contact(tip, tip_radius, upper0)[0], 0.0)
        R = panel.resistance(dent, popped)
        F_av = float(lk.tip_force(th))
        event = ""
        if dent > 0 and frames and not frames[-1].extra["contact"]:
            event = "contact"
        if not popped and dent >= panel.pop_displacement:
            popped, event = True, "latch popped"
        p_need = R / (float(lk.mechanical_advantage(th)) * lk.piston_area) if R > 0 else 0.0
        upper = upper0.copy()
        if popped:
            upper[:, 1] += dent                     # rigid swing-away (translation for simplicity)
        else:
            upper[face, 1] += dent * np.exp(-((upper[face, 0] - tip[0]) / dent_width) ** 2)
        frames.append(Frame(t, x, th, float(lk.tip_gap(th)), min(p_need, lk.pressure_max), F_av, R,
                            lk.poses(th), {"upper": upper, "lower": upper * [1.0, -1.0]},
                            event, {"contact": dent > 0, "dent": dent, "popped": popped,
                                    "tips": (tip, lk.tip_point(th, upper=False))}))
        if th >= lk.theta_max - 1e-9:
            frames[-1].event = frames[-1].event or "full open"
            break
        # propose the next step; the relief valve refuses it if the panel needs more than p_max
        x_new = min(x + v * dt, lk.stroke)
        th_new = float(lk.theta_from_extension(x_new))
        d_new = max(circle_polygon_contact(lk.tip_point(th_new), tip_radius, upper0)[0], 0.0)
        if panel.resistance(d_new, popped) > float(lk.tip_force(th_new)):
            frames[-1].event = "stall: relief valve"
            stalled = True
            break
        x, t = x_new, t + dt
    for f in frames:
        f.extra["stalled"] = stalled
    return frames


def simulate_cutter(lk: Linkage | None = None, diameter=0.040, wall=0.002, uts=1500e6,
                    distance=0.05, mu=0.2, n_frames=40):
    """Blades close from theta_max to 0 on a round tube held `distance` from the pivot.

    The tube outline is clipped by both blade lines each frame (crush). Required force is the
    full-section shear estimate; available force is blade torque / distance.
    """
    lk = lk or default_cutter()
    D = _positive("diameter", diameter)
    d = _positive("distance", distance)
    if d > lk.tip_length:
        raise ValueError("distance is beyond the end of the blades")
    if D / 2 >= d * math.sin(lk.theta_max):
        raise ValueError("tube does not fit between the open blades at that distance")
    F_req = cut_force(shear_area_tube(D, wall), uts)
    tube = circle_polygon((d, 0.0), D / 2)
    frames = []
    for th in np.linspace(lk.theta_max, 0.0, n_frames):
        th = float(th)
        n_up = np.array([-math.sin(th), math.cos(th)])          # outward normal of upper blade line
        n_lo = np.array([-math.sin(th), -math.cos(th)])
        crushed = clip_polygon_halfplane(clip_polygon_halfplane(tube, (0, 0), n_up), (0, 0), n_lo)
        touching = D / 2 >= d * math.sin(th)
        F_av = float(lk.arm_torque(th)) / d
        event = ""
        if touching and (not frames or not frames[-1].extra["touching"]):
            event = "blades touch tube"
        stalled = touching and F_av < F_req
        if stalled:
            event = "stall: relief valve"
        frames.append(Frame(0.0, float(lk.piston_extension(th)), th, float(lk.tip_gap(th)),
                            lk.pressure_max if touching else 0.0, F_av, F_req if touching else 0.0,
                            lk.poses(th), {"tube": crushed},
                            event, {"touching": touching, "slides_out": slides_out(th, mu),
                                    "area_frac": abs(polygon_area(crushed)) / abs(polygon_area(tube))}))
        if stalled:
            break
    if frames and not frames[-1].event.startswith("stall"):
        frames[-1].event = "cut through"
    return frames


def default_cutter():
    """Illustrative cutter: 180 mm blades, 70 mm bore, 30 deg half-opening."""
    return Linkage(tip_length=0.18, ear_radius=0.09, link_length=0.15,
                   ear_angle=math.radians(50), bore=0.070, theta_max=math.radians(30))


# ---------------------------------------------------------------------------------------------
# 5. Cutting force vs metal, and a timed extrication (the mini-game)
# ---------------------------------------------------------------------------------------------

#: Ultimate tensile strengths [Pa], typical values written from memory (illustrative only).
METALS = {
    "aluminium 6061-T6": 310e6,
    "mild steel": 400e6,
    "high-strength steel (HSLA)": 600e6,
    "advanced high-strength (DP1000)": 1000e6,
    "boron / UHSS (press-hardened)": 1500e6,
}

#: Cross-sections: (name, shear area [m^2], half-width the blades must close over [m]).
SECTIONS = {
    "tube 40x2": (shear_area_tube(0.040, 0.002), 0.020),
    "box pillar 50x80x1.5": (shear_area_box(0.050, 0.080, 0.0015), 0.025),
}


def min_cutter_force(lk: Linkage, distance, half_width, pressure=None):
    """Smallest blade force over the closing from first contact (d sin theta = half_width) to 0."""
    d, hw = _positive("distance", distance), _positive("half_width", half_width)
    if hw >= d * math.sin(lk.theta_max):
        raise ValueError("section does not fit between the open blades at that distance")
    th = np.linspace(0.0, math.asin(hw / d), 400)
    return float(np.min(lk.arm_torque(th, pressure))) / d


def seat_distance(lk: Linkage, half_width, distance=0.05):
    """Where the operator can hold a section: `distance`, or deeper-out if it does not fit (2% margin)."""
    hw = _positive("half_width", half_width)
    return max(_positive("distance", distance), 1.02 * hw / math.sin(lk.theta_max))


def metal_cut_table(lk: Linkage | None = None, distance=0.05, shear_ratio=0.6, pressure=None):
    """Rows (metal, section, seat distance, required kN, available kN, cuts?) at the rated pressure.

    Each section is held as close to the pivot as it fits (see seat_distance).
    """
    lk = lk or default_cutter()
    rows = []
    for metal, uts in METALS.items():
        for sec, (area, hw) in SECTIONS.items():
            F_req = cut_force(area, uts, shear_ratio)
            d = seat_distance(lk, hw, distance)
            F_av = min_cutter_force(lk, d, hw, pressure)
            rows.append({"metal": metal, "section": sec, "uts_MPa": uts / 1e6, "distance_mm": d * 1e3,
                         "required_kN": F_req / 1e3, "available_kN": F_av / 1e3, "cuts": F_av >= F_req})
    return rows


def extrication_run(pillar_metal="mild steel", pillar_section="box pillar 50x80x1.5",
                    pump_flow_lpm=1.0, target_time=120.0, cut_distance=0.05, tool_swap=30.0,
                    spreader: Linkage | None = None, cutter: Linkage | None = None, panel: Panel | None = None):
    """Timed extrication on a car door: the clock is driven by pump flow (piston speed = Q / A).

    Steps (each with its own time budget): 1. spread the door gap until the latch pops;
    2. swap to the cutter (fixed handling time); 3. cut the door hinge pillar (one cutter
    close = cutter stroke volume / Q, or fail if the metal beats the tool); 4. swap back;
    5. open the door fully with the spreader. Score = target_time - total time (positive = beat it).
    """
    if pillar_metal not in METALS:
        raise ValueError(f"unknown metal {pillar_metal!r}; choose from {list(METALS)}")
    if pillar_section not in SECTIONS:
        raise ValueError(f"unknown section {pillar_section!r}; choose from {list(SECTIONS)}")
    _positive("target_time", target_time); _nonneg("tool_swap", tool_swap)
    sp_lk, cu_lk = spreader or Linkage(), cutter or default_cutter()
    frames = simulate_spreader(sp_lk, panel, pump_flow_lpm=pump_flow_lpm, dt=0.05, t_end=600.0)
    q = pump_flow_lpm / 1000 / 60                      # m^3/s
    steps = []
    pop = next((f for f in frames if f.event == "latch popped"), None)
    if pop is None:
        steps.append(("spread door gap", frames[-1].t, "FAILED: " + frames[-1].event))
        return {"steps": steps, "total": float("nan"), "target": target_time, "score": float("nan"),
                "success": False}
    steps.append(("spread door gap, latch pops", pop.t, "ok"))
    steps.append(("swap to cutter", tool_swap, "ok"))
    area, hw = SECTIONS[pillar_section]
    F_req = cut_force(area, METALS[pillar_metal])
    F_av = min_cutter_force(cu_lk, seat_distance(cu_lk, hw, cut_distance), hw)
    t_cut = cu_lk.stroke * cu_lk.piston_area / q
    if F_av < F_req:
        steps.append((f"cut hinge pillar ({pillar_metal})", t_cut,
                      f"FAILED: needs {F_req/1e3:.0f} kN, tool makes {F_av/1e3:.0f} kN at 700 bar"))
        total = sum(s[1] for s in steps)
        return {"steps": steps, "total": total, "target": target_time, "score": float("nan"),
                "success": False}
    steps.append((f"cut hinge pillar ({pillar_metal})", t_cut, "ok"))
    steps.append(("swap to spreader", tool_swap, "ok"))
    steps.append(("open door fully", frames[-1].t - pop.t, frames[-1].event or "ok"))
    total = sum(s[1] for s in steps)
    return {"steps": steps, "total": total, "target": target_time, "score": target_time - total,
            "success": total <= target_time}


def summary(lk: Linkage | None = None):
    """Headline numbers for the default spreader."""
    lk = lk or Linkage()
    th = np.linspace(0, lk.theta_max, 7)
    return {"piston_force_kN": lk.piston_force() / 1e3, "stroke_mm": lk.stroke * 1e3,
            "max_gap_mm": float(lk.tip_gap(lk.theta_max)) * 1e3,
            "tip_force_kN": dict(zip(np.round(np.degrees(th), 1), np.round(lk.tip_force(th) / 1e3, 2)))}


if __name__ == "__main__":
    for k, v in summary().items():
        print(k, v)
