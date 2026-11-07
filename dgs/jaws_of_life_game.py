"""Jaws of life, the game layer: a timed car-door extrication and a "what can this cutter cut?" metal table.

In plain words
--------------
dgs.jaws_of_life works out the tool: oil at ~700 bar on a 60 mm piston (Pascal: F = p A, about 200 kN),
a bell-crank linkage that turns that push into tip force (spreader) or blade force (cutter), and the
shear force a section of metal needs. This module plays a rescue level with it:

* **Clock.** The pump fixes the oil flow Q, so the piston moves at v = Q / A. Every action costs the
  time it takes to push the oil: opening or closing by a piston extension x takes x A / Q. Bigger
  piston = more force but slower; bigger pump = faster. Tool changes and repositioning add fixed time.
* **Spreading.** To pry the door the tips must reach a gap g (half-angle theta = asin(g / 2L)) while
  still pushing harder than the door resists. If the tip force at that opening is too small, the
  relief valve opens and the tool stalls: the step fails.
* **Cutting.** A pillar or hinge of box (or solid) section needs F = 0.6 x UTS x area to shear
  (full-section, an upper bound). The cutter is strongest deep in its throat, F = T / d, but the
  section must fit between the blades: 2 d sin(theta) >= depth, so d >= depth / (2 sin theta_max).
  The best spot is found on a fixed grid (same grid in MATLAB). Modern cars use hot-stamped boron
  steel (~1500 MPa) in the B-pillar, which is exactly where cutters run out of force.
* **Level.** A car (latch strength, hinge and pillar sections and metals), a plan (an ordered list of
  actions) and a tool kit (cutter/spreader linkages, pump flow). run_level() returns a timeline,
  success or the step that stalled, total time and a star score against a target time.

Honesty: every number (latch and hinge strengths, pillar sections, 0.6 shear ratio, tool-change
times, the 10-minute target, the return-stroke factor) is ILLUSTRATIVE and from memory. This is a
physics game for teaching F = pA, levers and shear, not rescue training, and the real NFPA 1936 /
EN 13204 ratings and extrication procedures are not reproduced here.

Built on (and does not modify): dgs.jaws_of_life (Linkage, blade_force, cut_force, shear_area_box).
MATLAB parity: learn/matlab/jaws_of_life_game_check.m prints the same "key value" table.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from dgs.jaws_of_life import Linkage, blade_force, cut_force, shear_area_box

SHEAR_RATIO = 0.6          # ultimate shear / UTS (rule of thumb)
RETURN_FACTOR = 0.6        # return stroke takes this fraction of the working-stroke time (smaller annulus volume)
D_GRID = 2001              # points in the best-cutting-distance search (MATLAB uses the same)
THROAT_MIN = 0.05          # m: blades cannot bite closer to the pivot than their throat (illustrative)

METALS = {   # name: (UTS in Pa, where you meet it) -- typical values, from memory
    "aluminium 6xxx-T6": (300e6, "bumper beams, some newer door frames"),
    "mild steel": (400e6, "older car bodies, door skins"),
    "HSLA steel": (600e6, "door rings, rails"),
    "dual-phase DP980": (980e6, "pillar reinforcements"),
    "boron steel (press-hardened)": (1500e6, "modern B-pillars and roof rails"),
}


def _pos(name, v):
    v = float(v)
    if not math.isfinite(v) or v <= 0:
        raise ValueError(f"{name} must be positive and finite, got {v!r}")
    return v


def shear_area_solid(width, thickness):
    return _pos("width", width) * _pos("thickness", thickness)


@dataclass
class Section:
    """What the cutter must go through: a box tube (pillar) or a solid plate (hinge)."""
    width: float
    depth: float                 # the dimension that must fit between the blades
    wall: float | None = None    # None -> solid
    metal: str = "mild steel"

    def __post_init__(self):
        _pos("width", self.width); _pos("depth", self.depth)
        if self.metal not in METALS:
            raise ValueError(f"unknown metal {self.metal!r}; choose from {list(METALS)}")

    @property
    def area(self):
        if self.wall is None:
            return shear_area_solid(self.width, self.depth)
        return shear_area_box(self.width, self.depth, self.wall)

    @property
    def uts(self):
        return METALS[self.metal][0]

    @property
    def force_needed(self):
        return cut_force(self.area, self.uts, SHEAR_RATIO)


# ---------------------------------------------------------------- physics of each action
def piston_time(lk: Linkage, extension, flow_lpm):
    """Seconds to move the piston by `extension` metres at pump flow Q: x A / Q."""
    if extension < 0:
        raise ValueError("extension must be >= 0")
    Q = _pos("flow_lpm", flow_lpm) / 1000 / 60
    return float(extension) * lk.piston_area / Q


def cut_capacity(lk: Linkage, depth, pressure=None):
    """Best blade force for a section of the given depth: max over d of T(theta(d)) / d, theta = asin(depth/2d).

    Returns (force N, distance d m, half-angle theta rad). Raises if the section cannot fit at all.
    """
    depth = _pos("depth", depth)
    d_min = max(depth / (2 * math.sin(lk.theta_max)), THROAT_MIN)
    if d_min > lk.tip_length:
        raise ValueError(f"a {depth*1e3:.0f} mm section does not fit in this cutter's jaws")
    ds = np.linspace(d_min, lk.tip_length, D_GRID)
    th = np.arcsin(np.clip(depth / (2 * ds), 0, 1))
    F = np.array([blade_force(lk, t, d, pressure) for t, d in zip(th, ds)])
    i = int(np.argmax(F))
    return float(F[i]), float(ds[i]), float(th[i])


def spread_check(lk: Linkage, gap, resistance, pressure=None):
    """Can the spreader reach tip gap `gap` while pushing `resistance` newtons? Returns dict."""
    gap = _pos("gap", gap)
    if gap > 2 * lk.tip_length * math.sin(lk.theta_max):
        return {"ok": False, "theta": None, "force": 0.0, "extension": lk.stroke,
                "why": "gap wider than the fully open tips"}
    theta = math.asin(gap / (2 * lk.tip_length))
    F = float(lk.tip_force(theta, pressure))
    ok = F >= resistance
    return {"ok": ok, "theta": theta, "force": F, "extension": float(lk.piston_extension(theta)),
            "why": "" if ok else f"relief valve: tip force {F/1e3:.1f} kN < resistance {resistance/1e3:.1f} kN"}


def metal_table(lk: Linkage | None = None, width=0.10, depth=0.05, wall=0.002):
    """Same box section in every metal: force needed vs this cutter's best force."""
    lk = lk or Linkage()
    cap, d, th = cut_capacity(lk, depth)
    rows = []
    for name, (uts, where) in METALS.items():
        s = Section(width, depth, wall, name)
        rows.append({"metal": name, "uts_MPa": uts / 1e6, "where": where, "area_mm2": s.area * 1e6,
                     "needed_kN": s.force_needed / 1e3, "capacity_kN": cap / 1e3, "cuts": cap >= s.force_needed,
                     "margin": cap / s.force_needed})
    return {"rows": rows, "best_distance_m": d, "theta_deg": math.degrees(th)}


def max_wall_that_cuts(lk: Linkage, metal, width=0.10, depth=0.05):
    """Thickest box wall this cutter can shear in one bite (bisection)."""
    cap, _, _ = cut_capacity(lk, depth)
    lo, hi = 1e-6, min(width, depth) / 2 - 1e-9
    if Section(width, depth, hi, metal).force_needed <= cap:
        return hi
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if Section(width, depth, mid, metal).force_needed <= cap:
            lo = mid
        else:
            hi = mid
    return lo


# ---------------------------------------------------------------- the level
@dataclass
class Car:
    name: str
    latch_force: float            # N to pop the latch by spreading
    latch_gap: float              # m of tip gap at which it pops
    purchase_force: float         # N to crush a purchase point in the door gap
    purchase_gap: float
    hinge: Section
    b_pillar: Section
    b_pillar_relief: Section      # a weaker spot (below the reinforcement) if the main cut stalls


@dataclass
class Kit:
    cutter: Linkage = field(default_factory=Linkage)
    spreader: Linkage = field(default_factory=Linkage)
    flow_lpm: float = 1.0
    tool_change_s: float = 20.0   # set one tool down, pick up the other
    reposition_s: float = 8.0     # move the same tool to the next spot

    def __post_init__(self):
        _pos("flow_lpm", self.flow_lpm)
        if self.tool_change_s < 0 or self.reposition_s < 0:
            raise ValueError("handling times must be >= 0")


LEVELS = {
    "1990s sedan": Car("1990s sedan", latch_force=14e3, latch_gap=0.10, purchase_force=6e3, purchase_gap=0.035,
                       hinge=Section(0.035, 0.008, None, "mild steel"),
                       b_pillar=Section(0.10, 0.05, 0.0015, "mild steel"),
                       b_pillar_relief=Section(0.10, 0.05, 0.0010, "mild steel")),
    "2020s SUV": Car("2020s SUV", latch_force=24e3, latch_gap=0.10, purchase_force=8e3, purchase_gap=0.035,
                     hinge=Section(0.040, 0.010, None, "HSLA steel"),
                     b_pillar=Section(0.10, 0.05, 0.004, "boron steel (press-hardened)"),
                     b_pillar_relief=Section(0.10, 0.05, 0.0012, "HSLA steel")),
}

PLANS = {
    "door removal": ["spread purchase", "spread latch", "cut upper hinge", "cut lower hinge"],
    "B-pillar rip": ["cut B-pillar top", "cut B-pillar bottom", "spread purchase", "spread latch"],
    "B-pillar rip, relief cuts": ["cut B-pillar top (relief)", "cut B-pillar bottom (relief)",
                                  "spread purchase", "spread latch"],
}
TARGET_S = 600.0    # illustrative 10-minute extrication target


def run_level(car: Car | str, plan: list[str] | str, kit: Kit | None = None, target_s=TARGET_S):
    """Play a plan. Returns dict(timeline, success, failed_step, total_s, stars)."""
    car = LEVELS[car] if isinstance(car, str) else car
    steps = PLANS[plan] if isinstance(plan, str) else list(plan)
    kit = kit or Kit()
    t, tool, timeline = 0.0, None, []

    def add(name, dt, ok, detail):
        nonlocal t
        timeline.append({"step": name, "t_start": t, "t_end": t + dt, "ok": ok, "detail": detail})
        t += dt

    for step in steps:
        want = "spreader" if step.startswith("spread") else "cutter"
        if step.split()[0] not in ("spread", "cut"):
            raise ValueError(f"unknown action {step!r}")
        lk = kit.spreader if want == "spreader" else kit.cutter
        handling = kit.tool_change_s if (tool is not None and tool != want) else (kit.reposition_s if tool else 0.0)
        if handling:
            add(f"{'tool change to ' + want if handling == kit.tool_change_s else 'reposition'}", handling, True, "")
        tool = want
        if want == "spreader":
            gap, R = ((car.purchase_gap, car.purchase_force) if "purchase" in step else (car.latch_gap, car.latch_force))
            chk = spread_check(lk, gap, R)
            x = chk["extension"]
            dt = piston_time(lk, x, kit.flow_lpm) * (1 + RETURN_FACTOR)
            add(step, dt, chk["ok"], f"gap {gap*1e3:.0f} mm, tip force {chk['force']/1e3:.1f} kN vs {R/1e3:.1f} kN"
                if chk["ok"] else chk["why"])
        else:
            sec = (car.hinge if "hinge" in step else
                   car.b_pillar_relief if "relief" in step else car.b_pillar)
            cap, d, th = cut_capacity(lk, sec.depth)
            x = float(lk.piston_extension(th))
            dt = piston_time(lk, x, kit.flow_lpm) * (1 + RETURN_FACTOR)
            ok = cap >= sec.force_needed
            add(step, dt, ok, f"{sec.metal}, needs {sec.force_needed/1e3:.0f} kN, cutter gives {cap/1e3:.0f} kN "
                f"at {d*1e3:.0f} mm from the pivot" + ("" if ok else ": STALLED"))
        if not timeline[-1]["ok"]:
            return {"timeline": timeline, "success": False, "failed_step": step, "total_s": t, "stars": 0}
    stars = 3 if t <= target_s / 2 else 2 if t <= target_s else 1
    return {"timeline": timeline, "success": True, "failed_step": None, "total_s": t, "stars": stars}


def best_plan(car: Car | str, kit: Kit | None = None):
    """Try every plan; return (name, result) of the fastest that succeeds, or (None, None)."""
    best = (None, None)
    for name in PLANS:
        r = run_level(car, name, kit)
        if r["success"] and (best[1] is None or r["total_s"] < best[1]["total_s"]):
            best = (name, r)
    return best


# ---------------------------------------------------------------- shared table (Python == MATLAB)
def table_lines(kit: Kit | None = None):
    """Fixed-format 'key value' lines for the MATLAB parity check (default tool, 1 L/min)."""
    kit = kit or Kit()
    lk = kit.cutter
    out = {"tool.piston_area_m2": lk.piston_area, "tool.piston_force_N": lk.piston_force(),
           "tool.stroke_m": lk.stroke, "tool.full_stroke_s": piston_time(lk, lk.stroke, kit.flow_lpm)}
    cap, d, th = cut_capacity(lk, 0.05)
    out.update({"cut.capacity_N": cap, "cut.best_distance_m": d, "cut.theta_rad": th})
    for name, (uts, _) in METALS.items():
        key = name.split()[0].lower().replace("-", "")
        out[f"metal.{key}.needed_N"] = Section(0.10, 0.05, 0.002, name).force_needed
    for lvl in LEVELS:
        for plan in PLANS:
            r = run_level(lvl, plan, kit)
            key = f"level.{lvl.split()[0]}.{plan.replace(' ', '_').replace(',', '')}"
            out[key + ".total_s"] = r["total_s"]
            out[key + ".success"] = float(r["success"])
    return [f"{k} {v:.12e}" for k, v in sorted(out.items())]


def parse_table(text):
    out = {}
    for line in text.splitlines():
        parts = line.split()
        if len(parts) == 2 and "." in parts[0]:
            try:
                out[parts[0]] = float(parts[1])
            except ValueError:
                pass
    return out


if __name__ == "__main__":
    print("\n".join(table_lines()))
