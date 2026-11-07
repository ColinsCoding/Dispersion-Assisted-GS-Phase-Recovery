"""Build notebooks/jaws_of_life.ipynb (hydraulic rescue spreader + cutter geometry engine)."""
import pathlib

import nbformat as nbf

md = lambda s: nbf.v4.new_markdown_cell(s)
co = lambda s: nbf.v4.new_code_cell(s)
nb = nbf.v4.new_notebook()

nb.cells = [
md("""# Jaws of life: a geometry engine for a hydraulic rescue tool
### SymPy linkage derivation · virtual work vs. force balance · a game-style rescue level (spreader + cutter)

**In plain words.** After a bad crash the doors are jammed and the paramedics in the ambulance can't
reach the patient. Firefighters bring two hydraulic tools. The **spreader** has two long jaws that
start closed, get pushed into a gap, and open with tens of kilonewtons of force until the door latch
tears loose. The **cutter** has two short curved blades that close on a roof pillar and shear it.

Both are the same machine. A pump pushes oil at about **700 bar** into a cylinder; pressure times
piston area is a ~200 kN straight push. That push drives a pin (the crosshead), two short links turn
two bent levers about one pivot, and the long ends (tips or blades) move. A short lever turning a long
one trades force for travel: the spreader tips travel ~700 mm for ~130 mm of piston, so tip force is
only a fraction of the piston force. Working out that fraction from geometry is the core of this
notebook. Then we put the tool in a small 2-D game engine (the kind a GTA-style ambulance level would
run every tick): rigid-body transforms, tip-vs-door collision, a door that dents and pops, a pillar that
gets crushed, a pump that sets the speed and a relief valve that makes the tool stall.

**All dimensions and ratings are illustrative and written from memory** (not from a datasheet or the
NFPA 1936 / EN 13204 standards). Library: `dgs/jaws_of_life.py`, tests: `tests/test_jaws_of_life.py`,
MATLAB cross-check: `learn/matlab/jaws_of_life_check.m`, Lean proof: `notebooks/jaws_of_life.lean`.
Figures use the Okabe-Ito palette, curves differ by line style too, and every figure has a printed
text description."""),

co("""import sys, pathlib, math
sys.path.insert(0, str(pathlib.Path.cwd().parent))
import numpy as np, sympy as sp, matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MPoly, Rectangle, Circle
from matplotlib import animation
from IPython.display import HTML, display
from dgs import jaws_of_life as jl
sp.init_printing()
OI = {"orange": "#E69F00", "sky": "#56B4E9", "green": "#009E73", "yellow": "#F0E442",
      "blue": "#0072B2", "red": "#D55E00", "pink": "#CC79A7", "black": "#000000"}
plt.rcParams["axes.prop_cycle"] = plt.cycler(color=list(OI.values()))
lk = jl.Linkage()
print(lk)"""),

md("""## 1. The linkage, derived (SymPy)

Pivot $O$ at the origin, tool axis $+x$ toward the tips. The upper arm is turned by the half-opening
$\\theta$: tip at $L(\\cos\\theta, \\sin\\theta)$. Its ear sits behind the pivot on the *other* side
(crossed links) at $E = -r(\\cos\\psi, \\sin\\psi)$ with $\\psi = \\theta + \\delta$ ($\\delta$ is the fixed
ear angle). The crosshead $C = (-c, 0)$ slides on the axis and is joined to $E$ by a link of length
$\\ell$. So $O$-$E$-$C$ is a **slider-crank**, and the link length fixes $c$."""),
co("""r, l, L, delta, th, c = sp.symbols("r ell L delta theta c", positive=True)
psi = th + delta
E = sp.Matrix([-r*sp.cos(psi), -r*sp.sin(psi)])
C = sp.Matrix([-c, 0])
closure = sp.expand(((E - C).T*(E - C))[0] - l**2)
print("loop closure  |E - C|^2 - l^2 = 0 :"); display(sp.Eq(sp.trigsimp(closure), 0))
c_sol = [s for s in sp.solve(closure, c) if s.subs({r: 1, l: 2, th: 0.1, delta: 0.3}) > 0][0]
print("forward kinematics c(theta):"); display(sp.Eq(sp.Symbol("c"), c_sol))
cos_inv = sp.solve(sp.Eq(l**2, c**2 + r**2 - 2*c*r*sp.Symbol("cpsi")), sp.Symbol("cpsi"))[0]
print("inverse kinematics (law of cosines):"); display(sp.Eq(sp.Symbol("cos psi"), cos_inv))"""),

md("""Piston extension is $x_p = c(0) - c(\\theta)$ and the tip gap is $w = 2L\\sin\\theta$.

**Virtual work.** With no friction, the work the piston puts in equals the work the tips put into the
door: $F_p\\,dx_p = F_s\\,dw$. So the mechanical advantage is a ratio of two derivatives."""),
co("""xp = c_sol.subs(th, 0) - c_sol
w = 2*L*sp.sin(th)
dxp = sp.simplify(sp.diff(xp, th))
MA_vw = sp.simplify(dxp/sp.diff(w, th))
print("dx_p/dtheta ="); display(dxp)
print("MA = F_s/F_p = (dx_p/dtheta)/(dw/dtheta) ="); display(MA_vw)"""),

md("""**Force balance, independently.** Crosshead: the piston force $F_p$ is shared by two links in
compression $N$ along unit vector $\\hat u = (E-C)/\\ell$, so $F_p = 2N\\hat u_x$. Upper arm: torque of
$N\\hat u$ at the ear about $O$ is balanced by the door pushing down on the tip,
$\\tau = F_s L\\cos\\theta$. SymPy shows the two routes agree identically."""),
co("""Fp = sp.symbols("F_p", positive=True)
root = sp.sqrt(l**2 - r**2*sp.sin(psi)**2)
u = (E - C.subs(c, c_sol))/l
N = Fp/(2*u[0])
F_ear = N*u
tau = sp.simplify(E[0]*F_ear[1] - E[1]*F_ear[0])
Fs_fb = sp.simplify(tau/(L*sp.cos(th)))
print("torque on arm tau ="); display(tau)
print("F_s (force balance) - F_p*MA (virtual work) simplifies to:", sp.simplify(Fs_fb - Fp*MA_vw))
vals = {r: 0.12, l: 0.19, L: 0.40, delta: sp.rad(25), th: sp.rad(37), Fp: 1}
print("numeric spot check at 37 deg:", sp.N(Fs_fb.subs(vals)), sp.N(MA_vw.subs(vals)))"""),

md("""**Limits.** As $\\ell\\to\\infty$ the link pushes the ear straight along $x$ and the slider-crank
becomes a plain lever: $\\mathrm{MA}\\to r\\sin\\psi / (2L\\cos\\theta)$. With $\\delta = 0$ and the
tips closed, the link points straight through the pivot (a *toggle*) and the tool has **no** force:
that is why real spreaders offset the ears."""),
co("""print("l -> oo:"); display(sp.limit(MA_vw, l, sp.oo))
print("delta = 0, theta = 0:", sp.simplify(MA_vw.subs({delta: 0, th: 0})))"""),

md("""## 2. Numbers, checked against the closed forms

Default spreader (illustrative): bore 60 mm at 700 bar, arms 400 mm, ear radius 120 mm, links 190 mm,
ear angle 25°, mechanical stop at 60° half-opening."""),
co("""for k, v in jl.summary(lk).items():
    print(f"{k:16s} {v}")
print(f"\\nclosed form F_p = p*pi*b^2/4 = {700e5*math.pi*0.06**2/4/1e3:.2f} kN")
print(f"closed form max gap = 2 L sin 60 = {2*0.4*math.sin(math.radians(60))*1e3:.1f} mm")
print("\\n theta   x_p[mm]  F_s virtual work [kN]  F_s force balance [kN]  rel. diff")
for deg in (0, 15, 30, 45, 60):
    t = math.radians(deg)
    fb = lk.force_balance(t)["separating_force"]
    vw = float(lk.tip_force(t))
    print(f"{deg:5d}  {float(lk.piston_extension(t))*1e3:8.2f}  {vw/1e3:20.3f}  {fb/1e3:22.3f}  {abs(fb-vw)/vw:9.1e}")
th_f = np.linspace(0, lk.theta_max, 20001)
W_tip = np.trapezoid(lk.tip_force(th_f)*2*lk.tip_length*np.cos(th_f), th_f)
print(f"\\nenergy: piston F_p*stroke = {lk.piston_force()*lk.stroke:.2f} J, tips integral F_s dw = {W_tip:.2f} J")
x = lk.piston_extension(np.radians([10, 40]))
print("inverse kinematics recovers theta [deg]:", np.degrees(lk.theta_from_extension(x)))"""),

co("""th_d = np.linspace(0, lk.theta_max, 300)
xp_mm = lk.piston_extension(th_d)*1e3
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].plot(xp_mm, lk.tip_gap(th_d)*1e3, color=OI["blue"], ls="-", lw=2, label="tip gap w")
ax[0].plot(xp_mm, xp_mm, color=OI["black"], ls=":", label="w = x_p (1:1 reference)")
ax[0].set(xlabel="piston extension x_p [mm]", ylabel="tip gap [mm]", title="Kinematics: travel multiplied")
ax[0].legend(); ax[0].grid(alpha=.3)
ax[1].plot(xp_mm, lk.tip_force(th_d)/1e3, color=OI["red"], ls="-", lw=2, label="virtual work")
fb = [lk.force_balance(t)["separating_force"]/1e3 for t in th_d[::15]]
ax[1].plot(xp_mm[::15], fb, "o", color=OI["green"], mfc="none", label="force balance")
ax[1].plot(xp_mm, lk.mechanical_advantage(th_d)*100, color=OI["orange"], ls="--", label="MA x 100")
ax[1].set(xlabel="piston extension x_p [mm]", ylabel="kN  (or MA x 100)", title="Tip force at 700 bar")
ax[1].legend(); ax[1].grid(alpha=.3)
plt.tight_layout(); plt.show()
print("Figure: left, tip gap (solid blue) against piston extension, rising from 0 to about 690 mm over a "
      "130 mm stroke, far above the dotted 1:1 line, so the jaws multiply travel about 5x. Right, separating "
      "tip force (solid red) from virtual work with force-balance points (green circles) on top of it, rising "
      "from about 20 kN closed to about 63 kN fully open; the dashed orange curve is the mechanical advantage x 100.")"""),

md("""## 3. The cutter: cut close to the pivot

Each blade gets torque $T = F_p\\,(dx_p/d\\theta)/2$. A force at distance $d$ from the pivot is $T/d$:
the same torque gives twice the force at half the distance, which is why rescuers push the pillar
deep into the blade throat. A round bar of diameter $D$ is first touched when $d\\sin\\theta = D/2$, and
straight blades squeeze it out if $\\tan\\theta > \\mu$ (hence hooked blade shapes). The force needed to
shear a full cross-section is roughly $F = \\tau_u A$ with $\\tau_u \\approx 0.6\\,\\sigma_\\text{UTS}$; that
is an upper bound because blades cut progressively."""),
co("""cut = jl.Linkage(tip_length=0.18, ear_radius=0.09, link_length=0.15, ear_angle=math.radians(50),
                 bore=0.070, theta_max=math.radians(30))
print(f"cutter piston force {cut.piston_force()/1e3:.0f} kN, blade torque at 15 deg {float(cut.arm_torque(math.radians(15)))/1e3:.2f} kN m")
A_tube = jl.shear_area_tube(0.040, 0.002)
A_box = jl.shear_area_box(0.050, 0.080, 0.0015)
cases = [("mild-steel tube 40x2", A_tube, 400e6), ("boron-steel tube 40x2", A_tube, 1500e6),
         ("mild box pillar 50x80x1.5", A_box, 400e6), ("boron box pillar 50x80x1.5", A_box, 1500e6)]
for name, A, s in cases:
    print(f"{name:28s} A = {A*1e6:6.1f} mm^2  F_cut = {jl.cut_force(A, s)/1e3:6.1f} kN  (0.6*UTS*A = {0.6*s*A/1e3:6.1f})")
print(f"\\ncontact distance for D = 40 mm at 20 deg: {jl.contact_distance(0.04, math.radians(20))*1e3:.1f} mm "
      f"(closed form 20/sin20 = {20/math.sin(math.radians(20)):.1f} mm)")
print("bar slides out at 20 deg with mu = 0.2?", jl.slides_out(math.radians(20), 0.2), f"(tan20 = {math.tan(math.radians(20)):.3f})")"""),
co("""d = np.linspace(0.02, 0.18, 200)
fig, ax = plt.subplots(figsize=(7, 4))
for deg, ls, col in ((0, "-", OI["blue"]), (15, "--", OI["green"]), (30, "-.", OI["pink"])):
    ax.plot(d*1e3, cut.arm_torque(math.radians(deg))/d/1e3, ls=ls, color=col, lw=2, label=f"blade force, theta = {deg} deg")
for (name, A, s), ls in zip(cases[:2], (":", (0, (5, 1)))):
    ax.axhline(jl.cut_force(A, s)/1e3, color=OI["red"] if s > 1e9 else OI["orange"], ls=ls, label=f"needed: {name}")
ax.set(xlabel="distance from pivot d [mm]", ylabel="force [kN]", ylim=(0, 800), title="Cutter: force = torque / distance")
ax.legend(fontsize=8); ax.grid(alpha=.3); plt.tight_layout(); plt.show()
print("Figure: blade force against distance from the pivot for three blade half-angles (solid blue 0 deg, dashed "
      "green 15 deg, dash-dot pink 30 deg), each falling like 1/d. The dotted orange line is the ~58 kN needed for a "
      "mild-steel 40x2 tube and the dense-dashed red line the ~215 kN for a boron-steel one: mild steel cuts anywhere "
      "on the blade, boron steel only within roughly 55-65 mm of the pivot.")"""),

md("""## 4. The geometry engine (what a game runs each tick)

Every body is a polygon in its own frame, placed in the world by a `Transform2D` (mirror, rotate,
translate; composable and invertible). Each tick of the spreader level:

1. The pump (1 L/min) advances the piston by $v\\,\\Delta t$ with $v = Q/A$.
2. Inverse kinematics gives $\\theta$; the arm transforms place the tip circles.
3. `circle_polygon_contact` measures how deep the tip is into the door's original surface (the dent).
4. The door's elastic-plastic law gives the force it needs; if that exceeds what the tool can make at
   700 bar the relief valve opens and the tool **stalls**. Past a dent of 45 mm the latch pops.
5. The door polygon's face vertices are pushed out by the dent with a Gaussian falloff (a soft dent).

Both panels (door and the pillar edge) deform symmetrically so the tool stays centred."""),
co("""frames = jl.simulate_spreader(lk)
for f in frames:
    if f.event:
        print(f"t = {f.t:5.2f} s  gap = {f.gap*1e3:6.1f} mm  needed {f.force_required/1e3:5.1f} kN of "
              f"{f.force_available/1e3:5.1f} kN available  pressure {f.pressure/1e5:5.0f} bar  -> {f.event}")
v = 1/1000/60/lk.piston_area
print(f"\\npiston speed Q/A = {v*1e3:.2f} mm/s; full stroke takes {lk.stroke/v:.1f} s (closed form), "
      f"simulation reached full open at t = {frames[-1].t:.2f} s")
strong = jl.simulate_spreader(lk, panel=jl.Panel(yield_force=80e3, pop_displacement=0.08))
print(f"stiffer door (80 kN plateau): last event '{strong[-1].event}' at gap {strong[-1].gap*1e3:.1f} mm")"""),

co("""def draw_tool(ax, bodies, arm_col=OI["blue"]):
    for k in ("upper_arm", "lower_arm"):
        ax.add_patch(MPoly(bodies[k], closed=True, fc=arm_col, ec="k", lw=0.8, alpha=0.9))
    for k in ("upper_link", "lower_link"):
        ax.plot(*bodies[k].T, color=OI["orange"], lw=3, solid_capstyle="round")
    ax.add_patch(MPoly(bodies["cylinder"], closed=True, fc=OI["yellow"], ec="k", lw=0.8))
    ax.plot(*bodies["rod"].T, color="0.4", lw=4)
    ax.plot(*bodies["crosshead"], "ko", ms=5); ax.plot(*bodies["pivot"], "o", color="w", mec="k", ms=6)

def draw_spreader_frame(ax, f):
    for k, col, hatch in (("upper", OI["sky"], "//"), ("lower", OI["green"], "\\\\\\\\")):
        ax.add_patch(MPoly(f.targets[k], closed=True, fc=col, ec="k", hatch=hatch, alpha=0.6))
    draw_tool(ax, f.bodies)
    ax.set(xlim=(-0.65, 0.7), ylim=(-0.45, 0.45), aspect="equal", xticks=[], yticks=[])
    ax.set_title(f"t={f.t:.1f}s gap={f.gap*1e3:.0f}mm\\n{f.event or (f'{f.pressure/1e5:.0f} bar')}", fontsize=9)

idx = [0, 3, 6, 9, 30, len(frames)-1]
fig, axs = plt.subplots(2, 3, figsize=(12, 6.5))
for a, i in zip(axs.flat, idx):
    draw_spreader_frame(a, frames[i])
plt.suptitle("Spreader level: tips dent the door, latch pops, door swings clear"); plt.tight_layout(); plt.show()
print("Figure: six frames of the spreader simulation, top view. The tool (blue arms, orange links, yellow "
      "cylinder, grey rod, white pivot) points right into the gap between the door (sky blue, hatched //) and the "
      "pillar edge (green, hatched \\\\). Frame 1 closed; frames 2-3 the tips press soft dents into both panels "
      "at a few hundred bar; frame 4 the latch has popped; frames 5-6 the freed panels ride outward on the tips "
      "until the jaws reach the 60 deg stop with a ~690 mm gap.")"""),

co("""t = np.array([f.t for f in frames])
fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
ax[0].plot(t, [f.force_available/1e3 for f in frames], color=OI["blue"], ls="-", lw=2, label="available at 700 bar")
ax[0].plot(t, [f.force_required/1e3 for f in frames], color=OI["red"], ls="--", lw=2, label="door needs")
ax[0].set(xlabel="time [s]", ylabel="tip force [kN]", xlim=(0, 6)); ax[0].legend(); ax[0].grid(alpha=.3)
ax[1].plot(t, [f.pressure/1e5 for f in frames], color=OI["pink"], ls="-.", lw=2, label="system pressure")
ax[1].axhline(700, color="k", ls=":", label="relief valve 700 bar")
ax[1].set(xlabel="time [s]", ylabel="bar", xlim=(0, 6)); ax[1].legend(); ax[1].grid(alpha=.3)
plt.tight_layout(); plt.show()
print("Figure: first 6 s of the spreader run. Left: force the tool can make at 700 bar (solid blue, ~20-26 kN "
      "here) versus what the door needs (dashed red: zero until contact, a steep elastic rise to the 22 kN plateau, "
      "then a drop to 2 kN when the latch pops at ~2.2 s). Right: the pressure the pump must reach (dash-dot pink) "
      "climbs to just under the 700 bar relief line (dotted black) on the plateau, then falls once the door is free.")"""),

md("""### The cutter level
The tube is held 50 mm from the pivot. Each frame both blade lines clip the tube outline
(`clip_polygon_halfplane`, Sutherland-Hodgman), so the remaining area shrinks to zero as the blades
meet. Held 70 mm out, a boron-steel tube stalls the tool."""),
co("""cframes = jl.simulate_cutter()
far = jl.simulate_cutter(distance=0.07)
for name, fr in (("50 mm", cframes), ("70 mm", far)):
    ev = [(round(math.degrees(f.theta), 1), f.event, round(f.force_available/1e3), round(f.force_required/1e3)) for f in fr if f.event]
    print(f"tube at {name}: (blade half-angle deg, event, available kN, needed kN) ->", ev)

def draw_cutter_frame(ax, f, tube0):
    ax.add_patch(MPoly(tube0, closed=True, fill=False, ec="0.6", ls=":"))
    if len(f.targets["tube"]):
        ax.add_patch(MPoly(f.targets["tube"], closed=True, fc=OI["red"], ec="k", hatch="xx", alpha=0.85, zorder=5))
    draw_tool(ax, f.bodies, arm_col=OI["green"])
    ax.set(xlim=(-0.35, 0.22), ylim=(-0.12, 0.12), aspect="equal", xticks=[], yticks=[])
    ax.set_title(f"half-angle {math.degrees(f.theta):.1f} deg, area {f.extra['area_frac']*100:.0f}%\\n{f.event}", fontsize=9)

tube0 = jl.circle_polygon((0.05, 0.0), 0.02)
fig, axs = plt.subplots(2, 3, figsize=(12, 5))
for a, i in zip(axs.flat, [0, 8, 16, 24, 32, len(cframes)-1]):
    draw_cutter_frame(a, cframes[i], tube0)
plt.suptitle("Cutter level: blades crush and shear a 40 mm boron-steel tube held 50 mm from the pivot")
plt.tight_layout(); plt.show()
print("Figure: six frames of the cutter closing from 30 deg to 0 deg half-angle. Green blades, orange links, yellow "
      "cylinder. The red cross-hatched shape is what is left of the tube after clipping by both blade lines; the "
      "dotted grey circle is the original outline. The remaining area falls from 100% through partial crush to 0% "
      "(cut through) when the blades meet.")"""),

md("""## 5. Cutting force versus metal

**Pascal's law** says the pump's pressure is the same everywhere in the oil, so the cutter's 70 mm
piston at 700 bar pushes with $pA \approx 269$ kN no matter how long the hose is. Whether that cuts
depends on the metal. Each section is held as close to the pivot as it fits, and "available" is the
*smallest* blade force over the whole closing (from first contact to blades shut), so a "yes" means
the tool never stalls. Strengths are typical values written from memory."""),
co("""rows = jl.metal_cut_table()
print(f"{'metal':34s} {'section':22s} {'UTS MPa':>8s} {'d mm':>6s} {'needs kN':>9s} {'has kN':>7s}  cuts?")
for r in rows:
    print(f"{r['metal']:34s} {r['section']:22s} {r['uts_MPa']:8.0f} {r['distance_mm']:6.1f} "
          f"{r['required_kN']:9.1f} {r['available_kN']:7.1f}  {'YES' if r['cuts'] else 'NO - stalls'}")
area, _ = jl.SECTIONS["box pillar 50x80x1.5"]
print(f"\nclosed-form check, boron box pillar: 0.6 x 1500 MPa x {area*1e6:.1f} mm^2 = {0.6*1500e6*area/1e3:.1f} kN")
print(f"pressure that WOULD cut it: {700*0.6*1500e6*area/(rows[-1]['available_kN']*1e3):.0f} bar (beyond this tool's 700 bar)")"""),
co("""metals = list(jl.METALS)
fig, ax = plt.subplots(figsize=(10, 4))
xx = np.arange(len(metals))
for k, (sec, hatch, col) in enumerate((("tube 40x2", "//", OI["sky"]), ("box pillar 50x80x1.5", "xx", OI["orange"]))):
    req = [next(r for r in rows if r["metal"] == m and r["section"] == sec)["required_kN"] for m in metals]
    av = next(r for r in rows if r["section"] == sec)["available_kN"]
    ax.bar(xx + (k - 0.5)*0.38, req, 0.38, color=col, hatch=hatch, ec="k", label=f"needed: {sec}")
    ax.axhline(av, color=OI["blue"] if k == 0 else OI["red"], ls="--" if k == 0 else ":", lw=2,
               label=f"tool makes (min over closing): {sec}")
ax.set_xticks(xx, [m.replace(" (", "\n(") for m in metals], fontsize=8)
ax.set(ylabel="force [kN]", title="Cutter at 700 bar vs. metal"); ax.legend(fontsize=8); ax.grid(alpha=.3, axis="y")
plt.tight_layout(); plt.show()
print("Figure: bars are the full-section shear force needed for each metal, weakest (aluminium) to strongest "
      "(boron/UHSS); sky-blue // bars for the 40x2 tube, orange xx bars for the 50x80x1.5 box pillar. The dashed blue "
      "(~266 kN) and dotted red (~261 kN) lines are the least force the cutter makes while closing on each section. "
      "Every bar is below its line except the boron box pillar (~343 kN): the classic cutter-defeating case.")"""),

md("""## 6. The mini-game: a timed extrication

The clock is the pump. Oil flow $Q$ moves a piston of area $A$ at $v = Q/A$, so every hydraulic step
takes (volume swept) / $Q$. Steps: spread the door gap until the latch pops, swap to the cutter, cut
the hinge pillar (one full close), swap back, open the door fully. Tool swaps are a fixed 30 s
(a guess). Beat the 120 s target to score."""),
co("""def play(**kw):
    run = jl.extrication_run(**kw)
    print(f"--- {kw or 'defaults (mild-steel box pillar, 1 L/min pump)'}")
    clock = 0.0
    for name, dt_step, status in run["steps"]:
        clock += dt_step
        print(f"  {name:48s} {dt_step:6.1f} s   clock {clock:6.1f} s   {status}")
    verdict = (f"score {run['score']:+.1f} s vs target {run['target']:.0f} s" if run["success"]
               else ("FAILED" if math.isnan(run["score"]) else f"too slow by {-run['score']:.1f} s"))
    print(f"  TOTAL {run['total']:.1f} s  ->  {verdict}\n")
    return run

runs = {"1 L/min": play(), "2 L/min": play(pump_flow_lpm=2.0), "0.5 L/min": play(pump_flow_lpm=0.5),
        "boron": play(pillar_metal="boron / UHSS (press-hardened)")}
cu = jl.default_cutter()
print(f"closed-form cut time at 1 L/min: stroke {cu.stroke*1e3:.1f} mm x area {cu.piston_area*1e4:.2f} cm^2 / Q = "
      f"{cu.stroke*cu.piston_area/(1/1000/60):.2f} s")"""),
co("""fig, ax = plt.subplots(figsize=(10, 3.2))
cols = {"spread": OI["blue"], "swap": "0.7", "cut": OI["red"], "open": OI["green"]}
hatches = {"spread": "//", "swap": "", "cut": "xx", "open": "\\\\"}
for row, (label, run) in enumerate(runs.items()):
    t0 = 0.0
    for name, dt_step, status in run["steps"]:
        kind = next(k for k in cols if name.startswith(k))
        ax.barh(row, dt_step, left=t0, color=cols[kind], hatch=hatches[kind], ec="k",
                alpha=0.4 if status.startswith("FAILED") else 1.0)
        t0 += dt_step
ax.axvline(120, color="k", ls="--", label="target 120 s")
ax.set_yticks(range(len(runs)), list(runs)); ax.invert_yaxis()
ax.set(xlabel="time [s]", title="Extrication timelines"); ax.legend(loc="lower right"); ax.grid(alpha=.3, axis="x")
plt.tight_layout(); plt.show()
print("Figure: one horizontal timeline per run (1, 2, 0.5 L/min pump, and a boron pillar). Segments: blue // spread "
      "until the latch pops, grey tool swaps, red xx the pillar cut, green \\ opening the door fully; the dashed black "
      "line is the 120 s target. Doubling the flow halves every hydraulic segment but not the swaps; the 0.5 L/min run "
      "overshoots the target; the boron run ends in a faded red segment because the cut fails.")"""),

md("""## 7. The rescue scene, game style

A top-down view in the spirit of the original GTA games: the crashed car, the ambulance waiting with
its light bar flashing, and the spreader working the gap between the driver's door and the B-pillar.
Top-down is the honest view for a door, because a door swings about a vertical hinge, which is a
plain in-plane rotation here (one more `Transform2D`). The tool is the same frame data, placed by a
`Transform2D` that turns its axis to point into the car."""),
co("""tool_T = jl.Transform2D(angle=math.pi/2, tx=2.2, ty=-0.40)   # tool axis -> +y, tips start at the door gap

def scene(ax, f):
    ax.clear()
    ax.add_patch(Rectangle((-0.6, -1.6), 10.2, 4.0, fc="0.35"))                      # asphalt
    for xd in np.arange(-0.4, 9.6, 1.2):
        ax.add_patch(Rectangle((xd, -1.25), 0.6, 0.06, fc=OI["yellow"]))             # lane dashes
    car = np.array([[0, 0], [4.4, 0], [4.5, 0.2], [4.5, 1.6], [4.4, 1.8], [0, 1.8], [-0.1, 1.6], [-0.1, 0.2]])
    ax.add_patch(MPoly(car, closed=True, fc=OI["sky"], ec="k"))
    ax.add_patch(Rectangle((2.9, 0.15), 0.5, 1.5, fc="0.2"))                          # windshield
    ax.add_patch(Rectangle((0.6, 0.15), 0.4, 1.5, fc="0.2"))                          # rear window
    ax.add_patch(Rectangle((2.2, -0.02), 0.08, 0.2, fc="0.15"))                        # B-pillar
    # door: hinged at its front end (x = 3.3); swings out once the latch has popped
    pop = next((g.t for g in frames if g.event == "latch popped"), np.inf)
    swing = 0.0 if f.t < pop else min(1.2, (f.gap - 0.11)/0.45)
    swing += 0.03*f.extra["dent"]/0.045 if f.t < pop else 0.0
    door_local = np.array([[-1.08, -0.09], [0, -0.09], [0, 0], [-1.08, 0]])
    door = jl.Transform2D(angle=swing, tx=3.3, ty=0.0).apply(door_local)
    ax.add_patch(MPoly(door, closed=True, fc=OI["blue"], ec="k", hatch="//"))
    ax.plot(3.3, 0.0, "o", color="k", ms=4)
    # ambulance, top view
    amb = np.array([[5.6, 0.0], [9.2, 0.0], [9.2, 1.9], [5.6, 1.9], [5.4, 1.6], [5.4, 0.3]])
    ax.add_patch(MPoly(amb, closed=True, fc="w", ec="k"))
    ax.add_patch(Rectangle((6.8, 0.82), 1.4, 0.26, fc=OI["red"])); ax.add_patch(Rectangle((7.37, 0.25), 0.26, 1.4, fc=OI["red"]))
    on = int(f.t*4) % 2
    ax.add_patch(Rectangle((5.9, 0.15), 0.18, 0.75, fc=OI["blue"] if on else OI["red"]))
    ax.add_patch(Rectangle((5.9, 1.0), 0.18, 0.75, fc=OI["red"] if on else OI["blue"]))
    b = {k: tool_T.apply(v) if np.ndim(v) == 2 else tool_T.apply(v[None])[0] for k, v in f.bodies.items()}
    draw_tool(ax, b, arm_col=OI["orange"])
    ax.text(-0.5, 2.15, f"RESCUE  t={f.t:4.1f}s  gap {f.gap*1e3:4.0f} mm  {f.pressure/1e5:4.0f} bar  {f.event}",
            family="monospace", fontsize=9, bbox=dict(fc="k", alpha=0.85), color=OI["yellow"])
    ax.set(xlim=(-0.6, 9.6), ylim=(-1.6, 2.45), aspect="equal", xticks=[], yticks=[])

fig, ax = plt.subplots(figsize=(10, 4))
sel = frames[::3] + [frames[-1]]
anim = animation.FuncAnimation(fig, lambda i: scene(ax, sel[i]), frames=len(sel), interval=120)
html = anim.to_jshtml(); plt.close(fig)
display(HTML(html))
fig, ax = plt.subplots(figsize=(10, 4)); scene(ax, frames[40]); plt.show()
print("Figure (animation, then one still at t = 10 s): GTA-style top-down view on grey asphalt with yellow lane "
      "dashes. Left, the crashed light-blue car (dark windows); its driver's door (dark blue, hatched //) is hinged at "
      "its front end (black dot) next to the B-pillar. Right, a white ambulance with a red cross and a light bar that "
      "flashes red/blue frame to frame. The spreader (orange arms, orange links, yellow cylinder) comes in from below, "
      "tips in the door gap. The HUD bar shows time, tip gap, pump pressure and events. The tool and pressures come "
      "from the simulation; the door swing angle is a cartoon tied to the gap after the latch pops, not computed "
      "hinge mechanics.")"""),

md("""## What is and is not modelled

* **Verified here:** the slider-crank kinematics (loop closure, inverse kinematics round trip), the
  virtual-work mechanical advantage against a separate free-body force balance (SymPy: identical;
  numerics: ~1e-16), energy over a full stroke, the $\\ell\\to\\infty$ lever limit and the $\\delta=0$
  toggle limit. Also in `tests/test_jaws_of_life.py`, in MATLAB (`learn/matlab/jaws_of_life_check.m`),
  and the monotonic piston-extension fact in Lean (`notebooks/jaws_of_life.lean`).
* **Illustrative, from memory:** 700 bar, the 60 mm / 70 mm bores, arm and blade lengths, 1 L/min
  pump, 400 / 1500 MPa steel strengths, $\\tau_u\\approx0.6\\,\\sigma_\\text{UTS}$, the door's 22 kN plateau
  and 45 mm latch pop. Real tools are rated by NFPA 1936 / EN 13204 tests I have not checked here.
* **Game physics, not crash mechanics:** straight arms, frictionless pins, symmetric panels, a
  Gaussian dent, full-section shear as an upper bound, and a cartoon door swing in the top-down scene.

**Connections in this repo:** the same "work in = work out" step is the transformer turns ratio in
`dgs/ac_circuits.py`, and the rigid-transform + polygon-contact pattern is what `dgs/algorithms.py`
style data structures would accelerate (a bounding-volume tree) in a real game engine.
**Tests:** `PYTHONPATH=. python -m pytest tests/test_jaws_of_life.py -q`."""),
]

nb.metadata["kernelspec"] = {"name": "py312", "display_name": "py312", "language": "python"}
out = pathlib.Path(__file__).resolve().parent.parent / "notebooks" / "jaws_of_life.ipynb"
nbf.write(nb, out)
print("wrote", out)
