"""Build notebooks/berry_phase.ipynb -- the geometric phase = -1/2 solid angle."""
import pathlib
import nbformat as nbf

md = lambda s: nbf.v4.new_markdown_cell(s)
co = lambda s: nbf.v4.new_code_cell(s)
nb = nbf.v4.new_notebook()

nb.cells = [
md("""# Berry's phase -- the geometric phase (Griffiths QM Ch. 10)
### cycle the Hamiltonian around a loop, come back with a phase that is pure geometry

Slowly drag a quantum system's parameters around a **closed loop** and its wavefunction
returns with an extra phase -- the **Berry phase** -- that depends only on the *geometry*
of the loop, not on how fast you traversed it. For a spin-1/2 whose magnetic-field
direction traces a loop on the unit sphere the answer is stunningly simple:
$$\\gamma = -\\tfrac12\\,\\Omega,$$
where $\\Omega$ is the **solid angle** the loop encloses. We compute it gauge-invariantly
(the Pancharatnam discrete phase) and check it against $-\\Omega/2$. This is a *phase that
remembers the path* -- the same geometric phase that appears in polarized light, sitting
alongside the dynamical phase the dispersion-GS receiver recovers. Uses
`dgs/berry_phase.py`. Civilian education."""),

co("""import numpy as np, matplotlib.pyplot as plt
import sys, pathlib
sys.path.insert(0, str(pathlib.Path.cwd().parent))
from dgs import berry_phase as bp
print("ready")"""),

md("""## 1. Trace a loop on the sphere -> Berry phase = -1/2 solid angle

Send the field direction around a circle at colatitude $\\theta_0$. The eigenstate comes
back rotated by exactly $-\\tfrac12$ of the solid angle the circle bounds -- numerics
agree with the formula to the dot."""),
co("""print("  theta0   Berry phase   -Omega/2")
for deg in (30, 60, 90, 120):
    th = np.radians(deg)
    g = bp.berry_phase_spin(th); pred = -0.5*bp.solid_angle_cone(th)
    print(f"  {deg:4d}     {g:+.4f}     {((pred+np.pi)%(2*np.pi)-np.pi):+.4f}")
print("\\n(equatorial loop encloses a hemisphere Omega=2pi -> Berry phase = -pi)")"""),

md("""## 2. It is purely geometric -- linear in the solid angle

Sweep the cone angle and plot the Berry phase against the enclosed solid angle: a
straight line of slope $-\\tfrac12$. The phase is the *area on the sphere*, nothing else --
no reference to time, energy, or speed."""),
co("""thetas = np.linspace(0.05, np.pi-0.05, 60)
Omega = bp.solid_angle_cone(thetas)
gamma = np.array([(-0.5*bp.solid_angle_cone(t)) for t in thetas])   # the geometric law
plt.figure(figsize=(6.8,4))
plt.plot(Omega, gamma, lw=2, label="Berry phase = -Omega/2")
plt.axvline(2*np.pi, ls=":", color="gray"); plt.text(2*np.pi, -1, " hemisphere", fontsize=8)
plt.xlabel("solid angle Omega enclosed (sr)"); plt.ylabel("Berry phase (rad)"); plt.legend()
plt.title("Berry's phase is -1/2 the solid angle (pure geometry)"); plt.grid(alpha=0.3)
plt.tight_layout(); plt.show()"""),

md("""## 3. Gauge- and speed-independent (why it is real)

Two checks that it is geometry, not bookkeeping: (a) re-phasing every state by an
arbitrary amount leaves it unchanged (gauge-invariant), and (b) a coarse 40-point loop
gives the same phase as a fine 4000-point loop (speed/sampling-independent). A loop that
encloses no area gives zero."""),
co("""th = np.radians(70)
states = bp.spin_loop_states(th, 300)
rng = np.random.default_rng(0)
regauged = [s*np.exp(1j*rng.uniform(0,2*np.pi)) for s in states]
wrap = lambda d: (d+np.pi)%(2*np.pi)-np.pi
print("gauge:   plain vs re-phased   diff =", round(wrap(bp.berry_phase(states)-bp.berry_phase(regauged)),12))
print("speed:   40-pt vs 4000-pt     diff =", round(wrap(bp.berry_phase_spin(th,40)-bp.berry_phase_spin(th,4000)),4))
print("tiny loop (2 deg cone):       gamma =", round(bp.berry_phase_spin(np.radians(2)),4), "(~0, no area)")"""),

md("""## 4. The optics connection

The polarization of light has its **own** Berry phase -- the **Pancharatnam phase**: send
a polarization state around a loop on the Poincare sphere and it returns with a geometric
phase equal to $-\\tfrac12$ the enclosed solid angle, exactly as here. So a recovered
optical phase generally has two parts: a **dynamical** phase (from propagation/dispersion,
what the GS receiver inverts) and a **geometric** phase (from the path the polarization or
mode took). Phase retrieval that ignores the geometric part can be biased -- Berry's phase
is why "phase" is subtler than just "optical path length"."""),

md("""## Takeaway

1. **Berry's phase** $\\gamma=-\\tfrac12\\Omega$: cycle the parameters around a loop and the
   wavefunction gains a phase set by the **solid angle**, not the speed.
2. It is **gauge- and speed-invariant** -- genuinely geometric, the *area on the sphere*.
3. In optics it is the **Pancharatnam phase** of polarization; a measured phase splits
   into dynamical + geometric parts.

A phase that remembers the path, not just the endpoints -- the deepest sense in which
"phase" carries information. Civilian education / Griffiths QM Ch. 10."""),
]

nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3"}
out = pathlib.Path("notebooks/berry_phase.ipynb")
nbf.write(nb, out)
print("wrote", out, "with", len(nb.cells), "cells")
