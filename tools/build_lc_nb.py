"""Build notebooks/lc_quantum_oscillator.ipynb: SymPy, numerics, torch residuals, and Lean cells.

The Lean cells are cut from lean/LCOscillator.lean (the single source of truth) and run through
the %%lean magic in dgs/lean_magic.py; %lean_check compiles them all in one go.

Run:  py -3.12 tools/build_lc_nb.py
Then: py -3.12 -m jupyter nbconvert --to notebook --execute --inplace notebooks/lc_quantum_oscillator.ipynb
"""
import re
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "lc_quantum_oscillator.ipynb"
LEAN_SRC = ROOT / "lean" / "LCOscillator.lean"
cells = []


def md(s):
    cells.append(nbf.v4.new_markdown_cell(s.strip("\n")))


def code(s):
    cells.append(nbf.v4.new_code_cell(s.strip("\n")))


def lean_sections():
    text = LEAN_SRC.read_text(encoding="utf-8")
    body, footer = text.rsplit("end LCOsc", 1)
    parts = re.split(r"(?m)^(?=/-! ## \d+\.)", body)
    secs = {int(re.match(r"/-! ## (\d+)\.", p).group(1)): p for p in parts[1:]}
    return parts[0], secs, "end LCOsc" + footer


HEADER, LEAN, FOOTER = lean_sections()


def lean(name, src):
    code(f"%%lean {name}\n{src.strip()}")


md(r"""
# Is the quantum harmonic oscillator an LC circuit? Yes.

| mass-spring | LC circuit | quantum |
|---|---|---|
| position x | charge Q | operator Q |
| momentum p = m ẋ | flux Φ = L I | [Q, Φ] = iħ |
| mass m | inductance L | |
| spring k | 1/C | |
| ω = √(k/m) | ω = 1/√(LC) | Eₙ = ħω(n + ½) |

Cool a superconducting LC circuit to millikelvin and it *is* a quantum harmonic oscillator. Add a
Josephson junction, a nonlinear inductor, and the levels stop being evenly spaced: that is a
transmon qubit. This notebook covers:

1. The mapping and **symmetry**: no explicit time in the Lagrangian → energy is conserved (Noether)
2. The solution ω = 1/√(LC)
3. Simulation: a symplectic integrator keeps that symmetry; resistance breaks it
4. Quantum levels, zero-point fluctuations, and why it takes a dilution fridge
5. Terahertz: where P = VI and lumped circuits stop working, and Jalali's time stretch
6. Residuals in torch on the CPU and the CUDA GPU (a one-parameter PINN)
7. Compiling every Lean cell

| Label | Meaning |
|---|---|
| **SymPy** | exact computer algebra |
| **Lean** | machine-checked; counts only if `%lean_check` (§7) reports VERIFIED |
| **Numerical** | agreement to a stated tolerance: evidence, not proof |
""")

code(r"""
import sys, time, math
from pathlib import Path
sys.path.insert(0, str(Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()))
import numpy as np, pandas as pd, sympy as sp
import matplotlib.pyplot as plt
from IPython.display import display, Markdown
sp.init_printing(use_unicode=True)
from dgs import lc_oscillator as O
L0, C0 = 10e-9, 100e-15          # 10 nH, 100 fF: a 5 GHz qubit-style resonator
""")
code("%load_ext dgs.lean_magic\n%lean_reset")
lean("header", HEADER)

md(r"""
## 1. The mapping and the symmetry

The LC Lagrangian is ℒ = ½ L Q̇² − Q²/(2C): "kinetic" magnetic energy minus "potential" electric
energy. It has no explicit `t`, so it is the same today and tomorrow. Noether's theorem turns that
time-translation symmetry into a conserved quantity: the energy E = ½ L Q̇² + Q²/(2C).
**(SymPy)**
""")
code(r"""
display(pd.DataFrame(O.MAPPING, columns=["quantity", "mass-spring", "LC circuit", "SI unit"]))
t = sp.symbols("t", real=True)
L, C, w = sp.symbols("L C omega", positive=True)
Q = sp.Function("Q")(t)
Lag = L * Q.diff(t)**2 / 2 - Q**2 / (2 * C)
EL = sp.euler_equations(Lag, Q, t)[0]
display(EL)                                                   # L Q'' + Q/C = 0 (up to sign)
E = Q.diff(t) * sp.diff(Lag, Q.diff(t)) - Lag                 # Noether energy function
display(sp.Eq(sp.Symbol("E"), sp.simplify(E)))
dEdt = sp.diff(E, t).subs(Q.diff(t, 2), sp.solve(EL.lhs, Q.diff(t, 2))[0])
print("dE/dt on solutions:", sp.simplify(dEdt))
""")
lean("mapping", LEAN[1])
lean("energy", LEAN[2])

md("## 2. The solution: ω = 1/√(LC)  **(SymPy, Lean)**")
code(r"""
sol = sp.dsolve(sp.Eq(L * Q.diff(t, 2) + Q / C, 0), Q)
display(sol)
p = O.lc_params(L0, C0)
print(f"L = 10 nH, C = 100 fF:  f0 = {p['f0']/1e9:.3f} GHz,  Z = {p['Z']:.1f} ohm")
""")
lean("solution", LEAN[3])

md(r"""
## 3. Simulation: keeping the symmetry numerically  **(Numerical)**

Leapfrog is *symplectic*: it preserves the structure of Hamilton's equations, so the energy
error stays bounded forever instead of drifting. A resistor breaks time-reversal symmetry, and
the energy decays as e^(−Rt/L).
""")
code(r"""
fig, ax = plt.subplots(1, 2, figsize=(11, 3.4))
for R, lab in [(0.0, "R = 0"), (5.0, "R = 5 ohm")]:
    s = O.simulate_lc(L0, C0, Q0=1e-15, periods=100, R=R)
    ax[0].plot(s["t"] * 1e9, s["Q"] / 1e-15, lw=0.6, label=lab)
    ax[1].plot(s["t"] * 1e9, s["E"] / s["E"][0], label=lab)
ax[0].set(xlabel="t (ns)", ylabel="Q (fC)", title="charge rings at 5 GHz"); ax[0].set_xlim(0, 2)
ax[1].set(xlabel="t (ns)", ylabel="E / E0", title="energy: conserved vs damped")
for a in ax: a.legend()
plt.tight_layout(); plt.show()
s = O.simulate_lc(L0, C0, Q0=1e-15, periods=1000)
print("R = 0, 1000 periods: max |E/E0 - 1| =", f"{np.max(np.abs(s['E']/s['E'][0]-1)):.1e}")
""")

md(r"""
## 4. Quantum: levels, zero-point, and the fridge

Quantize with [Q, Φ] = iħ and you get Eₙ = ħω(n + ½). Even the ground state has energy ħω/2,
and charge and flux fluctuate with ΔQ·ΔΦ = ħ/2 exactly. At 5 GHz, one quantum is
ħω/k_B ≈ 0.24 K, so you need ~10 mK (a dilution refrigerator) to keep thermal photons out.
**(SymPy, Lean, Numerical)**
""")
code(r"""
n = sp.symbols("n", integer=True, nonnegative=True); hb = sp.symbols("hbar", positive=True)
En = hb * w * (n + sp.Rational(1, 2))
print("E_(n+1) - E_n =", sp.simplify(En.subs(n, n + 1) - En))
q = O.quantum_levels(L0, C0, 4)
display(pd.DataFrame({"n": q["n"], "E (ueV)": q["E_ueV"].round(3), "E / k_B (K)": q["E_K"].round(4)}))
z = O.zero_point(L0, C0)
print(f"zero-point: dQ = {z['dQ_electrons']:.2f} electrons, dV = {z['dV_V']*1e6:.2f} uV, dQ dPhi / hbar = {z['product_over_hbar']:.3f}")
T = np.logspace(-3, 0.5, 200)
plt.figure(figsize=(6, 3.2)); plt.loglog(T * 1e3, [O.thermal_occupation(L0, C0, x) for x in T])
plt.axvline(10, ls="--", c="gray"); plt.text(11, 1e-6, "10 mK fridge")
plt.xlabel("T (mK)"); plt.ylabel("mean thermal photons"); plt.title("5 GHz resonator"); plt.tight_layout(); plt.show()
""")
lean("ladder", LEAN[4])
lean("zero_point", LEAN[5])

md(r"""
## 5. Terahertz: where P = VI stops being enough

P = VI assumes one voltage per node, which needs the circuit to be small compared with the
wavelength (rule of thumb: under λ/10). At 1 THz, λ = 0.3 mm in air. A 1 mm part is then
longer than a wavelength: it's a transmission line or waveguide, and power is the Poynting flux
E × H, not a single V times I. That's why THz electronics is hard, and why Jalali's **photonic time
stretch** helps: slow the signal down optically by a factor M, then digitize it with an ordinary ADC.
**(Numerical, Lean)**
""")
code(r"""
rows = []
for f in [1e6, 1e9, 10e9, 100e9, 1e12, 10e12]:
    r = O.lumped_ok(f, 1e-3)
    rows.append((f"{f:.0e} Hz", f"{r['lambda_m']*1e3:.3g} mm", f"{r['size_over_lambda']:.3g}", "lumped" if r["lumped_ok"] else "distributed (waves)"))
display(pd.DataFrame(rows, columns=["frequency", "wavelength", "1 mm / lambda", "treat a 1 mm part as"]))
ts = O.time_stretch(20e9, 50, 1e-9)
print(f"time stretch M = 50 with a 20 GHz digitizer: effective {ts['effective_bandwidth_Hz']/1e12:.1f} THz, "
      f"captured window {ts['captured_window_s']*1e12:.0f} ps per frame")
print("P = VI averaged over a cycle with V, I 90 degrees apart (pure LC):", round(O.average_power(1.0, 1.0, math.pi/2), 12), "W")
""")
code(r"""
display(pd.DataFrame([(n_, f"${lo:,.0f}", f"${hi:,.0f}") for n_, lo, hi in O.COSTS],
                     columns=["how to look at fast signals", "low", "high"]))
print("Rough 2026 order-of-magnitude ranges, not quotes.")
""")
lean("thz", LEAN[6])

md(r"""
## 6. Residuals in torch on the CPU and CUDA  **(Numerical)**

A *residual* is how badly a candidate solution misses the equation: r(t) = L Q'' + Q/C, with
Q'' from autograd. The true solution gives r ≈ 0, and a wrong frequency doesn't. Minimizing the
mean r² over ω is the smallest possible physics-informed neural network (PINN), and it finds
ω = 1/√(LC) from the equation alone, with no data.
""")
code(r"""
import torch
devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
w0 = O.lc_params(L0, C0)["omega0"]
tg = np.linspace(0, 8 * math.pi / w0, 2000)
for d in devices:
    dev = O.torch_device(d)
    good = O.lc_residual(lambda tt: torch.cos(w0 * tt), L0, C0, tg, dev).abs().max().item()
    bad = O.lc_residual(lambda tt: torch.cos(1.1 * w0 * tt), L0, C0, tg, dev).abs().max().item()
    t0 = time.perf_counter(); fit = O.fit_omega(L0, C0, device=dev); dt = time.perf_counter() - t0
    print(f"{dev}: max|r| true = {good:.1e}, 10% off = {bad:.3f};  fit omega/omega0 = {fit['omega_fit']/w0:.6f} in {dt:.2f} s")
plt.figure(figsize=(6, 3.2)); plt.semilogy(fit["loss"]); plt.xlabel("Adam step"); plt.ylabel("mean r^2")
plt.title("learning omega from the residual"); plt.tight_layout(); plt.show()
""")

md(r"""
## 7. Compile every Lean cell at once

`%lean_check` joins the cells in order and runs `lake env lean` once.
""")
lean("footer", FOOTER)
code(r"""
lean_result = %lean_check
from dgs.lean_magic import SESSION
display(pd.DataFrame({"Lean theorem": SESSION.last.theorems, "status": SESSION.last.status}))
""")

nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"] = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                  "language_info": {"name": "python"}}
nbf.write(nb, OUT)
print("wrote", OUT, "with", len(cells), "cells")
