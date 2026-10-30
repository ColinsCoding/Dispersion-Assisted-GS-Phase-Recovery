"""Build notebooks/dispersion_phase_retrieval_lean.ipynb: Lean proofs + NumPy/SymPy + MATLAB check.

The Lean cells are cut from lean/DispersionPhaseRetrieval.lean and compiled once by %lean_check.

Run:  py -3.12 tools/build_dapr_lean_nb.py
Then: py -3.12 -m jupyter nbconvert --to notebook --execute --inplace notebooks/dispersion_phase_retrieval_lean.ipynb
"""
import re
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "dispersion_phase_retrieval_lean.ipynb"
LEAN_SRC = ROOT / "lean" / "DispersionPhaseRetrieval.lean"
cells = []


def md(s):
    cells.append(nbf.v4.new_markdown_cell(s.strip("\n")))


def code(s):
    cells.append(nbf.v4.new_code_cell(s.strip("\n")))


text = LEAN_SRC.read_text(encoding="utf-8")
body, footer = text.rsplit("end DAPR", 1)
parts = re.split(r"(?m)^(?=/-! ## \d+\.)", body)
HEADER, LEAN, FOOTER = parts[0], {int(re.match(r"/-! ## (\d+)\.", p).group(1)): p for p in parts[1:]}, "end DAPR" + footer


def lean(name, src):
    code(f"%%lean {name}\n{src.strip()}")


md(r"""
# Dispersion-assisted phase retrieval: what Lean proves, and what it doesn't

The forward model is H_D(ν) = exp(iπDν²), Jalali's time stretch. Measure two intensities
|H_{D1}E|² and |H_{D2}E|², then alternate projections between them (Gerchberg-Saxton).
Each step below has a **Lean theorem** (machine-checked) and a **numerical check** on the real
`dgs/gs_core.py` code.

| # | Claim | Lean | Numerics |
|---|---|---|---|
| 1 | phase'' = 2πD (D is the group-delay dispersion) | `phase_second_deriv` | SymPy |
| 2 | \|H\| = 1, so energy is conserved | `H_norm`, `energy_conserved` | Parseval on the FFT |
| 3 | H_{D1}H_{D2} = H_{D1+D2}, and undisperse inverts disperse | `H_add`, `H_inverse` | round trip ≈ 1e-16 |
| 4 | unitary steps preserve distance | `dist_preserved` | |
| 5 | the amplitude projection is the **nearest** point | `amplitude_projection_nearest` | random trials |
| 6 | the constraint set is **not convex** | `magnitude_set_not_convex` | midpoint of 1 and −1 |
| 7 | alternating nearest-point projections never increase the gap | `error_nonincreasing` | plain GS, measured |

**Not proved:** that GS reaches the *true* phase (uniqueness), anything about noise, or the
extra unit-amplitude step. §7 shows numerically that this extra step breaks the guarantee.
""")
code(r"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()))
import numpy as np, sympy as sp, pandas as pd, matplotlib.pyplot as plt
from IPython.display import display
sp.init_printing(use_unicode=True)
from dgs import gs_core as G
N, D1, D2 = 256, -5000.0, -5750.0
k = np.arange(N)
E_true = np.exp(1j * np.pi / 2 * np.mod(k // 8 * 7 + 3, 4))       # QPSK-like test field
I1, I2 = np.abs(G.disperse(E_true, D1))**2, np.abs(G.disperse(E_true, D2))**2
""")
code("%load_ext dgs.lean_magic\n%lean_reset")
lean("header", HEADER)

md(r"""
## 1. Second derivative of the phase

φ(ν) = πDν², so φ'' = 2πD. In fiber optics that constant is the group-delay dispersion.
Dispersion works by giving each frequency a different delay, τ(ν) = φ'(ν)/2π = Dν.
**Diffraction is the same math in space:** Fresnel propagation multiplies the spatial
spectrum by exp(−iπλz f_x²), the same quadratic phase. This is space-time duality, and it's why
"dispersion-assisted" phase retrieval mirrors the classic defocus-diversity method in imaging.
""")
code(r"""
nu, D, lam, z, fx = sp.symbols("nu D lambda z f_x", real=True)
phi = sp.pi * D * nu**2
display(sp.Eq(sp.Symbol("phi''"), sp.diff(phi, nu, 2)))
display(sp.Eq(sp.Symbol("H_fresnel"), sp.exp(-sp.I * sp.pi * lam * z * fx**2)))
print("time stretch <-> diffraction:  D  <->  -lambda z")
""")
lean("phase", LEAN[1])

md("## 2–4. Unit modulus, composition, inversion, distances")
code(r"""
Ed = G.disperse(E_true, D1)
print("Parseval: energy after / before =", np.sum(abs(Ed)**2) / np.sum(abs(E_true)**2))
print("undisperse(disperse(E)) - E, max =", np.max(abs(G.undisperse(Ed, D1) - E_true)))
print("disperse(disperse(E, D1), D2) vs disperse(E, D1 + D2), max diff =",
      np.max(abs(G.disperse(G.disperse(E_true, D1), D2) - G.disperse(E_true, D1 + D2))))
y = np.exp(1j * np.random.default_rng(0).uniform(0, 2 * np.pi, N))
print("|Hx - Hy| / |x - y| =", np.linalg.norm(G.disperse(E_true, D1) - G.disperse(y, D1)) / np.linalg.norm(E_true - y))
""")
lean("unit", LEAN[2]); lean("group", LEAN[3]); lean("distance", LEAN[4])

md(r"""
## 5. The amplitude projection is the nearest point

`apply_amplitude_constraint` replaces |z| with the measured a and keeps the phase. Lean proves
no other point with magnitude a is closer to z. The proof is the reverse triangle inequality:
|z − w| ≥ ||z| − |w|| = ||z| − a|, which is exactly the distance to a·z/|z|.
""")
code(r"""
rng = np.random.default_rng(1)
worst = np.inf
for _ in range(20000):
    zc = complex(*rng.normal(size=2)); a = rng.uniform(0, 3)
    proj = a * zc / abs(zc)
    w = a * np.exp(1j * rng.uniform(0, 2 * np.pi))
    worst = min(worst, abs(zc - w) - abs(zc - proj))
print("min over 20000 random trials of |z - w| - |z - P(z)| =", worst, "(never negative)")
""")
lean("projection", LEAN[5])

md(r"""
## 6. Midpoint: the magnitude set is not convex

The midpoint of 1 and −1 is 0, which isn't on the unit circle. Projections onto **convex** sets
always converge to a common point when one exists. The magnitude sets aren't convex, so GS has
no such guarantee: it can stall in a wrong answer. That's why the dispersion pair needs enough
diversity (|D| ≥ 5000 in this repo's experiments).
""")
lean("nonconvex", LEAN[6])

md(r"""
## 7. Error reduction: what the theorem guarantees, measured

Let x_k be on set B (consistent with I₂) and a_k = P_A(x_k) the nearest point on set A
(consistent with I₁). Because x_{k+1} = P_B(a_k) is a nearest point, the gap
‖x_k − a_k‖ can never grow. **Plain GS** (two projections) obeys this exactly.
`gs_iteration(unit_amplitude=True)` adds a third projection onto |E| = 1, which the theorem
doesn't cover. Its intensity error can go up, and here it does.
""")
code(r"""
PA = lambda E: G.undisperse(np.sqrt(I1) * np.exp(1j * np.angle(G.disperse(E, D1))), D1)
PB = lambda E: G.undisperse(np.sqrt(I2) * np.exp(1j * np.angle(G.disperse(E, D2))), D2)
x = PB(G.undisperse(np.sqrt(I1).astype(complex), D1))
gaps = []
for _ in range(60):
    a = PA(x); gaps.append(np.linalg.norm(x - a)); x = PB(a)
gaps = np.array(gaps)
print("plain GS: largest increase of the gap =", np.max(np.diff(gaps)), "(<= 0 up to rounding)")
E = G.undisperse(np.sqrt(I1).astype(complex), D1); ua = []
for _ in range(60):
    E = G.gs_iteration(E, I1, I2, D1, D2, unit_amplitude=True)
    ua.append(np.sqrt(np.mean((np.abs(G.disperse(E, D2))**2 - I2)**2)))
fig, ax = plt.subplots(1, 2, figsize=(11, 3.3))
ax[0].semilogy(gaps); ax[0].set(title="plain GS: gap ||x_k - P_A x_k|| (Lean: non-increasing)", xlabel="iteration")
ax[1].plot(ua, "C3"); ax[1].set(title="unit_amplitude=True: intensity error (no guarantee)", xlabel="iteration")
plt.tight_layout(); plt.show()
""")
lean("error_reduction", LEAN[7])

md(r"""
## 8. The same pipeline in MATLAB

`matlab/mp_toolkit.m` reimplements the box energies, the finite-difference Schrödinger solver,
tunneling, dispersion, and this GS loop. `tests/test_matlab_toolkit.py` checks that MATLAB R2025b
agrees with Python (box and tunneling to 1e-6, the GS error history to 1e-6).
""")
code(r"""
import glob, json, shutil, subprocess, tempfile
exe = shutil.which("matlab") or (sorted(glob.glob(r"C:\Program Files\MATLAB\R20*\bin\matlab.exe")) or [None])[-1]
root = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
if exe:
    out = Path(tempfile.gettempdir()) / "mp_toolkit_nb.json"
    subprocess.run([exe, "-batch", f"addpath('{(root / 'matlab').as_posix()}'); mp_toolkit('{out.as_posix()}')"],
                   capture_output=True, text=True, timeout=600)
    m = json.loads(out.read_text())
    E = G.undisperse(np.sqrt(I1).astype(complex), D1); py = []
    for _ in range(30):
        E = G.gs_iteration(E, I1, I2, D1, D2, unit_amplitude=True)
        py.append(np.sqrt(np.mean((np.abs(G.disperse(E, D2))**2 - I2)**2)))
    print("box energies (MATLAB):", np.round(m["box_E"], 4))
    print("GS error history, max |MATLAB - Python| =", np.max(np.abs(np.array(m["gs_errors"]) - py)))
else:
    print("MATLAB not installed: skipped")
""")

md("## 9. Compile every Lean cell")
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
