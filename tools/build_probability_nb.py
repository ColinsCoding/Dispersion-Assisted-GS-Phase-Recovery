"""Build notebooks/probability_computer_engineering.ipynb: probability derived from hardware physics.

The Lean cells are cut from lean/ProbabilityCE.lean and compiled once by %lean_check.

Run:  py -3.12 tools/build_probability_nb.py
Then: py -3.12 -m jupyter nbconvert --to notebook --execute --inplace notebooks/probability_computer_engineering.ipynb
"""
import re
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "probability_computer_engineering.ipynb"
LEAN_SRC = ROOT / "lean" / "ProbabilityCE.lean"
cells = []


def md(s):
    cells.append(nbf.v4.new_markdown_cell(s.strip("\n")))


def code(s):
    cells.append(nbf.v4.new_code_cell(s.strip("\n")))


text = LEAN_SRC.read_text(encoding="utf-8")
body, footer = text.rsplit("end ProbCE", 1)
parts = re.split(r"(?m)^(?=/-! ## \d+\.)", body)
HEADER, LEAN, FOOTER = parts[0], {int(re.match(r"/-! ## (\d+)\.", p).group(1)): p for p in parts[1:]}, "end ProbCE" + footer


def lean(name, src):
    code(f"%%lean {name}\n{src.strip()}")


md(r"""
# Probability for computer engineering, derived from the physics

Every random variable here comes from hardware: photons arriving at a receiver, thermal noise in
a resistor, bits flipping in a packet, electrons filling states in a semiconductor. Each one is
**derived with calculus** (SymPy), **simulated** (NumPy Monte Carlo), and **proved** in Lean where
the statement is general.

| # | Physics | Probability | Lean |
|---|---|---|---|
| 1 | photons / bit errors | Binomial → Poisson, mean = variance | `binomial_sum_one` |
| 2 | bits in a packet | P(packet error) ≤ n p | `packet_union_bound` |
| 3 | shot noise, `np.var` | Var = E[X²] − E[X]² | `variance_identity` |
| 4 | worst-case guarantees | Markov's inequality | `markov_finite` |
| 5 | packet arrivals, decay | exponential is memoryless | `memoryless` |
| 6 | Johnson noise → BER | Gaussian tail integral | (SymPy) |
| 7 | semiconductors | Fermi-Dirac symmetry | `fermi_symmetry` |
""")
code(r"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()))
import numpy as np, sympy as sp, pandas as pd, matplotlib.pyplot as plt
from IPython.display import display
sp.init_printing(use_unicode=True)
from dgs import probability_ce as P
""")
code("%load_ext dgs.lean_magic\n%lean_reset")
lean("header", HEADER)

md(r"""
## 1. Counting photons: Binomial → Poisson

Split one bit slot into n tiny time pieces; each holds a photon with probability λ/n. The count
is Binomial(n, λ/n). As n → ∞, C(n,k)/nᵏ → 1/k! and (1 − λ/n)ⁿ → e^{−λ}, giving Poisson(λ).
Summing the series gives mean = variance = λ, so the signal-to-noise ratio of a photon count is
λ/√λ = √λ (**shot noise**). Bit errors in a long stream follow the same limit.
""")
code(r"""
pmf, limit = P.binomial_to_poisson()
display(pmf, sp.Eq(sp.Symbol("limit"), limit))
mean, var = P.poisson_moments()
print("Poisson mean =", mean, "  variance =", var)
N = P.photons_per_bit(1e-6, 10e9, 1550e-9)
print(f"1 uW at 10 Gb/s, 1550 nm: {N:.0f} photons per bit, shot-noise SNR = {P.shot_noise_snr(N):.1f}")
""")
lean("binomial", LEAN[1])

md(r"""
## 2. A packet of n bits

P(no error) = (1 − p)ⁿ, so P(packet error) = 1 − (1 − p)ⁿ. Bernoulli's inequality bounds it by np:
the **union bound** engineers use for quick link budgets. Lean proves it for every n and p ≤ 1.
""")
code(r"""
for p in (1e-6, 1e-4, 1e-3):
    exact, bound = P.packet_error(p, 1500)
    print(f"p = {p:g}: exact {exact:.4g}, bound n p = {bound:.4g}")
""")
lean("union", LEAN[2])

md(r"""
## 3–4. Variance and Markov's inequality

Var = Σ w(x − μ)² = Σ wx² − μ² is what `np.var` computes. Markov's inequality, P(X ≥ a) ≤ E[X]/a,
needs only X ≥ 0, so it is a worst-case guarantee with no distribution assumed (Chebyshev is
Markov applied to (X − μ)²).
""")
lean("variance", LEAN[3]); lean("markov", LEAN[4])

md(r"""
## 5. Exponential arrival times are memoryless

Poisson arrivals (photons, packets, radioactive decays) have exponential waiting times with
P(T > t) = e^{−λt}. Having already waited s tells you nothing: P(T > s + t | T > s) = P(T > t).
This is why queueing models of routers and caches are tractable.
""")
code(r"""
print("P(T > s+t | T > s) - P(T > t) =", P.memoryless())
print("expected transmissions with failure prob q:", P.expected_transmissions())
""")
lean("memoryless", LEAN[5])

md(r"""
## 6. Thermal noise → bit error rate

A resistor at temperature T produces Gaussian noise with σ = √(4kTRB). Send +A or −A and decide
by sign. An error on +A happens when the noise is below −A, so BER = ∫_{−∞}^{−A} (Gaussian pdf) dx.
""")
code(r"""
integral, closed = P.ber_derivation()
display(sp.Eq(integral, closed))
print(f"Johnson noise, 1 kOhm, 300 K, 1 MHz: sigma = {P.johnson_sigma(300, 1e3, 1e6) * 1e6:.2f} uV")
snr = np.linspace(0.5, 6, 100)
plt.figure(figsize=(6, 3.4)); plt.semilogy(snr, [P.ber(a, 1) for a in snr])
plt.xlabel("A / sigma"); plt.ylabel("BER"); plt.title("Bit error rate from Gaussian thermal noise"); plt.grid(alpha=.3)
plt.tight_layout(); plt.show()
""")

md(r"""
## 7. Fermi-Dirac occupancy in a semiconductor

f(E) = 1/(1 + e^{(E−μ)/kT}). A state x above μ is full exactly as often as a state x below μ is
empty: f(μ + x) + f(μ − x) = 1. This symmetry is why electron and hole populations mirror each other
around the Fermi level.
""")
code(r"""
print("f(x) + f(-x) - 1 =", P.fermi_symmetry())
E = np.linspace(0, 1, 300)
for T in (77, 300, 600):
    plt.plot(E, P.fermi(E, 0.5, T), label=f"{T} K")
plt.xlabel("E (eV)"); plt.ylabel("occupancy"); plt.legend(); plt.title("Fermi-Dirac, mu = 0.5 eV")
plt.tight_layout(); plt.show()
""")
lean("fermi", LEAN[6])

md("## 8. Monte Carlo: every formula against a simulation")
code(r"""
rows = P.monte_carlo()
display(pd.DataFrame(rows, index=["formula", "simulated"]).T)
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
