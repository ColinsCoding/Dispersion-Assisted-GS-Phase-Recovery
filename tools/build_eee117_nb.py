"""Build notebooks/eee117_circuits_sympy_lean.ipynb: SymPy (init_printing) + Lean for EEE 117 unit 1.

The Lean cells are cut from lean/EEE117Circuits.lean and compiled once by %lean_check.

Run:  py -3.12 tools/build_eee117_nb.py
Then: py -3.12 -m jupyter nbconvert --to notebook --execute --inplace notebooks/eee117_circuits_sympy_lean.ipynb
"""
import re
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "eee117_circuits_sympy_lean.ipynb"
LEAN_SRC = ROOT / "lean" / "EEE117Circuits.lean"
cells = []


def md(s):
    cells.append(nbf.v4.new_markdown_cell(s.strip("\n")))


def code(s):
    cells.append(nbf.v4.new_code_cell(s.strip("\n")))


text = LEAN_SRC.read_text(encoding="utf-8")
body, footer = text.rsplit("end EEE117", 1)
parts = re.split(r"(?m)^(?=/-! ## \d+\.)", body)
HEADER, LEAN, FOOTER = parts[0], {int(re.match(r"/-! ## (\d+)\.", p).group(1)): p for p in parts[1:]}, "end EEE117" + footer


def lean(name, src):
    code(f"%%lean {name}\n{src.strip()}")


md(r"""
# EEE 117 unit 1, solved two ways: SymPy and Lean

| Topic | SymPy (`dgs.eee117_circuits`) | Lean (`lean/EEE117Circuits.lean`) |
|---|---|---|
| Voltage, current, power, energy; Ohm's law | `ohm`, `power_forms` | P = I²R = V²/R |
| Sources, polarity, absorbed power | `Circuit.element_table`, Tellegen | series power balance |
| Series and parallel | `series`, `parallel` | R₁R₂/(R₁+R₂) < R₁ |
| Voltage and current dividers | `voltage_divider`, `current_divider` | the parts add back to the whole |
| Measuring resistance: Wheatstone bridge | `wheatstone_output`, `wheatstone_unknown` | balance ⇒ zero output |
| Delta-to-wye | `delta_to_wye`, `wye_to_delta` | terminal equivalence |
| Node-voltage method | `Circuit` (KCL) | |
| Mesh-current method | `Mesh` (KVL) | |
| Operational amplifiers | `inverting_amp`, `noninverting_amp`, `summing_amp` | the two gain formulas |
| Binary | `r2r_dac` (a 3-bit DAC) | ladder output = binary weights |

**SymPy** gives exact algebra. **Lean** is machine-checked and counts only if `%lean_check`
(the last section) reports VERIFIED. Work each problem by hand first, then compare.
""")
code(r"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()))
import pandas as pd, sympy as sp
from IPython.display import display, Markdown
sp.init_printing(use_unicode=True)
from dgs import eee117_circuits as E
from dgs.field_circuits import Circuit
R1, R2, R3, R4, Rf, Rin, Vs, Is = sp.symbols("R_1 R_2 R_3 R_4 R_f R_in V_s I_s", positive=True)
""")
code("%load_ext dgs.lean_magic\n%lean_reset")
lean("header", HEADER)

md(r"""
## 1. Voltage, current, power, and energy; Ohm's law

V = IR. Power is the rate of energy, p = vi in watts, and energy is w = ∫p dt in joules.
With the passive sign convention (current enters the + terminal), p > 0 means *absorbed*.
""")
code(r"""
display(E.ohm(V=12, R=4))
V, I, R = sp.symbols("V I R", positive=True)
display(E.power_forms(I * R, I, R))
""")
lean("ohm", LEAN[1])

md("## 2. Series and parallel")
code(r"""
display(sp.Eq(sp.Symbol("R_series"), E.series(R1, R2, R3)), sp.Eq(sp.Symbol("R_parallel"), E.parallel(R1, R2)))
print("6 ohm || 3 ohm =", E.parallel(6, 3), "ohm;  1 + 2 + 3 =", E.series(1, 2, 3), "ohm")
""")
lean("series_parallel", LEAN[2])

md("## 3. Voltage divider and current divider")
code(r"""
display(sp.Eq(sp.Symbol("v_2"), E.voltage_divider(Vs, R2, R1)), sp.Eq(sp.Symbol("i_1"), E.current_divider(Is, R1, R2)))
print("10 V across 3 ohm + 2 ohm: the 3 ohm sees", E.voltage_divider(10, 3, 2), "V")
print("9 A into 3 ohm || 6 ohm: the 3 ohm takes", E.current_divider(9, 3, 6), "A (the SMALLER resistor takes MORE current)")
""")
lean("dividers", LEAN[3])

md(r"""
## 4. Measuring resistance: the Wheatstone bridge

Adjust the known resistors until the meter between the two midpoints reads zero. Then
R₁R₄ = R₂R₃, so the unknown is R₄ = R₂R₃/R₁. At balance no current flows through the meter, so
its own resistance doesn't matter.
""")
code(r"""
display(sp.Eq(sp.Symbol("V_ab"), E.wheatstone_output(Vs, R1, R2, R3, R4)))
print("R1 = 100, R2 = 200, R3 = 300 at balance -> unknown R4 =", E.wheatstone_unknown(100, 200, 300), "ohm")
""")
lean("wheatstone", LEAN[4])

md("## 5. Delta-to-wye")
code(r"""
y = E.delta_to_wye(R1, R2, R3)
display(dict(zip(["R_a", "R_b", "R_c"], y)))
print("round trip back to delta:", [sp.simplify(v) for v in E.wye_to_delta(*y)])
print("equal delta 30, 30, 30 -> wye", E.delta_to_wye(30, 30, 30), "(R_Y = R_delta / 3)")
""")
lean("delta_wye", LEAN[5])

md(r"""
## 6. Node-voltage and mesh-current methods on the same circuit

A 10 V source and R₁ = 2 Ω feed a node b, which has R₂ = 4 Ω and R₃ = 4 Ω to ground.
- **Node-voltage:** one unknown (V_b), from KCL.
- **Mesh-current:** two unknowns (I_A, I_B), from KVL.

Same answer either way. Choose the method with fewer unknowns.
""")
code(r"""
ckt = Circuit().V("Vs", "a", "0", 10).R("R1", "a", "b", 2).R("R2", "b", "0", 4).R("R3", "b", "0", 4)
print("NODE-VOLTAGE\n" + ckt.paper())
m = E.Mesh().V("Vs", 10, "A").R("R1", 2, "A").R("R2", 4, "A", "B").R("R3", 4, "B")
print("\nMESH-CURRENT\n" + m.paper())
A, b = m.matrix(); display(sp.Eq(sp.MatMul(A, sp.Matrix([sp.Symbol("I_A"), sp.Symbol("I_B")])), b))
print("check: V_b from mesh = I_R2 * 4 =", m.branch_current("R2") * 4, "  from node-voltage =", ckt.solve()[sp.Symbol("V_b")])
display(pd.DataFrame(ckt.element_table()))
print("Tellegen (sum of absorbed power):", ckt.tellegen())
""")

md(r"""
## 7. Operational amplifiers

The ideal op-amp rules are: no current into the inputs, and V₊ = V₋ when there's negative feedback.
The node-voltage solver takes the op-amp's output current as one more unknown.
""")
code(r"""
for name, fn in [("inverting", E.inverting_amp), ("non-inverting", E.noninverting_amp)]:
    ckt_oa, vout = fn(Rin, Rf)
    display(Markdown(f"**{name}**")); display(sp.Eq(sp.Symbol("V_out"), sp.factor(vout)))
    print("  Tellegen with the op-amp's supply included:", ckt_oa.tellegen())
_, vsum = E.summing_amp([1000, 2000], 4000, [1, 2])
print("summing amp, Rf = 4k, inputs 1 V via 1k and 2 V via 2k: V_out =", vsum, "V")
""")
lean("opamps", LEAN[6])
lean("energy", LEAN[7])

md(r"""
## 8. Binary: a 3-bit R-2R ladder DAC

A computer-engineering bridge: switch each bit between 0 and V_ref, and the ladder outputs
V_ref × code / 2ⁿ. Here node-voltage analysis solves the full ladder for all 8 codes.
""")
code(r"""
from itertools import product
rows = [("".join(map(str, b)), int("".join(map(str, b)), 2), E.r2r_dac(b, Vref=8)[1]) for b in product((0, 1), repeat=3)]
display(pd.DataFrame(rows, columns=["bits", "code", "V_out (Vref = 8 V)"]))
""")
lean("r2r", LEAN[8])

md("## 9. Compile every Lean cell at once")
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
