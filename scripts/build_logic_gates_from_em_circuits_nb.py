"""Build notebooks/logic_gates_from_em_circuits.ipynb

The 7 basic logic gates (AND, OR, NOT, NAND, NOR, XOR, XNOR), each built
from real diode I-V physics and a transistor switch -- the historical
Diode-Transistor Logic (DTL) family -- extending dgs.computer_engineering's
existing diode_iv/diode_or_gate. Includes an honest account of a real
zero-noise-margin problem (and an outright cascade misfire in an early
un-restored prototype) this module's own development caught, and the
genuine historical fix (series level-shifting diodes).

Research-partner notebook template: diode I-V recap -> AND (the OR gate's
dual) -> NOT (the transistor switch) -> DTL NAND/NOR -> the zero-margin
problem, honestly -> XOR/XNOR cascade -> full verification -> engineering
interpretation -> research discussion -> possible experiments -> future
improvements.

Engine: dgs/logic_gates_from_em_circuits.py (built on
dgs/computer_engineering.py; numpy only).
"""
import json, pathlib

cells = []

def md(src): cells.append({"cell_type":"markdown","metadata":{},"source":[s+"\n" for s in src.splitlines()]})
def code(src): cells.append({"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":[s+"\n" for s in src.splitlines()]})

# ── Title ─────────────────────────────────────────────────────────────────────
md("""# The 7 Basic Logic Gates, From Real Circuit Physics

Not abstract boolean algebra with gate names attached: every one of the 7
basic gates here is built from the same Shockley diode I-V curve
(`dgs.computer_engineering.diode_iv`) and a transistor switch, using the
real 1960s-70s **Diode-Transistor Logic (DTL)** family topology --
AND/OR from diode-resistor logic, NOT from a transistor, NAND/NOR by
cascading the diode stage's actual output VOLTAGE into the transistor's
base, and XOR/XNOR from the classic 4-/5-NAND gate networks.

This module's own development caught a genuine circuit problem along the
way -- not invented for the notebook, found while verifying the cascade --
and the fix is the real historical one, not a hack. Engine:
`dgs/logic_gates_from_em_circuits.py`.
""")

code("""%matplotlib inline
import sys, pathlib
sys.path.insert(0, str(pathlib.Path('..').resolve()))

import numpy as np
import matplotlib.pyplot as plt

from dgs.computer_engineering import diode_iv, diode_or_gate
from dgs import logic_gates_from_em_circuits as lgc

print('Setup complete.')
""")

# ── 1. Diode I-V recap ───────────────────────────────────────────────────────
md("""## 1. The Shockley Diode I-V Curve (Already in the Repo)

Every gate below rests on this one nonlinear device curve --
`dgs.computer_engineering.diode_iv`, reused directly, not reimplemented.
""")

code("""V = np.linspace(-0.5, 1.0, 400)
I = diode_iv(V)

fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(V, I * 1000, color='steelblue')
ax.axvline(0.7, color='firebrick', ls='--', label='~0.7V turn-on (Si)')
ax.set_xlabel('V (volts)'); ax.set_ylabel('I (mA)')
ax.set_ylim(-0.5, 20)
ax.set_title('Diode I-V curve -- the ONE nonlinearity every gate below relies on')
ax.legend()
plt.tight_layout()
plt.savefig('logic_gates_diode_iv.png', dpi=100, bbox_inches='tight')
plt.show()
""")

# ── 2. AND / OR ───────────────────────────────────────────────────────────────
md("""## 2. AND and OR: Diode-Resistor Logic (DRL)

`diode_or_gate` (already in `dgs.computer_engineering`) pulls the output
UP when any input is HIGH. `diode_and_gate` here is its dual circuit --
diodes flipped, pull-up resistor instead of pull-down -- clamping the
output DOWN unless every input is HIGH.
""")

code("""print('OR (existing):')
for a in (0, 1):
    for b in (0, 1):
        r = diode_or_gate(a, b)
        print(f\"  A={a} B={b} -> V_out={r['V_out']:.2f}  logic={r['logic_out']}\")

print('\\nAND (dual circuit):')
for a in (0, 1):
    for b in (0, 1):
        r = lgc.diode_and_gate(a, b)
        print(f\"  A={a} B={b} -> V_out={r['V_out']:.2f}  logic={r['logic_out']}\")
""")

# ── 3. NOT ────────────────────────────────────────────────────────────────────
md("""## 3. NOT: a Transistor Switch

Base-emitter forward bias (another diode junction) saturates the
transistor, pulling the output LOW; cut off lets the collector resistor
pull it HIGH.
""")

code("""for a in (0, 1):
    r = lgc.not_gate(a)
    print(f\"A={a} -> V_out={r['V_out']:.2f}  logic={r['logic_out']}\")
""")

# ── 4. DTL NAND / NOR ─────────────────────────────────────────────────────────
md("""## 4. DTL: Cascading the Diode Stage's REAL Output Voltage

`dtl_nand_gate` feeds the AND stage's actual output voltage into the
transistor's base -- not a boolean composition.
""")

code("""print('NAND:')
for a in (0, 1):
    for b in (0, 1):
        r = lgc.dtl_nand_gate(a, b)
        print(f\"  A={a} B={b} -> V_and_stage={r['V_and_stage']:.2f}, V_out={r['V_out']:.2f}, logic={r['logic_out']}\")

print('\\nNOR:')
for a in (0, 1):
    for b in (0, 1):
        r = lgc.dtl_nor_gate(a, b)
        print(f\"  A={a} B={b} -> V_or_stage={r['V_or_stage']:.2f}, V_out={r['V_out']:.2f}, logic={r['logic_out']}\")
""")

# ── 5. The zero-margin problem ────────────────────────────────────────────────
md("""## 5. A Real Problem This Module's Development Caught

A DTL gate's LOW-input AND-stage output sits at exactly $V_D=0.7$V -- the
SAME value as the transistor's bare turn-on threshold $V_{BE,on}=0.7$V
(both are silicon PN-junction drops, not a made-up coincidence). That's
**zero noise margin**: any real-world component tolerance could tip it
either way. Worse, an early prototype of the XOR network here that fed
raw stage voltages forward (instead of restoring each gate's output to a
clean rail voltage) produced an outright wrong answer.
""")

code("""value_low_input = lgc.diode_and_gate(0, 1)['V_out']
print(f'LOW-input AND-stage voltage: {value_low_input:.2f}V')
print(f'bare transistor threshold:   0.70V')
print(f'margin: {value_low_input - 0.70:.2f}V  <- exactly zero')

raw_case_voltage = min(5.0, 5.0 + 0.7, 0.2 + 0.7)   # AND(HIGH-node, prior-LOW-output) internal voltage
print(f\"\\na deeper cascade case combining a HIGH node with a prior stage's LOW (0.2V) output:\")
print(f'  internal voltage = {raw_case_voltage:.2f}V  (meant to read as LOW)')
print(f'  bare threshold = 0.70V -- {raw_case_voltage:.2f}V EXCEEDS it: an actual misfire')
""")

code("""with_shift = lgc.bjt_inverter(value_low_input, n_level_shift_diodes=lgc.DTL_LEVEL_SHIFT_DIODES)
effective_threshold = 0.7 * (1 + lgc.DTL_LEVEL_SHIFT_DIODES)
print(f'With {lgc.DTL_LEVEL_SHIFT_DIODES} series level-shifting diodes (the real historical DTL fix):')
print(f'  effective threshold = {effective_threshold:.2f}V')
print(f'  margin = {effective_threshold - value_low_input:.2f}V  <- real headroom now')
""")

# ── 6. XOR / XNOR ─────────────────────────────────────────────────────────────
md("""## 6. XOR and XNOR: the Classic 4-/5-NAND Networks

$$n_1=\\text{NAND}(A,B),\\quad n_2=\\text{NAND}(A,n_1),\\quad
n_3=\\text{NAND}(B,n_1),\\quad \\text{XOR}=\\text{NAND}(n_2,n_3)$$

built by cascading the ACTUAL DTL NAND gate's output voltage forward,
with level-shift diodes AND rail-voltage restoration at each gate
boundary -- both real fixes at once.
""")

code("""print('XOR (4-NAND network):')
for a in (0, 1):
    for b in (0, 1):
        r = lgc.xor_gate_from_nand(a, b)
        print(f\"  A={a} B={b} -> V_out={r['V_out']:.2f}, logic={r['logic_out']}, \"
              f\"stages={r['stage_voltages']}\")

print('\\nXNOR:')
for a in (0, 1):
    for b in (0, 1):
        r = lgc.xnor_gate(a, b)
        print(f\"  A={a} B={b} -> V_out={r['V_out']:.2f}, logic={r['logic_out']}\")
""")

# ── 7. Full verification ──────────────────────────────────────────────────────
md("""## 7. All 7 Gates, Verified Exhaustively
""")

code("""check = lgc.verify_all_seven_gates()
for gate, ok in check['per_gate'].items():
    print(f'  {gate:>5}: {\"PASS\" if ok else \"FAIL\"}')
print(f\"\\nall 7 gates correct, every input row: {check['all_seven_correct']}\")
""")

# ── 8. Engineering interpretation ────────────────────────────────────────────
md("""## 8. Engineering Interpretation

- Section 5 is the actual point of building gates from real device
  physics instead of boolean algebra: a purely logical model has no way
  to discover a noise-margin problem, because "0.7 > 0.7" and "0.9 > 0.7"
  aren't questions that exist without real voltages attached.
- The zero-margin case (Section 5's first finding) didn't cause a wrong
  answer in THIS exact idealized model -- but that's exactly the kind of
  fragile "works by coincidence" result real engineers distrust, which is
  why real DTL circuits added the level-shift diodes regardless of
  whether a specific simulation happens to get lucky.
- `xor_gate_from_nand` fixing the problem two independent ways at once
  (level-shift diodes AND rail-voltage restoration between gates) mirrors
  how real multi-package logic design actually layers defenses rather
  than relying on one fix alone.
""")

# ── 9. Research discussion ───────────────────────────────────────────────────
md("""## 9. Research Discussion

- `dgs.boolean_algebra`'s Karnaugh-map minimization operates purely on
  truth tables; a natural bridge is minimizing a boolean expression there,
  then realizing the minimized circuit here with actual DTL gates and
  confirming the resulting voltage-level circuit still matches the
  original (unminimized) truth table exactly.
- `dgs.computer_engineering.majority_gate`/`full_adder` are currently pure
  boolean functions; rebuilding the full adder from THIS module's
  voltage-level XOR/AND/OR gates (rather than Python `^`/`and`/`or`)
  would extend the "real circuit physics all the way down" posture to a
  genuinely useful combinational circuit, not just the 7 primitives.
- Real DTL's level-shift diodes also affect SPEED (each added junction is
  another capacitance to charge/discharge) -- a natural follow-up ties
  this module's static (DC) voltage model to `dgs.interconnect_delay` or
  `dgs.spacetime_circuit_timing`'s timing-domain treatment.
""")

# ── 10. Possible experiments ──────────────────────────────────────────────────
md("""## 10. Possible Experiments

1. Sweep `V_D` and `V_BE_on` independently (rather than keeping them
   equal) and find the exact combination where the zero-margin condition
   in Section 5 becomes a genuine functional failure even WITH rail-
   voltage restoration between gates -- mapping out the real failure
   boundary rather than the one specific case found here.
2. Build the 5-NAND XNOR network explicitly (rather than XOR + one more
   inverter, `xnor_gate`'s current approach) and confirm it gives
   identical results -- two different real gate-count topologies for the
   same truth table.
3. Rebuild `dgs.computer_engineering.full_adder` using only this module's
   voltage-level gates (no Python booleans) and verify its full 8-row
   truth table end to end through real circuit voltages.
""")

# ── 11. Future improvements ───────────────────────────────────────────────────
md("""## 11. Future Improvements

- This module's transistor model (`bjt_inverter`) is a two-state
  saturated/cutoff switch, not `diode_iv`'s continuous exponential I-V --
  a full Ebers-Moll BJT model would let the noise-margin analysis in
  Section 5 be quantitative (an actual voltage-transfer-characteristic
  curve, not a hard threshold) rather than a step function.
- `diode_and_gate`/`dtl_nand_gate`/etc. assume ideal (zero-resistance)
  pull-up/pull-down resistors implicitly baked into the voltage formulas;
  an explicit `R` parameter (matching `dgs.computer_engineering.diode_or_gate`'s
  own `R` argument, currently unused for the logic-level voltage here)
  would let fan-out / loading effects be modeled directly.
""")

# ── Write notebook ────────────────────────────────────────────────────────────
nb = {
    "nbformat": 4, "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.13.0"},
    },
    "cells": cells,
}
out = pathlib.Path("notebooks/logic_gates_from_em_circuits.ipynb")
out.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print(f"Written: {out}  ({len(cells)} cells)")
