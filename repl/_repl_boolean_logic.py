"""
repl/_repl_boolean_logic.py
Truth tables, Boolean algebra, gates, ECE fundamentals.
"""
import numpy as np
import pandas as pd
import sympy as sp
from sympy.logic.boolalg import (And, Or, Not, Xor, Nand, Nor, Xnor,
                                  truth_table, simplify_logic)
from sympy.abc import A, B, C

print("=" * 55)
print("ECE BOOLEAN LOGIC -- TRUTH TABLES + ALGEBRA")
print("=" * 55)
print()

# ============================================================
# 1. All 2-input gates
# ============================================================
print("=== 2-input gate truth tables ===")

vals = [(0,0),(0,1),(1,0),(1,1)]
rows = []
for a,b in vals:
    rows.append({
        'A': a, 'B': b,
        'AND':  int(a and b),
        'OR':   int(a or b),
        'NAND': int(not(a and b)),
        'NOR':  int(not(a or b)),
        'XOR':  int(a ^ b),
        'XNOR': int(not(a ^ b)),
        'IMPLY':int((not a) or b),
    })

df = pd.DataFrame(rows)
print(df.to_string(index=False))
print()

# ============================================================
# 2. Boolean algebra laws (SymPy symbolic)
# ============================================================
print("=== Boolean algebra laws ===")
laws = [
    ("Identity",      "A AND 1 = A",     sp.Eq(And(A, sp.true), A)),
    ("Identity",      "A OR  0 = A",     sp.Eq(Or(A, sp.false), A)),
    ("Null",          "A AND 0 = 0",     sp.Eq(And(A, sp.false), sp.false)),
    ("Null",          "A OR  1 = 1",     sp.Eq(Or(A, sp.true), sp.true)),
    ("Idempotent",    "A AND A = A",     sp.Eq(And(A, A), A)),
    ("Complement",    "A AND ~A = 0",    sp.Eq(And(A, Not(A)), sp.false)),
    ("Complement",    "A OR  ~A = 1",    sp.Eq(Or(A, Not(A)), sp.true)),
    ("DeMorgan 1",    "~(A AND B) = ~A OR ~B",
                      sp.Eq(Not(And(A,B)), Or(Not(A),Not(B)))),
    ("DeMorgan 2",    "~(A OR B) = ~A AND ~B",
                      sp.Eq(Not(Or(A,B)), And(Not(A),Not(B)))),
    ("Absorption",    "A AND (A OR B) = A",
                      sp.Eq(And(A, Or(A,B)), A)),
    ("Distribution",  "A AND (B OR C) = (A AND B) OR (A AND C)",
                      sp.Eq(And(A,Or(B,C)), Or(And(A,B),And(A,C)))),
]
for law, desc, eq in laws:
    check = '+' if sp.simplify(eq) == sp.true else 'FAIL'
    print(f"  [{check}] {law:14s}  {desc}")
print()

# ============================================================
# 3. SymPy simplify a messy expression
# ============================================================
print("=== SymPy: simplify Boolean expression ===")
expr = Or(And(A, B), And(A, Not(B)), And(Not(A), B))
simplified = simplify_logic(expr)
print(f"  Original:   A*B + A*~B + ~A*B")
print(f"  Simplified: {simplified}")
print(f"  = A OR B  (absorb middle term)")
print()

# SOP from truth table
print("=== Sum of Products (SOP) from truth table ===")
print("F(A,B,C): minterms {1,3,5,7} = F is 1 when C=1")
minterms = [1,3,5,7]   # binary: 001,011,101,111 -> C=1 always
expr_sop = sp.SOPform([A,B,C], minterms)
print(f"  SOP: {expr_sop}")
print(f"  Simplified: {simplify_logic(expr_sop)}")
print()

# ============================================================
# 4. Half adder and full adder
# ============================================================
print("=== Half Adder and Full Adder ===")
print("Half adder (1-bit + 1-bit -> sum, carry):")
print("  Sum   = A XOR B")
print("  Carry = A AND B")
print()

rows_ha = []
for a,b in vals:
    rows_ha.append({'A':a,'B':b,'Sum':a^b,'Carry':a&b})
print(pd.DataFrame(rows_ha).to_string(index=False))
print()

print("Full adder (A + B + Cin -> Sum, Cout):")
rows_fa = []
for a in [0,1]:
    for b in [0,1]:
        for cin in [0,1]:
            s = a^b^cin
            cout = (a&b)|(b&cin)|(a&cin)
            rows_fa.append({'A':a,'B':b,'Cin':cin,'Sum':s,'Cout':cout})
print(pd.DataFrame(rows_fa).to_string(index=False))
print()

# ============================================================
# 5. Electrostatics f=0 Hz: truth table analog
# ============================================================
print("=== Electrostatics f=0 Hz: charge states as logic ===")
print("""
At DC (f=0):  Maxwell's equations reduce to:
  div(E) = rho/eps0     (Gauss's law)
  curl(E) = 0           (no time-varying B)

Charge states map to logic:
  no charge   -> 0     (conductor at ground, V=0)
  charge +Q   -> 1     (conductor at V=+1)

Logic gate in CMOS:
  PMOS (p-type): conducts when gate = 0 (pulls to VDD=1)
  NMOS (n-type): conducts when gate = 1 (pulls to GND=0)

NAND gate CMOS:
  Out = NAND(A,B) = ~(A AND B)
  If A=1, B=1: both NMOS on -> Out pulled to GND -> 0  correct
  If A=0, B=1: top PMOS on  -> Out pulled to VDD -> 1  correct
""")

# Electrostatic energy stored in a capacitor = charge on plate
print("Capacitor as memory bit:")
eps0 = 8.854e-12
A_plate = (1e-6)**2  # 1 um^2 = 1e-12 m^2
d_gap   = 10e-9      # 10 nm
C_cap   = eps0 * A_plate / d_gap
V_high  = 1.0     # 1V = logic 1
Q_bit   = C_cap * V_high
E_bit   = 0.5 * C_cap * V_high**2
print(f"  C = eps0*A/d = {C_cap:.3e} F  ({C_cap*1e15:.2f} fF)")
print(f"  Q = C*V      = {Q_bit:.3e} C  (~{Q_bit/1.6e-19:.0f} electrons)")
print(f"  E = (1/2)CV^2= {E_bit:.3e} J  ({E_bit/1.6e-19:.2f} eV per bit)")
print()
print("1 bit = ~560 electrons on a 1um^2 capacitor at 1V")
print("DRAM: refresh every 64ms because leakage discharges the cap")
print()

# ============================================================
# 6. ECE <-> photonics <-> GS
# ============================================================
print("=== ECE -> Photonics -> GS: the chain ===")
print("""
Boolean logic         Photonics equivalent       GS
-------------         --------------------       ---
0 / 1 voltage         0 / pi phase shift         phi = 0 or pi  (BPSK)
XOR gate              MZI (Mach-Zehnder)         interference
AND gate              coincidence detector        |E1 * E2|
NOT gate              pi phase shifter            phi -> phi + pi
memory (D flip-flop)  optical cavity / loop       GS fixed point
clock edge            optical pulse               symbol period T
setup/hold time       coherence length            GS convergence window
""")
