"""
repl/_repl_digital_logic_64.py
Core digital logic: combinational, sequential, 64-bit arithmetic.
SymPy init_printing. Karnaugh, FSM, ALU, overflow, two's complement.
"""
import numpy as np
import sympy as sp
from sympy.logic.boolalg import (And, Or, Not, Xor, simplify_logic, SOPform)
from sympy import symbols, Matrix, pretty, init_printing
import pandas as pd

init_printing(use_unicode=False, wrap_line=False)

A, B, C, D = symbols('A B C D')

print("=" * 60)
print("CORE DIGITAL LOGIC + 64-BIT ARITHMETIC")
print("=" * 60)
print()

# ============================================================
# 1. Canonical forms: SOP and POS
# ============================================================
print("=== 1. SOP / POS Canonical Forms ===")

# 3-variable majority function: output 1 when >= 2 inputs are 1
minterms_maj = [3, 5, 6, 7]   # 011, 101, 110, 111
maxterms_maj = [0, 1, 2, 4]

sop = SOPform([A, B, C], minterms_maj)
pos = sp.POSform([A, B, C], maxterms_maj)

print(f"Majority function (out=1 when >=2 inputs high):")
print(f"  SOP: {sop}")
print(f"  POS: {pos}")
print(f"  Simplified SOP: {simplify_logic(sop)}")
print()

# verify with truth table
print("Truth table verification:")
rows = []
for i in range(8):
    a,b,c = (i>>2)&1,(i>>1)&1,i&1
    out = 1 if (a+b+c) >= 2 else 0
    minterm = 1 if i in minterms_maj else 0
    rows.append({'A':a,'B':b,'C':c,'Majority':out,'minterm':minterm})
print(pd.DataFrame(rows).to_string(index=False))
print()

# ============================================================
# 2. Karnaugh map (4-variable)
# ============================================================
print("=== 2. Karnaugh Map (4-variable) ===")
print("""
      CD
AB  | 00 | 01 | 11 | 10 |
----|----|----|----|----|
 00 |  0 |  0 |  1 |  1 |    <- group: A'B' * (CD + C'D) = A'B'C...
 01 |  0 |  0 |  1 |  1 |    <- same column -> C is the grouping
 11 |  1 |  1 |  1 |  1 |    <- whole row -> AB
 10 |  1 |  1 |  1 |  1 |    <- whole row -> A'B... wait

Reading the map:
  Top-right 2x4 block (columns 11,10, all rows): C = 1  -> term: C
  Bottom 2x4 block (rows 11,10, all cols):       A = 1  -> term: A

  F = A + C
""")

minterms_4var = [2,3,6,7,8,9,10,11,12,13,14,15]
A4,B4,C4,D4 = symbols('A B C D')
sop4 = SOPform([A4,B4,C4,D4], minterms_4var)
print(f"SymPy SOP: {sop4}")
print(f"Simplified: {simplify_logic(sop4)}")
print()

# ============================================================
# 3. Two's complement and 64-bit overflow
# ============================================================
print("=== 3. Two's Complement + 64-bit Overflow ===")
print("""
Two's complement (N-bit):
  Positive:  0 to 2^(N-1)-1
  Negative:  -2^(N-1) to -1
  Negate:    flip all bits + 1
  Overflow:  result doesn't fit in N bits

64-bit signed range: -9,223,372,036,854,775,808  to  9,223,372,036,854,775,807
                      -(2^63)                         2^63 - 1
""")

# two's complement demo
def twos_complement(n, bits=8):
    if n >= 0:
        return format(n, f'0{bits}b')
    return format((1 << bits) + n, f'0{bits}b')

def from_twos(b):
    n = int(b, 2)
    bits = len(b)
    if n >= (1 << (bits-1)):
        n -= (1 << bits)
    return n

print("8-bit two's complement examples:")
for n in [0, 1, 127, -1, -128, 42, -42]:
    tc = twos_complement(n, 8)
    back = from_twos(tc)
    print(f"  {n:5d}  ->  {tc}  ->  {back:5d}  {'OK' if back==n else 'ERR'}")
print()

# 64-bit overflow cases
print("64-bit overflow scenarios:")
MAX64 = (1 << 63) - 1
MIN64 = -(1 << 63)

cases = [
    ("MAX + 1",          MAX64,  1),
    ("MIN - 1",          MIN64, -1),
    ("MAX + MAX",        MAX64, MAX64),
    ("normal: 100+200",  100,   200),
]
for desc, a, b in cases:
    result_python = a + b   # Python int: no overflow
    overflow = result_python > MAX64 or result_python < MIN64
    print(f"  {desc:25s}  overflow={'YES' if overflow else 'no ':3s}  "
          f"result={result_python if not overflow else 'WRAPS'}")
print()

# ============================================================
# 4. 64-bit ALU operations
# ============================================================
print("=== 4. 64-bit ALU Operations ===")

a64 = np.uint64(0xDEAD_BEEF_CAFE_BABE)
b64 = np.uint64(0x1234_5678_9ABC_DEF0)

ops = [
    ('AND',    int(a64 & b64)),
    ('OR',     int(a64 | b64)),
    ('XOR',    int(a64 ^ b64)),
    ('NOT a',  int(~a64)),
    ('a+b',    int((int(a64) + int(b64)) & 0xFFFFFFFFFFFFFFFF)),
    ('a>>4',   int(a64 >> np.uint64(4))),
    ('a<<4',   int((int(a64) << 4) & 0xFFFFFFFFFFFFFFFF)),
    ('popcount a', bin(int(a64)).count('1')),
]

print(f"a = {int(a64):#018x}")
print(f"b = {int(b64):#018x}")
print()
for op, result in ops:
    if op == 'popcount a':
        print(f"  {op:15s} = {result} ones")
    else:
        print(f"  {op:15s} = {result:#018x}")
print()

# ============================================================
# 5. Sequential logic: D flip-flop and counter
# ============================================================
print("=== 5. Sequential Logic: D Flip-Flop + 4-bit Counter ===")
print("""
D flip-flop:
  On rising clock edge: Q_next = D
  Truth table:
    D | Q_next
    0 |   0
    1 |   1

  Used for: registers, counters, state machines

4-bit synchronous counter (mod-16):
  Q[3:0] increments on each clock edge
  Carry out when Q = 1111
""")

# simulate 4-bit counter
state = 0
print("4-bit counter simulation (20 cycles):")
print(f"  {'clk':>4}  {'Q[3:0]':>8}  {'binary':>8}  {'carry':>6}")
for clk in range(20):
    binary = format(state, '04b')
    carry = 1 if state == 15 else 0
    print(f"  {clk:4d}  {state:8d}  {binary:>8}  {carry:6d}")
    state = (state + 1) % 16
print()

# ============================================================
# 6. FSM: traffic light controller
# ============================================================
print("=== 6. FSM: Traffic Light Controller ===")
print("""
States: RED(0), GREEN(1), YELLOW(2)
Transitions (after timer expires):
  RED    -> GREEN
  GREEN  -> YELLOW
  YELLOW -> RED

Moore machine: output depends only on state
""")

states = {0:'RED   ', 1:'GREEN ', 2:'YELLOW'}
outputs = {0:'stop ', 1:'go   ', 2:'slow '}
transitions = {0:1, 1:2, 2:0}
timings = {0:30, 1:25, 2:5}   # seconds

state = 0
print(f"  {'t':>6}  {'state':>8}  {'output':>8}  {'next':>8}")
t = 0
for _ in range(6):
    duration = timings[state]
    nxt = transitions[state]
    print(f"  {t:6d}s  {states[state]:>8}  {outputs[state]:>8}  {states[nxt]:>8}")
    t += duration
    state = nxt
print()

# ============================================================
# 7. SymPy: Boolean minimization with init_printing
# ============================================================
print("=== 7. SymPy Boolean Minimization ===")

# Full adder SOP -> minimized
print("Full adder Sum = A XOR B XOR Cin:")
A_fa, B_fa, C_fa = symbols('A B C')
minterms_sum = [1, 2, 4, 7]   # odd number of 1s
expr_sum = SOPform([A_fa, B_fa, C_fa], minterms_sum)
expr_min = simplify_logic(expr_sum)
print(f"  SOP (raw):   {expr_sum}")
print(f"  Minimized:   {expr_min}")
print(f"  Expected:    A ^ B ^ C")
print()

print("Full adder Cout = majority(A,B,C):")
minterms_cout = [3, 5, 6, 7]
expr_cout = SOPform([A_fa, B_fa, C_fa], minterms_cout)
expr_cout_min = simplify_logic(expr_cout)
print(f"  SOP (raw):   {expr_cout}")
print(f"  Minimized:   {expr_cout_min}")
print(f"  Expected:    (A&B)|(B&C)|(A&C)")
print()

# show as pretty matrix for Sum truth table
print("Sum truth table as SymPy Matrix:")
tt = Matrix([[int((a^b^c)) for c in range(2)]
             for a in range(2) for b in range(2)])
print(pretty(tt))
print("rows: AB in {00,01,10,11}, cols: C in {0,1}")
print("1s at positions (1,2,4,7) in row-major order")
