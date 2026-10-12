"""Build notebooks/polynomials_boolean_hex_gf2.ipynb -- "Polynomials ->
Boolean Algebra -> Hexadecimal: Arithmetic Over GF(2), CRCs, and LFSRs".

Companion to dgs/gf2_polynomials.py.

Build:   py -3.13 scripts/build_gf2_polynomials_nb.py
Execute: py -3.13 -m jupyter nbconvert --to notebook --execute --inplace
         notebooks/polynomials_boolean_hex_gf2.ipynb
"""
import pathlib
import nbformat as nbf

nb = nbf.v4.new_notebook()
md = lambda s: nbf.v4.new_markdown_cell(s)
co = lambda s: nbf.v4.new_code_cell(s)
cells = []

cells.append(md(r"""# Polynomials → Boolean Algebra → Hexadecimal

Redo ordinary algebra with only two numbers, $\{0, 1\}$, where

$$a + b = a \oplus b\ (\text{XOR}),\qquad a\cdot b = a \wedge b\ (\text{AND}),\qquad 1 + 1 = 0$$

This number system is called **GF(2)**. Three things you've met separately turn out
to be the same object:

| § | polynomial view | Boolean / hardware view | hex view |
|---|---|---|---|
| 1 | $x^8 + x^2 + x + 1$ | bits `1 0000 0111` | `0x107` |
| 2 | $a + b + ab$ | `a OR b` | truth table `0b1110` |
| 3 | remainder of $M(x)\,x^r \bmod g(x)$ | a shift register with XOR gates | the CRC in every Ethernet frame, ZIP file, and PNG |
| 4 | a primitive polynomial | an LFSR | the PRBS7 test pattern in `eye_diagram.py` |

Module: [`dgs/gf2_polynomials.py`](../dgs/gf2_polynomials.py). Every CRC is checked
against Python's own `zlib` / `binascii`."""))

cells.append(co("""import sys, pathlib
sys.path.insert(0, str(pathlib.Path.cwd().parent))
import sympy as sp
import matplotlib.pyplot as plt
from IPython.display import display, Math
from sympy.logic.boolalg import to_anf
sp.init_printing()
from dgs.gf2_polynomials import (
    X, poly_to_int, int_to_poly, to_hex, gf2_add, gf2_mul, gf2_divmod, sympy_divmod,
    anf_coefficients, anf_expression, truth_table, CRC_SPECS, crc_bitwise,
    crc_polynomial_remainder, crc_reference, detects_all_single_bit_errors, lfsr_period, is_irreducible,
)"""))

# ── 1 ───────────────────────────────────────────────────────────────
cells.append(md(r"""# 1. A GF(2) polynomial *is* a binary number

The coefficients can only be 0 or 1, so write them down as bits, highest power on
the left, and read the result as hex. Arithmetic then becomes bit operations:

- **add** = XOR, so there are no carries. $x + x = 2x = 0$.
- **multiply** = shift-and-XOR, a *carry-less* multiply. Modern CPUs have an instruction for it (`PCLMULQDQ`).
- **divide** = long division, but subtract with XOR."""))

cells.append(co("""for p in (X**8 + X**2 + X + 1, X**16 + X**12 + X**5 + 1, X**3 + X + 1):
    v = poly_to_int(p)
    display(Math(sp.latex(p) + rf"\\;\\to\\; \\texttt{{{v:b}}}_2 \\;\\to\\; \\texttt{{{hex(v)}}}"))

a, b = 0b1011, 0b111          # (x^3 + x + 1)(x^2 + x + 1)
print(f"carry-less  {a:#06b} × {b:#05b} = {gf2_mul(a, b):#09b}   (ordinary integer product {a*b:#09b})")
display(Math(sp.latex(sp.expand(int_to_poly(a)*int_to_poly(b))) + r"\\pmod 2 \\;=\\; " + sp.latex(int_to_poly(gf2_mul(a, b)))))
q, r = gf2_divmod(0x1234, 0x107)
print(f"0x1234 ÷ 0x107 over GF(2): quotient {q:#x}, remainder {r:#x}   (SymPy: {tuple(hex(t) for t in sympy_divmod(0x1234, 0x107))})")"""))

# ── 2 ───────────────────────────────────────────────────────────────
cells.append(md(r"""# 2. Every Boolean function is a GF(2) polynomial

Any logic gate can be written using only XOR (+) and AND (×). This is the
**algebraic normal form** (ANF, or Zhegalkin polynomial), and it's unique. The
coefficients come from the truth table by a *binary Möbius transform*, a butterfly
of XORs with the same shape as an FFT:"""))

cells.append(co("""a, b, c = sp.symbols("a b c")
gates = [("NOT a", lambda p: 1 - p, "a", ~a),
         ("a AND b", lambda p, q: p & q, "a b", a & b),
         ("a OR b", lambda p, q: p | q, "a b", a | b),
         ("a NAND b", lambda p, q: 1 - (p & q), "a b", ~(a & b)),
         ("a XOR b", lambda p, q: p ^ q, "a b", a ^ b),
         ("majority(a,b,c)", lambda p, q, r: (p & q) | (q & r) | (p & r), "a b c", (a & b) | (b & c) | (a & c))]
print(f"{'gate':18s} {'truth table':>12s}  ANF (our transform)            SymPy to_anf")
for name, f, names, expr in gates:
    n = len(names.split()); tt = truth_table(f, n)
    tt_hex = hex(int("".join(map(str, tt[::-1])), 2))
    print(f"{name:18s} {tt_hex:>12s}  {str(anf_expression(tt, names)):30s} {to_anf(expr)}")"""))

cells.append(md(r"""Read it as algebra: **OR is $a + b + ab$**. Adding $a$ and $b$ counts the $(1,1)$
case twice, $1 + 1 = 0$, so the $ab$ term puts it back. That's inclusion–exclusion
in GF(2). The "truth table" column is the function's output bits packed into hex, so
`0xe` = `1110` is OR and `0xe8` is majority. Logic synthesis tools and
cryptographers both use this form, since the degree of the ANF measures how
"nonlinear" a function is."""))

# ── 3 ───────────────────────────────────────────────────────────────
cells.append(md(r"""# 3. CRC: the checksum is a polynomial remainder

Treat a message as one giant polynomial $M(x)$ (its bits are the coefficients). A
**cyclic redundancy check** of width $r$ is

$$\text{CRC} = M(x)\,x^r \bmod g(x)$$

where the generator $g(x)$ is usually quoted as **hex**. In hardware, it's a
shift register with an XOR gate at every 1-bit of $g$. Below, a bit-by-bit shift
register is compared with plain polynomial division and with Python's built-in CRC
routines on the standard test string `"123456789"`:"""))

cells.append(co("""d = b"123456789"
print(f"{'CRC':20s} {'g(x) hex':>12s} {'shift register':>15s} {'library':>12s} {'published check':>16s}")
for name, (w, poly, init, refl, xo, check) in CRC_SPECS.items():
    ref = crc_reference(d, name)
    print(f"{name:20s} {poly:#12x} {crc_bitwise(d, name):#15x} {('—' if ref is None else hex(ref)):>12s} {check:#16x}")
print(f"\\nCRC-8 as pure polynomial division, (M(x)·x^8) mod (x^8 + 0x07): {crc_polynomial_remainder(d, 8, 0x07):#x}")
g32 = int_to_poly((1 << 32) | 0x04C11DB7)
display(Math(r"g_{32}(x) = " + sp.latex(g32)))"""))

cells.append(md(r"""**Why it catches errors.** Flipping bit $k$ adds $x^k$ to the message, so the CRC
changes unless $g(x)$ divides $x^{k+r}$. Any $g$ with at least two terms never divides
a single power of $x$, so **every single-bit error is detected**. With good choices
of $g$, every burst shorter than $r$ bits is detected too."""))

cells.append(co("""print("CRC-8 (0x07) detects every single-bit error in a 64-bit message:", detects_all_single_bit_errors(8, 0x07))
msg = bytearray(b"hello, waveguide"); good = crc_bitwise(bytes(msg), "CRC-32"); missed = 0
for i in range(len(msg)*8):
    e = bytearray(msg); e[i//8] ^= 1 << (i % 8); missed += crc_bitwise(bytes(e), "CRC-32") == good
print(f"CRC-32, all {len(msg)*8} single-bit flips of {bytes(msg)!r}: missed {missed}")"""))

# ── 4 ───────────────────────────────────────────────────────────────
cells.append(md(r"""# 4. LFSRs: when a polynomial is "primitive"

A linear-feedback shift register (n flip-flops plus XOR taps) cycles through states.
Its feedback polynomial decides how long the cycle is. If the polynomial is
**primitive**, the register visits all $2^n - 1$ nonzero states before repeating,
which makes a maximal-length pseudo-random bit sequence. PRBS7, $x^7 + x^6 + 1$, is
the standard test pattern for serial links and for the eye diagrams in this repo."""))

cells.append(co("""cases = [("x^7 + x^6 + 1  (PRBS7)", (1<<7)|(1<<6)|1, 7),
         ("x^4 + x^3 + 1", (1<<4)|(1<<3)|1, 4),
         ("x^4 + x^2 + 1 = (x^2+x+1)^2", (1<<4)|(1<<2)|1, 4),
         ("x^5 + x^3 + 1", (1<<5)|(1<<3)|1, 5)]
print(f"{'feedback polynomial':30s} {'hex':>6s} {'irreducible':>12s} {'period':>7s} {'2^n − 1':>8s}")
for name, v, n in cases:
    print(f"{name:30s} {v:#6x} {str(is_irreducible(v)):>12s} {lfsr_period(v, n):7d} {2**n - 1:8d}")"""))

cells.append(md(r"""The reducible polynomial $x^4 + x^2 + 1 = (x^2+x+1)^2$ falls short of the maximum
period (6 instead of 15). Factoring a polynomial predicts how a circuit will behave.

# Summary

| idea | polynomial | Boolean / circuit | hex | checked by |
|---|---|---|---|---|
| coefficients = bits | $x^8+x^2+x+1$ | 9 flip-flop taps | `0x107` | SymPy `Poly(modulus=2)` |
| add / multiply | $+$, $\times$ mod 2 | XOR, carry-less multiply | — | SymPy |
| every gate is a polynomial | $a+b+ab$ | OR | `0xe` | `sympy.to_anf` |
| CRC | $M(x)x^r \bmod g$ | shift register + XOR | `0x04C11DB7` | `zlib.crc32`, `binascii.crc_hqx` |
| LFSR period | primitive $g$ | maximal sequence | `0xc1` | simulation |

## Exercises
1. Compute the ANF of a 2-to-1 multiplexer $s\,?\,b : a$. What is its degree?
2. Find all irreducible polynomials of degree 4 over GF(2) by brute force with `is_irreducible`.
3. Implement CRC-32 with a 256-entry lookup table (bytewise). Why is it 8× faster?
4. Show that a CRC with $g(x) = (x+1)\,h(x)$ detects every error with an *odd* number of flipped bits.
5. In PowerShell, `$?` is a Boolean (`True`/`False`), but in bash it's an *integer* exit code where `0` means success. Write the bash → Boolean map as a GF(2) function."""))

nb["cells"] = cells
nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
out = pathlib.Path(__file__).resolve().parent.parent / "notebooks" / "polynomials_boolean_hex_gf2.ipynb"
nbf.write(nb, out)
print(f"wrote {out}")
