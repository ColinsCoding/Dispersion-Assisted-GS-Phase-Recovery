"""Build notebooks/rc_calculus_problems.ipynb -- a small self-study problem
set on the RC time constant, each problem posed, solved symbolically with
SymPy (sp.init_printing()), and checked numerically against the already-
tested dgs/circuits.py (solve_rc_symbolic, rc_step, rc_bandwidth) rather
than re-deriving those results from an unchecked independent guess.

Build with `py -3.13 scripts/build_rc_calculus_problems_nb.py`, execute
with `py -3.13 -m jupyter nbconvert --to notebook --execute --inplace
notebooks/rc_calculus_problems.ipynb`.
"""
import pathlib
import nbformat as nbf

nb = nbf.v4.new_notebook()
md = lambda s: nbf.v4.new_markdown_cell(s)
co = lambda s: nbf.v4.new_code_cell(s)
cells = []

cells.append(md("""# RC time constant -- a calculus problem set

Four problems on the RC circuit, from first-principles calculus to the
frequency-domain bandwidth result, each checked against `dgs/circuits.py`
(already tested, `tests/test_circuits.py`) rather than trusted blind."""))

cells.append(co("""import sys, pathlib
sys.path.insert(0, str(pathlib.Path('..').resolve()))
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from dgs import circuits as ct

sp.init_printing()
print("loaded dgs.circuits")"""))

cells.append(md("""## Problem 1 -- solve the charging ODE from scratch

**Statement.** A capacitor charges through a resistor from a step voltage
$V_{in}$, starting uncharged. Kirchhoff's voltage law gives
$RC\\,\\dot V + V = V_{in}$, $V(0)=0$. Solve for $V(t)$.

**Method.** Separate variables: $\\frac{dV}{V_{in}-V} = \\frac{dt}{RC}$,
integrate both sides, apply $V(0)=0$."""))

cells.append(co("""t, R, C, Vin = sp.symbols('t R C V_in', positive=True)
V = sp.Function('V')

# solve by hand, symbolically, via separation of variables
Vt = sp.Symbol('V_t')
lhs_integral = sp.integrate(1/(Vin - Vt), (Vt, 0, sp.Symbol('V')))
rhs_integral = sp.Symbol('t') / (R * C)
# -ln((Vin-V)/Vin) = t/(RC)  ->  V = Vin*(1 - exp(-t/(RC)))
by_hand = Vin * (1 - sp.exp(-t / (R * C)))
by_hand"""))

cells.append(co("""# CHECK: cross-check against circuits.py's independent dsolve() call
sympy_result = ct.solve_rc_symbolic()
print("dgs.circuits.solve_rc_symbolic():", sympy_result)

# substitute the by-hand solution into the original ODE and confirm it's satisfied
ode_lhs = R * C * sp.diff(by_hand, t) + by_hand
residual = sp.simplify(ode_lhs - Vin)
print(f"\\nODE residual (should be 0): {residual}")
assert residual == 0
print("CHECK PASSED: by-hand solution satisfies RC*dV/dt + V = Vin exactly")"""))

cells.append(md("""## Problem 2 -- what "time constant" physically means

**Statement.** Show that at $t=\\tau=RC$, the capacitor has charged to
$1-e^{-1}\\approx63.2\\%$ of $V_{in}$, independent of the actual values of
$R$ and $C$ -- this is WHY $\\tau=RC$ is called *the* time constant: it's
a universal charging-fraction milestone, not just a convenient unit."""))

cells.append(co("""fraction_at_tau = by_hand.subs(t, R * C) / Vin
fraction_at_tau_simplified = sp.simplify(fraction_at_tau)
print("V(RC)/Vin =", fraction_at_tau_simplified, "=", float(fraction_at_tau_simplified))

# numeric check against circuits.rc_step for several unrelated (R,C) pairs
for R_val, C_val in [(1e3, 1e-6), (50.0, 2e-9), (1e6, 1e-12)]:
    tau = R_val * C_val
    frac = ct.rc_step(tau, R_val, C_val, Vin=1.0)
    print(f"R={R_val:.0e} C={C_val:.0e}: V(tau)/Vin = {frac:.6f}")
    assert abs(frac - (1 - np.exp(-1))) < 1e-9
print("\\nCHECK PASSED: 63.2% fraction at t=tau holds regardless of R, C individually")"""))

cells.append(md("""## Problem 3 -- energy dissipated while charging (the "missing half" result)

**Statement.** A source charges a capacitor to $V_{in}$ through a
resistor. Show that the energy DISSIPATED in the resistor during charging
equals the energy finally STORED in the capacitor -- i.e. only half the
total energy the source supplies ends up stored; the other half is heat,
REGARDLESS of the resistance value.

**Method.** $I(t)=\\frac{V_{in}}{R}e^{-t/RC}$ (from $V_R=V_{in}-V_C$).
Energy dissipated: $E_R=\\int_0^\\infty I^2R\\,dt$. Energy stored:
$E_C=\\frac12CV_{in}^2$."""))

cells.append(co("""I_t = (Vin / R) * sp.exp(-t / (R * C))
E_R = sp.integrate(I_t**2 * R, (t, 0, sp.oo))
E_C = sp.Rational(1, 2) * C * Vin**2

print("Energy dissipated in R:", sp.simplify(E_R))
print("Energy stored in C:    ", E_C)
print("\\nEqual (R-independent):", sp.simplify(E_R - E_C) == 0)
assert sp.simplify(E_R - E_C) == 0"""))

cells.append(md("""**INTERPRETATION:** exactly half the energy the source delivers while
charging a capacitor through ANY resistor is lost as heat -- a genuinely
counterintuitive result (it doesn't depend on $R$ at all, even though
$R$ sets HOW FAST charging happens) that shows up again in switching
power supplies and CMOS gate charging energy."""))

cells.append(md("""## Problem 4 -- from time constant to bandwidth

**Statement.** Derive the RC low-pass's $-3\\,dB$ cutoff frequency
$f_{3dB}=\\frac{1}{2\\pi RC}$ from the frequency-domain transfer function,
and connect it back to the time-domain time constant $\\tau=RC$ from
Problems 1-2 -- same circuit, two different (and consistent) ways of
describing "how fast" it responds."""))

cells.append(co("""omega, f = sp.symbols('omega f', positive=True)
H = 1 / (1 + sp.I * omega * R * C)             # RC low-pass transfer function
mag_sq = sp.simplify(H * sp.conjugate(H))       # |H(jw)|^2
print("|H(jw)|^2 =", mag_sq)

# solve |H|^2 = 1/2 (the -3dB, half-power point) for omega -- sympy's
# positive=True assumption on omega already discards the spurious
# negative root, leaving exactly one solution
omega_3dB_solutions = sp.solve(sp.Eq(mag_sq, sp.Rational(1, 2)), omega)
assert len(omega_3dB_solutions) == 1, f"expected one positive root, got {omega_3dB_solutions}"
f_3dB_derived = sp.simplify(omega_3dB_solutions[0] / (2 * sp.pi))
print("f_3dB (derived):", f_3dB_derived)

# CHECK against circuits.rc_bandwidth
for R_val, C_val in [(1e3, 1e-6), (50.0, 2e-9)]:
    derived = float(f_3dB_derived.subs({R: R_val, C: C_val}))
    from_module = ct.rc_bandwidth(R_val, C_val)
    print(f"R={R_val:.0e} C={C_val:.0e}: derived={derived:.4e} Hz  module={from_module:.4e} Hz")
    assert abs(derived - from_module) / from_module < 1e-9
print("\\nCHECK PASSED: frequency-domain derivation matches dgs.circuits.rc_bandwidth exactly")"""))

cells.append(md("""**INTERPRETATION:** $\\tau=RC$ (time domain, Problem 1-2) and
$f_{3dB}=\\frac{1}{2\\pi RC}$ (frequency domain, this problem) are the
SAME physical fact seen two ways -- $\\tau f_{3dB}=\\frac{1}{2\\pi}$
always, for any RC circuit. A slower time constant (big $\\tau$) is
exactly a lower bandwidth (small $f_{3dB}$), not a coincidence."""))

nb['cells'] = cells
nb['metadata'] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.13"},
}

out_path = pathlib.Path(__file__).resolve().parent.parent / "notebooks" / "rc_calculus_problems.ipynb"
out_path.parent.mkdir(parents=True, exist_ok=True)
with open(out_path, "w", encoding="utf-8") as f:
    nbf.write(nb, f)
print(f"wrote {out_path}  ({len(cells)} cells)")
