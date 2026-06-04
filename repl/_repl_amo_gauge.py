"""
repl/_repl_amo_gauge.py
AMO physics: Lennard-Jones, atomic energy levels, optical transitions.
Gauge invariance: U(1), SU(2), SU(3) -- Standard Model structure.
Conservation laws table. de Broglie + trap geometry.
"""
import numpy as np
import sympy as sp
import pandas as pd
sp.init_printing(use_unicode=False, wrap_line=False)

print("=" * 60)
print("AMO PHYSICS + GAUGE INVARIANCE + STANDARD MODEL")
print("=" * 60)
print()

# ============================================================
# 1. Lennard-Jones potential
# ============================================================
print("=== 1. Lennard-Jones Potential ===")
r, eps_lj, sigma = sp.symbols('r epsilon sigma', positive=True)

V_LJ = 4 * eps_lj * ((sigma/r)**12 - (sigma/r)**6)
dV   = sp.diff(V_LJ, r)
r_eq = sp.solve(dV, r)[0]
V_eq = V_LJ.subs(r, r_eq)

print("V(r) =", sp.pretty(V_LJ))
print("dV/dr =", sp.pretty(sp.simplify(dV)))
print("Equilibrium r_eq =", sp.pretty(r_eq))
print("V(r_eq) =", sp.pretty(sp.simplify(V_eq)), "  (= -epsilon, the well depth)")
print()

# numerical for Ar-Ar
eps_ar  = 0.0103    # eV
sig_ar  = 3.40e-10  # m (3.4 Angstrom)
r_arr   = np.linspace(0.9*sig_ar, 4*sig_ar, 500)
V_arr   = 4*eps_ar*((sig_ar/r_arr)**12 - (sig_ar/r_arr)**6)

r_eq_n  = 2**(1/6) * sig_ar
V_min   = -eps_ar
print(f"Ar-Ar: epsilon={eps_ar} eV  sigma={sig_ar*1e10:.2f} A")
print(f"  Equilibrium distance: r_eq = {r_eq_n*1e10:.3f} A  (= 2^(1/6)*sigma)")
print(f"  Well depth:           V_min = {V_min:.4f} eV")
print(f"  At r=sigma:           V = 0  (repulsion = attraction)")
print(f"  r < sigma:            V > 0  (hard-core repulsion, Pauli exclusion)")
print(f"  r > r_eq:             V < 0  (van der Waals attraction)")
print()

# ============================================================
# 2. Atomic energy levels: hydrogen
# ============================================================
print("=== 2. Hydrogen Energy Levels ===")
n_sym = sp.Symbol('n', positive=True, integer=True)
E_n   = -13.6 / n_sym**2   # eV

print("E_n = -13.6 / n^2  eV")
print()
rows_H = []
for n in range(1, 7):
    E    = -13.6 / n**2
    lam  = 1240 / (13.6*(1 - 1/n**2)) if n > 1 else None
    rows_H.append({'n': n, 'E_eV': round(E,4),
                   'series': 'Lyman' if n<=2 else ('Balmer' if n<=4 else 'Paschen'),
                   'lambda_nm_from_1': round(lam,1) if lam else '-'})
print(pd.DataFrame(rows_H).to_string(index=False))
print()
print("Lyman:  UV (n->1),  Balmer: visible (n->2),  Paschen: IR (n->3)")
print()

# ============================================================
# 3. Optical transitions: selection rules
# ============================================================
print("=== 3. Optical Transitions + Selection Rules ===")
print("""
Electric dipole transitions (most common):
  Delta_n  = any
  Delta_l  = +/- 1      (angular momentum of photon = 1)
  Delta_ml = 0, +/-1    (helicity of photon)
  Delta_ms = 0          (spin unchanged in E1)

Why Delta_l = +/-1:
  Photon has spin-1. Angular momentum conserved.
  Atom absorbs photon -> angular momentum changes by 1.

Forbidden transitions:
  Delta_l = 0, +/-2     (E2 or M1, much slower)
  Rate ~ (r/lambda)^2 * E1_rate   (10^-6 x slower)

Optical qubit (trapped ion example):
  |0> = ground state  1S_0
  |1> = metastable    3P_0  (forbidden transition, long lifetime ~seconds)
  Coherence time: microseconds to seconds depending on isolation
""")

# Einstein A coefficient (spontaneous emission rate)
print("Spontaneous emission rate A (Einstein coefficient):")
omega_s, d_s, c_s, hbar_s, eps0_s = sp.symbols(
    'omega d c hbar epsilon_0', positive=True)
A_coeff = omega_s**3 * d_s**2 / (3 * sp.pi * eps0_s * hbar_s * c_s**3)
print("  A =", sp.pretty(A_coeff))
print("  Lifetime tau = 1/A")
print("  Linewidth (natural): delta_nu = A / (2*pi)")
print()

# ============================================================
# 4. de Broglie in a trap (optical tweezer / ion trap)
# ============================================================
print("=== 4. de Broglie Wavelength in a Trap ===")
print("""
Optical tweezer / ion trap: harmonic potential V=(1/2)*m*omega^2*r^2
Ground state wavefunction:  psi_0 = exp(-r^2 / 2*a_ho^2)
Harmonic oscillator length: a_ho = sqrt(hbar / m*omega)
de Broglie wavelength:      lambda = h/p = h/sqrt(2*m*kT)
""")

hbar_n = 1.055e-34
m_Rb   = 87 * 1.66e-27    # Rb-87
k_B    = 1.38e-23
h_n    = 6.626e-34

print(f"{'System':25s}  {'T':>8}  {'lambda_dB':>12}  {'a_ho':>12}")
print("-" * 65)
for name, mass, T_K, omega_trap in [
    ('Rb-87 BEC',         m_Rb,          1e-7,   2*np.pi*100),
    ('Rb-87 MOT',         m_Rb,          1e-4,   2*np.pi*100),
    ('Ca+ ion trap',      40*1.66e-27,   1e-3,   2*np.pi*1e6),
    ('electron (free)',   9.109e-31,     300,    None),
]:
    lam_dB = h_n / np.sqrt(2 * mass * k_B * T_K)
    a_ho   = np.sqrt(hbar_n / (mass * omega_trap)) if omega_trap else 0
    print(f"  {name:23s}  {T_K:>8.1e}  {lam_dB:>12.3e}  "
          f"{a_ho:>12.3e}" if omega_trap else
          f"  {name:23s}  {T_K:>8.1e}  {lam_dB:>12.3e}  {'N/A':>12}")
print()

# ============================================================
# 5. Gauge invariance: U(1) -> SU(2) -> SU(3)
# ============================================================
print("=== 5. Gauge Invariance: U(1), SU(2), SU(3) ===")
print("""
Gauge invariance: physics unchanged under local phase transformation.

U(1) -- Electromagnetism
  psi -> psi * exp(i*alpha(x))    (local phase rotation)
  To keep Schrodinger eq invariant: introduce photon field A_mu
  Conserved charge: electric charge Q
  Force carrier: photon (massless, spin-1)
  YOUR WORK: GS has global U(1) symmetry -> phase ambiguity

SU(2) -- Weak force
  psi -> psi * exp(i*alpha_a(x) * sigma_a/2)   (3 generators)
  Introduces W+, W-, Z bosons (massive via Higgs mechanism)
  Conserved charge: weak isospin T3
  Violates parity (left-handed only)

SU(3) -- Strong force (QCD)
  psi -> psi * exp(i*alpha_a(x) * lambda_a/2)  (8 generators)
  Introduces 8 gluons (massless, but confined)
  Conserved charge: color (red, green, blue)
  Quarks confined: never see free quark

Standard Model gauge group: SU(3) x SU(2) x U(1)
""")

# Conservation laws: full table
print("=== 6. Conservation Laws: full Standard Model table ===")
conserved = pd.DataFrame([
    ('Energy',           True,  True,  True,  True,  'Noether: time translation'),
    ('Momentum',         True,  True,  True,  True,  'Noether: space translation'),
    ('Angular momentum', True,  True,  True,  True,  'Noether: rotation'),
    ('Electric charge',  True,  True,  True,  True,  'U(1) gauge symmetry'),
    ('Color charge',     True,  'N/A', 'N/A', True,  'SU(3) gauge symmetry'),
    ('Baryon number',    True,  True,  True,  True,  'approx (violated by sphalerons)'),
    ('Lepton number',    True,  True,  True,  True,  'approx (violated by neutrino mass)'),
    ('Parity P',         True,  True,  False, True,  'Weak force violates P'),
    ('CP',               True,  True,  'part',True,  'CKM matrix, small violation'),
    ('Isospin',          True,  False, False, True,  'broken by quark masses'),
    ('Strangeness',      True,  False, False, True,  'violated by weak decays'),
], columns=['Quantity','EM','Weak','Strong','Gravity','Note'])
print(conserved.to_string(index=False))
print()

# ============================================================
# 7. Trapezoid rule vs exact: classical numerical integration
# ============================================================
print("=== 7. Trapezoid rule error: classical vs modern ===")
x_sym = sp.Symbol('x')
funcs = [
    ("sin(x) 0->pi",  sp.sin(x_sym), 0, np.pi),
    ("exp(-x) 0->5",  sp.exp(-x_sym), 0, 5),
    ("x^2 0->1",      x_sym**2, 0, 1),
    ("LJ potential",  None, None, None),
]

for name, f_sym, a, b in funcs[:3]:
    exact = float(sp.integrate(f_sym, (x_sym, a, b)))
    for N in [4, 16, 64, 256]:
        x_n = np.linspace(a, b, N+1)
        if name.startswith('sin'):
            y_n = np.sin(x_n)
        elif name.startswith('exp'):
            y_n = np.exp(-x_n)
        else:
            y_n = x_n**2
        trap = np.trapezoid(y_n, x_n)
        err  = abs(trap - exact)/abs(exact)
        if N == 4 or N == 256:
            print(f"  {name:20s}  N={N:4d}  trap={trap:.6f}  "
                  f"exact={exact:.6f}  err={err:.2e}")
    print()

print("Trapezoid error ~ O(h^2) = O(1/N^2): double N -> 4x more accurate")
print("Simpson's rule ~ O(h^4): much better for smooth functions")
