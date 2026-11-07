#!/usr/bin/env python3
"""
live_cas_ollama_win.py — 2-step live transcription → SymPy CAS, fully local.
Windows port of live_cas_ollama.py, tuned for Jalali-style photonics
(photonic time stretch / dispersive Fourier transform).

Step 1: faster-whisper transcribes mic audio (CUDA if available, else CPU)
Step 2: if math detected, a local Ollama model extracts → SymPy executes

Differences from the macOS original:
  - mlx_whisper (Apple Silicon only)  → faster-whisper (CTranslate2)
  - macOS `say` TTS                   → Windows SAPI via PowerShell
  - cas_config.py                     → self-contained (prompt, filters,
                                        JSONL logging to caps.jsonl)
  - vecdb.py                          → vecdb_win.py: SQLite + Ollama
                                        nomic-embed-text embeddings (same
                                        idea, fully local); optional
  - Metal --plot3d / --torch mps      → dropped (no Metal on Windows)
  - extraction prompt seeded with Jalali photonics: time-stretch factor,
    dispersive Fourier wavelength-time mapping, GVD <-> D conversion

Setup (once):
  winget install Ollama.Ollama        # or https://ollama.com/download
  ollama pull qwen2.5-coder:3b        # or :1.5b for speed, :7b for accuracy
  ollama pull nomic-embed-text        # optional: enables --search / --reindex
  pip install numpy requests sounddevice sympy faster-whisper matplotlib

Usage:
  python live_cas_ollama_win.py
  python live_cas_ollama_win.py --model qwen2.5-coder:7b
  python live_cas_ollama_win.py --tts       # speak result via Windows TTS
  python live_cas_ollama_win.py --plot      # live-plot 1-variable expressions
  python live_cas_ollama_win.py --once "differentiate sine of x squared"
  python live_cas_ollama_win.py --reindex   # backfill embeddings for caps.jsonl
  python live_cas_ollama_win.py --search "dispersive fourier wavelength mapping"
"""

import argparse
import contextlib
import io
import json
import math
import os
import queue
import re
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone

# cap BLAS thread pools before numpy loads — per-thread scratch buffers can
# fail to allocate on this commit-tight machine, and the CAS never needs them
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import requests
import sympy as sp
from sympy import init_printing

init_printing(use_unicode=True)

# Windows consoles default to cp1252, which can't print SymPy's unicode
# output (or → ─ 🎤). Force UTF-8 so pprint and status lines never crash.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

SAMPLE_RATE   = 16000
CHUNK         = 1024
SILENCE_DB    = -38
SILENCE_SEC   = 0.7
MODEL_WHISPER = "base.en"                 # faster-whisper tag, auto-downloaded
MODEL_OLLAMA  = "qwen2.5-coder:3b"        # fits 8GB VRAM; 1.5b was too weak on multi-step formulas
OLLAMA_URL    = os.environ.get("OLLAMA_URL", "http://localhost:11434")
MAX_TOKENS    = 2000
LOG_PATH      = os.path.join(os.path.dirname(os.path.abspath(__file__)), "caps.jsonl")

# ── extraction prompt (inlined; the macOS version keeps this in cas_config) ──

EXTRACT_PROMPT = """- A valid request must specify a mathematical operation, expression,
  equation, numerical problem, vector field, matrix, probability experiment,
  circuit quantity, or recognized physics formula.
- Reject metaphorical, political, historical, social, shopping, food,
  or entertainment language even if it contains an isolated math word.
- Never invent missing mathematical data.

You convert a spoken math or physics utterance into a short
Python script that computes the answer with SymPy.

Output rules (strict):
- Reply with ONLY a fenced ```python code block — no prose before or after.
- If the utterance contains nothing computable, reply with exactly: # none
- EXCEPTION: if the utterance is a bare math topic with no specific problem
  ("integral", "chain rule", "eigenvalues"), do NOT reply # none — instead
  compute one short canonical example of that topic and sp.pprint it, with a
  print() line saying what the example is. E.g. for "integral":
    x = sp.Symbol('x')
    print("Example: integral of x**2")
    sp.pprint(sp.integrate(x**2, x))
  E.g. for "Bayes theorem" (bare, no numbers given — invent ALL of them,
  every probability on the right-hand side must be a plain float literal,
  never a bare symbol):
    P_A = 0.01
    P_B_given_A = 0.9
    P_B_given_not_A = 0.05
    P_B = P_B_given_A*P_A + P_B_given_not_A*(1 - P_A)
    P_A_given_B = P_B_given_A*P_A / P_B
    print("Example: P(A|B) via Bayes' theorem, P(A)=1%, P(B|A)=90%, P(B|not A)=5%")
    print(f"P(A|B) = {P_A_given_B:.4f}")
  E.g. for "Bessel function" (bare topic — use sp.besselj for the ORDINARY
  Bessel function of the first kind, J_n(x); NEVER sp.besseli, which is the
  MODIFIED Bessel function and is a different function entirely):
    x = sp.Symbol('x')
    J0 = sp.besselj(0, x)
    J1 = sp.besselj(1, x)
    print("Example: derivative identity d/dx[J_0(x)] = -J_1(x)")
    sp.pprint(sp.diff(J0, x))
  E.g. for "capital F versus differential F" / "F vs dF" / "what is F" in a
  fundamental-theorem-of-calculus context (bare conceptual question, no
  specific f given — invent one): capital F(x) is the ANTIDERIVATIVE
  function (dF/dx = f(x)); dF is the DIFFERENTIAL, an infinitesimal
  quantity dF = f(x)*dx, NOT a function:
    x = sp.Symbol('x')
    f = x**2
    F = sp.integrate(f, x)
    print("Example: f(x) = x**2. F(x) is its ANTIDERIVATIVE (a function, dF/dx=f):")
    sp.pprint(F)
    print("dF = f(x)*dx is the DIFFERENTIAL (an infinitesimal, not a function)")
  E.g. for "Lorentz transformation" (bare topic, no specific v/x/t given —
  invent concrete numbers; this is PLAIN-NUMBER arithmetic once v, x, t are
  picked, NOT a symbolic sp.solve — never call sp.solve on an equation that
  doesn't contain the variable you're solving for, e.g. solving an x-prime
  equation for t2 when t2 never appears in it returns an EMPTY list and
  indexing [0] into it crashes):
    c = 299792458.0
    v = 0.8*c
    t1, x1 = 10.0, 0.0
    gamma = 1/(1-(v/c)**2)**0.5
    x2 = gamma*(x1 - v*t1)
    t2 = gamma*(t1 - v*x1/c**2)
    print("Example: Lorentz transformation, v=0.8c, x=0 m, t=10 s")
    print(f"gamma = {gamma:.4f}")
    print(f"x' = {x2:.4e} m")
    print(f"t' = {t2:.4e} s")
- `sp` (sympy) is already imported; do not import anything else.
- Define EVERY symbol before use: x = sp.Symbol('x'), never bare names.
- End every symbolic computation with sp.pprint(...) of the final expression.
- For pure-number arithmetic and unit conversions use ONLY plain Python floats
  and print(f"...") with the formula and units — NEVER SymPy on plain numbers.
- Keep it under 20 lines. No functions, no classes, no comments.
- FIRST define every number given in the utterance as a plain float in SI
  units (e.g. "1 nanometer" -> L = 1e-9; "5 picoseconds" -> t = 5e-12),
  THEN apply the formula. Never leave a given quantity as a symbol.
- Print small/large numeric results with {value:.4e}, NEVER {value:.4f}
  (an f-format shows 5e-7 as 0.0000).
- If the user asks for a specific unit (eV, ns, THz, ...), finish the
  conversion and print the value in that unit.

Physical constants — these are PREDEFINED VARIABLES already available in the
execution environment. Use the names directly; NEVER redefine them, NEVER
make them sp.Symbol:
  c = 299792458.0            # speed of light, m/s
  h = 6.62607015e-34         # Planck, J*s
  hbar = 1.054571817e-34     # J*s
  e_charge = 1.602176634e-19 # C (also: 1 eV = 1.602176634e-19 J)
  m_e = 9.1093837015e-31     # electron mass, kg
  m_p = 1.67262192369e-27    # proton mass, kg
  k_B = 1.380649e-23         # Boltzmann, J/K
  eps0 = 8.8541878128e-12    # vacuum permittivity, F/m
  mu0 = 1.25663706212e-6     # vacuum permeability
  G = 6.67430e-11            # gravitation, m^3/(kg s^2)
  g = 9.80665                # standard gravity, m/s^2
  N_A = 6.02214076e23        # Avogadro
  R_gas = 8.314462618        # gas constant, J/(mol K)
  b_wien = 2.897771955e-3    # Wien displacement, m*K

Domain: Jalali-style photonics (photonic time stretch / dispersive Fourier
transform). Formulas you may be asked about:
- time-stretch factor:  M = 1 + D2/D1   (D1, D2 = dispersion of fiber 1, 2)
  equivalently M = 1 + L2/L1 for equal-dispersion fiber
- wavelength-to-time mapping (dispersive Fourier): dt = D * L * dlambda.
  Work in SI: D [s/m^2] * L [m] * dlambda [m] = dt [s].
  Unit conversion: 1 ps/(nm*km) = 1e-6 s/m^2
- dispersion parameter <-> GVD:  D = -(2*pi*c/lam**2) * beta2
- group delay across bandwidth:  tau(omega) = beta2 * L * (omega - omega0)
- angular frequency: omega = 2*pi*f = 2*pi*c/lam
- photon energy: E = h*f = h*c/lam
- Gaussian spectrum: sp.exp(-((omega-omega0)/sigma)**2 / 2)
- nonlinear phase: phi_NL = gamma * P * L_eff

Standard physics formulas (use these exact forms):
- free fall: t = sp.sqrt(2*height/g);  v = g*t
- particle in a box: E_n = n**2 * h**2 / (8 * m * L**2)   (m=m_e, L in meters)
- hydrogen levels: E_n = -13.605693 / n**2   (already in eV)
- de Broglie: lam = h / (m * v)
- photon: E = h*c/lam;  p = h/lam
- Coulomb: F = q1*q2 / (4*sp.pi*eps0*r**2)
- Wien peak: lam_max = b_wien / T
- ideal gas: P*V = n*R_gas*T
- time dilation: gamma = 1/sp.sqrt(1 - (v/c)**2)
- resistors: series R1+R2, parallel R1*R2/(R1+R2); RC time constant tau = R*C
- harmonic oscillator: omega = sp.sqrt(k/m); quantum E_n = hbar*omega*(n + 0.5)

Domain: vector calculus & Lagrangian mechanics — PREDEFINED functions
(call directly, never redefine):
  grad(f, [x,y,z]) -> list of partials     laplacian(f, [x,y,z])
  div([Fx,Fy,Fz], [x,y,z])                 curl([Fx,Fy,Fz], [x,y,z])
  euler_lagrange(L, q, t) -> left side of the equation of motion (= 0)
Lagrangian pattern (L = T - V, kinetic minus potential):
  t, m, k = sp.symbols('t m k')
  x = sp.Function('x')(t)
  L = m*sp.diff(x, t)**2/2 - k*x**2/2
  print("equation of motion (= 0):")
  sp.pprint(euler_lagrange(L, x, t))
For a pendulum use theta = sp.Function('theta')(t), V = -m*g*l*sp.cos(theta).

Domain: engineering economy (ENGR 140) — PREDEFINED FUNCTIONS are available
in the execution environment; call them directly (i is a decimal, n periods):
  fp(i,n) F/P    pf(i,n) P/F    fa(i,n) F/A    af(i,n) A/F
  pa(i,n) P/A    ap(i,n) A/P    ag(i,n) A/G    pg(i,n) P/G
  pa_geo(i,j,n) geometric-gradient P/A1 (payments grow at rate j)
  pe(cashflows,i) net present worth   fe(...) future   ae(...) annual
  irr(cashflows)  payback(cashflows[,i])  err(cashflows,marr)
  sir(cashflows,i)  cw(A,i) capitalized worth
  depreciation_sl(P,S,N), depreciation_soyd(P,S,N), depreciation_db(P,S,N),
  depreciation_ddb(P,S,N)  -> list of (year, D_n, book_value) tuples
cashflows is a plain list indexed by period, t=0 first (investment negative):
  pe([-650000] + [162500]*8, 0.15)
Examples: loan payment A = P*ap(i,n); present worth of savings P = A*pa(i,n);
"MARR" is just the i to use; accept a project if pe(...) > 0 or irr > MARR.
For depreciation print each row: for yr, D, bv in rows: print(f"...").

Domain: computer engineering — digital logic and probability.
- Boolean algebra: build expressions with sp.And, sp.Or, sp.Not, sp.Xor,
  sp.Nand, sp.Nor on sp.symbols('A B C'); simplify with
  sp.simplify_logic(expr) and sp.pprint the result.
- Sum-of-products from minterms: sp.SOPform([A, B, C], [1, 3, 5, 7]);
  product-of-sums: sp.POSform. The minterm list must be the plain spoken
  integers — NEVER convert them to binary or strings.
  Karnaugh-map simplification IS sp.simplify_logic(expr, form='dnf').
- Number systems: plain Python — bin(n), hex(n), int('1011', 2); print all
  requested bases. Two's complement of n in b bits: format((1 << b) + n if
  n < 0 else n, '0' + str(b) + 'b').
- Probability: exact answers with sp.Rational and sp.binomial —
  e.g. P(exactly k heads in n flips) = sp.binomial(n, k) / sp.Integer(2)**n.
  Print the exact fraction AND float(result).
- Bayes: P(A|B) = P(B|A)*P(A) / (P(B|A)*P(A) + P(B|~A)*P(~A)) with plain
  floats; print numerator, denominator, and the posterior as a float.
  "false positive rate r" means P(T|~D) = r exactly (use r itself, never
  1-specificity arithmetic on it). "sensitivity s" means P(T|D) = s.
- Expected value / variance of a die or discrete list: sp.Rational sums.
"""

# Extra guardrails small local models need (same spirit as the macOS version).
LOCAL_PROMPT_ADDENDUM = """

Extra rules (strict):
- NEVER sp.Eq, sp.pprint, or any SymPy call on plain numbers
  (sp.Eq(2e-11, 2e-11) collapses to True, which is useless output)
- Always print the formula used and the result with units, e.g.:
    print(f"1 / 50 GHz = {result:.4e} s = {result*1e9:.4f} ns")
- sp.diff(expr, var): var MUST be a plain sp.Symbol — NEVER an expression.
  For a chain rule just compute sp.diff(sp.sin(x**2), x) directly.
- NEVER use sp.Eq at all. sp.pprint(the_expression) directly — an sp.Eq of
  two equal expressions collapses to the useless output True.
- Do not name a variable `lambda` (Python keyword) — use lam."""

# ── formula retrieval bank ───────────────────────────────────────────────────
# Khan Academy / AP / intro-college coverage, electrodynamics emphasized.
# Only entries whose keywords appear in the utterance get injected into the
# prompt, so the bank can grow large without bloating every request.

FORMULA_BANK = [
    # electrodynamics
    (("coulomb", "point", "charge", "electric"),
     "E field of point charge: E = q/(4*sp.pi*eps0*r**2); force F = q1*q2/(4*sp.pi*eps0*r**2)"),
    (("gauss", "flux", "enclosed"),
     "Gauss's law: flux = q_enclosed/eps0"),
    (("capacitor", "capacitance", "farad", "plate"),
     "parallel plate C = eps0*A/d; Q = C*V; energy U = C*V**2/2"),
    (("lorentz", "magnetic", "force", "moving"),
     "magnetic force F = q*v*B (perpendicular); cyclotron radius r = m*v/(q*B), omega = q*B/m"),
    (("solenoid", "wire", "ampere", "field"),
     "B of long wire: B = mu0*I/(2*sp.pi*r); solenoid: B = mu0*n*I (n = turns/meter)"),
    (("faraday", "induction", "induced", "emf", "flux"),
     "Faraday: emf = -N*dPhi/dt with Phi = B*A*sp.cos(theta); use sp.diff for dPhi/dt"),
    (("inductor", "inductance", "henry"),
     "inductor: V = L*dI/dt; energy U = L*I**2/2"),
    (("poynting", "intensity", "electromagnetic", "wave"),
     "EM wave: c = 1/sp.sqrt(mu0*eps0); E = c*B; intensity S = E*B/mu0"),
    (("dipole", "torque"),
     "dipole moment p = q*d; torque = p*E*sp.sin(theta)"),
    (("drift", "electrons", "conduction"),
     "current I = n*A*e_charge*v_drift (n = carrier density)"),
    # mechanics
    (("projectile", "launched", "range", "thrown"),
     "projectile: range = v0**2*sp.sin(2*theta)/g; height = v0**2*sp.sin(theta)**2/(2*g); t_flight = 2*v0*sp.sin(theta)/g"),
    (("momentum", "collision", "impulse"),
     "p = m*v; impulse J = F*t = dp; conserve total m*v across collisions"),
    (("work", "kinetic", "potential", "spring"),
     "KE = m*v**2/2; PE = m*g*h; spring U = k*x**2/2; W = F*d*sp.cos(theta)"),
    (("circular", "centripetal"),
     "a_c = v**2/r; F_c = m*v**2/r"),
    (("orbit", "orbital", "satellite", "escape", "gravity"),
     "orbital v = sp.sqrt(G*M/r); escape v = sp.sqrt(2*G*M/r); F = G*m1*m2/r**2"),
    (("pendulum", "period", "spring", "oscillation"),
     "spring T = 2*sp.pi*sp.sqrt(m/k); pendulum T = 2*sp.pi*sp.sqrt(L/g)"),
    # waves & optics
    (("snell", "refraction", "refractive", "critical"),
     "Snell: n1*sp.sin(t1) = n2*sp.sin(t2); critical angle sp.asin(n2/n1)"),
    (("lens", "mirror", "focal", "image"),
     "thin lens: 1/f = 1/d_o + 1/d_i; magnification m = -d_i/d_o"),
    (("slit", "diffraction", "interference", "grating"),
     "double slit maxima: d*sp.sin(theta) = m*lam"),
    (("doppler",),
     "Doppler (sound): f_obs = f*(v_sound + v_observer)/(v_sound - v_source), signs toward"),
    (("wave", "speed", "wavelength", "frequency"),
     "v = f*lam"),
    # circuits
    (("resistor", "ohm", "current", "voltage", "power"),
     "V = I*R; P = I*V = I**2*R; series R1+R2; parallel R1*R2/(R1+R2)"),
    (("rc", "charging", "discharge", "capacitor", "time"),
     "RC: tau = R*C; charging V(t) = V0*(1 - sp.exp(-t/(R*C)))"),
    (("resonant", "resonance", "rlc", "impedance"),
     "resonance f0 = 1/(2*sp.pi*sp.sqrt(L*C)); Z = sp.sqrt(R**2 + (X_L - X_C)**2)"),
    # modern physics
    (("photoelectric", "work", "function", "threshold"),
     "photoelectric: KE_max = h*f - phi (phi = work function in J)"),
    (("bohr", "hydrogen", "level", "transition"),
     "hydrogen E_n = -13.605693/n**2 eV; photon from transition: dE = E_n2 - E_n1"),
    (("half", "life", "decay", "radioactive"),
     "N = N0*(sp.Rational(1,2))**(t/t_half); activity A = lam_decay*N, lam_decay = sp.log(2)/t_half"),
    (("mass", "energy", "rest"),
     "E = m*c**2; total E = gamma*m*c**2"),
    (("de", "broglie", "matter"),
     "de Broglie: lam = h/(m*v)"),
    # thermo
    (("heat", "specific", "temperature", "calorimetry"),
     "Q = m*c_specific*dT; melting/boiling Q = m*L_latent"),
    (("carnot", "efficiency", "engine"),
     "Carnot efficiency = 1 - T_cold/T_hot (kelvin)"),
    # engineering economy
    (("loan", "mortgage", "payment", "borrow"),
     "ANNUAL loan payment: A = P*ap(i, n) with annual i and n in years. "
     "ONLY if the user says monthly: A = P*ap(i/12, n*12)"),
    (("present", "worth", "future", "annuity", "deposit", "withdraw"),
     "P = F*pf(i,n); F = P*fp(i,n); P = A*pa(i,n); F = A*fa(i,n); A = F*af(i,n)"),
    (("gradient", "increases", "maintenance"),
     "arithmetic gradient: P = A1*pa(i,n) + G*pg(i,n); A = A1 + G*ag(i,n)"),
    (("npv", "npw", "marr", "acceptable", "profitable", "investment", "project", "saves"),
     "ALWAYS call the provided pe() — never derive NPW by hand: "
     "cashflows = [-initial_cost] + [annual_net]*years; "
     "result = pe(cashflows, marr_decimal); accept if result > 0"),
    (("irr", "rate", "return"),
     "irr(cashflows) -> decimal rate; accept if irr > MARR"),
    (("payback",),
     "payback(cashflows) simple; payback(cashflows, i) discounted"),
    (("depreciation", "salvage", "book", "straight", "declining"),
     "NEVER compute depreciation manually. Copy this pattern exactly "
     "(depreciation_sl / depreciation_soyd / depreciation_ddb as asked): "
     "rows = depreciation_sl(10000, 2000, 5)\n"
     "for yr, D, bv in rows:\n"
     "    print(f'Year {yr}: D = {D:.2f}, book value = {bv:.2f}')"),
    (("capitalized", "forever", "perpetual", "endowment"),
     "capitalized worth CW = A/i = cw(A, i); if A recurs every k years first convert: A_annual = A_k*af(i, k)"),
    # vector calculus / Lagrangian
    (("gradient", "divergence", "curl", "laplacian", "field"),
     "x, y, z = sp.symbols('x y z'); grad(f, [x,y,z]); div([Fx,Fy,Fz], [x,y,z]); "
     "curl([Fx,Fy,Fz], [x,y,z]); laplacian(f, [x,y,z]) — pass explicit variable lists"),
    (("lagrangian", "lagrange", "equation", "motion", "pendulum"),
     "L = kinetic - potential with q = sp.Function('q')(t); "
     "sp.pprint(euler_lagrange(L, q, t)) gives the equation of motion (= 0). "
     "Spring: L = m*sp.diff(x,t)**2/2 - k*x**2/2. "
     "Pendulum: L = m*l**2*sp.diff(theta,t)**2/2 + m*g*l*sp.cos(theta)"),
]


def retrieve_formulas(text: str) -> str:
    words = set(re.findall(r"[a-z]+", text.lower()))
    hits = [line for keys, line in FORMULA_BANK if words & set(keys)]
    if not hits:
        return ""
    return ("\n\nFormulas relevant to THIS request (use these exact forms):\n- "
            + "\n- ".join(hits[:8]))


# ── speech-text cleanup and gating (inlined cas_config equivalents) ──────────

HOMOPHONE_FIXES = [
    (re.compile(r"\bh bar\b", re.I), "hbar"),
    (re.compile(r"\bbeta (?:two|2)\b", re.I), "beta2"),
    (re.compile(r"\bbeta (?:one|1)\b", re.I), "beta1"),
    (re.compile(r"\bd (?:one|1)\b(?=.*(fiber|dispersion|stretch))", re.I), "D1"),
    (re.compile(r"\bd (?:two|2)\b(?=.*(fiber|dispersion|stretch))", re.I), "D2"),
    (re.compile(r"\bpico seconds?\b", re.I), "picoseconds"),
    (re.compile(r"\bnano meters?\b", re.I), "nanometers"),
    (re.compile(r"\bmicro meters?\b", re.I), "micrometers"),
    (re.compile(r"\bsign of\b", re.I), "sine of"),
    (re.compile(r"\bjelly\b", re.I), "Jalali"),
    (re.compile(r"\bex\s?or\b", re.I), "xor"),
    (re.compile(r"\bk\s?map\b", re.I), "karnaugh map"),
    (re.compile(r"\bmin terms?\b", re.I), "minterms"),
    (re.compile(r"\btwo'?s compliment\b", re.I), "two's complement"),
    (re.compile(r"\bbays theorem\b", re.I), "Bayes theorem"),
    (re.compile(r"\bbase theorem\b", re.I), "Bayes theorem"),
]

MATH_SIGNALS = re.compile(
    r"\d|plus|minus|times|divided|over|squared|cubed|root|equals?|"
    r"derivative|differen|integra|antideriv|solve|limit|expand|factor|simplify|"
    r"fundamental theorem|theorem of calculus|"
    r"sine?|cosine|tangent|log|exp\b|"
    r"wavelength|frequency|dispersion|stretch|gvd|beta2|soliton|photon|"
    r"energy|bandwidth|fiber|omega|lambda|hertz|joule|watt|"
    r"ohm|resistor|resistance|capacitor|capacitance|inductor|inductance|"
    r"impedance|resonance|resonant|solenoid|lorentz|faraday|coulomb|"
    r"poynting|dipole|drift|voltage|current|circuit|rlc|farad\b|henry\b|"
    r"boolean|logic|gate|xor|nand|nor\b|karnaugh|minterm|truth table|"
    r"binary|hex\b|two'?s complement|bits?\b|"
    r"probability|bayes|dice|die\b|coin|heads|tails|expected value|"
    r"variance|combinations?|permutations?|factorial|choose|"
    r"worth|present value|future value|net present value|annuity|loan|"
    r"mortgage|interest|irr\b|marr\b|payback|"
    r"depreciat|salvage|invest|cash flow|dollars?|percent|"
    r"deposit|withdraw|capitalized|gradient|"
    r"lagrangian|lagrange|curl\b|divergence|laplacian|pendulum|"
    r"odd function|even function|parity|symmetric|antisymmetric|"
    r"bessel|spherical harmonic|legendre|orthogonal|eigenvalue|eigenvector|"
    r"taylor series|maclaurin|fourier|convolution|"
    r"geometry|manifold|topology|vector space", re.I)

NOISE_PHRASES = {
    "you", "thank you", "thanks for watching", "bye", "so", "okay", "the",
    "thank you for watching", "i'm sorry", "hmm", "uh",
}


# ── Bayesian attention gate ──────────────────────────────────────────────────
# Naive Bayes over words: is this utterance math or chatter? Seeded with a
# vocabulary, then updated from caps.jsonl history — utterances that produced
# a CAS result are positive evidence, filtered hallucinations negative.
# 'skip' outcomes are deliberately NOT used for training: they're the gate's
# own (possibly wrong) decisions, and learning from them would entrench errors.

MATH_SEED = ("derivative derivatives integral integrate chain rule vector "
             "calculus gradient divergence curl matrix eigenvalue eigenvector "
             "laplace fourier taylor series limit boolean algebra logic gate "
             "minterm karnaugh truth table binary hex complement bit "
             "probability bayes coin dice heads tails expected variance "
             "factorial choose combinations permutations wavelength frequency "
             "dispersion stretch photon energy fiber soliton quantum "
             "oscillator resistor capacitor circuit voltage current ohm "
             "dilation relativity blackbody kelvin electron proton planck "
             "sine cosine tangent log exponential solve simplify expand "
             "factor plus minus times divided squared cubed root equals "
             "power pi theta omega lambda "
             "worth loan mortgage payment payback depreciation salvage "
             "marr irr invest investment annuity interest percent gradient "
             "capitalized cash flow "
             "odd even parity symmetric antisymmetric function functions "
             "bessel legendre orthogonal manifold topology geometry").split()

NOISE_SEED = ("thank you thanks watching okay bye hello sorry hmm uh yeah "
              "right change room interactions video subscribe like comment "
              "channel music applause laughter silence background so anyway "
              "stop wait hold nothing never mind").split()


class BayesGate:
    def __init__(self, log_path: str | None):
        self.pos = Counter(MATH_SEED)
        self.neg = Counter(NOISE_SEED)
        if log_path and os.path.exists(log_path):
            try:
                with open(log_path, encoding="utf-8") as f:
                    for line in f:
                        try:
                            row = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        self.observe(row.get("text", ""), row.get("outcome", ""))
            except OSError:
                pass

    STOPWORDS = frozenset(
        "the a an of to in is it and or for on with at by as be was were "
        "what how that this my me you your i we they he she girl guy so "
        "do does did can could would should will just now then there "
        "please give tell show us are am not no yes if when where".split())

    @classmethod
    def _words(cls, text: str) -> set:
        # strip possessive 's so "ohm's"/"newton's"/"coulomb's" match the
        # bare seed vocabulary ("ohm", "newton", "coulomb")
        words = (re.sub(r"'s$", "", w) for w in re.findall(r"[a-z']+", text.lower()))
        return {w for w in words if w not in cls.STOPWORDS and len(w) > 1}

    def observe(self, text: str, outcome: str) -> None:
        if outcome == "cas":
            self.pos.update(self._words(text))
        elif outcome == "filtered":
            # only true hallucinations are negative evidence; "no_extract"
            # often means the MODEL fumbled real math — training on it
            # would teach the gate to reject legitimate topics
            self.neg.update(self._words(text))

    def p_math(self, text: str) -> float:
        words = self._words(text)
        if not words:
            return 0.0
        vocab = len(set(self.pos) | set(self.neg)) + 1
        # normalise each class to an equal-size effective corpus so the
        # verdict doesn't drift as one class accumulates more history
        n_pos = max(sum(self.pos.values()), 1)
        n_neg = max(sum(self.neg.values()), 1)
        lp = ln = math.log(0.5)          # neutral prior
        for w in words:
            fp = self.pos[w] / n_pos
            fn = self.neg[w] / n_neg
            lp += math.log((fp * 1000 + 1) / (1000 + vocab))
            ln += math.log((fn * 1000 + 1) / (1000 + vocab))
        return 1.0 / (1.0 + math.exp(ln - lp))


BAYES_GATE = BayesGate(None)   # replaced in main() once LOG_PATH is known


def is_noise(text: str) -> bool:
    t = text.strip().strip(".!?,").lower()
    return len(t) < 3 or t in NOISE_PHRASES


def has_math_content(text: str) -> bool:
    """Signal fired but is there actually something computable?"""
    t = text.lower()
    if re.search(r"\d", t):
        return True
    return bool(re.search(
        r"derivative|differen|integra|antideriv|fundamental theorem|"
        r"theorem of calculus|solve|expand|factor|simplify|"
        r"stretch factor|dispersion|photon energy|group delay|limit of|"
        r"boolean|minterm|truth table|karnaugh|complement|"
        r"probability|bayes|expected value|variance|choose|factorial|"
        r"lagrangian|equation of motion|curl|divergence|laplacian|"
        r"gradient|vector field|pendulum|payback|depreciation|worth of|"
        r"present value|future value|net present value|"
        r"ohm|resistor|capacitor|inductor|impedance|resonance|resonant|"
        r"solenoid|lorentz|faraday|coulomb|poynting|dipole|drift|circuit|"
        # whisper spells small numbers as words ("one plus two")
        r"\b(zero|one|two|three|four|five|six|seven|eight|nine|ten|"
        r"eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|"
        r"eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|"
        r"eighty|ninety|hundred|thousand|million|half|quarter|pi)\b", t))


def fix_homophones(text: str) -> str:
    for pattern, replacement in HOMOPHONE_FIXES:
        text = pattern.sub(replacement, text)
    return text


# ── ollama client ─────────────────────────────────────────────────────────────

def check_ollama(model: str) -> None:
    try:
        resp = requests.get(f"{OLLAMA_URL}/api/tags", timeout=3)
        resp.raise_for_status()
    except requests.RequestException:
        sys.exit(
            f"[fatal] can't reach Ollama at {OLLAMA_URL}\n"
            "        start it with:  ollama serve   (or launch the Ollama app)"
        )
    names = [m["name"] for m in resp.json().get("models", [])]
    if model not in names and f"{model}:latest" not in names:
        sys.exit(
            f"[fatal] model {model!r} not found in Ollama (have: {', '.join(names) or 'none'})\n"
            f"        pull it with:  ollama pull {model}"
        )


_FORCE_CPU = False   # flips on after a CUDA-side crash; sticky for the session


def ollama_chat(model: str, system: str, user: str) -> tuple[str, bool]:
    """One non-streaming chat call. Returns (text, truncated).
    If the GPU runner crashes (OOM / JIT failure on tight machines), retry
    once on CPU and stay there for the rest of the session."""
    global _FORCE_CPU
    for attempt in ("gpu", "cpu") if not _FORCE_CPU else ("cpu",):
        options = {"temperature": 0, "num_predict": MAX_TOKENS}
        if attempt == "cpu" or _FORCE_CPU:
            options["num_gpu"] = 0
        resp = requests.post(
            f"{OLLAMA_URL}/api/chat",
            json={
                "model": model,
                "stream": False,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "options": options,
                "keep_alive": "30m",
            },
            timeout=300,
        )
        if resp.status_code == 500 and attempt == "gpu" and \
                re.search(r"cuda|cudaMalloc|PTX|buffer", resp.text, re.I):
            _FORCE_CPU = True
            print("[warn]    GPU runner failed — falling back to CPU inference", flush=True)
            continue
        resp.raise_for_status()
        data = resp.json()
        text = data.get("message", {}).get("content", "").strip()
        truncated = data.get("done_reason") == "length"
        return text, truncated
    raise RuntimeError("ollama chat failed on both gpu and cpu")


# ── step 1: audio → text ──────────────────────────────────────────────────────

def rms_db(chunk):
    rms = np.sqrt(np.mean(chunk.astype(float) ** 2) + 1e-10)
    return 20 * np.log10(rms / 32768)


PIPE_SPECTRUM = False
_SPEC_BANDS   = 24
_spec_window  = np.hanning(CHUNK)
_spec_edges   = np.unique(np.logspace(
    np.log10(2), np.log10(CHUNK // 2), _SPEC_BANDS + 1).astype(int))


def spectrum_bands(chunk) -> list[float]:
    x = chunk.astype(np.float32) / 32768.0
    mag = np.abs(np.fft.rfft(x * _spec_window))
    bands = np.zeros(_SPEC_BANDS, dtype=np.float32)
    for i in range(min(_SPEC_BANDS, len(_spec_edges) - 1)):
        lo, hi = _spec_edges[i], max(_spec_edges[i + 1], _spec_edges[i] + 1)
        bands[i] = mag[lo:hi].mean()
    bands = np.log10(1.0 + 40.0 * bands) / 2.0
    return [round(float(v), 2) for v in np.clip(bands, 0.0, 1.0)]


GATE = SILENCE_DB   # replaced at startup by calibrate_gate()
STREAM_RATE = SAMPLE_RATE


def open_input(callback):
    """Open the input stream at 16 kHz, falling back to the device's native
    rate (WASAPI/WDM-KS devices often refuse resampling)."""
    import sounddevice as sd
    global STREAM_RATE
    try:
        s = sd.InputStream(samplerate=SAMPLE_RATE, channels=1,
                           dtype="int16", blocksize=CHUNK, callback=callback)
        STREAM_RATE = SAMPLE_RATE
        return s
    except sd.PortAudioError:
        native = int(sd.query_devices(sd.default.device[0])["default_samplerate"])
        s = sd.InputStream(samplerate=native, channels=1,
                           dtype="int16", blocksize=CHUNK, callback=callback)
        STREAM_RATE = native
        print(f"[mic]     device refused 16 kHz — capturing at {native} Hz and resampling", flush=True)
        return s


def calibrate_gate(seconds=1.0):
    """Measure the ambient noise floor and place the speech gate 12 dB above
    it (quiet desktop mics sit far below the -38 dB default)."""
    global GATE
    levels = []

    def cb(indata, frames, t, status):
        levels.append(rms_db(indata[:, 0]))

    with open_input(cb):
        time.sleep(seconds)
    if levels:
        floor = float(np.median(levels))
        GATE = min(SILENCE_DB, max(floor + 12.0, -72.0))
        print(f"[mic]     noise floor {floor:.1f} dB → speech gate {GATE:.1f} dB", flush=True)
    return GATE


def record_utterance(timeout=20.0):
    q: queue.Queue = queue.Queue()
    buf, speech_started, silence_chunks = [], False, 0

    def cb(indata, frames, t, status):
        q.put(indata[:, 0].copy())

    stream = open_input(cb)
    silence_needed = int(SILENCE_SEC * STREAM_RATE / CHUNK)
    with stream:
        deadline = time.time() + timeout
        n_chunk = 0
        while time.time() < deadline:
            try:
                chunk = q.get(timeout=0.2)
            except queue.Empty:
                continue
            n_chunk += 1
            if PIPE_SPECTRUM and n_chunk % 2 == 0:
                print(json.dumps({"type": "spec", "b": spectrum_bands(chunk)}),
                      flush=True)
            db = rms_db(chunk)
            if db > GATE:
                speech_started = True
                silence_chunks = 0
                buf.append(chunk)
            elif speech_started:
                buf.append(chunk)
                silence_chunks += 1
                if silence_chunks >= silence_needed:
                    break

    if not buf:
        return None
    audio = np.concatenate(buf).astype(np.float32) / 32768.0
    if STREAM_RATE != SAMPLE_RATE:        # linear resample to whisper's 16 kHz
        n_out = int(len(audio) * SAMPLE_RATE / STREAM_RATE)
        audio = np.interp(np.linspace(0, len(audio) - 1, n_out),
                          np.arange(len(audio)), audio).astype(np.float32)
    return audio


_whisper_model = None


def get_whisper():
    """Load faster-whisper once: CUDA fp16 if the stack is present, else CPU
    int8 (fast enough for base.en)."""
    global _whisper_model
    if _whisper_model is not None:
        return _whisper_model
    # ctranslate2 imports torch optionally (try/except ImportError), but on a
    # commit-starved machine mapping torch's CUDA DLLs dies with OSError 1455,
    # which that except doesn't catch. We don't need torch — make its import
    # fail cleanly instead.
    sys.modules.setdefault("torch", None)
    from faster_whisper import WhisperModel
    # C: is nearly full on this machine — keep the model cache on D:
    cache = os.environ.get("WHISPER_CACHE", r"D:\whisper_models")
    for device, compute in (("cuda", "float16"), ("cpu", "int8")):
        try:
            m = WhisperModel(MODEL_WHISPER, device=device,
                             compute_type=compute, download_root=cache)
            list(m.transcribe(np.zeros(1600, dtype=np.float32))[0])  # warm-up probe
            _whisper_model = m
            print(f"[whisper] {MODEL_WHISPER} on {device} ({compute})", flush=True)
            return _whisper_model
        except Exception:
            continue
    sys.exit("[fatal] could not load faster-whisper on cuda or cpu")


def transcribe(audio):
    segments, _info = get_whisper().transcribe(audio, language="en", beam_size=1)
    return " ".join(s.text for s in segments).strip()


# ── step 2: text → SymPy ──────────────────────────────────────────────────────

# predefined in the exec namespace so generated code can use them directly
# (mirrors the constants table in EXTRACT_PROMPT)
CONSTANTS = {
    "c": 299792458.0,
    "h": 6.62607015e-34,
    "hbar": 1.054571817e-34,
    "e_charge": 1.602176634e-19,
    "m_e": 9.1093837015e-31,
    "m_p": 1.67262192369e-27,
    "k_B": 1.380649e-23,
    "eps0": 8.8541878128e-12,
    "mu0": 1.25663706212e-6,
    "G": 6.67430e-11,
    "g": 9.80665,
    "N_A": 6.02214076e23,
    "R_gas": 8.314462618,
    "b_wien": 2.897771955e-3,
}

# engineering-economy functions (engineering_economy.py, same folder) get
# injected into the exec namespace alongside the constants
try:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import engineering_economy as _ee
    FINANCE_FUNCS = {name: getattr(_ee, name) for name in (
        "fp", "pf", "fa", "af", "pa", "ap", "ag", "pg", "pa_geo",
        "pe", "fe", "ae", "irr", "err", "payback", "sir", "cw",
        "depreciation_sl", "depreciation_soyd", "depreciation_uop",
        "depreciation_db", "depreciation_ddb")}
except ImportError:
    FINANCE_FUNCS = {}

try:
    import vecdb_win as _vecdb
except ImportError:
    _vecdb = None


# vector-calculus and Lagrangian-mechanics helpers, injected like the
# finance functions so the model orchestrates instead of deriving

def _sorted_vars(f):
    return sorted(f.free_symbols, key=lambda s: s.name)


def grad(f, variables=None):
    """Gradient: list of df/dv."""
    return [sp.diff(f, v) for v in (variables or _sorted_vars(f))]


def div(F, variables):
    """Divergence of vector field [Fx, Fy, Fz]."""
    return sum(sp.diff(fi, v) for fi, v in zip(F, variables))


def curl(F, variables):
    """Curl of [Fx, Fy, Fz] with respect to [x, y, z]."""
    x, y, z = variables
    Fx, Fy, Fz = F
    return [sp.diff(Fz, y) - sp.diff(Fy, z),
            sp.diff(Fx, z) - sp.diff(Fz, x),
            sp.diff(Fy, x) - sp.diff(Fx, y)]


def laplacian(f, variables=None):
    """Sum of second partials."""
    return sum(sp.diff(f, v, 2) for v in (variables or _sorted_vars(f)))


def euler_lagrange(L, q, t):
    """Euler-Lagrange: d/dt(dL/dq') - dL/dq. Result = 0 is the equation of
    motion. q must be a sp.Function of t, e.g. x = sp.Function('x')(t)."""
    qdot = sp.diff(q, t)
    return sp.simplify(sp.diff(sp.diff(L, qdot), t) - sp.diff(L, q))


PHYSICS_FUNCS = {"grad": grad, "div": div, "curl": curl,
                 "laplacian": laplacian, "euler_lagrange": euler_lagrange}
INJECTED_FUNCS = {**FINANCE_FUNCS, **PHYSICS_FUNCS}


class _SpProxy:
    """sympy passthrough that defuses the sp.Eq(a, b) → True collapse local
    models keep producing: force evaluate=False so equations display."""

    def __getattr__(self, name):
        # injected helpers win name conflicts: sp.div is polynomial division
        # in sympy, but the prompt only ever teaches our vector-divergence
        if name in INJECTED_FUNCS:
            return INJECTED_FUNCS[name]
        return getattr(sp, name)

    @staticmethod
    def Eq(*args, **kwargs):
        if len(args) == 2 and "evaluate" not in kwargs:
            kwargs["evaluate"] = False
        return sp.Eq(*args, **kwargs)


def pick_plottable(ns: dict):
    """Scan an exec namespace for the first single-variable SymPy expression."""
    for name, val in ns.items():
        if name.startswith("_") or name == "sp":
            continue
        if isinstance(val, sp.Expr) and not val.is_number:
            free = val.free_symbols
            if len(free) == 1:
                return val, free.pop()
    return None, None


def extract_and_run(text: str, model: str) -> tuple[str | None, tuple]:
    """Returns (printed_result_or_None, (expr, symbol)_or_(None, None)).
    If the model's code prints only a bare boolean (== / .equals / Eq
    collapse), retry once with corrective feedback."""
    result, plot = _extract_and_run_once(text, model)
    if result is not None and result.strip() in ("True", "False", "[no output]"):
        print("[warn]    bare boolean output — retrying with feedback", flush=True)
        hint = (text + "\n\nIMPORTANT: your previous code printed only a bare "
                "boolean, which is useless. Do NOT use ==, .equals(), or sp.Eq "
                "anywhere. Compute the expression and sp.pprint it directly.")
        result2, plot2 = _extract_and_run_once(hint, model)
        if result2 is not None and result2.strip() not in ("True", "False", "[no output]"):
            return result2, plot2
    return result, plot


def _extract_and_run_once(text: str, model: str) -> tuple[str | None, tuple]:
    raw, truncated = ollama_chat(
        model, EXTRACT_PROMPT + LOCAL_PROMPT_ADDENDUM + retrieve_formulas(text), text)

    # reasoning models wrap deliberation in <think> tags — drop it
    raw = re.sub(r"<think>.*?(?:</think>|\Z)", "", raw, flags=re.DOTALL).strip()

    if not raw or raw == "# none" or "# none" in raw:
        return None, (None, None)

    match = re.search(r"```(?:python)?\s*\n(.*?)(?:```|\Z)", raw, re.DOTALL)
    if match:
        code = match.group(1).strip()
    else:
        lines = [l for l in raw.splitlines()
                 if not l.strip().startswith("```")
                 and (l.strip() and not l.startswith("#")
                      or l.strip().startswith(("import", "sp.", "print", "for", "if", "=")))]
        code = "\n".join(lines).strip()

    if truncated:
        code = "\n".join(code.splitlines()[:-1]).strip()
        print("[warn]    response hit num_predict — dropped final partial line", flush=True)

    if not code:
        return None, (None, None)

    # local models ignore the no-imports rule: rewrite sympy imports into
    # bindings on the sp proxy (so Eq stays defused), drop bare import lines
    fixed = []
    for line in code.splitlines():
        if re.match(r"\s*from\s+sympy(?:\.\w+)*\s+import\s+\*", line):
            continue                    # star import: sp proxy already covers it
        m = re.match(r"\s*from\s+sympy(?:\.\w+)*\s+import\s+([\w\s,]+)$", line)
        if m:
            for name in m.group(1).split(","):
                parts = [p.strip() for p in name.split(" as ")]
                src, target = parts[0], parts[-1]
                if src and src != "sp":
                    fixed.append(f"{target} = sp.{src}")
        elif re.match(r"\s*import\s+sympy", line):
            continue                        # sp is already in the namespace
        else:
            fixed.append(line)
    code = "\n".join(fixed)

    # models sometimes re-implement an injected finance function (badly),
    # shadowing the real one — strip any such def block so the injected
    # implementation stays in force
    for fname in INJECTED_FUNCS:
        code = re.sub(
            rf"(?m)^def {fname}\([^)]*\):\n(?:(?:[ \t]+.*)?\n)*?(?=^\S|\Z)",
            "", code)

    # sanitise known bad patterns local models produce
    # f"{expr:.4e}" dies when expr is a SymPy object (sp.pi in a product):
    # wrap numeric-format targets in float()
    code = re.sub(r"\{([^{}:]+):(\.?\d*[efg%])\}", r"{float(\1):\2}", code)
    code = re.sub(r'\blambda\b(?=\s*=)', 'lam', code)
    code = re.sub(r'[ħℏ]', 'hbar', code)
    code = re.sub(r'[∂∇]', '', code)
    code = re.sub(r"pprint\(\s*bool\(", "pprint((", code)
    # symbols('x')(t) is a Symbol call crash — the model means a Function of t
    code = re.sub(r"(?:sp\.)?symbols\(\s*'(\w+)'\s*\)\s*\(\s*(\w+)\s*\)",
                  r"sp.Function('\1')(\2)", code)
    code = code.replace("int(result)", "int(bool(result))")
    code = code.replace("int(output)", "int(bool(output))")
    code = re.sub(r'(\w)\{([^}]+)\}', r'\1_\2', code)

    buf = io.StringIO()
    ns = {"sp": _SpProxy(), "__builtins__": __builtins__,
          **CONSTANTS, **INJECTED_FUNCS}
    try:
        compile(code, "<cas>", "exec")
    except SyntaxError as se:
        numbered = "\n".join(f"{i+1:3}: {l}" for i, l in enumerate(code.splitlines()))
        print(f"[syntax error] line {se.lineno}: {se.msg}\n[bad code]\n{numbered}\n", flush=True)
        return f"[exec error] {se}", (None, None)

    # self-heal loop: define missing names as Symbols and retry (up to 3)
    for _attempt in range(4):
        try:
            with contextlib.redirect_stdout(buf):
                exec(code, ns)  # noqa: S102
            break
        except NameError as e:
            missing = re.search(r"name '(\w+)' is not defined", str(e))
            if missing and _attempt < 3:
                ns[missing.group(1)] = sp.Symbol(missing.group(1))
                buf = io.StringIO()
                print(f"[heal]    defined missing symbol {missing.group(1)!r} — retrying", flush=True)
                continue
            numbered = "\n".join(f"{i+1:3}: {l}" for i, l in enumerate(code.splitlines()))
            print(f"[exec error] {e}\n[bad code]\n{numbered}\n", flush=True)
            return f"[exec error] {e}", (None, None)
        except Exception as e:
            # a self-healed symbol (bare sp.Symbol, no value) can survive into
            # a later float()/format() call the model wrote for the "happy
            # path" numeric result — degrade to the symbolic form instead of
            # failing outright
            if "cannot convert expression to float" in str(e).lower():
                # models name their final expression all sorts of things
                # (E, V, F, ...), not just "result"/"output" -- scan the
                # namespace in REVERSE assignment order for the most
                # recently defined COMPOUND sympy expression (skip bare
                # Symbols like `r = sp.Symbol('r')`, which trivially have
                # themselves as a free symbol but aren't the answer)
                fallback = None
                for _name in reversed(list(ns.keys())):
                    _val = ns[_name]
                    if (isinstance(_val, sp.Basic) and not isinstance(_val, sp.Symbol)
                            and _val.free_symbols):
                        fallback = _val
                        break
                if fallback is None:
                    fallback = ns.get("result", ns.get("output"))
                if isinstance(fallback, sp.Basic) and fallback.free_symbols:
                    print(f"[warn]    {e} — input under-specified, "
                          "showing symbolic form instead", flush=True)
                    buf = io.StringIO()
                    with contextlib.redirect_stdout(buf):
                        sp.pprint(fallback)
                    break
            numbered = "\n".join(f"{i+1:3}: {l}" for i, l in enumerate(code.splitlines()))
            print(f"[exec error] {e}\n[bad code]\n{numbered}\n", flush=True)
            return f"[exec error] {e}", (None, None)

    result = buf.getvalue().strip() or "[no output]"

    # drop bare True/False collapse noise when real output surrounds it
    lines = result.splitlines()
    kept = [l for l in lines if l.strip() not in ("True", "False")]
    if kept and len(kept) < len(lines):
        result = "\n".join(kept).strip()

    return result, pick_plottable(ns)


# ── logging: JSONL (always) + optional vector index (vecdb_win.py) ───────────

def log_utterance(enabled: bool, ts: str, duration: float, text: str,
                  outcome: str, result: str | None = None):
    if not enabled:
        return
    try:
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "ts": ts, "duration": round(duration, 2), "text": text,
                "outcome": outcome, "result": result,
                "source": "live_cas_ollama_win",
            }) + "\n")
    except OSError as e:
        print(f"[log error] {e}", flush=True)

    # only successful extractions are worth semantic search; embedding every
    # filtered/skip utterance would add a network round-trip to the hot path
    # for no benefit
    if _vecdb is not None and outcome == "cas":
        _vecdb.add(LOG_PATH, ts, text, outcome, result)


# ── Windows TTS ───────────────────────────────────────────────────────────────

def speak(text: str) -> None:
    """Speak via Windows SAPI. Text goes over stdin to dodge quoting issues."""
    clean = re.sub(r"[^\x20-\x7e]", " ", text)[:300]
    ps = ("Add-Type -AssemblyName System.Speech; "
          "(New-Object System.Speech.Synthesis.SpeechSynthesizer)"
          ".Speak([Console]::In.ReadToEnd())")
    proc = subprocess.Popen(
        ["powershell", "-NoProfile", "-Command", ps],
        stdin=subprocess.PIPE,
        creationflags=subprocess.CREATE_NO_WINDOW,
    )
    proc.stdin.write(clean.encode("ascii", "replace"))
    proc.stdin.close()


# ── main loop ─────────────────────────────────────────────────────────────────

def process_text(text, args, history, conn_enabled, duration, plot_state):
    """Shared pipeline for mic utterances and --once text."""
    text = fix_homophones(text)

    if is_noise(text):
        if not args.pipe:
            print(f"[filter]  hallucination dropped: {text!r}\n", flush=True)
        log_utterance(conn_enabled, datetime.now(timezone.utc).isoformat(),
                      duration, text, "filtered")
        BAYES_GATE.observe(text, "filtered")
        return

    ts = datetime.now(timezone.utc).isoformat()

    if args.pipe:
        print(json.dumps({"ts": ts, "type": "speech", "text": text}), flush=True)
    else:
        print(f"\n[speech]  {text}", flush=True)

    p = BAYES_GATE.p_math(text)

    if not MATH_SIGNALS.search(text) and p < 0.60:
        if not args.pipe:
            print(f"[cas]     no math signal (P(math)={p:.2f}) — skipping\n", flush=True)
        log_utterance(conn_enabled, ts, duration, text, "skip")
        return

    if not has_math_content(text) and p < 0.75:
        if not args.pipe:
            print(f"[cas]     no math content (P(math)={p:.2f}) — skipping\n", flush=True)
        log_utterance(conn_enabled, ts, duration, text, "skip")
        return

    if not args.pipe:
        print(f"[cas]     extracting via {args.model} → running SymPy…", flush=True)

    try:
        result, (plot_expr, plot_symbol) = extract_and_run(text, args.model)
    except Exception as e:
        if args.pipe:
            print(json.dumps({"ts": ts, "type": "error", "msg": str(e)[:200]}), flush=True)
        else:
            print(f"[ollama error] {e}\n", flush=True)
        log_utterance(conn_enabled, ts, duration, text, "api_error")
        return

    if result is None:
        if not args.pipe:
            print("[cas]     no extractable expression\n", flush=True)
        log_utterance(conn_enabled, ts, duration, text, "no_extract")
        BAYES_GATE.observe(text, "no_extract")
        return

    history.append((text, result))
    log_utterance(conn_enabled, ts, duration, text, "cas", result)
    BAYES_GATE.observe(text, "cas")

    if plot_state is not None and plot_expr is not None:
        plt, fig, ax, line = plot_state
        try:
            f_numeric = sp.lambdify(plot_symbol, plot_expr, "numpy")
            xs = np.linspace(-10, 10, 400)
            ys = f_numeric(xs)
            line.set_data(xs, ys)
            ax.set_title(f"{plot_expr} vs {plot_symbol}", color="#dddddd")
            ax.relim()
            ax.autoscale_view()
            fig.canvas.draw()
            fig.canvas.flush_events()
            plt.pause(0.001)
        except Exception as e:
            if not args.pipe:
                print(f"[plot error] {e}", flush=True)

    if args.pipe:
        print(json.dumps({"ts": ts, "type": "cas", "result": result}), flush=True)
    else:
        bar = "─" * 52
        print(f"\n{bar}", flush=True)
        print(result, flush=True)
        print(f"{bar}\n", flush=True)

    if args.tts:
        speak(result)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=MODEL_OLLAMA,
                        help=f"Ollama model tag (default: {MODEL_OLLAMA})")
    parser.add_argument("--tts",   action="store_true", help="speak result via Windows TTS")
    parser.add_argument("--pipe",  action="store_true", help="stdout-only mode: emit JSON lines")
    parser.add_argument("--plot",  action="store_true", help="live-plot single-variable expressions")
    parser.add_argument("--once",  metavar="TEXT", help="run the pipeline on TEXT (no mic) and exit")
    parser.add_argument("--no-log", action="store_true", help="disable JSONL logging to caps.jsonl")
    parser.add_argument("--list-devices", action="store_true", help="list audio input devices and exit")
    parser.add_argument("--device", type=int, metavar="N", help="input device index from --list-devices")
    parser.add_argument("--meter", action="store_true", help="15s mic level meter (is my mic above the gate?)")
    parser.add_argument("--search", metavar="QUERY",
                        help="semantic search past results (needs: ollama pull nomic-embed-text) and exit")
    parser.add_argument("--search-k", type=int, default=5, metavar="N",
                        help="number of results for --search (default 5)")
    parser.add_argument("--reindex", action="store_true",
                        help="backfill embeddings for existing caps.jsonl history and exit")
    args = parser.parse_args()

    if args.search:
        if _vecdb is None:
            sys.exit("[fatal] vecdb_win.py not found next to this script")
        results = _vecdb.search(LOG_PATH, args.search, k=args.search_k)
        if not results:
            print("no matches (is `ollama pull nomic-embed-text` done, and Ollama running?)")
            return
        for score, ts, text, result in results:
            print(f"\n[{score:.3f}] {ts}")
            print(f"  Q: {text}")
            if result:
                print(f"  A: {result}")
        return

    if args.reindex:
        if _vecdb is None:
            sys.exit("[fatal] vecdb_win.py not found next to this script")
        n = _vecdb.backfill(LOG_PATH)
        print(f"[vecdb]   embedded {n} new utterance(s)")
        return

    if args.list_devices:
        import sounddevice as sd
        apis = sd.query_hostapis()
        for i, d in enumerate(sd.query_devices()):
            if d["max_input_channels"] > 0:
                star = "  <-- default" if i == sd.default.device[0] else ""
                print(f"{i:3}: {d['name']}  [{apis[d['hostapi']]['name']}]{star}")
        return

    if args.device is not None:
        import sounddevice as sd
        sd.default.device = (args.device, None)

    if args.meter:
        import sounddevice as sd
        q: queue.Queue = queue.Queue()

        def cb(indata, frames, t, status):
            q.put(indata[:, 0].copy())

        name = sd.query_devices(sd.default.device[0])["name"]
        print(f"calibrating noise floor on {name!r} — stay quiet a moment…")
        calibrate_gate()
        print("meter for 15s — speak now.")
        with open_input(cb):
            end = time.time() + 15
            while time.time() < end:
                try:
                    chunk = q.get(timeout=0.3)
                except queue.Empty:
                    continue
                db = rms_db(chunk)
                width = max(0, min(60, int(db) + 90))   # -90 dB .. -30 dB span
                mark = "  << ABOVE GATE" if db > GATE else ""
                print(f"{db:7.1f} dB |{'#' * width:<60}|{mark}", flush=True)
        return

    global PIPE_SPECTRUM
    PIPE_SPECTRUM = args.pipe

    check_ollama(args.model)

    global BAYES_GATE
    BAYES_GATE = BayesGate(LOG_PATH)   # learn from this machine's history

    log_enabled = not args.no_log
    if log_enabled and not args.pipe:
        print(f"[log]     appending to {LOG_PATH}", flush=True)

    plot_state = None
    if args.plot:
        import matplotlib.pyplot as plt
        plt.ion()
        fig, ax = plt.subplots(figsize=(8, 5))
        fig.patch.set_facecolor("#1e1e1e")
        ax.set_facecolor("#252525")
        ax.grid(True, color="#444444", linewidth=0.5)
        ax.tick_params(colors="#cccccc")
        for spine in ax.spines.values():
            spine.set_color("#555555")
        line, = ax.plot([], [], color="#4fd1c5", linewidth=2)
        fig.tight_layout()
        plot_state = (plt, fig, ax, line)

    history = []

    if args.once:
        process_text(args.once, args, history, log_enabled, 0.0, plot_state)
        return

    if not args.pipe:
        print("\n── Live CAS (Whisper → SymPy, 2 steps, LOCAL via Ollama) ──", flush=True)
        print(f"   extraction model: {args.model} @ {OLLAMA_URL}", flush=True)
        print("   Jalali photonics · time stretch · dispersive Fourier · GVD", flush=True)
        print("   Ctrl-C to quit\n", flush=True)

    get_whisper()   # load before the first utterance so latency is visible now
    calibrate_gate()   # place the speech gate just above this room's noise floor

    while True:
        if not args.pipe:
            print("🎤  listening…", flush=True)

        audio = record_utterance()
        if audio is None:
            continue
        duration = len(audio) / SAMPLE_RATE

        text = transcribe(audio)
        if not text:
            continue

        process_text(text, args, history, log_enabled, duration, plot_state)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nbye", flush=True)
    except BrokenPipeError:
        os._exit(0)
