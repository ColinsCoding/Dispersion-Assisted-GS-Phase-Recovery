"""Smoke-test griffiths.dielectrics against known Griffiths Ch.4 answers."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import sympy as sp
from griffiths import dielectrics as di
from griffiths import electrostatics as es

P, a, E0, chi = sp.symbols("P a E_0 chi_e", positive=True)
eps_r = sp.Symbol("epsilon_r", positive=True)
theta = sp.Symbol("theta")

# 4.1 polarizability of a sphere: alpha = 4 pi eps0 a^3
alpha = di.polarizability_sphere(a)
print("4.1 alpha =", alpha, " == 4 pi eps0 a^3 ?",
      sp.simplify(alpha - 4*sp.pi*es.eps0*a**3) == 0)

# bound surface charge of P along z: sigma_b = P cos theta (a P_1 distribution)
print("sigma_b =", di.bound_surface_charge(P, theta))

# uniformly polarized sphere: E_in = -P/(3 eps0)
E_in = di.uniformly_polarized_sphere(P)
print("\nuniformly polarized sphere E_in =", E_in, " == -P/(3 eps0) ?",
      sp.simplify(E_in + P/(3*es.eps0)) == 0)

# linear dielectric: eps_r = 1 + chi_e
print("\neps_r =", di.dielectric_constant(chi), " (expect 1 + chi_e)")

# dielectric sphere in uniform field: E_in/E0 = 3/(eps_r+2)
res = di.dielectric_sphere_in_field(eps_r, E0)
print("\ndielectric sphere: A =", res["A"], " B =", res["B"])
print("  E_in =", res["E_in"])
print("  E_in/E0 =", res["E_in_over_E0"], " == 3/(eps_r+2) ?",
      sp.simplify(res["E_in_over_E0"] - 3/(eps_r+2)) == 0)
# sanity: eps_r -> 1 gives E_in = E0 (vacuum); eps_r -> oo gives E_in -> 0 (conductor)
print("  eps_r->1:", sp.limit(res["E_in_over_E0"], eps_r, 1),
      " eps_r->oo:", sp.limit(res["E_in_over_E0"], eps_r, sp.oo))

# Clausius-Mossotti round trip
N = sp.Symbol("N", positive=True)
er_cm = di.clausius_mossotti(N, alpha)
print("\nClausius-Mossotti eps_r =", sp.simplify(er_cm))

# capacitor with dielectric
C0 = sp.Symbol("C_0", positive=True)
print("capacitor: C =", di.capacitor_with_dielectric(eps_r, C0), "(expect eps_r C0)")

# validation
try:
    di.capacitor_with_dielectric(sp.Rational(1, 2), C0)
except ValueError as e:
    print("err ok:", e)
print("SMOKE PASS")
