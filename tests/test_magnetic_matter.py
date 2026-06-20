"""Smoke-test griffiths.magnetic_matter against known Griffiths Ch.6 results
and the Ch.4 dielectric duality."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import sympy as sp
from griffiths import magnetic_matter as mm
from griffiths import magnetostatics as mag
from griffiths import dielectrics as di

M, chi, B0, C, T = sp.symbols("M chi_m B_0 C T", positive=True)
mu_r = sp.Symbol("mu_r", positive=True)
theta = sp.Symbol("theta")

# bound surface current of M along z: K_b = M sin(theta)
print("K_b =", mm.bound_surface_current(M, theta), "(expect M sin theta)")

# linear medium: mu_r = 1 + chi_m
print("mu_r =", mm.permeability_constant(chi), "(expect 1 + chi_m)")
print("classify:", mm.classify_magnetic(-0.001), "|", mm.classify_magnetic(0.001),
      "|", mm.classify_magnetic(5000))

# uniformly magnetized sphere: B_in = 2/3 mu0 M, H_in = -M/3
sph = mm.uniformly_magnetized_sphere(M)
print("\nuniformly magnetized sphere: B_in =", sph["B_in"], " H_in =", sph["H_in"])
print("  B_in == 2/3 mu0 M ?", sp.simplify(sph["B_in"] - sp.Rational(2,3)*mag.mu0*M) == 0)
print("  H_in == -M/3 ?", sp.simplify(sph["H_in"] + M/3) == 0)

# magnetizable sphere in field: H_in/H0 = 3/(mu_r+2), B_in/B0 = 3 mu_r/(mu_r+2)
res = mm.magnetizable_sphere_in_field(mu_r, B0)
print("\nmagnetizable sphere: H_in/H0 =", res["H_in_over_H0"], " B_in/B0 =", res["B_in_over_B0"])
print("  H_in/H0 == 3/(mu_r+2) ?", sp.simplify(res["H_in_over_H0"] - 3/(mu_r+2)) == 0)
print("  B_in/B0 == 3 mu_r/(mu_r+2) ?", sp.simplify(res["B_in_over_B0"] - 3*mu_r/(mu_r+2)) == 0)
# limits: mu_r->1 trivial; mu_r->oo: B concentrates (3 B0), H expelled (0)
print("  mu_r->1: B", sp.limit(res["B_in_over_B0"], mu_r, 1), " H", sp.limit(res["H_in_over_H0"], mu_r, 1))
print("  mu_r->oo: B", sp.limit(res["B_in_over_B0"], mu_r, sp.oo), " H", sp.limit(res["H_in_over_H0"], mu_r, sp.oo))

# DUALITY: H field of magnetic sphere == E field of dielectric sphere (eps_r<->mu_r)
eps_r = sp.Symbol("epsilon_r", positive=True)
E0 = sp.Symbol("E_0", positive=True)
diel = di.dielectric_sphere_in_field(eps_r, E0)
print("\nduality check: dielectric E_in/E0 =", diel["E_in_over_E0"],
      " == magnetic H_in/H0 with eps_r->mu_r ?",
      sp.simplify(diel["E_in_over_E0"].subs(eps_r, mu_r) - res["H_in_over_H0"]) == 0)

# Curie law
print("\nCurie law chi_m = C/T:", mm.curie_law(C, T))

# validation
for bad in [lambda: mm.curie_law(C, -1),
            lambda: mm.classify_magnetic(chi)]:
    try:
        bad()
    except ValueError as e:
        print("err ok:", str(e)[:45])
print("SMOKE PASS")
