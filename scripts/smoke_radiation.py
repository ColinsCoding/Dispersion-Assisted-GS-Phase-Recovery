"""Smoke-test griffiths.radiation against known Griffiths Ch.11 results."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import sympy as sp
from griffiths import radiation as rad

q, a, p0, omega = sp.symbols("q a p_0 omega", positive=True)
theta = sp.Symbol("theta", positive=True)

# Larmor power P = mu0 q^2 a^2 / 6 pi c
P = rad.larmor_power(q, a)
print("Larmor P =", P, " == mu0 q^2 a^2 / 6 pi c ?",
      sp.simplify(P - rad.mu0*q**2*a**2/(6*sp.pi*rad.c)) == 0)

# the angular distribution integrates to the total Larmor power
ang = rad.larmor_angular(q, a)
total = sp.integrate(ang * sp.sin(theta), (theta, 0, sp.pi)) * 2*sp.pi
print("integral of dP/dOmega over sphere =", sp.simplify(total), " == Larmor P ?",
      sp.simplify(total - P) == 0)

# oscillating dipole average power <P> = mu0 p0^2 omega^4 / 12 pi c
Pd = rad.dipole_average_power(p0, omega)
print("\ndipole <P> =", Pd, " (omega^4 -> Rayleigh/blue sky)")

# Hertzian pattern integral = 8 pi / 3, directivity = 3/2
print("\nhertzian pattern integral over sphere:", rad.total_pattern_solid_angle("hertzian"),
      " (expect 8 pi/3)")
print("hertzian directivity D =", rad.directivity("hertzian"), " (expect 3/2)")

# half-wave dipole directivity ~ 1.64
Dhw = rad.directivity("half_wave")
print("half-wave dipole directivity D =", sp.nsimplify(Dhw, rational=False),
      "~", float(Dhw), " (expect ~1.64)")

# odd vs even multipole parity
print("\nmultipole parity: dipole(1)=", rad.multipole_parity(1),
      " quadrupole(2)=", rad.multipole_parity(2),
      " octupole(3)=", rad.multipole_parity(3))

# validation
for bad in [lambda: rad.radiation_pattern("monopole"),
            lambda: rad.multipole_parity(0)]:
    try:
        bad()
    except ValueError as e:
        print("err ok:", e)

assert sp.simplify(total - P) == 0
assert rad.directivity("hertzian") == sp.Rational(3, 2)
assert rad.total_pattern_solid_angle("hertzian") == sp.Rational(8, 3)*sp.pi
print("SMOKE PASS")
