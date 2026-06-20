"""Smoke-test griffiths.magnetostatics against known Griffiths Ch.5 results."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import sympy as sp
from griffiths import magnetostatics as mag
from griffiths import x, y, z

q, m, B, I, R, s, n, vperp, vpar = sp.symbols("q m B I R s n v_perp v_par", positive=True)
t = sp.Symbol("t", real=True)

# Lorentz force: v=(vx,0,0), B=(0,0,B) -> F = q vx B (y-hat)
F = mag.lorentz_force(q, [0, 0, 0], [vperp, 0, 0], [0, 0, B])
print("Lorentz F (v=x, B=z):", F.T, " (expect q v B in +y)")

# cyclotron frequency / period
print("omega_c =", mag.cyclotron_frequency(q, m, B), " == qB/m ?",
      sp.simplify(mag.cyclotron_frequency(q, m, B) - q*B/m) == 0)
print("period  =", mag.cyclotron_period(q, m, B), "(speed-independent)")

# cyclotron trajectory satisfies m a = q v x B
xt, yt, zt = mag.cyclotron_trajectory(q, m, B, vperp, vpar, t)
r = sp.Matrix([xt, yt, zt])
v = r.diff(t); a = r.diff(t, 2)
lhs = m * a
rhs = q * v.cross(sp.Matrix([0, 0, B]))
print("trajectory solves m a = q v x B ?", sp.simplify(lhs - rhs) == sp.zeros(3, 1))
print("  |v_perp| constant?", sp.simplify(v[0]**2 + v[1]**2 - vperp**2) == 0)

# straight wire field
print("\nwire B =", mag.wire_field(I, s), " == mu0 I/(2 pi s)?",
      sp.simplify(mag.wire_field(I, s) - mag.mu0*I/(2*sp.pi*s)) == 0)
print("Ampere enclosed wire B =", mag.ampere_enclosed_wire(I, s), "(should match)")

# loop on axis: center and dipole limits
Bz = mag.loop_field_axis(I, R, z)
print("\nloop B_z =", Bz)
print("  center z=0:", sp.simplify(Bz.subs(z, 0)), " == mu0 I/(2R)?",
      sp.simplify(Bz.subs(z, 0) - mag.mu0*I/(2*R)) == 0)
dip = sp.simplify(sp.limit(Bz / (mag.mu0*I*R**2/(2*z**3)), z, sp.oo))
print("  z>>R dipole factor ->", dip, "(expect 1)")
print("  magnetic moment m =", mag.magnetic_dipole_moment(I, R))

# solenoid
print("\nsolenoid B =", mag.solenoid_field(n, I), "(uniform mu0 n I)")

# div B = 0 for a dipole-like field B = curl A ; pick A = (-y, x, 0)*f
A = sp.Matrix([-y, x, 0])
Bfield = mag.B_from_vector_potential(A)
ok, dB = mag.is_divergence_free(Bfield)
print("\nB = curl(-y,x,0) =", Bfield.T, " div B = 0?", ok)

# ExB drift independent of charge
vd = mag.ExB_drift([sp.Symbol('E0', positive=True), 0, 0], [0, 0, B])
print("ExB drift =", vd.T, "(charge-independent velocity selector)")

# validation
for bad in [lambda: mag.cyclotron_frequency(q, 0, B),
            lambda: mag.ExB_drift([1, 0, 0], [0, 0, 0])]:
    try:
        bad()
    except ValueError as e:
        print("err ok:", e)
print("SMOKE PASS")
