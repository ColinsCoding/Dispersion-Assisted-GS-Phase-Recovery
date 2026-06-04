"""
repl/_repl_igetc_physics.py
IGETC Physical Science: kinematics, momentum, rotation, collisions.
Special relativity. SymPy + numpy loops.
"""
import numpy as np
import sympy as sp
sp.init_printing(use_unicode=False, wrap_line=False)

print("=" * 60)
print("IGETC PHYSICAL SCIENCE -- MECHANICS + SPECIAL RELATIVITY")
print("=" * 60)
print()

# ============================================================
# 1. Kinematics: velocity and acceleration
# ============================================================
print("=== 1. Kinematics ===")
t, a_s, v0, x0 = sp.symbols('t a v_0 x_0', real=True)

x_t = x0 + v0*t + sp.Rational(1,2)*a_s*t**2
v_t = sp.diff(x_t, t)
a_t = sp.diff(v_t, t)

print("Position:     x(t) =", sp.pretty(x_t))
print("Velocity:     v(t) =", sp.pretty(v_t))
print("Acceleration: a(t) =", sp.pretty(a_t))
print()

# Big Four equations
print("The Big Four (constant acceleration):")
print("  1. v   = v0 + a*t")
print("  2. x   = x0 + v0*t + (1/2)*a*t^2")
print("  3. v^2 = v0^2 + 2*a*(x-x0)")
print("  4. x   = x0 + (v+v0)/2 * t")
print()

# numerical: projectile
g = 9.81
v0_n = 20.0   # m/s
theta = np.radians(45)
vx, vy0 = v0_n*np.cos(theta), v0_n*np.sin(theta)
t_flight = 2*vy0/g
t_arr = np.linspace(0, t_flight, 100)
x_arr = vx * t_arr
y_arr = vy0*t_arr - 0.5*g*t_arr**2

print(f"Projectile v0={v0_n} m/s theta=45 deg:")
print(f"  Range:        {x_arr[-1]:.2f} m")
print(f"  Max height:   {max(y_arr):.2f} m")
print(f"  Flight time:  {t_flight:.2f} s")
print()

# ============================================================
# 2. Momentum and impulse
# ============================================================
print("=== 2. Momentum and Impulse ===")
m1, m2, v1, v2 = sp.symbols('m1 m2 v1 v2', real=True)

p = m1 * v1
J = sp.Symbol('J')   # impulse
print("p = m*v:", sp.pretty(p))
print("J = delta_p = F*delta_t")
print("Conservation: m1*v1 + m2*v2 = const  (no external forces)")
print()

# ============================================================
# 3. Collisions
# ============================================================
print("=== 3. Collisions ===")
print("""
              Momentum    KE      Example
Elastic       conserved   conserved   billiard balls
Inelastic     conserved   LOST        car crash
Perfectly     conserved   max lost    clay sticking together
inelastic
""")

# elastic 1D collision formulas
print("Elastic 1D final velocities:")
v1f = sp.Symbol('v1f'); v2f = sp.Symbol('v2f')
# solve momentum + energy conservation
sol = sp.solve([
    sp.Eq(m1*v1 + m2*v2, m1*v1f + m2*v2f),
    sp.Eq(sp.Rational(1,2)*m1*v1**2 + sp.Rational(1,2)*m2*v2**2,
          sp.Rational(1,2)*m1*v1f**2 + sp.Rational(1,2)*m2*v2f**2)
], [v1f, v2f])
names = ['v1f', 'v2f']
for name, val in zip(names, sol[0]):
    print(f"  {name} =", sp.pretty(sp.simplify(val)))
print()

# numerical: equal mass elastic (v2=0 -> v1 stops, v2 gets all velocity)
m = 1.0; u1 = 5.0; u2 = 0.0
v1_f = (m-m)/(m+m)*u1 + 2*m/(m+m)*u2
v2_f = 2*m/(m+m)*u1 + (m-m)/(m+m)*u2
print(f"Equal mass elastic: u1={u1}, u2={u2} -> v1={v1_f:.2f}, v2={v2_f:.2f}")
print(f"  p before: {m*u1:.2f}  p after: {m*v1_f+m*v2_f:.2f}")
print(f"  KE before: {0.5*m*u1**2:.2f}  KE after: {0.5*m*v1_f**2+0.5*m*v2_f**2:.2f}")
print()

# perfectly inelastic
v_final = (m*u1 + m*u2)/(m+m)
KE_lost = 0.5*m*u1**2 - 0.5*(2*m)*v_final**2
print(f"Perfectly inelastic: v_final={v_final:.2f} m/s  KE_lost={KE_lost:.2f} J")
print()

# ============================================================
# 4. Rotation
# ============================================================
print("=== 4. Rotation ===")
omega, alpha_r, I_r, r_s = sp.symbols('omega alpha I r', positive=True)
tau = sp.Symbol('tau')

print("Rotational analogs:")
rows = [
    ("Linear",      "x",          "v=dx/dt",    "a=dv/dt",  "F=ma",    "p=mv",    "KE=(1/2)mv^2"),
    ("Rotational",  "theta",      "omega",       "alpha",    "tau=I*a", "L=I*w",   "KE=(1/2)I*w^2"),
]
for row in rows:
    print("  {:12s}  {:8s}  {:10s}  {:8s}  {:10s}  {:8s}  {}".format(*row))
print()

# moments of inertia
print("Moments of inertia I = integral r^2 dm:")
moments = [
    ("Point mass",           "m*r^2",          ""),
    ("Solid disk/cylinder",  "(1/2)*m*r^2",    "about center axis"),
    ("Hollow cylinder",      "m*r^2",          "about center axis"),
    ("Solid sphere",         "(2/5)*m*r^2",    "about diameter"),
    ("Hollow sphere",        "(2/3)*m*r^2",    "about diameter"),
    ("Rod (center)",         "(1/12)*m*L^2",   ""),
    ("Rod (end)",            "(1/3)*m*L^2",    ""),
]
for name, I_val, note in moments:
    print(f"  {name:25s}  I = {I_val:20s}  {note}")
print()

# angular momentum conservation: ice skater
I1_n = 5.0; w1_n = 1.0   # arms out
I2_n = 2.0               # arms in
w2_n = I1_n * w1_n / I2_n
print(f"Ice skater: I1={I1_n}, w1={w1_n} rad/s -> I2={I2_n}, w2={w2_n:.2f} rad/s")
print(f"  L conserved: {I1_n*w1_n:.2f} = {I2_n*w2_n:.2f}")
print(f"  KE1={0.5*I1_n*w1_n**2:.2f} J  KE2={0.5*I2_n*w2_n**2:.2f} J  (muscle does work)")
print()

# ============================================================
# 5. Special relativity
# ============================================================
print("=== 5. Special Relativity ===")
c_s, v_s, m_s = sp.symbols('c v m', positive=True)
beta  = v_s / c_s
gamma = 1 / sp.sqrt(1 - beta**2)

print("Lorentz factor:")
print("  gamma =", sp.pretty(gamma))
print()

print("The four effects:")
print("""
  Time dilation:      t' = gamma * t0          (moving clocks run slow)
  Length contraction: L' = L0 / gamma          (moving rulers shrink)
  Relativistic mass:  p  = gamma * m * v       (not m*v at high speed)
  Mass-energy:        E  = gamma * m * c^2     (rest: E0 = m*c^2)
""")

# numerical table
print("Gamma vs velocity:")
import pandas as pd
c_n = 3e8
rows2 = []
for pct in [0.1, 0.5, 0.8, 0.9, 0.99, 0.999, 0.9999]:
    v_n   = pct * c_n
    g_n   = 1/np.sqrt(1-(pct)**2)
    t_dil = g_n          # t' = gamma * t0
    L_con = 1/g_n        # L' = L0/gamma
    rows2.append({'v/c': pct, 'gamma': round(g_n,4),
                  't_dilate': round(t_dil,4), 'L_contract': round(L_con,4)})
df = pd.DataFrame(rows2)
print(df.to_string(index=False))
print()

# relativistic momentum
print("Relativistic momentum p = gamma*m*v:")
m_n = 9.109e-31   # electron
for pct in [0.1, 0.9, 0.99]:
    v_n = pct * c_n
    g_n = 1/np.sqrt(1-pct**2)
    p_rel = g_n * m_n * v_n
    p_class = m_n * v_n
    print(f"  v={pct}c  p_rel={p_rel:.3e}  p_classical={p_class:.3e}  "
          f"ratio={p_rel/p_class:.3f}")
print()

# ============================================================
# 6. Connection to your project
# ============================================================
print("=== Connection to GS phase recovery ===")
print("""
IGETC mechanics            GS / photonics equivalent
-----------------          -------------------------
v = dx/dt                  group velocity = d(omega)/dk
p = m*v                    photon momentum p = hbar*k
E = (1/2)mv^2              field energy = integral |E(t)|^2 dt
collision (elastic)        lossless beam splitter (unitary)
collision (inelastic)      lossy fiber (energy absorbed)
L = I*omega (conserved)    angular momentum of optical vortex
gamma = 1/sqrt(1-v^2/c^2)  GVD: group delay = -d phi/d omega
relativistic mass          effective photon mass in waveguide (cutoff)
""")
