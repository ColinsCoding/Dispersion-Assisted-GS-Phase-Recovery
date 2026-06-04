"""
_repl_particles.py -- standard model particles: formal names, properties, SymPy
"""
import numpy as np
import pandas as pd
import sympy as sp

# ── 1. Standard Model particle table ─────────────────────────────────────────
print("=== Standard Model Particles ===")

particles = pd.DataFrame([
    # name,          symbol, mass_MeV,   charge, spin, category
    ('electron',     'e-',   0.511,      -1,     0.5,  'lepton'),
    ('muon',         'mu-',  105.66,     -1,     0.5,  'lepton'),
    ('tau',          'tau-', 1776.86,    -1,     0.5,  'lepton'),
    ('e-neutrino',   'nu_e', 2.2e-6,      0,     0.5,  'lepton'),
    ('up quark',     'u',    2.2,         2/3,   0.5,  'quark'),
    ('down quark',   'd',    4.7,        -1/3,   0.5,  'quark'),
    ('charm quark',  'c',    1275.0,      2/3,   0.5,  'quark'),
    ('strange quark','s',    95.0,       -1/3,   0.5,  'quark'),
    ('top quark',    't',    172760.0,    2/3,   0.5,  'quark'),
    ('bottom quark', 'b',    4180.0,     -1/3,   0.5,  'quark'),
    ('photon',       'gamma', 0.0,         0,     1.0,  'boson'),
    ('W boson',      'W+-',  80379.0,    1,      1.0,  'boson'),
    ('Z boson',      'Z0',   91187.6,    0,      1.0,  'boson'),
    ('gluon',        'g',    0.0,         0,     1.0,  'boson'),
    ('Higgs',        'H0',   125250.0,    0,     0.0,  'boson'),
    ('proton',       'p',    938.272,     1,     0.5,  'hadron'),
    ('neutron',      'n',    939.565,     0,     0.5,  'hadron'),
], columns=['name','symbol','mass_MeV','charge','spin','category'])

# fix W mass display
particles.loc[particles.name=='W boson', 'mass_MeV'] = 80379.0

print(particles[['name','symbol','mass_MeV','charge','spin','category']].to_string(index=False))
print()

# ── 2. Conservation laws per interaction ──────────────────────────────────────
print("=== Conservation Laws ===")
laws = pd.DataFrame([
    ('Strong',        True,  True,  True,  True,  True,  True),
    ('Electromagnetic', True, True, True,  True,  False, True),
    ('Weak',          True,  True,  True,  False, False, True),
    ('Gravity',       True,  True,  False, False, False, True),
], columns=['Force','Energy','Momentum','Charge','Parity','Flavor','Baryon#'])
print(laws.to_string(index=False))
print()

# ── 3. SymPy: relativistic energy-momentum relation ──────────────────────────
print("=== SymPy: E^2 = (pc)^2 + (mc^2)^2 ===")
E, p, m, c = sp.symbols('E p m c', positive=True)

relation = sp.Eq(E**2, (p*c)**2 + (m*c**2)**2)
print("Relation:", relation)

# solve for E
E_sol = sp.solve(relation, E)[0]
print("E =", E_sol)

# non-relativistic limit: p << mc, Taylor expand
x = sp.Symbol('x')  # x = p/(mc)
E_nr = sp.series(sp.sqrt(1 + x**2), x, 0, 4)
print("Non-relativistic expansion (p/mc << 1):")
print("  E/mc^2 =", E_nr)
print("  -> E ~= mc^2 + p^2/2m  (rest energy + kinetic)")
print()

# ── 4. de Broglie wavelength sweep ───────────────────────────────────────────
print("=== de Broglie wavelength: lambda = h/p ===")
h  = 6.626e-34   # J*s
eV = 1.602e-19   # J
me = 9.109e-31   # kg electron mass
mp = 1.673e-27   # kg proton mass
c_light = 3e8

data = []
for name, mass_kg, KE_eV in [
    ('electron  1 eV',   me,  1),
    ('electron  1 keV',  me,  1e3),
    ('electron  1 MeV',  me,  1e6),
    ('proton    1 keV',  mp,  1e3),
    ('proton    1 MeV',  mp,  1e6),
    ('proton    1 GeV',  mp,  1e9),
]:
    KE_J = KE_eV * eV
    # relativistic: p = sqrt((KE+mc^2)^2-mc^4)/c
    mc2  = mass_kg * c_light**2
    p_kg = np.sqrt((KE_J + mc2)**2 - mc2**2) / c_light
    lam  = h / p_kg
    data.append((name, KE_eV, p_kg, lam))

df_dB = pd.DataFrame(data, columns=['particle','KE_eV','p_kgms','lambda_m'])
df_dB['lambda_pm'] = df_dB['lambda_m'] * 1e12   # picometers
df_dB['lambda_nm'] = df_dB['lambda_m'] * 1e9
print(df_dB[['particle','KE_eV','lambda_pm','lambda_nm']].to_string(index=False))
print()

# ── 5. Voltage & E-field from particle charge ─────────────────────────────────
print("=== Coulomb potential V(r) = kq/r ===")
k_c = 8.988e9   # N*m^2/C^2
q_e = 1.602e-19 # C (proton charge)

r_vals = np.array([1e-15, 1e-10, 1e-9, 1e-6, 1e-3])  # fm to mm
labels = ['1 fm (nuclear)', '1 A (atomic)', '1 nm', '1 um', '1 mm']

print(f"  {'r':>20}  {'V (proton) [V]':>18}  {'E-field [V/m]':>16}")
for r, lab in zip(r_vals, labels):
    V  = k_c * q_e / r
    Ef = k_c * q_e / r**2
    print(f"  {lab:>20}  {V:>18.3e}  {Ef:>16.3e}")
print()

# ── 6. Loop: random 2D particle scatter (x,y) with momentum (vx,vy) ──────────
print("=== Random particle scatter (N=10) ===")
rng = np.random.default_rng(7)
N_p = 10
x,  y  = rng.uniform(-1, 1, N_p), rng.uniform(-1, 1, N_p)
vx, vy = rng.normal(0, 1, N_p),   rng.normal(0, 1, N_p)
m_p    = rng.choice([me, mp], N_p)  # mix electrons & protons

KE = 0.5 * m_p * (vx**2 + vy**2)
p_mag = m_p * np.sqrt(vx**2 + vy**2)

df_sc = pd.DataFrame({'x':x.round(3), 'y':y.round(3),
                      'vx':vx.round(3), 'vy':vy.round(3),
                      'mass': ['e-' if mm==me else 'p' for mm in m_p],
                      'KE_J': KE, 'p_kgms': p_mag})
print(df_sc.to_string(index=False))
print()
print("Total KE:", df_sc['KE_J'].sum())
print("Total |p|:", df_sc['p_kgms'].sum())
