"""Charge-to-mass ratio (Knight Ch. 37.4): which of 2H+, p, 6Li3+, alpha is the odd one out?

TEXTBOOK LEVEL: q/m ~ Z_charge / A (mass number). 2H+, 6Li3+ and alpha all give
1/2 e/u; the proton gives 1 e/u -> the proton is different.

PRECISION LEVEL: real nuclear masses are not A u -- nuclear binding energy
(the mass defect, E = mc^2) makes them lighter, by different amounts. So the
three "1/2" ions differ at the ~0.6 % level; the alpha, the most tightly bound,
has the largest q/m of the three. Masses: nuclear masses from CODATA 2018 [1]
(p, d, alpha) and the AME2020 atomic mass of 6Li [2] minus three electrons.

THOMSON'S MEASUREMENT [3]: crossed fields E (V/m) and B select v = E/B; then
the deflection in E alone gives q/m = 2 y v^2 / (E L^2).

References (IEEE style)
[1] E. Tiesinga, P. J. Mohr, D. B. Newell, and B. N. Taylor, "CODATA recommended values
    of the fundamental physical constants: 2018," Rev. Mod. Phys., vol. 93, no. 2,
    Art. no. 025010, 2021.
[2] M. Wang, W. J. Huang, F. G. Kondev, G. Audi, and S. Naimi, "The AME 2020 atomic mass
    evaluation (II). Tables, graphs and references," Chin. Phys. C, vol. 45, no. 3,
    Art. no. 030003, 2021.
[3] J. J. Thomson, "Cathode rays," Philos. Mag., vol. 44, no. 269, pp. 293-316, 1897.
"""
import numpy as np
import sympy as sp

E_CHARGE = 1.602176634e-19       # C (exact)
U = 1.66053906660e-27            # kg per u  [1]
M_E_U = 5.48579909065e-4         # electron mass in u  [1]

# nuclear masses in u
M_PROTON_U = 1.007276466621       # [1]
M_DEUTERON_U = 2.013553212745     # [1]
M_ALPHA_U = 4.001506179127        # [1]
M_LI6_ATOM_U = 6.0151228874       # [2] atomic mass; nucleus = atom - 3 m_e (+ ~2e-7 u e-binding, neglected)

PARTICLES = {
    "2H+ ion (deuteron)": dict(q=1, A=2, m_u=M_DEUTERON_U),
    "proton": dict(q=1, A=1, m_u=M_PROTON_U),
    "6Li3+ ion": dict(q=3, A=6, m_u=M_LI6_ATOM_U - 3 * M_E_U),
    "alpha particle": dict(q=2, A=4, m_u=M_ALPHA_U),
}


def textbook_ratio(name):
    """q/A as an exact SymPy Rational (units e/u)."""
    p = PARTICLES[name]
    return sp.Rational(p["q"], p["A"])


def precise_ratio(name):
    """q/m with real nuclear masses, e/u."""
    p = PARTICLES[name]
    return p["q"] / p["m_u"]


def ratio_si(name):
    """q/m in C/kg."""
    return precise_ratio(name) * E_CHARGE / U


def odd_one_out(ratios):
    """The key whose value differs from all the others (None if no unique outlier)."""
    vals = list(ratios.values())
    out = [k for k, v in ratios.items() if sum(v == w for w in vals) == 1]
    return out[0] if len(out) == 1 and len(set(vals)) == 2 else None


def binding_energy_MeV(Z, N, m_nucleus_u):
    """B = (Z m_p + N m_n - m_nucleus) c^2, with m_n from [1]."""
    m_n = 1.00866491595
    return (Z * M_PROTON_U + N * m_n - m_nucleus_u) * 931.49410242


# ── Thomson's crossed-field measurement ──────────────────────────────

def thomson_symbolic():
    """Velocity selector qE = qvB -> v = E/B; deflection in E over length L:
    y = (1/2)(qE/m)(L/v)^2  ->  q/m = 2 y v^2 / (E L^2) = 2 y E / (B^2 L^2)."""
    q, m, E, B, L, y, v = sp.symbols("q m E B L y v", positive=True)
    v_sel = sp.solve(sp.Eq(q * E, q * v * B), v)[0]
    y_expr = sp.Rational(1, 2) * (q * E / m) * (L / v) ** 2
    q_solved = sp.solve(sp.Eq(y, y_expr), q)[0]                 # q = 2 y m v^2 / (E L^2)
    qm = sp.simplify((q_solved / m).subs(v, v_sel))
    return v_sel, y_expr, qm


def thomson_deflection(q_over_m, E, B, L):
    """Deflection y (m) after length L in field E, for particles passing the E/B selector."""
    if min(E, B, L) <= 0:
        raise ValueError("E, B, L must be positive")
    v = E / B
    return 0.5 * q_over_m * E * (L / v) ** 2, v


# ── Millikan oil drop (Round 2 question) ─────────────────────────────

def millikan_symbolic():
    """Balance: q E = m g with E = V / d  ->  q = m g d / V."""
    q, m, g, V, d = sp.symbols("q m g V d", positive=True)
    return sp.solve(sp.Eq(q * V / d, m * g), q)[0], (m, g, V, d)


def millikan_charge(m, V, d, g=9.80):
    """Charge (C) and its multiple of e for a suspended drop."""
    if min(m, V, d, g) <= 0:
        raise ValueError("m, V, d, g must be positive")
    q = m * g * d / V
    return q, q / E_CHARGE


def is_physical_charge(n_e, tol=0.05):
    """Charge must be a whole number of e (Millikan): is n_e within tol of an integer >= 1?"""
    return round(n_e) >= 1 and abs(n_e - round(n_e)) < tol
