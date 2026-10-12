"""Neutron beta decay  n -> p + e- + anti-nu  (Knight Ch. 37 adaptive question),
done rigorously in SymPy: exact rationals, unit-checked conversions, and a
diagnosis of the mistakes that land on each wrong multiple-choice option.

Energy conservation (neutron at rest, m_nu = 0):
    K_total = (m_n - m_p - m_e) c^2 = Delta_m c^2  = Q   (~0.782 MeV)

Three-body kinematics (exact, relativistic, m_nu = 0):
    max electron total energy  E_e,max = (m_n^2 + m_e^2 - m_p^2) c^2 / (2 m_n)
    max proton recoil KE       T_p,max = ((m_n - m_p)^2 - m_e^2) c^2 / (2 m_n)   (~0.75 keV)
The proton is ~1836x heavier than the electron, so it carries almost no
kinetic energy -- which is why "close to 1 keV" is a real number in this
problem, but the wrong one for the TOTAL.
"""
import numpy as np
import sympy as sp
from sympy.physics.units import kilogram, speed_of_light, electronvolt, joule, convert_to

# The question's data, as exact rationals (no float rounding before the subtraction)
M_N = sp.Rational("1.6749274e-27")
M_P = sp.Rational("1.6726219e-27")
M_E = sp.Rational("0.0009109e-27")
C = sp.Integer(299792458)                       # exact by SI definition
EV = sp.Rational("1.602176634e-19")             # exact by SI definition

OPTIONS = {"100 MeV": 1e8, "10 eV": 1e1, "1 keV": 1e3, "1 MeV": 1e6, "30 keV": 3e4}


def symbolic_q_value():
    """Solve m_n c^2 = sum(m c^2 + K) for K_total symbolically."""
    mn, mp, me, c, K = sp.symbols("m_n m_p m_e c K_total", positive=True)
    eq = sp.Eq(mn * c ** 2, mp * c ** 2 + me * c ** 2 + K)
    return sp.solve(eq, K)[0], (mn, mp, me, c)


def delta_m():
    return M_N - M_P - M_E


def q_value_joule():
    return delta_m() * C ** 2


def q_value_ev():
    return q_value_joule() / EV


def q_value_units():
    """Same calculation carried with sympy.physics.units (catches unit mistakes)."""
    E = delta_m() * kilogram * speed_of_light ** 2
    return convert_to(E, electronvolt), convert_to(E, joule)


def nearest_option(energy_ev):
    """Closest multiple-choice option on a log scale (order of magnitude)."""
    if energy_ev <= 0:
        raise ValueError("energy must be positive")
    return min(OPTIONS, key=lambda k: abs(np.log10(float(energy_ev)) - np.log10(OPTIONS[k])))


def significant_figures_lost():
    """Digits lost to cancellation: log10(m_n / Delta_m)."""
    return float(sp.log(M_N / delta_m(), 10))


def max_electron_kinetic_ev():
    Ee = (M_N ** 2 + M_E ** 2 - M_P ** 2) * C ** 2 / (2 * M_N)
    return (Ee - M_E * C ** 2) / EV


def max_proton_kinetic_ev():
    return ((M_N - M_P) ** 2 - M_E ** 2) * C ** 2 / (2 * M_N) / EV


def common_mistakes():
    """Plausible slips -> the energy each produces (eV) -> which option it picks."""
    c = C
    out = {}
    out["correct: (m_n - m_p - m_e) c^2"] = q_value_ev()
    out["forgot the electron: (m_n - m_p) c^2"] = (M_N - M_P) * c ** 2 / EV
    out["exponent slip: Delta_m = 1.3946e-33 kg"] = delta_m() * sp.Rational(1, 1000) * c ** 2 / EV
    out["divided by 1.6e-16 (J per keV) but read it as eV"] = q_value_joule() / (EV * 1000)
    out["used c, not c^2"] = delta_m() * c / EV
    out["reported proton recoil only (T_p,max)"] = max_proton_kinetic_ev()
    out["total rest energy of the neutron, m_n c^2"] = M_N * c ** 2 / EV
    return {k: (float(v), nearest_option(v)) for k, v in out.items()}


def beta_spectrum_allowed(T_ev, Q_ev=None, me_ev=None):
    """Allowed-shape electron spectrum (no Coulomb/Fermi correction):
    dN/dT ~ p E (Q - T)^2, with E = T + m_e c^2, p c = sqrt(E^2 - (m_e c^2)^2)."""
    Q = float(q_value_ev()) if Q_ev is None else Q_ev
    me = float(M_E * C ** 2 / EV) if me_ev is None else me_ev
    T = np.asarray(T_ev, float)
    E = T + me
    p = np.sqrt(np.clip(E ** 2 - me ** 2, 0, None))
    return np.where((T >= 0) & (T <= Q), p * E * (Q - T) ** 2, 0.0)


def endpoint_sum_identity():
    """SymPy proof (c = 1): T_e,max + T_p,max = m_n - m_p - m_e = Q exactly,
    even though the two maxima occur in different decay configurations."""
    mn, mp, me = sp.symbols("m_n m_p m_e", positive=True)
    Te = (mn ** 2 + me ** 2 - mp ** 2) / (2 * mn) - me
    Tp = ((mn - mp) ** 2 - me ** 2) / (2 * mn)
    return sp.simplify(Te + Tp - (mn - mp - me))
