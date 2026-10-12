"""Physics models used by the notebook. SI units throughout.

    photon energy        E = h c / lambda                      [J]
    Beer-Lambert         I = I0 exp(-alpha L)                  [W/m^2]
    RC step response     v(t) = V (1 - exp(-t / (R C)))        [V]
    series RLC impedance Z(f) = R + j(2 pi f L - 1/(2 pi f C)) [ohm]
    reflection           Gamma = (Z - Z0) / (Z + Z0)           [-]
    return loss          RL = -20 log10 |Gamma|                [dB]
    power                P = V^2 / R                           [W]

Each closed form has a SymPy derivation (functions ending in _symbolic) so the
notebook can show where it comes from, and a NumPy evaluator for data.
"""
import numpy as np
import sympy as sp

H_PLANCK = 6.62607015e-34       # J s (exact, SI 2019)
C_LIGHT = 299792458.0           # m/s (exact)
E_CHARGE = 1.602176634e-19      # C (exact)


def _positive(name, value):
    if np.any(np.asarray(value) <= 0):
        raise ValueError(f"{name} must be positive")


# ── closed forms ─────────────────────────────────────────────────────

def photon_energy_J(wavelength_m):
    _positive("wavelength_m", wavelength_m)
    return H_PLANCK * C_LIGHT / np.asarray(wavelength_m, float)


def photon_energy_eV(wavelength_m):
    return photon_energy_J(wavelength_m) / E_CHARGE


def beer_lambert(I0, alpha_per_m, length_m):
    if np.any(np.asarray(alpha_per_m) < 0) or np.any(np.asarray(length_m) < 0):
        raise ValueError("alpha and length must be non-negative")
    return np.asarray(I0, float) * np.exp(-np.asarray(alpha_per_m) * np.asarray(length_m))


def rc_step(t_s, V, R, C):
    _positive("R", R)
    _positive("C", C)
    t = np.asarray(t_s, float)
    return np.where(t >= 0, V * (1 - np.exp(-t / (R * C))), 0.0)


def series_rlc_impedance(f_hz, R, L, C):
    _positive("f_hz", f_hz)
    for n, v in (("R", R), ("L", L), ("C", C)):
        _positive(n, v)
    w = 2 * np.pi * np.asarray(f_hz, float)
    return R + 1j * (w * L - 1 / (w * C))


def resonance_hz(L, C):
    _positive("L", L)
    _positive("C", C)
    return 1 / (2 * np.pi * np.sqrt(L * C))


def quality_factor(R, L, C):
    return np.sqrt(L / C) / R


def reflection_coefficient(Z, Z0=50.0):
    _positive("Z0", Z0)
    Z = np.asarray(Z, complex)
    return (Z - Z0) / (Z + Z0)


def return_loss_db(gamma):
    g = np.abs(np.asarray(gamma, complex))
    return -20 * np.log10(np.maximum(g, 1e-12))


def power_W(V, R):
    _positive("R", R)
    return np.asarray(V, float) ** 2 / R


# ── SymPy derivations ────────────────────────────────────────────────

def photon_energy_symbolic():
    """E = h f and f = c / lambda -> E = h c / lambda."""
    h, c, lam, f = sp.symbols("h c lambda f", positive=True)
    E = (h * f).subs(f, c / lam)
    return sp.Eq(sp.Symbol("E"), E)


def beer_lambert_symbolic():
    """dI/dx = -alpha I, I(0) = I0  ->  I(L) = I0 exp(-alpha L)."""
    x, alpha, I0, L = sp.symbols("x alpha I_0 L", positive=True)
    I = sp.Function("I")
    sol = sp.dsolve(sp.Eq(I(x).diff(x), -alpha * I(x)), ics={I(0): I0})
    return sol.subs(x, L)


def rc_symbolic():
    """V = R C dv/dt + v, v(0) = 0  ->  v(t) = V (1 - exp(-t / RC))."""
    t, R, C, V = sp.symbols("t R C V", positive=True)
    v = sp.Function("v")
    return sp.dsolve(sp.Eq(R * C * v(t).diff(t) + v(t), V), ics={v(0): 0})


def rlc_impedance_symbolic():
    """Series R, L, C with d/dt -> j omega; resonance where Im Z = 0."""
    R, L, C, w = sp.symbols("R L C omega", positive=True)
    Z = R + sp.I * w * L + 1 / (sp.I * w * C)
    w0 = [s for s in sp.solve(sp.im(sp.expand_complex(Z)), w) if s.is_positive][0]
    return sp.simplify(Z), sp.simplify(w0)


def power_symbolic():
    """P = V I with Ohm's law I = V / R."""
    V, R, I = sp.symbols("V R I", positive=True)
    return sp.Eq(sp.Symbol("P"), (V * I).subs(I, V / R))


def reference_features():
    """The same deterministic quantities computed by matlab/optical_rf_reference.m,
    keyed by the same names, for cross-checking the CSV it exports."""
    out = {}
    for lam in (405, 532, 633, 850, 1310, 1550):
        out[f"photon_energy_eV_{lam}nm"] = float(photon_energy_eV(lam * 1e-9))
    for L in (0, 0.005, 0.01, 0.02, 0.05, 0.1):
        out[f"beer_lambert_I_L{L:g}m"] = float(beer_lambert(1.0, 120.0, L))
    R, f0, Q = 45.0, 2.0e9, 30.0
    Lh = Q * R / (2 * np.pi * f0)
    C = 1 / ((2 * np.pi * f0) ** 2 * Lh)
    f = np.linspace(1e9, 3e9, 4001)
    g = np.abs(reflection_coefficient(series_rlc_impedance(f, R, Lh, C)))
    k = int(np.argmin(g))
    out.update({"rlc_resonance_hz": float(resonance_hz(Lh, C)), "rlc_Q": float(quality_factor(R, Lh, C)),
                "rlc_min_abs_gamma": float(g[k]), "rlc_f_at_min_hz": float(f[k]),
                "rlc_return_loss_db": float(return_loss_db(g[k]))})
    P = float(power_W(5.0, 100.0))
    out.update({"power_W": P, "power_sigma_W": P * float(np.sqrt((2 * 0.05 / 5.0) ** 2 + (1.0 / 100.0) ** 2))})
    return out
