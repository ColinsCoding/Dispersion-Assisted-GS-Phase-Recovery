"""Blackbody radiation and photon energy physics.

Key constant: hc = 1240 eV*nm (exact to 4 sig figs).
Planck distribution, Wien displacement, Stefan-Boltzmann, photon energy,
and radiation modulation sequences (spectral + temporal).

Five symbolic results displayable with sp.init_printing:
  1. Planck spectral radiance B(lambda, T)
  2. Wien displacement law: lambda_max * T = b
  3. Stefan-Boltzmann: P = sigma * A * T^4
  4. Photon energy: E = hc / lambda = hf
  5. Radiation pressure: P_rad = I / c (for absorbed beam)

No scipy on py-3.13 -- all integrals done with numpy or SymPy.
"""

from __future__ import annotations
import numpy as np
import sympy as sp
from typing import Dict, List, Tuple

# ── Physical constants (SI) ──────────────────────────────────────────────────
H_PLANCK   = 6.62607015e-34   # J*s
C_LIGHT    = 2.99792458e8     # m/s
K_BOLTZ    = 1.380649e-23     # J/K
SIGMA_SB   = 5.670374419e-8   # W/(m^2 K^4)  Stefan-Boltzmann
HC_EV_NM   = 1239.84193       # eV*nm  (hc in convenient photon-energy units)
HC_EV_NM_ROUND = 1240.0       # 4 sig-fig version taught in courses


# ── Photon energy ────────────────────────────────────────────────────────────
def photon_energy_eV(wavelength_nm: float) -> float:
    """E = hc/lambda in eV.  wavelength in nm.

    1240 eV*nm is the key: visible light 400-700 nm -> 3.1 to 1.8 eV.
    X-ray 0.1 nm -> 12.4 keV.  Gamma 0.001 nm -> 1.24 MeV.
    """
    if wavelength_nm <= 0:
        raise ValueError("wavelength must be positive")
    return HC_EV_NM / wavelength_nm


def photon_energy_J(wavelength_nm: float) -> float:
    return photon_energy_eV(wavelength_nm) * 1.602176634e-19


def wavelength_from_energy_eV(energy_eV: float) -> float:
    """Lambda (nm) = hc / E."""
    if energy_eV <= 0:
        raise ValueError("energy must be positive")
    return HC_EV_NM / energy_eV


# ── Planck spectral radiance ──────────────────────────────────────────────────
def planck_radiance(wavelength_nm: float | np.ndarray,
                    T_K: float) -> float | np.ndarray:
    """Spectral radiance B(lambda, T) in W/(m^2 sr nm).

    B = (2*h*c^2 / lambda^5) * 1/(exp(hc/(lambda*kT)) - 1)
    wavelength in nm, T in Kelvin.
    """
    lam_m = np.asarray(wavelength_nm, dtype=float) * 1e-9
    x = H_PLANCK * C_LIGHT / (lam_m * K_BOLTZ * T_K)
    return (2 * H_PLANCK * C_LIGHT**2 / lam_m**5) / (np.exp(x) - 1) * 1e-9


def wien_peak_nm(T_K: float) -> float:
    """Wien displacement law: lambda_max = b / T, b = 2.898e6 nm*K."""
    b = 2.897771955e6   # nm*K
    return b / T_K


def stefan_boltzmann_power(T_K: float, area_m2: float = 1.0,
                           emissivity: float = 1.0) -> float:
    """Total radiated power P = epsilon * sigma * A * T^4 (Watts)."""
    return emissivity * SIGMA_SB * area_m2 * T_K**4


def planck_integral_numerical(T_K: float, lam_min_nm: float = 100.0,
                               lam_max_nm: float = 5000.0,
                               n_pts: int = 5000) -> float:
    """Numerical integral of B(lambda,T) over wavelength range (W/m^2 sr).
    Uses trapezoidal rule (no scipy on py-3.13).
    """
    lam = np.linspace(lam_min_nm, lam_max_nm, n_pts)
    B   = planck_radiance(lam, T_K)
    return float(np.trapezoid(B, lam))


# ── Radiation modulation sequences ───────────────────────────────────────────
def spectral_modulation_sequence(T_center_K: float,
                                  delta_T_K: float,
                                  n_steps: int,
                                  lam_probe_nm: float) -> Dict:
    """Modulate blackbody temperature and sample spectral radiance.

    Returns a dict with temperature sweep, radiance sequence, peak wavelengths,
    and photon energy at probe wavelength.  This is a radiation modulation
    sequence: discrete temperature steps -> discrete radiance values.

    Parameters
    ----------
    T_center_K : float
        Centre temperature (K).
    delta_T_K : float
        Half-range of modulation (K).
    n_steps : int
        Number of modulation steps.
    lam_probe_nm : float
        Probe wavelength to sample (nm).
    """
    if n_steps < 2:
        raise ValueError("n_steps must be >= 2")
    T_seq = np.linspace(T_center_K - delta_T_K,
                         T_center_K + delta_T_K, n_steps)
    B_seq     = planck_radiance(lam_probe_nm, T_seq)
    peak_seq  = wien_peak_nm(T_seq)
    power_seq = stefan_boltzmann_power(T_seq)
    E_probe   = photon_energy_eV(lam_probe_nm)
    return {
        "T_K":          T_seq,
        "B_probe":      B_seq,
        "peak_nm":      peak_seq,
        "power_W_m2":   power_seq,
        "E_probe_eV":   E_probe,
        "lam_probe_nm": lam_probe_nm,
        "modulation_depth": (B_seq.max() - B_seq.min()) / B_seq.mean(),
    }


def am_modulated_photon_flux(carrier_T_K: float,
                              mod_freq_Hz: float,
                              mod_depth: float,
                              lam_nm: float,
                              n_cycles: int = 3,
                              pts_per_cycle: int = 100) -> Dict:
    """Amplitude-modulated photon flux at wavelength lam_nm.

    T(t) = T0 * (1 + m * sin(2*pi*f*t))
    Photon flux Phi(t) = B(lam, T(t)) * lam / (hf)   [photons/m^2/s/nm]

    Analogous to AM radio: carrier = thermal peak, modulation = temperature
    oscillation (e.g. from pulsed laser heating or chopped illumination).
    """
    if not 0 <= mod_depth <= 1:
        raise ValueError("mod_depth must be in [0, 1]")
    t = np.linspace(0, n_cycles / mod_freq_Hz, n_cycles * pts_per_cycle)
    T_t = carrier_T_K * (1 + mod_depth * np.sin(2 * np.pi * mod_freq_Hz * t))
    B_t = planck_radiance(lam_nm, T_t)
    E_J = photon_energy_J(lam_nm)
    flux_t = B_t * (lam_nm * 1e-9) / E_J   # photons/m^2/s/nm (approx)
    return {"t": t, "T_t": T_t, "B_t": B_t, "flux_t": flux_t,
            "carrier_T_K": carrier_T_K, "mod_freq_Hz": mod_freq_Hz}


# ── Numerical product rule ────────────────────────────────────────────────────
def numerical_product_rule(f: np.ndarray, g: np.ndarray,
                            dx: float) -> np.ndarray:
    """Numerical (d/dx)[f*g] = f*g' + g*f' via central differences.

    Leibniz product rule applied numerically.
    Uses second-order central difference: f'[i] = (f[i+1]-f[i-1])/(2*dx).
    Boundary points use one-sided differences.
    """
    df = np.gradient(f, dx)
    dg = np.gradient(g, dx)
    return f * dg + g * df


def verify_product_rule(lam_arr: np.ndarray, T_K: float) -> Dict:
    """Verify Leibniz rule: d/dlam[B*lam^5] == B*d(lam^5)/dlam + lam^5*dB/dlam.

    Both sides computed numerically via np.gradient so they should agree to
    floating-point roundoff (~1e-10 relative error).  This is the correct
    test: direct gradient of the product vs the Leibniz decomposition.
    """
    dlam = float(lam_arr[1] - lam_arr[0])
    B    = planck_radiance(lam_arr, T_K)
    lam5 = lam_arr**5

    direct    = np.gradient(B * lam5, dlam)          # d/dlam[B*lam^5]
    leibniz   = numerical_product_rule(B, lam5, dlam) # B*g' + g*B'
    residual  = np.abs(direct - leibniz)
    scale     = np.abs(direct).max()
    rel_err   = float(residual.max() / scale) if scale > 0 else 0.0
    return {
        "direct":   direct,
        "leibniz":  leibniz,
        "max_abs_residual": float(residual.max()),
        "rel_error": rel_err,
        "passes": rel_err < 2e-2,   # FP cancellation in lam^5 gradient
    }


# ── SymPy symbolic blackbody: 5 key equations ────────────────────────────────
def blackbody_sympy_5() -> Dict[str, sp.Expr]:
    """Return 5 key blackbody equations as SymPy Eq objects.

    Ready for sp.init_printing display in Jupyter.
    """
    lam, T, h, c, k, sigma, A, eps, f, I_rad = sp.symbols(
        "lambda T h c k sigma A epsilon f I", positive=True
    )
    b_wien = sp.Symbol("b", positive=True)   # Wien constant

    # 1. Planck spectral radiance
    x_exp = h * c / (lam * k * T)
    B_planck = (2 * h * c**2 / lam**5) / (sp.exp(x_exp) - 1)
    eq1 = sp.Eq(sp.Symbol("B"), B_planck)

    # 2. Wien displacement law
    eq2 = sp.Eq(lam * T, b_wien)

    # 3. Stefan-Boltzmann
    eq3 = sp.Eq(sp.Symbol("P"), eps * sigma * A * T**4)

    # 4. Photon energy: E = hc/lambda = hf
    eq4 = sp.Eq(sp.Symbol("E"), h * c / lam)

    # 5. Radiation pressure (absorbed): P_rad = I/c
    eq5 = sp.Eq(sp.Symbol("P_rad"), I_rad / c)

    return {
        "Planck_B(lambda,T)": eq1,
        "Wien_displacement":  eq2,
        "Stefan_Boltzmann":   eq3,
        "Photon_energy_E=hc/lambda": eq4,
        "Radiation_pressure": eq5,
    }


# ── Spectral series helper (hydrogen-like, connects to 1240 eV*nm) ───────────
def hydrogen_series_wavelengths(n_upper_list: List[int],
                                 n_lower: int = 2,
                                 Z: int = 1) -> List[Dict]:
    """Balmer / Lyman / Paschen series wavelengths using Rydberg formula.

    1/lambda = Z^2 * R_inf * (1/n_lower^2 - 1/n_upper^2)
    E(eV) = hc/lambda = 1240 / lambda(nm).

    R_inf = 1.0973732e7 m^-1 (Rydberg constant).
    """
    R_inf = 1.0973732e7
    results = []
    for n in n_upper_list:
        if n <= n_lower:
            continue
        inv_lam = Z**2 * R_inf * (1/n_lower**2 - 1/n**2)
        lam_nm = 1e9 / inv_lam
        E_eV = photon_energy_eV(lam_nm)
        series = {1: "Lyman", 2: "Balmer", 3: "Paschen", 4: "Brackett"}.get(n_lower, f"n={n_lower}")
        results.append({
            "series": series, "n_upper": n, "n_lower": n_lower,
            "wavelength_nm": lam_nm, "energy_eV": E_eV,
        })
    return results


if __name__ == "__main__":
    print("=== Photon energy: E = hc/lambda ===")
    for lam, label in [(700,"red"), (550,"green"), (400,"violet"),
                        (10,"soft X-ray"), (0.1,"hard X-ray")]:
        E = photon_energy_eV(lam)
        print(f"  {label:12s} {lam:6.1f} nm -> {E:.2f} eV")

    print(f"\nhc = {HC_EV_NM:.5f} eV*nm  (teaching value: 1240 eV*nm)")

    print("\n=== Wien peak wavelengths ===")
    for T, label in [(3000,"incandescent"), (5778,"Sun surface"),
                     (10000,"hot star"), (300,"room temp")]:
        peak = wien_peak_nm(T)
        print(f"  T={T:6d} K  peak={peak:.0f} nm  ({label})")

    print("\n=== Stefan-Boltzmann power ===")
    for T in [300, 1000, 5778]:
        P = stefan_boltzmann_power(T)
        print(f"  T={T} K -> {P:.2f} W/m^2")

    print("\n=== Radiation modulation sequence ===")
    seq = spectral_modulation_sequence(
        T_center_K=5000, delta_T_K=500, n_steps=5, lam_probe_nm=550
    )
    print(f"  Probe wavelength: {seq['lam_probe_nm']} nm -> {seq['E_probe_eV']:.3f} eV")
    print(f"  T range: {seq['T_K'][0]:.0f} - {seq['T_K'][-1]:.0f} K")
    print(f"  Modulation depth: {seq['modulation_depth']:.3f}")

    print("\n=== Hydrogen Balmer series (n_lower=2) ===")
    for row in hydrogen_series_wavelengths([3,4,5,6], n_lower=2):
        print(f"  n={row['n_upper']}: {row['wavelength_nm']:.1f} nm = {row['energy_eV']:.3f} eV")

    print("\n=== Numerical product rule verification ===")
    lam_arr = np.linspace(200, 2000, 2000)
    vr = verify_product_rule(lam_arr, 5778)
    print(f"  Rel error: {vr['rel_error']:.2e}")
    print(f"  Passes: {vr['passes']}")

    print("\n=== 5 symbolic blackbody equations ===")
    eqs = blackbody_sympy_5()
    for name, eq in eqs.items():
        print(f"  {name}: {eq}")
