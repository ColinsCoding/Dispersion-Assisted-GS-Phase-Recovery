"""
dgs.photoelectric  --  Work function, photoelectric effect, EM waves in vacuum.

Classical EM is the parent theory; the photoelectric effect is the first crack
in it that forced quantum mechanics.  This module covers:

  1. Work function & stopping voltage  (Einstein 1905)
  2. EM wave in vacuum: E/B ratio, Poynting vector, irradiance
  3. Wave-equation derivation from Maxwell (curl-of-curl → wave eq)
  4. Phase velocity c = 1/sqrt(ε₀ μ₀)

All formulas are self-contained (numpy + scipy constants only).
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.constants import (
    h, hbar, c, e, epsilon_0, mu_0, m_e, k
)

# ── physical constants (explicit for readability) ───────────────────────────
eV = e                          # 1 eV in Joules


# ═══════════════════════════════════════════════════════════════════════════
# 1.  Work function & photoelectric effect
# ═══════════════════════════════════════════════════════════════════════════

# Common work functions in eV  (source: CRC Handbook)
WORK_FUNCTIONS = {
    "cesium":   2.1,
    "sodium":   2.75,
    "aluminum": 4.08,
    "copper":   4.65,
    "gold":     5.10,
    "platinum": 5.65,
    "tungsten": 4.55,
}


def threshold_frequency(phi_eV: float) -> float:
    """Minimum photon frequency that ejects an electron.

    φ = h ν₀  →  ν₀ = φ / h

    Parameters
    ----------
    phi_eV : work function in eV

    Returns
    -------
    ν₀ in Hz
    """
    return phi_eV * eV / h


def threshold_wavelength(phi_eV: float) -> float:
    """Maximum photon wavelength that ejects an electron (m)."""
    return c / threshold_frequency(phi_eV)


def max_kinetic_energy(nu_Hz: float, phi_eV: float) -> float:
    """KE_max = h ν − φ  (Einstein's photoelectric equation).

    Returns KE in Joules; negative means no emission.
    """
    return h * nu_Hz - phi_eV * eV


def stopping_voltage(nu_Hz: float, phi_eV: float) -> float:
    """Stopping potential V_stop = KE_max / e  (Volts).

    e V_stop = h ν − φ  →  V_stop = (h ν − φ) / e
    Negative result means no electron ejected.
    """
    return max_kinetic_energy(nu_Hz, phi_eV) / e


def photoelectric_summary(material: str = "sodium",
                           wavelength_nm: float = 250.0) -> dict:
    """Full photoelectric calculation for a given material and light wavelength.

    Parameters
    ----------
    material     : key into WORK_FUNCTIONS dict
    wavelength_nm: incident light wavelength in nm

    Returns
    -------
    dict with all derived quantities
    """
    if material not in WORK_FUNCTIONS:
        raise ValueError(f"Unknown material '{material}'. "
                         f"Choose from {list(WORK_FUNCTIONS)}")
    phi = WORK_FUNCTIONS[material]
    lam = wavelength_nm * 1e-9
    nu  = c / lam
    nu0 = threshold_frequency(phi)
    lam0 = threshold_wavelength(phi)
    KE   = max_kinetic_energy(nu, phi)
    Vs   = stopping_voltage(nu, phi)
    ejected = KE > 0

    return {
        "material":           material,
        "phi_eV":             phi,
        "wavelength_nm":      wavelength_nm,
        "photon_energy_eV":   h * nu / eV,
        "nu_Hz":              nu,
        "nu0_Hz":             nu0,
        "lambda0_nm":         lam0 * 1e9,
        "KE_max_eV":          KE / eV,
        "stopping_voltage_V": Vs,
        "ejected":            ejected,
    }


def plot_stopping_voltage(materials=None, nu_range_Hz=None, ax=None):
    """Plot V_stop vs photon energy for one or more materials."""
    if materials is None:
        materials = ["cesium", "sodium", "aluminum", "copper"]
    if nu_range_Hz is None:
        nu_range_Hz = np.linspace(3e14, 2e15, 400)

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 4))
    else:
        fig = ax.get_figure()

    E_eV = h * nu_range_Hz / eV

    for mat in materials:
        phi = WORK_FUNCTIONS[mat]
        Vs = np.array([max(stopping_voltage(nu, phi), 0)
                       for nu in nu_range_Hz])
        ax.plot(E_eV, Vs, label=f"{mat.capitalize()} (φ={phi} eV)")

    ax.set_xlabel("Photon energy (eV)")
    ax.set_ylabel("Stopping voltage (V)")
    ax.set_title("Photoelectric effect: stopping voltage vs photon energy")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# 2.  Classical EM: wave in vacuum
# ═══════════════════════════════════════════════════════════════════════════
#
# Maxwell's equations in vacuum (Gaussian SI):
#   ∇·E = 0        ∇·B = 0
#   ∇×E = −∂B/∂t   ∇×B = μ₀ε₀ ∂E/∂t
#
# Take curl of (∇×E):
#   ∇×(∇×E) = ∇(∇·E) − ∇²E = −∇²E   (since ∇·E=0)
#            = −∂/∂t (∇×B) = −μ₀ε₀ ∂²E/∂t²
#
# → Wave equation:  ∇²E = μ₀ε₀ ∂²E/∂t²
#   Phase velocity:  c = 1/√(μ₀ε₀)

def wave_speed_from_maxwell() -> float:
    """c = 1/√(ε₀ μ₀)  derived from Maxwell constants (m/s)."""
    return 1.0 / np.sqrt(epsilon_0 * mu_0)


def E_to_B_amplitude(E0: float, omega: float = None, k_mag: float = None) -> float:
    """B₀ = E₀ / c  for a plane wave in vacuum.

    For a plane wave E = E₀ cos(kx − ωt) ẑ,  B = E₀/c cos(kx − ωt) ŷ
    Follows from Faraday: ∂B/∂t = −∇×E → k E₀ = ω B₀ → B₀ = E₀/c.
    """
    return E0 / c


def poynting_magnitude(E0: float) -> float:
    """Time-averaged Poynting vector magnitude (irradiance) in W/m².

    ⟨S⟩ = (1/2) E₀² / (μ₀ c) = (1/2) ε₀ c E₀²
    """
    return 0.5 * epsilon_0 * c * E0**2


def plane_wave(E0: float, lam_nm: float, t: float,
               x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """E and B fields of a plane wave propagating in +x̂ at time t.

    E = E₀ cos(kx − ωt) ẑ
    B = (E₀/c) cos(kx − ωt) ŷ

    Parameters
    ----------
    E0      : amplitude (V/m)
    lam_nm  : wavelength (nm)
    t       : time (s)
    x       : spatial grid (m)

    Returns
    -------
    E_z, B_y arrays
    """
    lam = lam_nm * 1e-9
    k   = 2 * np.pi / lam
    omega = k * c
    phase = k * x - omega * t
    return E0 * np.cos(phase), (E0 / c) * np.cos(phase)


def plot_plane_wave(E0: float = 1.0, lam_nm: float = 500.0, t: float = 0.0):
    """Snapshot of E and B fields of a plane wave at time t."""
    x = np.linspace(0, 3 * lam_nm * 1e-9, 1000)
    Ez, By = plane_wave(E0, lam_nm, t, x)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
    ax1.plot(x * 1e9, Ez, color="steelblue")
    ax1.set_ylabel("E_z  (V/m)")
    ax1.set_title(f"EM plane wave in vacuum  (λ={lam_nm} nm, t={t:.2e} s)")
    ax1.axhline(0, color="k", lw=0.5)

    ax2.plot(x * 1e9, By * 1e9, color="tomato")
    ax2.set_ylabel("B_y  (nT)")
    ax2.set_xlabel("x  (nm)")
    ax2.axhline(0, color="k", lw=0.5)

    fig.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# 3.  Classical EM subset: what Maxwell gives you for free
# ═══════════════════════════════════════════════════════════════════════════

def maxwell_to_optics_summary() -> str:
    """Print the logical chain: Maxwell → wave eq → optics."""
    c_derived = wave_speed_from_maxwell()
    lines = [
        "Classical EM → Optics chain:",
        "─" * 50,
        "Maxwell (vacuum):",
        "  ∇×E = −∂B/∂t          (Faraday)",
        "  ∇×B =  μ₀ε₀ ∂E/∂t    (Ampere-Maxwell)",
        "  ∇·E = 0,  ∇·B = 0",
        "",
        "Curl-of-curl trick:",
        "  ∇²E = μ₀ε₀ ∂²E/∂t²",
        "",
        "Phase velocity:",
        f"  c = 1/√(μ₀ε₀) = {c_derived:.6e} m/s",
        f"  (NIST value:   {c:.6e} m/s)",
        "",
        "Free results (no quantum needed):",
        "  • transverse waves  (E⊥B⊥k)",
        "  • B₀ = E₀/c",
        "  • Poynting: S = (1/μ₀) E×B",
        "  • irradiance I = ½ε₀c E₀²",
        "  • reflection/refraction (Fresnel coefficients)",
        "  • polarization",
        "",
        "What classical EM cannot explain (needs quantum):",
        "  • photoelectric effect  → photon energy E=hν",
        "  • blackbody spectrum    → Planck distribution",
        "  • atomic emission lines → Bohr / QM",
        "─" * 50,
    ]
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print(maxwell_to_optics_summary())
    print()

    for mat in ["cesium", "sodium", "aluminum"]:
        r = photoelectric_summary(mat, wavelength_nm=250.0)
        tag = "EJECTED" if r["ejected"] else "no emission"
        print(f"{mat:12s}  φ={r['phi_eV']:.2f} eV  "
              f"KE_max={r['KE_max_eV']:.3f} eV  "
              f"V_stop={r['stopping_voltage_V']:.3f} V  [{tag}]")

    print(f"\nIrradiance of 1 V/m wave: {poynting_magnitude(1.0):.4f} W/m²")
    print(f"B₀ for E₀=1 V/m:         {E_to_B_amplitude(1.0):.4e} T")
