"""
fdtd_dispersion.py
------------------
SymPy + numpy analytic dispersion model for the spiral waveguide.
No Lumerical license required — closed-form GVD from waveguide geometry.

H(ν) = exp(i·π·D·ν²)    D = β₂·L
β₂   = (λ²/2πc) · d²n_eff/dλ²

Run:  python sim/fdtd_dispersion.py
"""

import numpy as np
import sympy as sp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ── SymPy analytic GVD model ──────────────────────────────────────────────────

lam, c_light = sp.symbols("lambda c", positive=True)

# Sellmeier-like effective index for 220nm × 450nm Si wire (fitted)
# n_eff(λ) ≈ a + b/λ² + d·λ²  (λ in µm)
a_coef = sp.Rational(2303, 1000)    # 2.303
b_coef = sp.Rational(12, 100)       # 0.12 µm²
d_coef = sp.Rational(-8, 1000)      # -0.008 µm⁻²

n_eff_expr = a_coef + b_coef / lam**2 + d_coef * lam**2

# Group velocity dispersion β₂ = (λ²/2πc) · d²n_eff/dλ²
d2n_dlam2   = sp.diff(n_eff_expr, lam, 2)
beta2_expr  = (lam**2 / (2 * sp.pi * c_light)) * d2n_dlam2

beta2_func  = sp.lambdify((lam, c_light), beta2_expr, "numpy")


def beta2_ps2_per_mm(lam_nm: float) -> float:
    """
    GVD of 220nm Si wire waveguide at lam_nm.
    Returns β₂ in ps²/mm.

    c = 3×10⁵ µm/ps  (speed of light in µm/ps)
    """
    lam_um = lam_nm / 1000.0
    c_um_ps = 3e5         # µm/ps
    beta2_um_ps2 = beta2_func(lam_um, c_um_ps)   # ps²/µm
    return float(beta2_um_ps2) * 1e3              # ps²/mm


def required_length_mm(D_ps2: float, lam_nm: float = 1550.0) -> float:
    """Spiral length needed to achieve D_ps2 of total dispersion."""
    b2 = beta2_ps2_per_mm(lam_nm)
    return D_ps2 / b2


def transfer_function(nu_GHz: np.ndarray, D_ps2: float) -> np.ndarray:
    """
    H(ν) = exp(i·π·D·ν²)
    nu_GHz : frequency offset from carrier (GHz)
    D_ps2  : dispersion (ps²)
    """
    nu_THz = nu_GHz * 1e-3    # GHz → THz
    return np.exp(1j * np.pi * D_ps2 * nu_THz**2)


# ── Plot ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    lam_range = np.linspace(1450, 1650, 200)
    b2_vals   = np.array([beta2_ps2_per_mm(l) for l in lam_range])

    D1_ps2 = -600.0
    D2_ps2 = -900.0
    L1_mm  = required_length_mm(D1_ps2)
    L2_mm  = required_length_mm(D2_ps2)

    nu = np.linspace(-100, 100, 1000)   # GHz
    H1 = transfer_function(nu, D1_ps2)
    H2 = transfer_function(nu, D2_ps2)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle("Chip Dispersion Model — TD-GS PIC", fontsize=12)

    # β₂ vs wavelength
    axes[0].plot(lam_range, b2_vals, "b-", lw=2)
    axes[0].axvline(1550, color="r", ls="--", label="1550 nm")
    axes[0].set_xlabel("Wavelength (nm)")
    axes[0].set_ylabel("β₂ (ps²/mm)")
    axes[0].set_title("Si Wire GVD")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Transfer function phase
    axes[1].plot(nu, np.angle(H1), label=f"D₁={D1_ps2} ps² L={L1_mm:.0f}mm")
    axes[1].plot(nu, np.angle(H2), label=f"D₂={D2_ps2} ps² L={L2_mm:.0f}mm")
    axes[1].set_xlabel("Frequency offset (GHz)")
    axes[1].set_ylabel("Phase H(ν) (rad)")
    axes[1].set_title("H(ν) = exp(iπDν²)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # |H(ν)| = 1 always (all-pass dispersion)
    axes[2].plot(nu, np.abs(H1), label="Arm 1", lw=2)
    axes[2].plot(nu, np.abs(H2), label="Arm 2", lw=2, ls="--")
    axes[2].set_xlabel("Frequency offset (GHz)")
    axes[2].set_ylabel("|H(ν)|")
    axes[2].set_title("|H(ν)| = 1  (all-pass, lossless)")
    axes[2].set_ylim(0, 1.2)
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("../docs/dispersion_model.png", dpi=150)
    print("Saved docs/dispersion_model.png")

    print(f"\nArm 1: D={D1_ps2} ps²  →  spiral L = {L1_mm:.1f} mm")
    print(f"Arm 2: D={D2_ps2} ps²  →  spiral L = {L2_mm:.1f} mm")
    print(f"β₂ @ 1550nm = {beta2_ps2_per_mm(1550):.4f} ps²/mm")
