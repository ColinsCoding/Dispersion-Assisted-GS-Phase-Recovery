"""acid_base_chemistry.py -- pH, pOH, and acid/base equilibria: chemistry
where the logarithm IS the measurement unit, not just a convenient
rescaling. pH = -log10[H+] exists because [H+] concentrations span many
orders of magnitude (roughly 1 M down to 1e-14 M) -- the same "compress a
huge dynamic range into a usable number" role logarithms play everywhere
else in this repo (decibels for intensity, ENOB for a digitizer's dynamic
range).

Water's autoionization, H2O <-> H+ + OH-, has an equilibrium constant
Kw = [H+][OH-] = 1e-14 at 25 C -- fixed regardless of what acid or base is
dissolved, which is why pH + pOH = 14 always holds at that temperature.
"""
from __future__ import annotations
import numpy as np
from typing import Dict, Optional

KW_25C = 1.0e-14  # water autoionization constant at 25 C


# ── 1. pH / pOH definitions ─────────────────────────────────────────────────

def pH_from_H_concentration(H_conc: float) -> float:
    """pH = -log10[H+]."""
    if H_conc <= 0:
        raise ValueError("H_conc must be positive")
    return -np.log10(H_conc)


def H_concentration_from_pH(pH: float) -> float:
    """[H+] = 10^(-pH), the inverse of pH_from_H_concentration."""
    return 10.0 ** (-pH)


def pOH_from_OH_concentration(OH_conc: float) -> float:
    """pOH = -log10[OH-]."""
    if OH_conc <= 0:
        raise ValueError("OH_conc must be positive")
    return -np.log10(OH_conc)


def OH_concentration_from_pOH(pOH: float) -> float:
    """[OH-] = 10^(-pOH), the inverse of pOH_from_OH_concentration."""
    return 10.0 ** (-pOH)


def pH_pOH_relationship(pH: Optional[float] = None, pOH: Optional[float] = None,
                         Kw: float = KW_25C) -> Dict:
    """Given exactly one of pH or pOH, compute the other via
    pH + pOH = -log10(Kw) (= 14.0 at 25 C, Kw=1e-14) -- water's
    autoionization constant links them regardless of what's dissolved."""
    if (pH is None) == (pOH is None):
        raise ValueError("provide exactly one of pH or pOH, not both or neither")
    if Kw <= 0:
        raise ValueError("Kw must be positive")
    total = -np.log10(Kw)
    if pH is None:
        pH = total - pOH
    else:
        pOH = total - pH
    return {"pH": float(pH), "pOH": float(pOH), "pH_plus_pOH": total}


def water_autoionization_check(H_conc: float, OH_conc: float, Kw: float = KW_25C,
                                rtol: float = 1e-6) -> Dict:
    """Verify [H+][OH-] ~= Kw for a given pair of concentrations -- a
    consistency check, not a derivation."""
    if H_conc <= 0 or OH_conc <= 0:
        raise ValueError("H_conc and OH_conc must be positive")
    product = H_conc * OH_conc
    consistent = bool(np.isclose(product, Kw, rtol=rtol))
    return {"product": product, "Kw": Kw, "consistent": consistent}


# ── 2. Strong and weak acids ────────────────────────────────────────────────

def strong_acid_pH(concentration: float) -> float:
    """A strong acid dissociates completely: [H+] = concentration exactly,
    so pH = -log10(concentration)."""
    if concentration <= 0:
        raise ValueError("concentration must be positive")
    return pH_from_H_concentration(concentration)


def pKa_from_Ka(Ka: float) -> float:
    """pKa = -log10(Ka), the same log-compression idea as pH itself."""
    if Ka <= 0:
        raise ValueError("Ka must be positive")
    return -np.log10(Ka)


def weak_acid_pH(Ka: float, concentration: float) -> Dict:
    """A weak acid HA <-> H+ + A- only partially dissociates:
    Ka = [H+][A-]/[HA] = x^2/(C-x) for initial concentration C and
    equilibrium [H+]=x. Solved EXACTLY via the quadratic x^2+Ka*x-Ka*C=0
    (positive root only), not the common small-x approximation x~=sqrt(Ka*C)
    -- the exact solution stays valid even when dissociation isn't small."""
    if Ka <= 0:
        raise ValueError("Ka must be positive")
    if concentration <= 0:
        raise ValueError("concentration must be positive")
    # x^2 + Ka*x - Ka*C = 0  ->  x = (-Ka + sqrt(Ka^2 + 4*Ka*C)) / 2
    x = (-Ka + np.sqrt(Ka ** 2 + 4 * Ka * concentration)) / 2.0
    pH = pH_from_H_concentration(x)
    fraction_dissociated = x / concentration
    return {"H_conc": float(x), "pH": float(pH),
            "fraction_dissociated": float(fraction_dissociated)}


def henderson_hasselbalch(pKa: float, base_conc: float, acid_conc: float) -> float:
    """pH = pKa + log10([A-]/[HA]) -- the buffer equation: a direct
    application of log10(product) = log10(numerator) - log10(denominator)
    to the equilibrium expression Ka = [H+][A-]/[HA]."""
    if base_conc <= 0 or acid_conc <= 0:
        raise ValueError("base_conc and acid_conc must be positive")
    return pKa + np.log10(base_conc / acid_conc)


# ── 3. Strong acid / strong base titration curve ────────────────────────────

def titration_curve(C_acid: float, V_acid: float, C_base: float,
                     V_base_max: Optional[float] = None, n_points: int = 200,
                     Kw: float = KW_25C) -> Dict:
    """pH vs. volume of strong base titrant added to a strong acid --
    moles balance at each point, not an approximation. V_acid/V_base_max in
    the same volume unit (e.g. mL); C_acid/C_base in mol/that-volume-unit.
    V_base_max defaults to 2x the equivalence volume so the curve shows the
    full jump and levels off on both sides."""
    if C_acid <= 0 or C_base <= 0 or V_acid <= 0:
        raise ValueError("C_acid, C_base, and V_acid must be positive")
    if n_points < 3:
        raise ValueError(f"n_points={n_points}: must be >= 3")
    equivalence_volume = C_acid * V_acid / C_base
    if V_base_max is None:
        V_base_max = 2.0 * equivalence_volume
    if V_base_max <= 0:
        raise ValueError("V_base_max must be positive")

    V_base = np.linspace(1e-9, V_base_max, n_points)  # avoid V_base=0 (handled analytically below anyway)
    moles_acid = C_acid * V_acid
    moles_base = C_base * V_base
    total_volume = V_acid + V_base

    pH = np.empty(n_points)
    for i in range(n_points):
        net = moles_acid - moles_base[i]
        if net > 0:
            H_conc = net / total_volume[i]
            pH[i] = pH_from_H_concentration(H_conc)
        elif net < 0:
            OH_conc = -net / total_volume[i]
            pOH = pOH_from_OH_concentration(OH_conc)
            pH[i] = pH_pOH_relationship(pOH=pOH, Kw=Kw)["pH"]
        else:
            pH[i] = -np.log10(np.sqrt(Kw))  # neutral: [H+]=[OH-]=sqrt(Kw), pH=7 at 25C

    return {"V_base": V_base, "pH": pH, "equivalence_volume": float(equivalence_volume)}


if __name__ == "__main__":
    print("=== 1. pH/pOH from concentration, and back ===")
    for H_conc in [1e-1, 1e-7, 1e-12]:
        pH = pH_from_H_concentration(H_conc)
        back = H_concentration_from_pH(pH)
        print(f"  [H+]={H_conc:.0e} M  ->  pH={pH:.2f}  ->  back to [H+]={back:.2e} M")

    print("\n=== 2. pH + pOH = 14 (at 25 C) ===")
    result = pH_pOH_relationship(pH=4.0)
    print(f"  pH=4.0  ->  pOH={result['pOH']:.2f}  (sum={result['pH_plus_pOH']:.1f})")

    print("\n=== 3. Strong vs. weak acid at the SAME concentration ===")
    C = 0.1  # 0.1 M
    strong = strong_acid_pH(C)
    weak = weak_acid_pH(Ka=1.8e-5, concentration=C)  # acetic acid's real Ka
    print(f"  0.1 M strong acid: pH={strong:.2f}")
    print(f"  0.1 M acetic acid (Ka=1.8e-5): pH={weak['pH']:.2f}, "
          f"only {weak['fraction_dissociated']*100:.2f}% dissociated")

    print("\n=== 4. Henderson-Hasselbalch: an acetate buffer ===")
    pKa_acetic = pKa_from_Ka(1.8e-5)
    pH_buffer = henderson_hasselbalch(pKa_acetic, base_conc=0.1, acid_conc=0.1)
    print(f"  pKa(acetic acid)={pKa_acetic:.2f}, equal-concentration buffer -> pH={pH_buffer:.2f} "
          f"(should equal pKa when [A-]=[HA])")

    print("\n=== 5. Titration curve: 0.1 M HCl (25 mL) with 0.1 M NaOH ===")
    curve = titration_curve(C_acid=0.1, V_acid=25.0, C_base=0.1)
    print(f"  equivalence volume: {curve['equivalence_volume']:.2f} mL")
    for frac in [0.0, 0.5, 0.99, 1.0, 1.01, 1.5]:
        idx = int(np.argmin(np.abs(curve["V_base"] - frac * curve["equivalence_volume"])))
        print(f"  V_base={curve['V_base'][idx]:6.2f} mL ({frac:.2f}x eq.vol)  pH={curve['pH'][idx]:.2f}")
