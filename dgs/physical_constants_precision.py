"""How to choose a data type for a physical constant: match the type's
precision to the constant's OWN known precision, no more and no less.

Since the 2019 SI redefinition, several fundamental constants are no
longer MEASURED at all -- they are DEFINED to be exact numbers (c, h, e,
k_B, N_A), because the SI base units themselves are now defined in terms
of them. Others are still genuinely measured, with real experimental
uncertainty -- and some of those are shockingly poorly known: the
gravitational constant G has a relative uncertainty around 2e-5, roughly
four orders of magnitude worse than most other fundamental constants.

This matters computationally (dgs.c_type_precision already established
float32 gives ~7 significant decimal digits, float64 gives ~15-17):
  * storing c in float32 silently THROWS AWAY exact information -- c is
    defined to infinite precision, and float32 can't represent it exactly.
  * storing G in float64 is mostly WASTED precision -- G's own experimental
    uncertainty (~5 significant digits) is far looser than float32's ~7,
    let alone float64's ~16. The extra bits encode false confidence.

QED's coupling constant, the fine-structure constant alpha, sits at the
opposite extreme from G: it is one of the most precisely MEASURED
quantities in all of physics (relative uncertainty ~1.5e-10, comparable
to float64's own rounding floor) -- a genuinely interesting contrast to
G's four-order-of-magnitude-worse uncertainty, despite both being
"just constants."
"""

import numpy as np

# each entry: (value, relative_uncertainty, is_exact_by_definition, unit)
# 2019 SI redefinition + CODATA 2018 recommended values
CONSTANTS = {
    "c (speed of light)":        (2.99792458e8,     0.0,        True,  "m/s"),
    "h (Planck constant)":       (6.62607015e-34,   0.0,        True,  "J*s"),
    "e (elementary charge)":     (1.602176634e-19,  0.0,        True,  "C"),
    "k_B (Boltzmann constant)":  (1.380649e-23,      0.0,        True,  "J/K"),
    "N_A (Avogadro constant)":   (6.02214076e23,     0.0,        True,  "1/mol"),
    "alpha (fine-structure, QED coupling)": (7.2973525693e-3, 1.5e-10, False, "dimensionless"),
    "m_e (electron mass)":       (9.1093837015e-31, 3.0e-10,    False, "kg"),
    "G (gravitational constant)": (6.67430e-11,      2.2e-5,     False, "m^3/(kg*s^2)"),
}

FLOAT32_EPS = np.finfo(np.float32).eps    # ~1.19e-7, ~7 significant decimal digits
FLOAT64_EPS = np.finfo(np.float64).eps    # ~2.22e-16, ~15-17 significant decimal digits


def significant_digits_justified(relative_uncertainty):
    """How many significant decimal digits are actually meaningful, given
    a constant's relative uncertainty -- digits beyond this encode false
    precision no matter what data type stores them. Exact constants
    (uncertainty=0) return None (limited only by the chosen data type,
    not by physics)."""
    if relative_uncertainty < 0:
        raise ValueError("relative_uncertainty must be non-negative")
    if relative_uncertainty == 0:
        return None
    return int(np.floor(-np.log10(relative_uncertainty)))


def recommend_dtype(relative_uncertainty, is_exact):
    """The actual data-type recommendation: an EXACT constant deserves
    float64 (or better) so the type's own precision is the only limit;
    a MEASURED constant only needs enough bits to represent its own
    uncertainty -- anything past that is wasted storage/bandwidth, not
    added accuracy."""
    if is_exact:
        return "float64 (exact constants deserve the type's full precision -- no upper limit from physics)"
    digits = significant_digits_justified(relative_uncertainty)
    if digits <= 6:
        return f"float32 is SUFFICIENT ({digits} digits justified, float32 gives ~7)"
    else:
        return f"float64 needed ({digits} digits justified, exceeds float32's ~7)"


def precision_waste_ratio(relative_uncertainty, dtype_eps):
    """How many orders of magnitude MORE precise the data type is than
    the constant's own known precision -- a direct measure of "wasted
    bits" (values >> 1 mean the type is overkill for this constant)."""
    if relative_uncertainty <= 0:
        raise ValueError("only meaningful for constants with nonzero uncertainty")
    return relative_uncertainty / dtype_eps


def float32_roundoff_vs_true_uncertainty(name):
    """For a given constant, compare float32's OWN rounding error (from
    just storing the value) against the constant's real physical
    uncertainty -- confirms whether float32 is 'hiding' new error beyond
    what physics already doesn't know, or whether it's well within the
    physical uncertainty already."""
    value, rel_unc, is_exact, unit = CONSTANTS[name]
    value32 = np.float32(value)
    roundoff_rel_error = abs(float(value32) - value) / value if value != 0 else 0.0
    return roundoff_rel_error, rel_unc, is_exact


if __name__ == "__main__":
    print("Constant                                    | rel. uncertainty | digits justified | dtype recommendation")
    print("-" * 115)
    for name, (value, rel_unc, is_exact, unit) in CONSTANTS.items():
        digits = significant_digits_justified(rel_unc)
        digits_str = "exact (unlimited)" if digits is None else str(digits)
        rec = recommend_dtype(rel_unc, is_exact)
        print(f"{name:44s} | {rel_unc:16.2e} | {digits_str:>17s} | {rec}")

    print("\n=== float32 roundoff vs. the constant's OWN physical uncertainty ===")
    for name in ["c (speed of light)", "alpha (fine-structure, QED coupling)", "G (gravitational constant)"]:
        roundoff, rel_unc, is_exact = float32_roundoff_vs_true_uncertainty(name)
        if is_exact:
            print(f"{name}: float32 roundoff={roundoff:.2e}  <-- this IS new error, "
                  f"since the true value is EXACT (zero uncertainty)")
        else:
            ratio = roundoff / rel_unc
            verdict = "well within" if ratio < 1 else "EXCEEDS"
            print(f"{name}: float32 roundoff={roundoff:.2e} vs physical uncertainty={rel_unc:.2e} "
                  f"(ratio={ratio:.2e}, {verdict} the real uncertainty)")

    print("\n=== The G vs. alpha contrast ===")
    print(f"G's relative uncertainty:     {CONSTANTS['G (gravitational constant)'][1]:.2e}")
    print(f"alpha's relative uncertainty: {CONSTANTS['alpha (fine-structure, QED coupling)'][1]:.2e}")
    ratio = CONSTANTS['G (gravitational constant)'][1] / CONSTANTS['alpha (fine-structure, QED coupling)'][1]
    print(f"G is {ratio:.0e}x MORE uncertain than alpha -- both are 'just constants,' but")
    print("alpha (QED's coupling strength) is among the best-measured numbers in physics,")
    print("while G remains one of the worst-measured, despite over a century of experiments.")
