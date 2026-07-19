"""Food science kinetics: thermal-death kinetics (D-value/z-value/F-value, the
Bigelow model behind every canning/pasteurization safety spec) and Q10/shelf-life
prediction -- the food-industry dialect of the SAME kinetics in dgs.reaction_rates,
just conventionally written in log10("decades") and degrees instead of ln(e-foldings).

Microbial death under heat is first-order decay (dgs.reaction_rates order=1) with
the SAME physics, different bookkeeping:
    reaction_rates:  [A](t) = [A]0 * exp(-k t),        half_life = ln(2)/k
    food science:    N(t)   = N0   * 10^(-t/D),        D = ln(10)/k
D is "how long to kill 90% of the population" the way half-life is "how long to
lose half of it" -- same exponential, decimal instead of binary. The z-value plays
the same role for TEMPERATURE dependence that dgs.reaction_rates.arrhenius_rate_constant's
activation energy Ea plays -- both describe how fast the rate constant grows with T,
just parameterized differently (a food-science habit, not different physics).

NumPy only. Education.
"""

import numpy as np


def D_value(N0, N, t):
    """Decimal reduction time from one measured data point: time for a
    microbial population to drop by one log10 cycle (90% killed).
    D = t / log10(N0/N) -- the food-microbiology D = ln(10)/k relative of
    dgs.reaction_rates.half_life for a first-order (order=1) death process."""
    if N0 <= 0 or N <= 0 or t <= 0:
        raise ValueError("N0, N, t must all be positive")
    if N >= N0:
        raise ValueError("N must be less than N0 (population must have decreased)")
    return t / np.log10(N0 / N)


def microbial_survivors(N0, t, D):
    """N(t) = N0 * 10^(-t/D) -- first-order microbial death curve (Bigelow
    model), the log10/decade-based twin of
    dgs.reaction_rates.integrated_concentration(order=1)."""
    if D <= 0:
        raise ValueError(f"D-value must be positive, got {D}")
    t = np.asarray(t, dtype=float)
    return N0 * 10.0 ** (-t / D)


def z_value(D1, D2, T1, T2):
    """z-value: the temperature increase that reduces D by a factor of 10
    (log10(D) vs T is linear with slope -1/z): z = (T2-T1) / log10(D1/D2)."""
    if D1 <= 0 or D2 <= 0:
        raise ValueError("D-values must be positive")
    if T1 == T2:
        raise ValueError("T1 and T2 must differ")
    return (T2 - T1) / np.log10(D1 / D2)


def D_at_temperature(D_ref, T, T_ref, z):
    """Bigelow model: D(T) = D_ref * 10^(-(T-T_ref)/z) -- how fast the
    decimal reduction time shrinks (faster kill) as process temperature rises
    above the reference T_ref."""
    if D_ref <= 0 or z <= 0:
        raise ValueError("D_ref and z must be positive")
    T = np.asarray(T, dtype=float)
    return D_ref * 10.0 ** (-(T - T_ref) / z)


def lethality_rate(T, T_ref, z):
    """Instantaneous lethal rate L(T) = 10^((T-T_ref)/z) relative to a
    reference temperature (canonically 121.1 C / 250 F for sterilization,
    z=10 C / 18 F for Clostridium botulinum) -- how many
    'reference-temperature-seconds' of kill one real second at T delivers.
    L(T_ref) = 1 by construction."""
    if z <= 0:
        raise ValueError(f"z must be positive, got {z}")
    T = np.asarray(T, dtype=float)
    return 10.0 ** ((T - T_ref) / z)


def F_value(t, T_profile, T_ref, z):
    """F-value: accumulated lethality F = integral L(T(t)) dt over a real
    (possibly time-varying, e.g. oven come-up-time) temperature profile --
    the actual quantity a thermal process is validated against
    (F0 = 3 min at T_ref=121.1 C, z=10 C is the canonical botulinum safety
    target for low-acid canned foods)."""
    t = np.asarray(t, dtype=float)
    L = lethality_rate(T_profile, T_ref, z)
    return float(np.trapezoid(L, t))


def q10_coefficient(k1, k2, T1, T2):
    """Q10 = (k2/k1)^(10/(T2-T1)) -- the food-science rule-of-thumb rate
    multiplier per 10 degrees, the shelf-life-testing dialect of an Arrhenius
    activation energy (dgs.reaction_rates.arrhenius_rate_constant): both
    describe how strongly a rate constant grows with temperature."""
    if k1 <= 0 or k2 <= 0:
        raise ValueError("rate constants must be positive")
    if T1 == T2:
        raise ValueError("T1 and T2 must differ")
    return (k2 / k1) ** (10.0 / (T2 - T1))


def shelf_life_first_order(k, quality_fraction_remaining):
    """Time until a first-order-decaying quality attribute (vitamin C, color,
    flavor) falls to a given fraction of its initial value: solve
    exp(-k t) = quality_fraction_remaining for t. Shelf life IS just a
    threshold-crossing time on dgs.reaction_rates.integrated_concentration's
    order=1 curve."""
    if not (0 < quality_fraction_remaining < 1):
        raise ValueError("quality_fraction_remaining must be in (0, 1)")
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    return -np.log(quality_fraction_remaining) / k


if __name__ == "__main__":
    # canonical low-acid canning numbers: T_ref=121.1 C (250 F), z=10 C (18 F)
    T_ref, z = 121.1, 10.0
    D_ref = 0.21  # minutes, typical D_121 for C. botulinum spores

    D_130 = D_at_temperature(D_ref, 130.0, T_ref, z)
    print(f"D at 130 C = {D_130:.4f} min (D at {T_ref} C = {D_ref} min)")
    print(f"recovered z from (D_ref, D_130): {z_value(D_ref, D_130, T_ref, 130.0):.2f} C "
          f"(expect {z})")

    # isothermal process held exactly at T_ref for 3 minutes -> F-value = 3 min exactly
    t = np.linspace(0, 3, 500)
    T_profile = np.full_like(t, T_ref)
    F0 = F_value(t, T_profile, T_ref, z)
    print(f"\nisothermal F-value at T_ref for 3 min: {F0:.4f} min (expect 3.0000)")

    # shelf life: vitamin C first-order loss, k=0.02 /day, until 50% remains
    k = 0.02
    t_half_life_style = shelf_life_first_order(k, 0.5)
    print(f"\nvitamin C shelf life to 50% remaining (k={k}/day): {t_half_life_style:.1f} days")

    print(f"\nQ10 check: k doubles over 10 degrees -> Q10 = "
          f"{q10_coefficient(1.0, 2.0, 0.0, 10.0):.2f} (expect 2.00)")
