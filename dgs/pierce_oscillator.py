"""The Pierce oscillator: the actual circuit that generates a clock signal
in nearly every microcontroller and digital system -- an EEE fundamentals
topic done for real, reusing dgs.ac_circuits' impedance primitives rather
than re-deriving Z=jwL etc.

A quartz crystal is modeled by its standard equivalent circuit: a series
"motional" branch (Rm, Lm, Cm -- the crystal's mechanical resonance,
electrically equivalent) in PARALLEL with a shunt capacitance C0 (the
electrodes' physical capacitance). This gives the crystal TWO resonances:

  * series resonance f_s = 1/(2*pi*sqrt(Lm*Cm))       -- Z minimum (purely resistive)
  * parallel resonance f_p > f_s (with C0 included)   -- Z maximum

Between f_s and f_p the crystal's impedance is INDUCTIVE (positive
reactance) -- this narrow band is the only place a Pierce oscillator (which
needs an inductive element to resonate against its two load capacitors,
exactly like an LC tank) can actually oscillate. The external load
capacitance C_L "pulls" the actual oscillation frequency up from f_s by a
small, calculable amount -- the datasheet "load capacitance" spec is
precisely this pulling effect.
"""

import numpy as np

from dgs.ac_circuits import impedance_R, impedance_L, impedance_C


def series_resonance_freq(Lm, Cm):
    """f_s = 1/(2*pi*sqrt(Lm*Cm)): where the motional RLC branch alone is
    purely resistive (minimum |Z|)."""
    if Lm <= 0 or Cm <= 0:
        raise ValueError("Lm and Cm must be positive")
    return 1.0 / (2 * np.pi * np.sqrt(Lm * Cm))


def parallel_resonance_freq(Lm, Cm, C0):
    """f_p: where the motional branch's inductive reactance cancels the
    combined series capacitance of Cm and the shunt C0 in series
    (Cm*C0/(Cm+C0)) -- the crystal's maximum-impedance (anti-resonance)
    point, always slightly above f_s."""
    if Lm <= 0 or Cm <= 0 or C0 <= 0:
        raise ValueError("Lm, Cm, and C0 must be positive")
    C_series = Cm * C0 / (Cm + C0)
    return 1.0 / (2 * np.pi * np.sqrt(Lm * C_series))


def crystal_impedance(f, Rm, Lm, Cm, C0):
    """Full crystal equivalent-circuit impedance at frequency f: the
    motional branch (Rm+jwLm+1/(jwCm), built from dgs.ac_circuits'
    impedance_R/L/C) in PARALLEL with the shunt capacitance C0."""
    if f <= 0:
        raise ValueError("f must be positive")
    if Rm <= 0 or Lm <= 0 or Cm <= 0 or C0 <= 0:
        raise ValueError("Rm, Lm, Cm, C0 must be positive")
    omega = 2 * np.pi * f
    Z_motional = impedance_R(Rm) + impedance_L(Lm, omega) + impedance_C(Cm, omega)
    Z_shunt = impedance_C(C0, omega)
    return (Z_motional * Z_shunt) / (Z_motional + Z_shunt)


def is_inductive_region(f, Lm, Cm, C0):
    """True iff f lies strictly between the series and parallel resonance
    (the crystal presents a positive/inductive reactance there) -- the ONLY
    frequency band where a Pierce oscillator can lock onto this crystal."""
    f_s = series_resonance_freq(Lm, Cm)
    f_p = parallel_resonance_freq(Lm, Cm, C0)
    return f_s < f < f_p


def pierce_load_frequency(Lm, Cm, C0, C_L):
    """The actual (pulled) oscillation frequency of a Pierce oscillator
    loaded by external capacitance C_L: f_L = f_s*(1 + Cm/(2*(C0+C_L))),
    the standard first-order crystal "load pulling" formula -- always
    slightly above f_s, and must land inside the inductive region."""
    if C_L <= 0:
        raise ValueError("C_L must be positive")
    f_s = series_resonance_freq(Lm, Cm)
    f_p = parallel_resonance_freq(Lm, Cm, C0)
    f_L = f_s * (1.0 + Cm / (2.0 * (C0 + C_L)))
    if not (f_s < f_L < f_p):
        raise ValueError(f"pulled frequency {f_L:.6e} Hz falls outside the "
                          f"inductive region ({f_s:.6e}, {f_p:.6e}) Hz -- "
                          f"this load capacitance is not physically achievable for this crystal")
    return f_L


def load_capacitance_from_two_caps(C1, C2, C_stray=0.0):
    """The Pierce oscillator's two external load capacitors (crystal to
    ground on each pin) appear to the crystal as their SERIES combination
    plus board/pin stray capacitance: C_L = C1*C2/(C1+C2) + C_stray."""
    if C1 <= 0 or C2 <= 0:
        raise ValueError("C1 and C2 must be positive")
    if C_stray < 0:
        raise ValueError("C_stray must be non-negative")
    return (C1 * C2) / (C1 + C2) + C_stray


if __name__ == "__main__":
    # a real 32.768 kHz tuning-fork watch crystal: typical datasheet
    # equivalent-circuit values (e.g. Epson MC-306 series), Lm solved so
    # f_s lands exactly on the target 32768 Hz for this Cm
    f_target, Rm, Cm, C0 = 32768.0, 40e3, 1.9e-15, 1.0e-12   # Hz, ohm, F, F
    Lm = 1.0 / ((2 * np.pi * f_target) ** 2 * Cm)             # ~12.4 kH, matches datasheet order
    f_s = series_resonance_freq(Lm, Cm)
    f_p = parallel_resonance_freq(Lm, Cm, C0)
    print(f"crystal: Rm={Rm:.0f} ohm, Lm={Lm:.1f} H, Cm={Cm:.2e} F, C0={C0:.1e} F")
    print(f"series resonance   f_s = {f_s:.3f} Hz")
    print(f"parallel resonance f_p = {f_p:.3f} Hz  (pullability window: {f_p-f_s:.3f} Hz wide)")

    C1, C2, C_stray = 12e-12, 12e-12, 3e-12
    C_L = load_capacitance_from_two_caps(C1, C2, C_stray)
    f_L = pierce_load_frequency(Lm, Cm, C0, C_L)
    print(f"\nPierce oscillator: C1={C1*1e12:.0f}pF, C2={C2*1e12:.0f}pF, "
          f"stray={C_stray*1e12:.0f}pF -> C_L={C_L*1e12:.2f}pF")
    print(f"pulled oscillation frequency f_L = {f_L:.3f} Hz "
          f"(target {f_target:.3f} Hz, error = {f_L-f_target:+.3f} Hz, "
          f"{1e6*(f_L-f_target)/f_target:+.1f} ppm)")
    print(f"inductive region check: {is_inductive_region(f_L, Lm, Cm, C0)}")

    Z_at_fL = crystal_impedance(f_L, Rm, Lm, Cm, C0)
    print(f"\ncrystal impedance at f_L: {Z_at_fL:.1f} ohm "
          f"(reactance {'inductive (+)' if Z_at_fL.imag > 0 else 'capacitive (-)'})")
