"""Reflected waves: what happens when a wave hits a boundary, across strings, cables, and light.

Whenever a wave meets a change of medium it partly reflects and partly transmits, and the split
is governed by one number -- the mismatch in WAVE IMPEDANCE Z. The same story runs through every
kind of wave:

    * a wave on a string hitting a heavier string or a wall,
    * a voltage pulse reaching the end of a coax cable,
    * light crossing from air into glass.

The amplitude REFLECTION COEFFICIENT for a wave going from medium 1 (impedance Z1) into medium 2
(Z2) is
        Gamma = (Z2 - Z1) / (Z2 + Z1),   tau = 1 + Gamma = 2 Z2 / (Z1 + Z2),
so a matched boundary (Z2=Z1) gives Gamma=0 (no reflection), an OPEN end (Z2->inf) gives Gamma=+1,
and a SHORT / fixed end (Z2=0) gives Gamma=-1 (the famous phase flip when a string is tied down).
Power splits as R=|Gamma|^2 reflected, T=1-R transmitted -- energy is conserved.

The reflected wave interferes with the incident one to make a STANDING WAVE whose peak-to-null
ratio is the VSWR = (1+|Gamma|)/(1-|Gamma|), the number every RF engineer watches. For LIGHT the
impedance is Z ~ 1/n, so Gamma=(n1-n2)/(n1+n2): air->glass reflects 4%, and at the BREWSTER angle
the reflection of p-polarized light vanishes, while past the CRITICAL angle it totally reflects.
A matched load (or a quarter-wave transformer, Z0=sqrt(Z_s Z_L)) kills the reflection entirely.

Complements dgs.transmission_line_tdr (which uses reflections to *range* a cable fault); here the
focus is the reflected/standing wave itself. NumPy; complex impedances allowed; py-3.13.
"""

import numpy as np


def reflection_coefficient(Z1, Z2):
    """Amplitude reflection coefficient for a wave going from medium 1 into medium 2:
    Gamma = (Z2 - Z1)/(Z2 + Z1). Z2=inf (open) -> +1, Z2=0 (short/fixed) -> -1."""
    if np.isinf(Z2):
        return 1.0
    if Z1 + Z2 == 0:
        raise ValueError("Z1 + Z2 = 0 makes Gamma singular")
    return (Z2 - Z1) / (Z2 + Z1)


def transmission_coefficient(Z1, Z2):
    """Amplitude transmission coefficient tau = 1 + Gamma = 2 Z2/(Z1 + Z2)."""
    if np.isinf(Z2):
        return 2.0                      # limiting value (field doubles at an open)
    return 2.0 * Z2 / (Z1 + Z2)


def power_reflectance(gamma):
    """Fraction of incident power reflected, R = |Gamma|^2."""
    return float(np.abs(gamma) ** 2)


def power_transmittance(gamma):
    """Fraction of incident power transmitted, T = 1 - |Gamma|^2 (lossless: R + T = 1)."""
    return 1.0 - power_reflectance(gamma)


def vswr(gamma):
    """Voltage standing-wave ratio (1+|Gamma|)/(1-|Gamma|): 1 = perfectly matched,
    infinity = total reflection."""
    g = np.abs(gamma)
    if g >= 1:
        return np.inf
    return (1 + g) / (1 - g)


def standing_wave_pattern(gamma, beta_z):
    """Envelope of the total wave |1 + Gamma e^{-2 i beta z}| along the line. Its maximum is
    1+|Gamma| (antinode), its minimum 1-|Gamma| (node); the ratio is the VSWR."""
    beta_z = np.asarray(beta_z, float)
    return np.abs(1 + gamma * np.exp(-2j * beta_z))


def string_end_reflection(end):
    """Displacement reflection coefficient for a wave on a string:
    a FIXED end inverts the wave (Gamma=-1), a FREE end reflects it upright (Gamma=+1)."""
    if end == "fixed":
        return -1.0
    if end == "free":
        return 1.0
    raise ValueError("end must be 'fixed' or 'free'")


def fresnel_normal_incidence(n1, n2):
    """Field reflection coefficient for light at normal incidence, r=(n1-n2)/(n1+n2), and the
    power reflectance R=r^2. Returns (r, R). Air (1.0) -> glass (1.5) reflects 4%."""
    if n1 <= 0 or n2 <= 0:
        raise ValueError("refractive indices must be > 0")
    r = (n1 - n2) / (n1 + n2)
    return r, r ** 2


def brewster_angle(n1, n2):
    """Brewster angle (radians): tan(theta_B)=n2/n1. Reflected light is fully polarized
    (p-reflection vanishes)."""
    if n1 <= 0 or n2 <= 0:
        raise ValueError("refractive indices must be > 0")
    return np.arctan2(n2, n1)


def critical_angle(n1, n2):
    """Critical angle (radians) for total internal reflection, sin(theta_c)=n2/n1.
    Only exists going into a rarer medium (n1 > n2)."""
    if not n1 > n2 > 0:
        raise ValueError("total internal reflection needs n1 > n2 > 0")
    return np.arcsin(n2 / n1)


def input_impedance(Z_load, Z0, beta_l):
    """Impedance seen looking into a lossless line of electrical length beta*l terminated in
    Z_load: Z_in = Z0 (Z_L + j Z0 tan bl)/(Z0 + j Z_L tan bl). Repeats every half wavelength."""
    if Z0 <= 0:
        raise ValueError("Z0 must be > 0")
    t = np.tan(beta_l)
    if np.isinf(Z_load):
        return -1j * Z0 / t if t != 0 else np.inf     # open-circuited stub
    return Z0 * (Z_load + 1j * Z0 * t) / (Z0 + 1j * Z_load * t)


def quarter_wave_transformer(Z_source, Z_load):
    """Characteristic impedance of a quarter-wave section that matches a real Z_load to a real
    source Z_source with zero reflection: Z0 = sqrt(Z_source * Z_load)."""
    if Z_source <= 0 or Z_load <= 0:
        raise ValueError("impedances must be > 0")
    return np.sqrt(Z_source * Z_load)


if __name__ == "__main__":
    print("=== a voltage pulse reaching the end of a 50 ohm cable ===")
    for name, ZL in [("matched 50", 50), ("open (inf)", np.inf), ("short (0)", 0),
                     ("100 ohm", 100)]:
        g = reflection_coefficient(50, ZL)
        print(f"  {name:12s}: Gamma={g:+.3f}  R={power_reflectance(g):.2f}  "
              f"T={power_transmittance(g):.2f}  VSWR={vswr(g):.2f}")

    print("\n=== light from air into glass (normal incidence) ===")
    r, R = fresnel_normal_incidence(1.0, 1.5)
    print(f"  r={r:+.3f}  R={R*100:.1f}%  (the 4% per glass surface)")
    print(f"  Brewster angle air->glass = {np.degrees(brewster_angle(1.0, 1.5)):.1f} deg")
    print(f"  critical angle glass->air = {np.degrees(critical_angle(1.5, 1.0)):.1f} deg")

    print("\n=== quarter-wave transformer: match 100 ohm load to a 50 ohm source ===")
    Z0 = quarter_wave_transformer(50, 100)
    Zin = input_impedance(100, Z0, np.pi / 2)          # quarter wave: beta*l = pi/2
    print(f"  Z0 = sqrt(50*100) = {Z0:.2f} ohm  ->  Z_in = {Zin.real:.2f} ohm "
          f"(= 50, matched, Gamma={reflection_coefficient(50, Zin.real):+.3f})")

    print("\n=== string tied to a wall (fixed end) ===")
    print(f"  Gamma = {string_end_reflection('fixed'):+.0f}  (wave flips over on reflection)")
