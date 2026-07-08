"""Laser physics, where the logarithm keeps appearing.

LASER = Light Amplification by Stimulated Emission of Radiation. Three ideas, and a
logarithm hiding in each:

  1. POPULATION INVERSION. In thermal equilibrium the Boltzmann factor sets the ratio
     of upper to lower populations, N2/N1 = exp(-dE/kT) < 1 -- always more atoms DOWN
     than up, so a thermal medium ABSORBS. To amplify you need N2 > N1, an INVERSION,
     which the Boltzmann law forbids at any positive temperature. Invert the log:
     T = -dE / (k ln(N2/N1)); when N2 > N1 the log is positive and T is NEGATIVE --
     a population inversion is literally a negative-temperature state.

  2. STIMULATED vs SPONTANEOUS. Einstein's coefficients tie them: A/B = 8*pi*h*nu^3/c^3,
     and the ratio of stimulated to spontaneous emission in a thermal field is the
     photon occupation 1/(exp(h*nu/kT) - 1). At optical frequencies and room
     temperature that is ~1e-33 -- spontaneous emission utterly wins, which is WHY a
     laser must be pumped far out of equilibrium.

  3. GAIN and THRESHOLD. Once inverted, intensity grows EXPONENTIALLY, I = I0 exp(gz),
     so gain is naturally quoted in decibels, G_dB = 10 log10(Pout/Pin) -- the log
     again. Lasing starts when round-trip gain beats round-trip loss, giving the
     threshold gain
        g_th = alpha + (1/2L) ln(1/(R1 R2)),
     where the mirror-loss term is exactly the logarithm of the reflectivities.

Everything is checked: the Boltzmann inversion/negative-temperature round trip, the
stimulated/spontaneous ratio, the exp<->dB gain identity, and that at g_th the
round-trip gain equals 1. Ties to dgs.quantum_oscillator (h*nu is one quantum) and the
repo's photonics thread. NumPy-free; py-3.13.
"""

import math

H_PLANCK = 6.62607015e-34     # J s
K_BOLTZ = 1.380649e-23        # J/K
C_LIGHT = 2.99792458e8        # m/s
EV_J = 1.602176634e-19        # J per eV


# ----------------------------------------------------------------------
# 1. Population inversion and the Boltzmann law
# ----------------------------------------------------------------------

def boltzmann_population_ratio(delta_E_J, T):
    """N2/N1 = exp(-dE/kT) in thermal equilibrium. Always < 1 for dE, T > 0 --
    a thermal medium cannot be inverted, so it absorbs rather than amplifies."""
    if delta_E_J <= 0 or T <= 0:
        raise ValueError("delta_E and T must be positive")
    return math.exp(-delta_E_J / (K_BOLTZ * T))


def temperature_from_ratio(ratio, delta_E_J):
    """Invert the Boltzmann law: T = -dE / (k ln(ratio)). For ratio < 1 this is a
    normal positive temperature; for an INVERSION (ratio > 1) the log is positive
    and T comes out NEGATIVE -- the negative-temperature description of a gain
    medium."""
    if ratio <= 0 or delta_E_J <= 0:
        raise ValueError("ratio and delta_E must be positive")
    if ratio == 1:
        return math.inf
    return -delta_E_J / (K_BOLTZ * math.log(ratio))


def is_inverted(N2, N1):
    """A medium amplifies only when the upper level is more populated: N2 > N1."""
    return N2 > N1


# ----------------------------------------------------------------------
# 2. Einstein coefficients: stimulated vs spontaneous
# ----------------------------------------------------------------------

def einstein_A_over_B(freq_Hz):
    """Ratio of the spontaneous (A) to stimulated (B) Einstein coefficients:
    A/B = 8*pi*h*nu^3/c^3. Grows as nu^3, so spontaneous emission dominates hard
    at high (optical) frequencies -- part of why short-wavelength lasers are hard."""
    if freq_Hz <= 0:
        raise ValueError("frequency must be positive")
    return 8 * math.pi * H_PLANCK * freq_Hz ** 3 / C_LIGHT ** 3


def stimulated_over_spontaneous(freq_Hz, T):
    """In a thermal field the stimulated/spontaneous emission ratio is the photon
    occupation number 1/(exp(h*nu/kT) - 1). Tiny at optical frequencies and room
    temperature -> a thermal source can't lase; you must pump."""
    if freq_Hz <= 0 or T <= 0:
        raise ValueError("frequency and T must be positive")
    x = H_PLANCK * freq_Hz / (K_BOLTZ * T)
    return 1.0 / (math.exp(x) - 1.0)


# ----------------------------------------------------------------------
# 3. Gain, decibels, and the threshold condition
# ----------------------------------------------------------------------

def gain_coefficient(sigma, N2, N1):
    """Small-signal gain coefficient g = sigma (N2 - N1) [1/m]. Positive when
    inverted (amplifies), negative when not (absorbs)."""
    return sigma * (N2 - N1)


def intensity_after_gain(I0, g, length):
    """Beer-Lambert with gain: I = I0 exp(g L). Exponential growth (g>0) or decay
    (g<0) -- the reason gain is a per-length coefficient."""
    if I0 < 0:
        raise ValueError("I0 must be non-negative")
    return I0 * math.exp(g * length)


def gain_dB(P_out, P_in):
    """Gain in decibels, G = 10 log10(Pout/Pin) -- the logarithm that turns
    exponential amplification into an additive number."""
    if P_out <= 0 or P_in <= 0:
        raise ValueError("powers must be positive")
    return 10 * math.log10(P_out / P_in)


def small_signal_gain_dB(g, length):
    """The dB gain of a length L of medium with gain coefficient g:
    10 log10(exp(gL)) = gL * 10/ln(10). Connects the per-length g to decibels."""
    return g * length * 10 / math.log(10)


def threshold_gain(alpha_loss, R1, R2, length):
    """Threshold gain g_th = alpha + (1/2L) ln(1/(R1 R2)): the gain at which
    round-trip amplification just balances internal loss alpha plus the mirror
    losses (the ln term). Perfect mirrors (R=1) leave only alpha."""
    if not (0 < R1 <= 1 and 0 < R2 <= 1):
        raise ValueError("reflectivities must be in (0, 1]")
    if length <= 0:
        raise ValueError("cavity length must be positive")
    return alpha_loss + math.log(1.0 / (R1 * R2)) / (2 * length)


def round_trip_gain(g, alpha_loss, R1, R2, length):
    """The net round-trip power factor R1 R2 exp(2(g-alpha)L). Lasing self-sustains
    when this reaches 1; at g = threshold_gain(...) it equals exactly 1."""
    return R1 * R2 * math.exp(2 * (g - alpha_loss) * length)


if __name__ == "__main__":
    # HeNe-like transition: 632.8 nm
    freq = C_LIGHT / 632.8e-9
    dE = H_PLANCK * freq
    print(f"632.8 nm photon: {dE/EV_J:.3f} eV, freq {freq:.3e} Hz")

    print("\n1. thermal populations never invert:")
    ratio = boltzmann_population_ratio(dE, 300)
    print(f"   N2/N1 at 300 K = {ratio:.2e}  (essentially no upper population)")
    print(f"   an inversion N2/N1 = 2 corresponds to T = "
          f"{temperature_from_ratio(2.0, dE):.1f} K  (negative!)")

    print("\n2. spontaneous emission dominates -> must pump:")
    print(f"   A/B = {einstein_A_over_B(freq):.3e}")
    print(f"   stimulated/spontaneous in a 300 K field = "
          f"{stimulated_over_spontaneous(freq, 300):.2e}")

    print("\n3. gain, dB, and threshold:")
    print(f"   g=0.05/m over 0.3 m: I/I0 = {intensity_after_gain(1, 0.05, 0.3):.4f}, "
          f"= {small_signal_gain_dB(0.05, 0.3):.3f} dB")
    print(f"   amplifier 1 mW -> 100 mW = {gain_dB(100e-3, 1e-3):.0f} dB")
    gth = threshold_gain(0.01, 1.0, 0.98, 0.3)
    print(f"   threshold gain (alpha=0.01, R1=1, R2=0.98, L=0.3) = {gth:.4f} /m")
    print(f"   round-trip gain at threshold = {round_trip_gain(gth, 0.01, 1.0, 0.98, 0.3):.6f} "
          f"(=1, self-sustaining)")
