"""Average power = <cos^2> = 1/2: the same factor in AC circuits and in light.

Anything that oscillates carries power that oscillates too, and what you measure is the
TIME AVERAGE. For sinusoids that average is fixed by two trig identities:
        <cos^2(wt)>           = 1/2,
        <cos(wt) cos(wt+phi)> = 1/2 cos(phi),
because cos A cos B = 1/2[cos(A-B) + cos(A+B)] and the double-frequency cos(A+B) term
averages to zero over a period. Everything about average power is those two facts.

The SAME 1/2 appears everywhere a field oscillates -- which is the point:

  AC ELECTRICAL POWER.  With v = V0 cos(wt), i = I0 cos(wt-phi),
        <P> = <v i> = 1/2 V0 I0 cos(phi) = V_rms I_rms cos(phi),
  and the RMS value of a sinusoid is peak/sqrt(2) (the sqrt(2) is 1/sqrt(<cos^2>)).
  cos(phi) is the POWER FACTOR: purely resistive (phi=0) delivers full power, purely
  reactive (phi=pi/2) delivers ZERO average power (energy sloshes, none dissipated).

  OPTICAL INTENSITY.  A light wave's intensity is the time-averaged Poynting vector,
        I = <S> = <c eps0 E^2> = 1/2 c eps0 E0^2 = c eps0 E_rms^2,
  the exact same <cos^2>=1/2 -- an optical detector reads the mean of E^2, just as a
  wattmeter reads the mean of v*i. This is why the repo's square-law detector sees
  intensity, and why brightness ~ amplitude^2.

So "power average, trig identities, photonics, modern physics" really is one thing: the
time average of an oscillation's square. Verified numerically (integrate over a period)
against the closed forms, and the identities proven in SymPy. NumPy + SymPy; py-3.13.
"""

import numpy as np
import sympy as sp

C_LIGHT = 2.99792458e8
EPS0 = 8.8541878128e-12


# ----------------------------------------------------------------------
# The two trig-identity averages, symbolic and numeric
# ----------------------------------------------------------------------

def average_identities_symbolic():
    """Prove the two averages in SymPy by integrating over one period:
    <cos^2> = 1/2 and <cos(t)cos(t+phi)> = 1/2 cos(phi)."""
    t, phi = sp.symbols("t phi", real=True)
    avg_sq = sp.integrate(sp.cos(t) ** 2, (t, 0, 2 * sp.pi)) / (2 * sp.pi)
    avg_prod = sp.simplify(
        sp.integrate(sp.cos(t) * sp.cos(t + phi), (t, 0, 2 * sp.pi)) / (2 * sp.pi))
    return {"cos_squared": avg_sq, "cos_product": avg_prod}


def average_cos_squared_numeric(n=200000):
    """Numerically average cos^2 over a period -> 1/2 (the root of every RMS)."""
    t = np.linspace(0, 2 * np.pi, n)
    return float(np.trapezoid(np.cos(t) ** 2, t) / (2 * np.pi))


def average_product_numeric(phase_diff, n=200000):
    """Numerically average cos(t) cos(t+phi) over a period -> 1/2 cos(phi). Zero at
    phi=pi/2 (orthogonal), the reason reactive current carries no average power."""
    t = np.linspace(0, 2 * np.pi, n)
    return float(np.trapezoid(np.cos(t) * np.cos(t + phase_diff), t) / (2 * np.pi))


# ----------------------------------------------------------------------
# RMS and AC power
# ----------------------------------------------------------------------

def rms_sinusoid(peak):
    """RMS of a sinusoid = peak / sqrt(2), since <cos^2> = 1/2."""
    if peak < 0:
        raise ValueError("peak must be non-negative")
    return peak / np.sqrt(2)


def rms(signal):
    """Root-mean-square of any sampled signal: sqrt(<x^2>)."""
    x = np.asarray(signal, float)
    if x.size == 0:
        raise ValueError("signal must be non-empty")
    return float(np.sqrt(np.mean(x ** 2)))


def average_power_ac(v_peak, i_peak, phase_diff):
    """Average real power <P> = 1/2 V0 I0 cos(phi) = V_rms I_rms cos(phi)."""
    if v_peak < 0 or i_peak < 0:
        raise ValueError("peak amplitudes must be non-negative")
    return 0.5 * v_peak * i_peak * np.cos(phase_diff)


def power_factor(phase_diff):
    """cos(phi): the fraction of apparent power that is real. 1 = resistive,
    0 = purely reactive (no average power delivered)."""
    return float(np.cos(phase_diff))


def average_power_numeric(v_peak, i_peak, phase_diff, n=200000):
    """Directly time-average v(t)*i(t) over a period -- an independent check of
    average_power_ac, not the same formula restated."""
    t = np.linspace(0, 2 * np.pi, n)
    v = v_peak * np.cos(t)
    i = i_peak * np.cos(t - phase_diff)
    return float(np.trapezoid(v * i, t) / (2 * np.pi))


# ----------------------------------------------------------------------
# Optical intensity: the same average, for a light wave
# ----------------------------------------------------------------------

def optical_intensity(E0, n_index=1.0):
    """Time-averaged intensity of an EM wave I = 1/2 n c eps0 E0^2 = n c eps0 E_rms^2
    -- the mean of the Poynting vector, the same <cos^2>=1/2 as AC power. [W/m^2]."""
    if n_index <= 0:
        raise ValueError("refractive index must be positive")
    return 0.5 * n_index * C_LIGHT * EPS0 * E0 ** 2


def intensity_numeric(E0, n_index=1.0, n=200000):
    """Average of n c eps0 E(t)^2 with E = E0 cos(wt) -- confirms optical_intensity."""
    t = np.linspace(0, 2 * np.pi, n)
    E = E0 * np.cos(t)
    return float(np.trapezoid(n_index * C_LIGHT * EPS0 * E ** 2, t) / (2 * np.pi))


if __name__ == "__main__":
    sym = average_identities_symbolic()
    print("trig identities (SymPy):  <cos^2> =", sym["cos_squared"],
          "   <cos*cos(+phi)> =", sym["cos_product"])
    print(f"numeric: <cos^2> = {average_cos_squared_numeric():.4f}, "
          f"<cos*cos(+pi/3)> = {average_product_numeric(np.pi/3):.4f} "
          f"(= 0.5 cos60 = 0.25)")

    print("\nAC power (V0=170 V, I0=10 A):")
    for name, phi in [("resistive phi=0", 0.0), ("phi=60 deg", np.pi/3),
                      ("reactive phi=90 deg", np.pi/2)]:
        P = average_power_ac(170, 10, phi)
        print(f"  {name:22s}: <P> = {P:7.1f} W, pf = {power_factor(phi):.2f}  "
              f"(numeric {average_power_numeric(170,10,phi):7.1f})")
    print(f"  V_rms = {rms_sinusoid(170):.1f} V (= 170/sqrt2 = 120 V mains)")

    print("\noptical intensity of a 1000 V/m field:")
    I = optical_intensity(1000.0)
    print(f"  I = 1/2 c eps0 E0^2 = {I:.4f} W/m^2  (numeric {intensity_numeric(1000.0):.4f})")
    print(f"  in glass (n=1.5): {optical_intensity(1000.0, 1.5):.4f} W/m^2")
    print("  -> same <cos^2>=1/2 as the wattmeter: light intensity ~ amplitude^2")
