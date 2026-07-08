"""Power electronics: turning AC into DC, and one DC voltage into another.

Two workhorses power every device you own:

  RECTIFIERS turn AC into DC by letting current pass only one way. Averaging the
  rectified sinusoid over a cycle gives the DC output; how much it still wiggles is the
  RIPPLE. For a peak Vp:
        half-wave:  V_dc = Vp/pi   ~= 0.318 Vp,  V_rms = Vp/2,     ripple factor ~1.21
        full-wave:  V_dc = 2Vp/pi  ~= 0.637 Vp,  V_rms = Vp/sqrt2, ripple factor ~0.48
  Full-wave uses both halves, so it delivers twice the DC and a third of the ripple.

  DC-DC CONVERTERS trade voltage for current by switching an inductor at duty cycle D
  (fraction of each cycle the switch is on). In steady state the inductor's volt-seconds
  must balance, which fixes the output:
        buck (step-DOWN):  V_out = D V_in            (0 < V_out < V_in)
        boost (step-UP):   V_out = V_in / (1 - D)    (V_out > V_in)
        buck-boost:        V_out = -V_in D / (1 - D) (inverts, either magnitude)
  The switching also leaves an INDUCTOR RIPPLE CURRENT (bigger with smaller L or slower
  switching) and a residual OUTPUT VOLTAGE RIPPLE the capacitor smooths.

Everything is a short closed form, checked: the rectifier DC/RMS against a numerical
average of the rectified wave, the converter laws by volt-second balance, and the ripple
scalings. Ties to dgs.average_power (RMS, <cos^2>) and the circuit thread. NumPy only;
py-3.13.
"""

import numpy as np


# ----------------------------------------------------------------------
# Rectifiers: AC -> DC
# ----------------------------------------------------------------------

def ripple_factor(v_rms, v_dc):
    """Ripple factor r = sqrt((V_rms/V_dc)^2 - 1): the AC (wiggle) content relative
    to the DC level. 0 is perfectly flat DC; larger is more ripple."""
    if v_dc <= 0:
        raise ValueError("v_dc must be positive")
    return float(np.sqrt(max((v_rms / v_dc) ** 2 - 1, 0.0)))


def halfwave_rectifier(v_peak):
    """Half-wave rectifier output: passes only the positive half. V_dc = Vp/pi,
    V_rms = Vp/2, ripple factor ~1.21 -- poor, but one diode."""
    if v_peak <= 0:
        raise ValueError("v_peak must be positive")
    v_dc = v_peak / np.pi
    v_rms = v_peak / 2
    return {"v_dc": v_dc, "v_rms": v_rms, "ripple_factor": ripple_factor(v_rms, v_dc)}


def fullwave_rectifier(v_peak):
    """Full-wave rectifier output: flips the negative half. V_dc = 2Vp/pi,
    V_rms = Vp/sqrt2, ripple factor ~0.48 -- twice the DC, far less ripple."""
    if v_peak <= 0:
        raise ValueError("v_peak must be positive")
    v_dc = 2 * v_peak / np.pi
    v_rms = v_peak / np.sqrt(2)
    return {"v_dc": v_dc, "v_rms": v_rms, "ripple_factor": ripple_factor(v_rms, v_dc)}


def rectifier_numeric(v_peak, full_wave=True, n=200000):
    """Numerically average the rectified sine over a period -> (V_dc, V_rms), an
    independent check of the closed forms."""
    t = np.linspace(0, 2 * np.pi, n)
    s = v_peak * np.sin(t)
    rect = np.abs(s) if full_wave else np.maximum(s, 0.0)
    v_dc = float(np.trapezoid(rect, t) / (2 * np.pi))
    v_rms = float(np.sqrt(np.trapezoid(rect ** 2, t) / (2 * np.pi)))
    return v_dc, v_rms


# ----------------------------------------------------------------------
# DC-DC converters: volt-second balance sets V_out
# ----------------------------------------------------------------------

def _check_duty(D):
    if not 0 < D < 1:
        raise ValueError("duty cycle D must be in (0, 1)")


def buck_output(v_in, duty):
    """Buck (step-down): V_out = D V_in, always below V_in."""
    _check_duty(duty)
    return duty * v_in


def boost_output(v_in, duty):
    """Boost (step-up): V_out = V_in/(1-D), always above V_in."""
    _check_duty(duty)
    return v_in / (1 - duty)


def buck_boost_output(v_in, duty):
    """Buck-boost: V_out = -V_in D/(1-D) (polarity-inverting; magnitude can be
    below or above V_in depending on D)."""
    _check_duty(duty)
    return -v_in * duty / (1 - duty)


def duty_for_buck(v_in, v_out):
    """Duty needed for a buck to make V_out from V_in: D = V_out/V_in (0<V_out<V_in)."""
    if not 0 < v_out < v_in:
        raise ValueError("buck requires 0 < v_out < v_in")
    return v_out / v_in


def duty_for_boost(v_in, v_out):
    """Duty for a boost: D = 1 - V_in/V_out (V_out > V_in > 0)."""
    if not 0 < v_in < v_out:
        raise ValueError("boost requires 0 < v_in < v_out")
    return 1 - v_in / v_out


def inductor_ripple_current(v_in, duty, L, f_sw, topology="buck"):
    """Peak-to-peak inductor ripple current. For a buck, during the on-time D/f_sw the
    inductor sees V_in - V_out = V_in(1-D), so dI = V_in(1-D)D/(L f_sw); for a boost the
    inductor sees V_in for D/f_sw, so dI = V_in D/(L f_sw). Smaller L or faster switching
    -> less ripple."""
    _check_duty(duty)
    if L <= 0 or f_sw <= 0:
        raise ValueError("L and f_sw must be positive")
    if topology == "buck":
        return v_in * (1 - duty) * duty / (L * f_sw)
    if topology == "boost":
        return v_in * duty / (L * f_sw)
    raise ValueError("topology must be 'buck' or 'boost'")


def output_voltage_ripple(ripple_current, C, f_sw):
    """Residual output ripple of a buck after the capacitor: dV = dI/(8 C f_sw)."""
    if C <= 0 or f_sw <= 0:
        raise ValueError("C and f_sw must be positive")
    return ripple_current / (8 * C * f_sw)


def efficiency(p_out, p_in):
    """Converter efficiency eta = P_out/P_in (the rest is switching + conduction loss)."""
    if p_in <= 0 or p_out < 0:
        raise ValueError("need p_in > 0 and p_out >= 0")
    return p_out / p_in


if __name__ == "__main__":
    Vp = 170.0    # ~120 Vrms mains peak
    hw, fw = halfwave_rectifier(Vp), fullwave_rectifier(Vp)
    print("rectifying 170 V peak (120 Vrms) mains:")
    print(f"  half-wave: V_dc={hw['v_dc']:.1f} V, V_rms={hw['v_rms']:.1f} V, "
          f"ripple={hw['ripple_factor']:.2f}")
    print(f"  full-wave: V_dc={fw['v_dc']:.1f} V, V_rms={fw['v_rms']:.1f} V, "
          f"ripple={fw['ripple_factor']:.2f}  (2x DC, ~1/3 the ripple)")
    dc_n, rms_n = rectifier_numeric(Vp, full_wave=True)
    print(f"  numeric full-wave check: V_dc={dc_n:.1f}, V_rms={rms_n:.1f}")

    print("\nDC-DC converters from 12 V at D=0.5:")
    print(f"  buck  -> {buck_output(12, 0.5):.1f} V (step down)")
    print(f"  boost -> {boost_output(12, 0.5):.1f} V (step up)")
    print(f"  buck-boost -> {buck_boost_output(12, 0.5):.1f} V (inverted)")
    print(f"  to make 5 V from 12 V, a buck needs D = {duty_for_buck(12, 5):.3f}")

    dI = inductor_ripple_current(12, 5/12, L=100e-6, f_sw=500e3, topology="buck")
    dV = output_voltage_ripple(dI, C=47e-6, f_sw=500e3)
    print(f"\n5 V buck (100 uH, 470 kHz-ish, 47 uF): inductor ripple {dI*1e3:.1f} mA, "
          f"output ripple {dV*1e3:.2f} mV")
