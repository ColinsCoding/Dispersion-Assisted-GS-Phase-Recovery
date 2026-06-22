"""Rectifiers -- turning AC into DC, and the nonlinearity that is also ReLU.

A diode is a one-way valve for current. Build circuits from it and you get:
  * HALF-WAVE rectifier (one diode): passes the positive half, blocks the negative ->
    output = max(vin, 0). That is EXACTLY ReLU, the standard neural-network activation.
  * FULL-WAVE rectifier (a 4-diode bridge): both half-cycles come out positive ->
    output = |vin|. Twice the average DC and twice the ripple frequency.
  * SMOOTHING capacitor after the bridge: a peak-following RC (the same tau=RC as
    dgs.spice / dgs.membrane_biophysics) that turns the bumpy rectified wave into DC
    with a small ripple.

So the same rectifying nonlinearity powers a phone charger AND gates signals in a
deep net -- electronics and machine learning meeting at max(x, 0). NumPy. Education.
"""

import numpy as np


def diode_iv(v, Is=1e-12, Vt=0.02585):
    """Shockley diode equation I = Is*(exp(v/Vt) - 1) [A]. Exponential forward current
    for v>0, ~ -Is (negligible) for reverse v<0. Vt = kT/q ~ 25.85 mV at room temp.
    The exponent is clipped to avoid overflow."""
    return Is * (np.exp(np.clip(np.asarray(v, float) / Vt, -50, 50)) - 1.0)


def half_wave_rectify(vin):
    """One ideal diode: pass the positive half, block the negative -> max(vin, 0).
    This IS ReLU, the most common neural-network activation."""
    return np.maximum(np.asarray(vin, float), 0.0)


def full_wave_rectify(vin):
    """Diode bridge (4 diodes): both half-cycles come out positive -> |vin|. Double
    the average output and double the ripple frequency of the half-wave version."""
    return np.abs(np.asarray(vin, float))


def average_output(vpeak, mode="full"):
    """DC (time-average) of a rectified sinusoid of amplitude vpeak:
    full-wave = (2/pi) vpeak ~ 0.637 vpeak;  half-wave = vpeak/pi ~ 0.318 vpeak."""
    return (2.0 / np.pi) * vpeak if mode == "full" else vpeak / np.pi


def rc_smooth(vin, t, R, C):
    """Smoothing capacitor after the rectifier: a peak-following RC. When the rectified
    input exceeds the cap voltage the diode conducts and the cap follows the peak;
    otherwise the diode is off and the cap discharges through the load (tau=RC). Returns
    the smoothed DC-with-ripple waveform."""
    vin = np.asarray(vin, float)
    V = np.empty_like(vin)
    V[0] = vin[0]
    dt = t[1] - t[0]
    decay = np.exp(-dt / (R * C))
    for n in range(1, len(vin)):
        V[n] = vin[n] if vin[n] >= V[n - 1] else V[n - 1] * decay
    return V


def ripple_voltage(I_load, C, f, full_wave=True):
    """Approximate peak-to-peak ripple after smoothing: V_r ~ I_load/(f_ripple * C),
    with f_ripple = 2f for full-wave (both halves) or f for half-wave. Bigger C -> less
    ripple; full-wave ripples HALF as much as half-wave for the same C (double freq)."""
    f_ripple = 2 * f if full_wave else f
    return I_load / (f_ripple * C)


if __name__ == "__main__":
    t = np.linspace(0, 0.04, 4000)        # 40 ms, two cycles of 50 Hz
    vin = 5.0 * np.sin(2 * np.pi * 50 * t)
    hw, fw = half_wave_rectify(vin), full_wave_rectify(vin)
    print(f"half-wave avg = {hw.mean():.3f} V  (Vpeak/pi = {average_output(5,'half'):.3f})")
    print(f"full-wave avg = {fw.mean():.3f} V  (2 Vpeak/pi = {average_output(5,'full'):.3f})")
    print(f"half_wave_rectify IS ReLU: {np.allclose(hw, np.maximum(vin,0))}")
    sm = rc_smooth(fw, t, R=1000, C=100e-6)
    tail = sm[t > 0.03]                    # steady state (skip the startup charge-up)
    print(f"smoothed (RC=100ms) steady ripple = {tail.max()-tail.min():.3f} V "
          f"(vs unsmoothed {fw.max()-fw.min():.2f} V) -- the cap holds near the peak")
