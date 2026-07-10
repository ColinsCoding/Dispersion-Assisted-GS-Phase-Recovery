"""The photonic op-amp: a transimpedance amplifier turning photocurrent into a voltage.

The front end of every optical receiver -- including the dispersion-GS / time-stretch pipeline
this repo is built around -- is a photodiode feeding a TRANSIMPEDANCE AMPLIFIER (TIA): an op-amp
with a feedback resistor R_f that converts the tiny photocurrent into a readable voltage. Light of
power P lands on the diode, which sources a current

    I_ph = R_lambda * P,   R_lambda = eta q lambda / (h c)   (responsivity, ~1.25 A/W at 1550 nm),

and the TIA outputs V_out = -I_ph R_f. The transimpedance gain is just R_f (units V/A = ohms).

THE CENTRAL TRADE-OFF.  A big R_f gives big gain and -- crucially -- LOW thermal noise, but the
feedback resistor works against the total input capacitance C (photodiode + amplifier) to set the
bandwidth f_3dB = 1/(2 pi R_f C). So

    R_f * f_3dB = 1 / (2 pi C) = constant:

you buy gain and low noise with bandwidth. That is the whole design tension of an optical receiver.

THE NOISE BUDGET (input-referred current, over bandwidth B):
    * SHOT noise of the photocurrent + dark current:  i^2 = 2 q (I_ph + I_dark) B      (signal-dependent)
    * THERMAL (Johnson) noise of R_f:                  i^2 = 4 kT B / R_f               (smaller for big R_f)
    * AMPLIFIER voltage noise e_n through C:           i^2 = (2 pi C e_n)^2 B^3 / 3     (rises steeply with B)
A receiver is "shot-noise limited" (the ideal) when the first term dominates, and
"thermal/amplifier limited" otherwise. From the total noise current we get the SNR, the
noise-equivalent power NEP = i_n / R_lambda, and the sensitivity (minimum P for a target SNR).

NumPy; SI units throughout (watts, amps, ohms, farads, hertz). py-3.13.
"""

import numpy as np

Q_E = 1.602176634e-19       # elementary charge, C
K_B = 1.380649e-23          # Boltzmann, J/K
H_PL = 6.62607015e-34       # Planck, J s
C_LIGHT = 299792458.0       # m/s


def responsivity(wavelength_nm, quantum_efficiency=1.0):
    """Photodiode responsivity R_lambda = eta q lambda /(h c) in A/W. Ideal InGaAs at 1550 nm
    is ~1.25 A/W; a real diode has eta < 1."""
    if wavelength_nm <= 0 or not (0 < quantum_efficiency <= 1):
        raise ValueError("wavelength_nm > 0 and 0 < quantum_efficiency <= 1")
    lam = wavelength_nm * 1e-9
    return quantum_efficiency * Q_E * lam / (H_PL * C_LIGHT)


def photocurrent(P_opt, resp):
    """Photocurrent I_ph = R_lambda * P_opt (amps)."""
    if P_opt < 0:
        raise ValueError("optical power must be >= 0")
    return resp * P_opt


def output_voltage(I_ph, R_f):
    """TIA output magnitude V_out = I_ph * R_f (the transimpedance gain is R_f, V/A)."""
    if R_f <= 0:
        raise ValueError("R_f must be > 0")
    return I_ph * R_f


def bandwidth_3db(R_f, C):
    """-3 dB bandwidth of the TIA, f = 1/(2 pi R_f C) (Hz)."""
    if R_f <= 0 or C <= 0:
        raise ValueError("R_f and C must be > 0")
    return 1.0 / (2 * np.pi * R_f * C)


def gain_bandwidth_product(C):
    """R_f * f_3dB = 1/(2 pi C): the fixed budget you split between gain and bandwidth."""
    if C <= 0:
        raise ValueError("C must be > 0")
    return 1.0 / (2 * np.pi * C)


def shot_noise_current(I_dc, B):
    """RMS shot-noise current sqrt(2 q I_dc B) over bandwidth B (I_dc = photocurrent + dark)."""
    if I_dc < 0 or B <= 0:
        raise ValueError("I_dc >= 0 and B > 0")
    return np.sqrt(2 * Q_E * I_dc * B)


def thermal_noise_current(R_f, B, T=300.0):
    """RMS Johnson noise current of the feedback resistor, sqrt(4 kT B / R_f). LARGER R_f
    means LESS current noise -- the reason high-gain TIAs are quiet."""
    if R_f <= 0 or B <= 0 or T <= 0:
        raise ValueError("R_f, B, T must be > 0")
    return np.sqrt(4 * K_B * T * B / R_f)


def amplifier_noise_current(e_n, C, B):
    """RMS input-referred current from the amplifier voltage noise e_n acting on the total
    capacitance C: integral of (2 pi f C e_n)^2 df from 0 to B = (2 pi C e_n)^2 B^3/3."""
    if e_n < 0 or C < 0 or B <= 0:
        raise ValueError("e_n, C >= 0 and B > 0")
    return 2 * np.pi * C * e_n * np.sqrt(B ** 3 / 3.0)


def total_noise_current(P_opt, resp, R_f, C, B, I_dark=0.0, e_n=0.0, T=300.0):
    """RMS input-referred noise current: quadrature sum of shot, thermal, and amplifier terms."""
    I_ph = photocurrent(P_opt, resp)
    i_shot = shot_noise_current(I_ph + I_dark, B)
    i_therm = thermal_noise_current(R_f, B, T)
    i_amp = amplifier_noise_current(e_n, C, B)
    return np.sqrt(i_shot ** 2 + i_therm ** 2 + i_amp ** 2)


def snr(P_opt, resp, R_f, C, B, I_dark=0.0, e_n=0.0, T=300.0):
    """Electrical signal-to-noise ratio I_ph / i_n (linear, amplitude). SNR_power = this^2."""
    I_ph = photocurrent(P_opt, resp)
    i_n = total_noise_current(P_opt, resp, R_f, C, B, I_dark, e_n, T)
    return I_ph / i_n


def noise_equivalent_power(resp, R_f, C, B, I_dark=0.0, e_n=0.0, T=300.0):
    """NEP: optical power giving SNR(power)=1, referred through the responsivity (W). At P=0 the
    noise is thermal+amplifier only (dark shot included)."""
    i_n = total_noise_current(0.0, resp, R_f, C, B, I_dark, e_n, T)
    return i_n / resp


def sensitivity(target_snr, resp, R_f, C, B, I_dark=0.0, e_n=0.0, T=300.0):
    """Minimum optical power (W) for a target amplitude SNR, solving I_ph = SNR * i_n with the
    signal-dependent shot term included (quadratic in P)."""
    if target_snr <= 0:
        raise ValueError("target_snr must be > 0")
    # thermal+amplifier+dark noise power (P-independent part)
    i0_sq = (thermal_noise_current(R_f, B, T) ** 2
             + amplifier_noise_current(e_n, C, B) ** 2
             + 2 * Q_E * I_dark * B)
    # solve (R P)^2 = S^2 (2 q R P B + i0^2)  ->  R^2 P^2 - S^2 2qB R P - S^2 i0^2 = 0
    a = resp ** 2
    b = -target_snr ** 2 * 2 * Q_E * B * resp
    c = -target_snr ** 2 * i0_sq
    P = (-b + np.sqrt(b ** 2 - 4 * a * c)) / (2 * a)
    return P


# ── stability & feedback compensation (the op-amp part of the "photonic op-amp") ──────────
# A TIA is an op-amp inside a feedback loop, and the photodiode capacitance C_in makes that loop
# want to RING. The feedback factor rolls off at the input pole 1/(2 pi R_f C_in), adding phase lag
# that eats the loop's phase margin, so the closed-loop response peaks (or oscillates). The fix is a
# small feedback capacitor C_f across R_f, whose zero 1/(2 pi R_f C_f) boosts the phase back before
# crossover. The best achievable bandwidth is the GEOMETRIC MEAN of the amplifier's gain-bandwidth
# and the input pole -- you cannot out-run the op-amp.

def input_pole_frequency(R_f, C_in):
    """The input-network pole f_in = 1/(2 pi R_f C_in) (Hz) -- where the feedback factor starts
    rolling off and the loop begins to lose phase margin."""
    if R_f <= 0 or C_in <= 0:
        raise ValueError("R_f and C_in must be > 0")
    return 1.0 / (2 * np.pi * R_f * C_in)


def closed_loop_bandwidth(R_f, C_in, f_gbw):
    """Maximally-flat closed-loop -3 dB bandwidth of a compensated TIA: the geometric mean of the
    op-amp gain-bandwidth f_gbw and the input pole, f_3dB = sqrt(f_gbw / (2 pi R_f C_in)). You
    can't beat the op-amp: bandwidth is limited by sqrt(f_gbw * f_input_pole)."""
    if f_gbw <= 0:
        raise ValueError("f_gbw must be > 0")
    return np.sqrt(f_gbw * input_pole_frequency(R_f, C_in))


def compensation_capacitor(R_f, C_in, f_gbw):
    """Feedback capacitor C_f for a maximally-flat (Butterworth) response: place its zero at the
    closed-loop bandwidth, C_f = sqrt(C_in / (2 pi R_f f_gbw)). Bigger C_in or slower op-amp
    needs more compensation (and costs bandwidth)."""
    if f_gbw <= 0:
        raise ValueError("f_gbw must be > 0")
    return np.sqrt(C_in / (2 * np.pi * R_f * f_gbw))


def phase_margin(R_f, C_in, C_f, f_gbw):
    """Loop phase margin (degrees) for a single-pole op-amp model A(s)=2 pi f_gbw/s with feedback
    factor beta(s)=(1+s R_f C_f)/(1+s R_f (C_in+C_f)). Finds the 0 dB crossover of |A*beta| and
    reports 180 + angle. ~65 deg is maximally flat; below ~45 deg the TIA rings."""
    if R_f <= 0 or C_in <= 0 or C_f < 0 or f_gbw <= 0:
        raise ValueError("need R_f>0, C_in>0, C_f>=0, f_gbw>0")
    f_in = input_pole_frequency(R_f, C_in)
    f = np.logspace(np.log10(f_in) - 3, np.log10(f_gbw) + 2, 6000)
    s = 2j * np.pi * f
    L = (2 * np.pi * f_gbw / s) * (1 + s * R_f * C_f) / (1 + s * R_f * (C_in + C_f))
    mag = np.abs(L)
    crossings = np.where(np.diff(np.sign(mag - 1.0)))[0]
    if len(crossings) == 0:
        return 180.0
    k = crossings[-1]                                        # last (highest-f) unity crossing
    # log-interpolate the crossover frequency
    m0, m1 = np.log(mag[k]), np.log(mag[k + 1])
    frac = (0.0 - m0) / (m1 - m0)
    f_c = f[k] * (f[k + 1] / f[k]) ** frac
    phase = -90.0 + np.degrees(np.arctan2(2 * np.pi * f_c * R_f * C_f, 1.0)) \
        - np.degrees(np.arctan2(2 * np.pi * f_c * R_f * (C_in + C_f), 1.0))
    return 180.0 + phase


if __name__ == "__main__":
    lam, eta = 1550.0, 0.8
    R = responsivity(lam, eta)
    print(f"responsivity at {lam} nm (eta={eta}): {R:.3f} A/W  (ideal {responsivity(lam):.3f})")

    C, e_n = 0.5e-12, 2e-9          # 0.5 pF total, 2 nV/rtHz amplifier
    print("\ngain-bandwidth trade-off (fixed C = 0.5 pF):")
    for R_f in (1e3, 1e4, 1e5):
        print(f"  R_f={R_f:7.0f} ohm -> gain {R_f:7.0f} V/A, f_3dB = "
              f"{bandwidth_3db(R_f, C)/1e6:8.1f} MHz  (R_f*f = {R_f*bandwidth_3db(R_f,C):.3e})")
    print(f"  fixed budget R_f*f_3dB = 1/(2 pi C) = {gain_bandwidth_product(C):.3e}")

    print("\nnoise budget at P = 1 uW, R_f = 10 kohm, B = 100 MHz:")
    P, R_f, B = 1e-6, 1e4, 100e6
    I_ph = photocurrent(P, R)
    print(f"  I_ph = {I_ph*1e6:.2f} uA, V_out = {output_voltage(I_ph,R_f)*1e3:.1f} mV")
    print(f"  shot      = {shot_noise_current(I_ph, B)*1e9:6.2f} nA")
    print(f"  thermal   = {thermal_noise_current(R_f, B)*1e9:6.2f} nA")
    print(f"  amplifier = {amplifier_noise_current(e_n, C, B)*1e9:6.2f} nA")
    print(f"  total     = {total_noise_current(P,R,R_f,C,B,e_n=e_n)*1e9:6.2f} nA  ->  "
          f"SNR = {snr(P,R,R_f,C,B,e_n=e_n):.1f} ({20*np.log10(snr(P,R,R_f,C,B,e_n=e_n)):.1f} dB)")
    print(f"  NEP = {noise_equivalent_power(R,R_f,C,B,e_n=e_n)*1e9:.3f} nW ; "
          f"sensitivity (SNR=7) = {sensitivity(7,R,R_f,C,B,e_n=e_n)*1e9:.1f} nW")

    print("\nstability / compensation (R_f = 10 kohm, C_in = 2 pF, op-amp GBW = 1 GHz):")
    R_f, C_in, f_gbw = 1e4, 2e-12, 1e9
    Cf = compensation_capacitor(R_f, C_in, f_gbw)
    print(f"  input pole            = {input_pole_frequency(R_f, C_in)/1e6:6.2f} MHz")
    print(f"  best closed-loop BW   = {closed_loop_bandwidth(R_f, C_in, f_gbw)/1e6:6.2f} MHz "
          f"(geometric mean of GBW and input pole)")
    print(f"  recommended C_f       = {Cf*1e15:6.1f} fF")
    print(f"  phase margin: uncompensated (C_f=0) = {phase_margin(R_f,C_in,0.0,f_gbw):5.1f} deg "
          f"(rings) ;  with C_f = {phase_margin(R_f,C_in,Cf,f_gbw):5.1f} deg (flat)")
