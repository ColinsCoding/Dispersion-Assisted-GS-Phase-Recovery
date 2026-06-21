"""AC power and op-amps -- where calculus becomes a circuit.

Two ideas, tightly linked:

  * AC power. With v(t) and i(t) sinusoidal, the instantaneous power p=vi has an
    average that is NOT V*I but V_rms*I_rms*cos(phi) -- the phase angle phi between
    voltage and current sets the power factor. Complex power S = P + jQ splits it
    into real (P, watts, does work) and reactive (Q, VAR, sloshes in L and C).

  * Op-amps. An op-amp with a feedback capacitor INTEGRATES its input; with a
    feedback resistor and an input capacitor it DIFFERENTIATES. So integration and
    differentiation -- inverse operations by the Fundamental Theorem of Calculus --
    are inverse *circuits*: differentiator(integrator(v)) = v. And a summing
    amplifier ADDS voltages: 1 V + 2 V -> 3 V, the analog cousin of the digital
    adder in dgs.logic_timing (1 + 2 = 3).

Numerical, leaning on dgs.numerical_methods for the calculus. Education.
"""

import numpy as np
from dgs import numerical_methods as nm


# ── AC power ─────────────────────────────────────────────────────────
def rms(signal):
    """Root-mean-square value: sqrt(mean(v^2)). For a sinusoid of amplitude A this
    is A/sqrt(2) -- the DC value delivering the same average power."""
    s = np.asarray(signal, float)
    return float(np.sqrt(np.mean(s ** 2)))


def average_power(v, i, t):
    """Average power = (1/T) integral of p(t)=v(t)i(t) over the record (numerically).
    Equals V_rms*I_rms*cos(phi) for sinusoids -- the active power that does work."""
    v, i, t = np.asarray(v, float), np.asarray(i, float), np.asarray(t, float)
    p = v * i
    return float(nm.trapezoid(p, t) / (t[-1] - t[0]))


def power_factor(phase_v, phase_i):
    """cos of the angle between voltage and current. 1 = resistive (all real power),
    0 = purely reactive (no net work, current just sloshes)."""
    return float(np.cos(phase_v - phase_i))


def complex_power(Vrms, Irms, phase_diff):
    """Complex power S = V_rms*I_rms*exp(j*phi) = P + jQ.
    P = real part (active, W); Q = imag part (reactive, VAR); |S| = apparent (VA)."""
    return Vrms * Irms * np.exp(1j * phase_diff)


# ── AC impedance (Griffiths' phasor circuits) ───────────────────────
def impedance_R(R):
    """Resistor: Z = R (real, in phase)."""
    return complex(R, 0.0)


def impedance_L(L, omega):
    """Inductor: Z = j*omega*L (current LAGS voltage by 90 deg)."""
    return complex(0.0, omega * L)


def impedance_C(C, omega):
    """Capacitor: Z = 1/(j*omega*C) = -j/(omega*C) (current LEADS voltage by 90 deg)."""
    return complex(0.0, -1.0 / (omega * C))


def series_impedance(*Z):
    """Series impedances add."""
    return sum(Z, complex(0))


# ── op-amps: gain, analog addition, and calculus ────────────────────
def inverting_gain(Rf, Rin):
    """Inverting amplifier gain = -Rf/Rin."""
    return -Rf / Rin


def noninverting_gain(Rf, Rin):
    """Non-inverting amplifier gain = 1 + Rf/Rin."""
    return 1.0 + Rf / Rin


def summing_amplifier(voltages, R_ins, Rf):
    """Summing (inverting) amp: Vout = -Rf * sum(V_k / R_k). With all R equal to Rf it
    is plain analog ADDITION: 1 V + 2 V -> -3 V. The analog twin of a digital adder."""
    return float(-Rf * sum(v / r for v, r in zip(voltages, R_ins)))


def opamp_integrator(vin, t, R, C):
    """Integrating op-amp: Vout(t) = -1/(RC) * integral vin dt. The circuit that
    performs integration (FTC, left half)."""
    return -1.0 / (R * C) * nm.cumulative_integral(vin, t)


def opamp_differentiator(vin, t, R, C):
    """Differentiating op-amp: Vout(t) = -RC * d(vin)/dt. The circuit that performs
    differentiation (FTC, right half)."""
    return -R * C * nm.gradient(vin, t)


if __name__ == "__main__":
    # AC power on an RL load: current lags voltage -> power factor < 1
    t = np.linspace(0, 1, 2000)          # one second; f = 1 Hz
    w = 2 * np.pi
    Vm, Im, phi = 10.0, 2.0, np.pi / 3   # 60 deg lag
    v = Vm * np.cos(w * t)
    i = Im * np.cos(w * t - phi)
    P_num = average_power(v, i, t)
    P_formula = rms(v) * rms(i) * power_factor(0, phi)
    print(f"avg power: numeric {P_num:.4f} W  vs  Vrms Irms cos(phi) {P_formula:.4f} W")
    S = complex_power(rms(v), rms(i), phi)
    print(f"complex power S = {S.real:.3f} + j{S.imag:.3f}  (P watts, Q VAR), "
          f"power factor = {power_factor(0, phi):.3f}")

    # analog addition: 1 V + 2 V -> 3 V (magnitude)
    print("summing amp 1V + 2V =", abs(summing_amplifier([1.0, 2.0], [1e3, 1e3], 1e3)), "V")

    # FTC in hardware: differentiate the integral of a sine -> get the sine back
    vin = np.sin(w * t)
    R, C = 1e3, 1e-3
    recovered = opamp_differentiator(opamp_integrator(vin, t, R, C), t, R, C)
    err = np.max(np.abs(recovered[5:-5] - vin[5:-5]))
    print(f"differentiator(integrator(sin)) recovers sin: max err {err:.2e}  (FTC)")
