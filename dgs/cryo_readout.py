"""
Why a superconducting qubit needs a dilution refrigerator, and how it is read out.

In plain words
--------------
Two course topics explain most of a quantum computer's cryostat.

**PHYS 106 (Boltzmann factor):** a qubit is two energy levels separated by hf. In thermal
equilibrium at temperature T, the upper level holds the fraction

    P1 = 1 / (1 + exp(hf / k_B T)).

For f = 5 GHz, hf / k_B is only 0.24 K. At liquid-helium temperature (4 K) the two levels are
almost equally populated, so the qubit starts in a random state. At 20 mK, P1 ~ 6e-6. Only a
dilution refrigerator reaches that. A resonator mode at frequency f holds
n_th = 1 / (exp(hf / k_B T) - 1) thermal photons on average (Bose-Einstein), and stray photons
in the readout resonator blur the qubit's phase.

**EEE 117 (RLC resonance):** the qubit is read out through a microwave resonator, which is an
RLC circuit: f0 = 1 / (2 pi sqrt(LC)), Q = (1/R) sqrt(L/C) for the series form, and
bandwidth = f0 / Q. The transfer function depends only on f/f0 and Q, so a 10 kHz bench RLC in
EEE 117L draws the same normalized Bode plot as a 7 GHz on-chip resonator.
In a "hanger" readout, the resonator hangs off a feedline and makes a dip in transmission S21.
The qubit state shifts the dip by +/- chi, so one transmission measurement gives one bit.

**Both together, the attenuator chain:** the room-temperature drive line carries thermal
noise at 300 K. Each attenuator at a cold stage passes 1/A of the incoming noise and adds
(1 - 1/A) of its own thermal noise: n_out = n_in / A + (1 - 1/A) n_th(T_stage). That is why
the fridge's coax lines have attenuators bolted to each plate.

Units: SI throughout (Hz, K, H, F, ohm). Attenuation in dB.

Sources (IEEE style)
--------------------
[1] D. M. Pozar, Microwave Engineering, 4th ed. Wiley, 2012, ch. 6 (RLC resonators, Q).
[2] P. Krantz et al., "A quantum engineer's guide to superconducting qubits," Appl. Phys.
    Rev., vol. 6, 021318, 2019 (dispersive readout, attenuation and thermal photons).
[3] M. S. Khalil et al., "An analysis method for asymmetric resonator transmission
    applied to superconducting devices," J. Appl. Phys., vol. 111, 054510, 2012 (hanger S21).
"""
import math

import numpy as np

H = 6.62607015e-34      # J s (exact)
K_B = 1.380649e-23      # J / K (exact)


def _pos(name, val):
    if not np.all(np.asarray(val) > 0):
        raise ValueError(f"{name} must be > 0, got {val}")


def _x(f_hz, T_k):
    """hf / k_B T."""
    _pos("frequency f_hz", f_hz)
    _pos("temperature T_k", T_k)
    return H * np.asarray(f_hz, dtype=float) / (K_B * np.asarray(T_k, dtype=float))


# --- PHYS 106: Boltzmann and Bose-Einstein ------------------------------------------------
def quantum_temperature(f_hz):
    """hf / k_B in kelvin: the temperature where thermal energy equals one level spacing."""
    _pos("frequency f_hz", f_hz)
    return H * f_hz / K_B


def excited_population(f_hz, T_k):
    """Thermal upper-level population of a two-level system, 1 / (1 + exp(hf/kT))."""
    return 1.0 / (1.0 + np.exp(_x(f_hz, T_k)))


def level_populations(f_hz, T_k, n_levels=3, anharmonicity_hz=0.0):
    """Boltzmann populations of levels E_n = n h f + (alpha/2) n (n - 1) h (n = 0, 1, ...).

    anharmonicity_hz = 0 is a harmonic ladder; a transmon has alpha ~ -200 to -300 MHz.
    """
    if int(n_levels) != n_levels or n_levels < 2:
        raise ValueError(f"n_levels must be an integer >= 2, got {n_levels}")
    _pos("frequency f_hz", f_hz)
    _pos("temperature T_k", T_k)
    n = np.arange(int(n_levels))
    E = H * (n * f_hz + 0.5 * anharmonicity_hz * n * (n - 1))
    if np.any(np.diff(E) <= 0):
        raise ValueError("anharmonicity too negative: levels must increase in energy")
    w = np.exp(-(E - E[0]) / (K_B * T_k))
    return w / w.sum()


def effective_temperature(f_hz, p1):
    """Invert excited_population: the T that gives upper-level population p1 (0 < p1 < 0.5)."""
    if not 0 < p1 < 0.5:
        raise ValueError(f"p1 must be in (0, 0.5) for a positive temperature, got {p1}")
    return quantum_temperature(f_hz) / math.log((1 - p1) / p1)


def mean_photon_number(f_hz, T_k):
    """Bose-Einstein occupation of one mode, 1 / (exp(hf/kT) - 1)."""
    return 1.0 / np.expm1(_x(f_hz, T_k))


# --- EEE 117: RLC resonance ---------------------------------------------------------------
def rlc_resonance(L, C):
    """f0 = 1 / (2 pi sqrt(LC)) in Hz."""
    _pos("L", L)
    _pos("C", C)
    return 1.0 / (2 * math.pi * math.sqrt(L * C))


def series_rlc_q(R, L, C):
    """Quality factor of a series RLC: Q = (1/R) sqrt(L/C) = omega0 L / R."""
    _pos("R", R)
    _pos("L", L)
    _pos("C", C)
    return math.sqrt(L / C) / R


def parallel_rlc_q(R, L, C):
    """Quality factor of a parallel RLC: Q = R sqrt(C/L) = omega0 R C."""
    _pos("R", R)
    _pos("L", L)
    _pos("C", C)
    return R * math.sqrt(C / L)


def series_rlc_bandpass(f_hz, R, L, C):
    """H = V_R / V_in = R / (R + j w L + 1/(j w C)): peak 1 at f0, -3 dB width f0/Q."""
    _pos("frequency f_hz", f_hz)
    _pos("R", R)
    w = 2 * math.pi * np.asarray(f_hz, dtype=float)
    return R / (R + 1j * w * L + 1 / (1j * w * C))


def hanger_s21(f_hz, f0, Qi, Qc):
    """Notch (hanger) resonator transmission, symmetric model.

    S21 = 1 - (Q/Qc) / (1 + 2 j Q (f - f0)/f0), with 1/Q = 1/Qi + 1/Qc.
    Minimum |S21| = Qc / (Qi + Qc) at f = f0.
    """
    _pos("f0", f0)
    _pos("Qi", Qi)
    _pos("Qc", Qc)
    Q = 1.0 / (1.0 / Qi + 1.0 / Qc)
    d = (np.asarray(f_hz, dtype=float) - f0) / f0
    return 1 - (Q / Qc) / (1 + 2j * Q * d)


def dispersive_readout(f_probe, f0, chi_hz, Qi, Qc):
    """S21 at f_probe when the qubit is in |0> (resonator at f0 - chi) or |1> (f0 + chi).

    Returns (S21_0, S21_1, separation |S21_1 - S21_0|). Probing at f0 (the midpoint)
    maximizes the separation when chi is comparable to the linewidth.
    """
    _pos("chi_hz", chi_hz)
    s0 = hanger_s21(f_probe, f0 - chi_hz, Qi, Qc)
    s1 = hanger_s21(f_probe, f0 + chi_hz, Qi, Qc)
    return s0, s1, np.abs(s1 - s0)


# --- both: the attenuator chain -----------------------------------------------------------
def attenuator_chain(f_hz, T_source, stages):
    """Thermal photon number after each attenuator on a drive line.

    stages: list of (attenuation_dB, T_stage_K), from warm to cold.
    Each stage: n_out = n_in / A + (1 - 1/A) n_th(T_stage), with A = 10^(dB/10).
    Returns a list of photon numbers, starting with the source.
    """
    n = [float(mean_photon_number(f_hz, T_source))]
    for dB, T in stages:
        if dB < 0:
            raise ValueError(f"attenuation must be >= 0 dB, got {dB}")
        A = 10 ** (dB / 10)
        n.append(n[-1] / A + (1 - 1 / A) * float(mean_photon_number(f_hz, T)))
    return n
