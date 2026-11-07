"""The microwave chain of a superconducting qubit, designed like an RF transmitter and receiver.

In plain words: a superconducting qubit is a ~5 GHz resonator that only has two usable levels. To
flip it you send a short, precisely shaped microwave burst; to read it you bounce a second tone off a
resonator and listen to the faint echo. Both are radio problems, the same ones taught in an RF and
microwave transmitter design course (UCLA ECE 279AS), just at the single-photon end:

  * TRANSMIT (drive). An AWG makes slow I and Q envelopes (~100 MHz). An IQ mixer moves them up to
    f_LO + f_IF (single sideband). Real mixers are imperfect: a gain mismatch g and phase error phi
    leave an IMAGE at f_LO - f_IF, with image rejection
        IRR = (1 + 2 g cos phi + g^2) / (1 - 2 g cos phi + g^2),
    and DC offsets leak the LO itself. iq_upconvert() builds the RF waveform and tone_amplitude()
    measures each line, so the formula is checked against the waveform.
  * GET COLD (drive line). Room-temperature electronics radiate thermal photons: at 5 GHz and 300 K
    there are ~1250 per mode, and each one can dephase the qubit. Attenuators bolted to colder and
    colder stages of the fridge throw most of the hot noise away and replace it with cold noise:
        n_out = A n_in + (1 - A) n_th(T_stage),  A = power transmission.
    drive_line() walks the chain stage by stage (powers in dBm and photon numbers).
  * RECEIVE (readout). The echo is around -130 dBm, so the receiver's first amplifier sets the noise
    (Friis): T_sys = T1 + T2/G1 + T3/(G1 G2) + ... A near-quantum-limited parametric amplifier
    first, then a 4 K HEMT, then room temperature. readout_fidelity() turns T_sys, signal power and
    integration time into a single-shot error rate with the matched-filter result
        d/sigma = sqrt(2 E / N0),  E = P_diff tau,  N0 = k_B T_sys,  error = erfc(d / (2 sqrt 2 sigma)) / 2.
  * THE STACK. Like the OSI model for networks, a quantum computer is layers: algorithm -> circuit ->
    error correction -> calibrated gate -> pulse -> IQ samples -> microwave volts. QUANTUM_STACK lists
    them next to the OSI layer each one resembles, and compile_gate() walks one gate down the stack
    (rotation angle -> Gaussian pulse by the pulse-area theorem theta = integral Omega dt -> I/Q samples)
    and simulates the qubit to confirm the rotation.

What is illustrative rather than measured, stated plainly: the stage temperatures and attenuations
(a common textbook layout, written from memory), amplifier gains and noise temperatures, the volts-to-
Rabi-frequency coupling, and the readout signal power. Every formula is checked; the numbers are
plausible orders of magnitude, not a specific lab's fridge.

Connects to: dgs.qubit_scope (Rabi/Ramsey and the scope view), dgs.gates_to_internet (OSI stack),
dgs.qubit_analog_bench (a classical bench analog you can build with a scope), dgs.circuit_rules.
"""
from __future__ import annotations

import math

import numpy as np

KB = 1.380649e-23
H = 6.62607015e-34


# ---------------------------------------------------------------- units
def dbm_to_w(p_dbm):
    return 1e-3 * 10 ** (np.asarray(p_dbm, float) / 10)


def w_to_dbm(p_w):
    p_w = np.asarray(p_w, float)
    if np.any(p_w <= 0):
        raise ValueError("power must be positive to express in dBm")
    return 10 * np.log10(p_w / 1e-3)


def vpeak_from_dbm(p_dbm, z0=50.0):
    """Peak voltage of a sine carrying p_dbm into z0: P = V^2 / (2 Z0)."""
    return np.sqrt(2 * dbm_to_w(p_dbm) * z0)


def thermal_photons(f, T):
    """Bose-Einstein occupation n = 1 / (exp(hf / kT) - 1) of one mode at frequency f, temperature T."""
    if f <= 0:
        raise ValueError("frequency must be positive")
    if T <= 0:
        return 0.0
    x = H * f / (KB * T)
    return 1.0 / math.expm1(x)


# ---------------------------------------------------------------- transmit: IQ upconversion
def iq_upconvert(t, I, Q, f_lo, gain_imbalance=1.0, phase_error=0.0, dc_i=0.0, dc_q=0.0):
    """RF = (I + dc_i) cos(w t) - g (Q + dc_q) sin(w t + phi). With I = cos, Q = sin -> f_LO + f_IF."""
    w = 2 * np.pi * f_lo
    return (np.asarray(I) + dc_i) * np.cos(w * t) - gain_imbalance * (np.asarray(Q) + dc_q) * np.sin(w * t + phase_error)


def tone_amplitude(x, t, f):
    """Amplitude of the frequency-f component of x(t) (projection; exact for an integer number of cycles)."""
    return 2 * abs(np.mean(np.asarray(x) * np.exp(-2j * np.pi * f * np.asarray(t))))


def image_rejection_db(gain_imbalance, phase_error):
    """Power ratio wanted sideband / image, in dB."""
    g, p = gain_imbalance, phase_error
    num = 1 + 2 * g * math.cos(p) + g * g
    den = 1 - 2 * g * math.cos(p) + g * g
    if den <= 0:
        return math.inf
    return 10 * math.log10(num / den)


def lo_leakage_amplitude(dc_i, dc_q, gain_imbalance=1.0, phase_error=0.0):
    """Carrier amplitude from DC offsets: |dc_i + j g dc_q e^{j phi}|."""
    return abs(dc_i + 1j * gain_imbalance * dc_q * np.exp(1j * phase_error))


# ---------------------------------------------------------------- the cold drive line
DEFAULT_DRIVE_LINE = [   # (name, attenuation dB, stage temperature K) -- illustrative, from memory
    ("cables, room temp", 6.0, 300.0),
    ("4 K plate", 20.0, 4.0),
    ("still (~0.8 K)", 10.0, 0.8),
    ("cold plate (~0.1 K)", 10.0, 0.1),
    ("mixing chamber", 20.0, 0.015),
]


def drive_line(p_in_dbm, f, stages=None, T_source=300.0):
    """Walk the drive line. Returns list of dicts: stage, power (dBm), thermal photons after the stage."""
    stages = DEFAULT_DRIVE_LINE if stages is None else stages
    n = thermal_photons(f, T_source)
    p = float(p_in_dbm)
    rows = [{"stage": "source", "p_dbm": p, "n_th": n, "T_K": T_source}]
    for name, att_db, T in stages:
        if att_db < 0:
            raise ValueError(f"attenuation must be >= 0 dB ({name})")
        A = 10 ** (-att_db / 10)
        n = A * n + (1 - A) * thermal_photons(f, T)
        p -= att_db
        rows.append({"stage": name, "p_dbm": p, "n_th": n, "T_K": T})
    return rows


def effective_temperature(n, f):
    """Temperature whose Bose-Einstein occupation at f equals n."""
    if n <= 0:
        return 0.0
    return H * f / (KB * math.log1p(1 / n))


def rabi_frequency(p_dbm_at_qubit, kappa_rad_per_s_per_volt, z0=50.0):
    """Omega = kappa * V_peak (rad/s). kappa is a calibration constant you measure with a Rabi sweep."""
    return kappa_rad_per_s_per_volt * vpeak_from_dbm(p_dbm_at_qubit, z0)


# ---------------------------------------------------------------- receive: readout chain
DEFAULT_READOUT_AMPS = [   # (name, gain dB, noise temperature K) -- illustrative
    ("TWPA (parametric, ~quantum limited)", 20.0, 0.3),
    ("HEMT at 4 K", 40.0, 3.0),
    ("room-temperature LNA", 30.0, 100.0),
]


def friis_temperature(amps, loss_before_db=0.0, T_loss=0.015):
    """System noise temperature (K) at the input of the chain, optional lossy line before it."""
    T, G = 0.0, 1.0
    if loss_before_db:
        A = 10 ** (-loss_before_db / 10)
        T += (1 / A - 1) * T_loss
        G *= A
    for name, g_db, Tn in amps:
        T += Tn / G
        G *= 10 ** (g_db / 10)
    return T


def readout_fidelity(p_diff_dbm, tau, T_sys):
    """Single-shot assignment fidelity for a matched filter: 1 - erfc(d / (2 sqrt 2 sigma)) / 2.

    p_diff_dbm: power of the DIFFERENCE between the |0> and |1> echoes; tau: integration time (s).
    """
    if tau <= 0 or T_sys <= 0:
        raise ValueError("tau and T_sys must be positive")
    E = dbm_to_w(p_diff_dbm) * tau
    snr = math.sqrt(2 * E / (KB * T_sys))           # d / sigma
    err = 0.5 * math.erfc(snr / (2 * math.sqrt(2)))
    return {"d_over_sigma": snr, "error": err, "fidelity": 1 - err}


def tau_for_fidelity(p_diff_dbm, T_sys, fidelity=0.99):
    """Integration time to reach a target fidelity (inverts readout_fidelity)."""
    lo, hi = 1e-12, 1.0
    for _ in range(200):
        mid = math.sqrt(lo * hi)
        if readout_fidelity(p_diff_dbm, mid, T_sys)["fidelity"] < fidelity:
            lo = mid
        else:
            hi = mid
    return hi


# ---------------------------------------------------------------- the quantum stack (next to OSI)
QUANTUM_STACK = [
    dict(n=7, name="Algorithm", unit="problem", example="Grover search, VQE, phase estimation",
         osi="L7 Application"),
    dict(n=6, name="Compiler / transpiler", unit="circuit", example="Qiskit transpile: map to native gates, route on the chip",
         osi="L6 Presentation (re-encoding the meaning)"),
    dict(n=5, name="Error correction", unit="logical qubit", example="repetition / Shor / surface code syndrome rounds",
         osi="L4 Transport (reliability on an unreliable channel)"),
    dict(n=4, name="Native gate", unit="gate", example="X, SX, RZ, CZ / CNOT with calibrated parameters",
         osi="L3 Network (the addressable operation)"),
    dict(n=3, name="Pulse schedule", unit="pulse", example="Gaussian / DRAG envelope, frequency, phase, timing",
         osi="L2 Data link (framing in time on one channel)"),
    dict(n=2, name="Waveform", unit="I/Q samples", example="AWG at ~1-2 GS/s, triggers, sequencer",
         osi="L1 Physical, digital side (symbols)"),
    dict(n=1, name="Microwave", unit="volts / photons", example="IQ mixer, LO, attenuators, coax, the qubit; readout amps, ADC",
         osi="L1 Physical, analog side (volts on the wire)"),
]


def gaussian_envelope(t, t_center, sigma, n_sigma=2.0):
    """Gaussian truncated at +/- n_sigma and shifted to start and end at zero (no step at the edges)."""
    g = np.exp(-0.5 * ((t - t_center) / sigma) ** 2)
    edge = math.exp(-0.5 * n_sigma ** 2)
    g = (g - edge) / (1 - edge)
    return np.where(np.abs(t - t_center) <= n_sigma * sigma, g, 0.0)


def compile_gate(theta, axis="x", sigma=10e-9, n_sigma=2.0, kappa=2 * math.pi * 40e6, sample_rate=2e9,
                 drag_beta=0.0, anharmonicity=-2 * math.pi * 250e6):
    """One gate down the stack: rotation theta about x or y -> Gaussian pulse -> I/Q samples (volts).

    Pulse-area theorem: theta = integral Omega(t) dt = kappa * A * integral envelope dt, so the AWG
    amplitude is A = theta / (kappa * area). The axis is the carrier phase (x: I, y: Q). Optional DRAG
    adds a quadrature derivative term -beta * dOmega/dt / anharmonicity (it suppresses leakage to the
    third level of a real transmon; in a pure two-level model it slightly tilts the rotation axis,
    so labs re-calibrate the amplitude with DRAG on).
    """
    if axis not in ("x", "y"):
        raise ValueError("axis must be 'x' or 'y'")
    if sigma <= 0 or kappa <= 0 or sample_rate <= 0:
        raise ValueError("sigma, kappa, sample_rate must be positive")
    dur = 2 * n_sigma * sigma
    t = np.arange(0, dur, 1 / sample_rate) + 0.5 / sample_rate
    env = gaussian_envelope(t, dur / 2, sigma, n_sigma)
    area = env.sum() / sample_rate
    A = theta / (kappa * area)
    phi = 0.0 if axis == "x" else math.pi / 2
    main = A * env
    deriv = np.gradient(main, 1 / sample_rate)
    drag = -drag_beta * deriv / anharmonicity
    I = main * math.cos(phi) - drag * math.sin(phi)
    Q = main * math.sin(phi) + drag * math.cos(phi)
    return {"t": t, "I": I, "Q": Q, "amplitude_V": A, "duration": dur, "kappa": kappa,
            "layers": {"L4 gate": f"R{axis}({theta:.4g})", "L3 pulse": f"Gaussian sigma={sigma*1e9:g} ns, "
                       f"{n_sigma:g} sigma each side, amplitude {A*1e3:.3g} mV",
                       "L2 samples": f"{len(t)} I/Q pairs at {sample_rate/1e9:g} GS/s"}}


def simulate_iq_pulse(pulse, Delta=0.0):
    """Apply the (I, Q) pulse to a two-level qubit starting in |0>; return the Bloch vector (x, y, z)."""
    from dgs.qubit_scope import drive_hamiltonian, unitary
    psi = np.array([1, 0], complex)
    dt = pulse["t"][1] - pulse["t"][0]
    for i, q in zip(pulse["I"], pulse["Q"]):
        Om = pulse["kappa"] * math.hypot(i, q)
        phase = math.atan2(q, i)
        psi = unitary(drive_hamiltonian(Om, Delta, phase), dt) @ psi
    rho = np.outer(psi, psi.conj())
    return (2 * rho[0, 1].real, -2 * rho[0, 1].imag, (rho[0, 0] - rho[1, 1]).real)


# ---------------------------------------------------------------- block diagram
def block_diagram(ax=None):
    """Draw the drive + readout chain. Returns (ax, text description)."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch
    OI = ["#E69F00", "#56B4E9", "#009E73", "#D55E00", "#CC79A7", "#0072B2"]
    if ax is None:
        _, ax = plt.subplots(figsize=(13, 6.5))
    ax.set_xlim(0, 13); ax.set_ylim(0, 7); ax.axis("off")
    # temperature bands
    bands = [(0, 4.2, "300 K: room-temperature rack", "#f4f4f4"), (4.2, 7.0, "4 K", "#e3eef8"),
             (7.0, 9.8, "0.8 K to 0.1 K", "#cfe2f2"), (9.8, 13, "15 mK: mixing chamber", "#b9d4ea")]
    for x0, x1, lab, c in bands:
        ax.axvspan(x0, x1, color=c, zorder=0)
        ax.text((x0 + x1) / 2, 6.75, lab, ha="center", fontsize=9)

    def box(x, y, text, color, w=1.6, h=0.75, style="-"):
        ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.04", fc="white",
                                    ec=color, lw=2, ls=style, zorder=2))
        ax.text(x, y, text, ha="center", va="center", fontsize=8, zorder=3)

    def arrow(x0, y0, x1, y1, color, ls="-"):
        ax.annotate("", (x1, y1), (x0, y0), arrowprops=dict(arrowstyle="-|>", color=color, lw=1.6, ls=ls), zorder=1)

    # drive (top row, solid orange)
    d = OI[0]
    box(0.95, 5.6, "PC: circuit ->\npulse schedule", d)
    box(0.95, 4.4, "AWG\nI(t), Q(t) ~100 MHz", d)
    box(2.9, 4.4, "IQ mixer\nf_LO + f_IF", d)
    box(2.9, 5.6, "LO synth\n~5 GHz", d)
    box(5.6, 4.4, "-20 dB", d, w=1.0)
    box(8.4, 4.4, "-10, -10 dB", d, w=1.3)
    box(10.9, 4.4, "-20 dB", d, w=1.0)
    box(12.2, 3.2, "qubit +\nreadout\nresonator", OI[4], w=1.3, h=1.1)
    arrow(0.95, 5.2, 0.95, 4.8, d); arrow(1.75, 4.4, 2.1, 4.4, d); arrow(2.9, 5.2, 2.9, 4.8, d)
    for x0, x1 in ((3.7, 5.1), (6.1, 7.75), (9.05, 10.4)):
        arrow(x0, 4.4, x1, 4.4, d)
    arrow(11.4, 4.4, 12.1, 3.8, d)
    # readout (bottom row, dashed blue)
    r = OI[5]
    box(10.9, 1.8, "circulators /\nisolators", r, style="--", w=1.4)
    box(10.9, 0.7, "TWPA\n+20 dB, ~0.3 K", r, style="--", w=1.4)
    box(5.6, 1.2, "HEMT 4 K\n+40 dB, ~3 K", r, style="--", w=1.4)
    box(2.9, 1.2, "IQ demod\n+ LNA, filters", r, style="--")
    box(0.95, 1.2, "ADC + FPGA\nthreshold -> bit", r, style="--")
    arrow(12.2, 2.6, 11.6, 1.95, r, "--"); arrow(10.9, 1.4, 10.9, 1.1, r, "--")
    arrow(10.2, 0.8, 6.3, 1.15, r, "--"); arrow(4.9, 1.2, 3.7, 1.2, r, "--"); arrow(2.1, 1.2, 1.75, 1.2, r, "--")
    arrow(2.9, 5.2, 2.9, 1.6, OI[2], ":")
    ax.text(3.0, 2.9, "same LO\n(down-converts\nthe echo)", fontsize=7, color=OI[2])
    arrow(0.95, 1.6, 0.95, 4.0, OI[3], "-.")
    ax.text(0.05, 2.75, "feedback /\nnext shot", fontsize=7, color=OI[3])
    ax.set_title("Qubit control and readout: a microwave transmitter and receiver spanning 300 K to 15 mK")
    desc = ("Block diagram on four temperature bands (300 K, 4 K, 0.8 to 0.1 K, 15 mK). Drive path, solid orange "
            "boxes along the top: the PC compiles a circuit into a pulse schedule, the AWG outputs I and Q envelopes "
            "near 100 MHz, an IQ mixer with a ~5 GHz LO up-converts them, then attenuators of 20 dB at 4 K, 10 + 10 dB "
            "on the intermediate plates and 20 dB at the mixing chamber cool the line before it reaches the qubit "
            "and readout resonator (pink box, bottom right of the coldest band). Readout path, dashed blue boxes "
            "along the bottom, right to left: circulators/isolators, a TWPA parametric amplifier (+20 dB, ~0.3 K "
            "noise), a 4 K HEMT (+40 dB, ~3 K), then IQ demodulation with the same LO (dotted green link), and an "
            "ADC/FPGA that thresholds the signal into a bit; a dash-dot vermillion arrow closes the loop for "
            "feedback or the next shot.")
    return ax, desc
