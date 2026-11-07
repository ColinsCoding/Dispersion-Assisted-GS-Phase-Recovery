"""
nfc_card.py - a contactless (tap-to-pay) card, voltage vs. time.

In plain words: a contactless card has no battery. The reader drives a coil with a 13.56 MHz current;
its magnetic field threads a few turns of wire around the card's edge and induces a voltage there
(Faraday's law, V = -M dI/dt). A capacitor tunes the card's coil to 13.56 MHz so that small induced
voltage rings up into a few volts, enough to power the chip. The reader talks to the card by briefly
switching its field off (pauses). The card answers without transmitting anything: it switches a load
resistor on and off at 847.5 kHz, which changes how much energy it pulls from the field, and the reader
sees that as a tiny ripple on its own coil voltage (load modulation).

What's here:
  * SymPy: mutual inductance of two coaxial loops (Neumann's formula, elliptic integrals) and its
    far-field (dipole) limit; the induced EMF; the tuned card circuit as a phasor; the resonance.
  * Numbers: ISO/IEC 14443-style values (13.56 MHz carrier, 847.5 kHz subcarrier = fc/16,
    106 kbit/s = fc/128, Manchester coding card -> reader).
  * A time-domain simulation of the card's tank circuit (RK4): induced EMF, card supply voltage,
    reader pauses, load modulation, and the voltage the reader sees, then decoding the card's bits
    back out of the reader's coil voltage.

Simplifications, said plainly: coaxial circular coils (a real ID-1 card coil is a rounded rectangle,
here an equal-area circle); the chip is a resistor that is switched for load modulation, plus a shunt
regulator that clamps the chip voltage at V_clamp (real chips do this so a card held right on the
reader doesn't see hundreds of volts); there is no rectifier; the reader's coil current is set directly, so the card's load doesn't reduce it. The
reflected signal on the reader is therefore computed as M di2/dt, the first-order coupling term.
"""
import math

import numpy as np
import sympy as sp

FC = 13.56e6                 # carrier (Hz)
F_SUB = FC / 16              # load-modulation subcarrier, 847.5 kHz
BIT_RATE = FC / 128          # 105.9 kbit/s
MU0 = 4e-7 * math.pi

# ID-1 card (85.6 x 54 mm): a 4-turn antenna close to the edge, as an equal-area circle
CARD = dict(radius=math.sqrt(0.080 * 0.048 / math.pi), turns=4, L=2.8e-6, R=3.0, R_load=1.5e3, R_mod=150.0,
            V_clamp=3.3, R_shunt=40.0)
READER = dict(radius=0.035, turns=2, I_peak=0.25)
H_ISO = (1.5, 7.5)          # ISO/IEC 14443 operating field, A/m rms


# ---------------------------------------------------------------- SymPy
def derivation():
    """Symbolic results, as a dict of SymPy expressions.

    M_exact : Neumann mutual inductance of coaxial loops a, b at distance d (per turn pair),
              mu0 sqrt(ab) [(2/k - k) K(k^2) - (2/k) E(k^2)],  k^2 = 4ab / ((a+b)^2 + d^2)
    M_dipole: small-card limit mu0 pi a^2 b^2 / (2 (a^2 + d^2)^(3/2))
    emf     : V = -M dI/dt for I = I0 sin(w t)
    V_card  : phasor voltage across the chip for the series coil (L2, R2) feeding C2 || R_L
    w0      : resonance 1/sqrt(L2 C2)
    H_axis  : reader field on axis, N I a^2 / (2 (a^2 + d^2)^(3/2))
    """
    a, b, d, mu0, I0, w, t, M, L2, R2, C2, RL, N, I = sp.symbols(
        "a b d mu_0 I_0 omega t M L_2 R_2 C_2 R_L N I", positive=True)
    k2 = 4 * a * b / ((a + b) ** 2 + d ** 2)
    k = sp.sqrt(k2)
    M_exact = mu0 * sp.sqrt(a * b) * ((2 / k - k) * sp.elliptic_k(k2) - (2 / k) * sp.elliptic_e(k2))
    M_dipole = mu0 * sp.pi * a ** 2 * b ** 2 / (2 * (a ** 2 + d ** 2) ** sp.Rational(3, 2))
    emf = -M * sp.diff(I0 * sp.sin(w * t), t)
    Vi = sp.I * w * M * I0                                         # phasor EMF magnitude source
    Z_coil = R2 + sp.I * w * L2
    Y_load = sp.I * w * C2 + 1 / RL
    V_card = sp.simplify(Vi / (1 + Z_coil * Y_load))
    w0 = 1 / sp.sqrt(L2 * C2)
    H_axis = N * I * a ** 2 / (2 * (a ** 2 + d ** 2) ** sp.Rational(3, 2))
    return dict(symbols=dict(a=a, b=b, d=d, mu0=mu0, I0=I0, w=w, t=t, M=M, L2=L2, R2=R2, C2=C2, RL=RL, N=N, I=I),
                M_exact=M_exact, M_dipole=M_dipole, emf=emf, V_card=V_card, w0=w0, H_axis=H_axis)


def dipole_limit_check():
    """SymPy check that the field of a small loop reproduces the dipole mutual inductance:
    flux of the on-axis field B = mu0 I a^2 / (2 (a^2+d^2)^(3/2)) through a small loop of area pi b^2."""
    D = derivation()
    s = D["symbols"]
    flux_per_amp = s["mu0"] * D["H_axis"].subs({s["N"]: 1, s["I"]: 1}) * sp.pi * s["b"] ** 2
    return sp.simplify(flux_per_amp - D["M_dipole"]) == 0


# ---------------------------------------------------------------- numbers
def mutual_inductance(a, b, d, n1=1, n2=1):
    """Coaxial loops (radii a, b, separation d), n1 x n2 turns: Neumann formula via elliptic integrals."""
    from scipy.special import ellipe, ellipk
    m = 4 * a * b / ((a + b) ** 2 + d ** 2)
    k = math.sqrt(m)
    return n1 * n2 * MU0 * math.sqrt(a * b) * ((2 / k - k) * ellipk(m) - (2 / k) * ellipe(m))


def reader_field(d, reader=READER):
    """On-axis field strength H (A/m, amplitude) of the reader coil at distance d."""
    a, n, i = reader["radius"], reader["turns"], reader["I_peak"]
    return n * i * a ** 2 / (2 * (a ** 2 + d ** 2) ** 1.5)


def tuning_capacitor(L=CARD["L"], f=FC):
    return 1 / ((2 * math.pi * f) ** 2 * L)


def card_voltage_phasor(d, R_load=None, card=CARD, reader=READER, C=None, f=FC):
    """Steady-state peak voltage across the chip from the linear phasor solution of the tuned circuit
    (no clamp: valid while the result stays below card['V_clamp'])."""
    M = mutual_inductance(reader["radius"], card["radius"], d, reader["turns"], card["turns"])
    w = 2 * math.pi * f
    C = tuning_capacitor(card["L"], f) if C is None else C
    RL = card["R_load"] if R_load is None else R_load
    Vi = 1j * w * M * reader["I_peak"]
    return abs(Vi / (1 + (card["R"] + 1j * w * card["L"]) * (1j * w * C + 1 / RL)))


def manchester(bits):
    """ISO 14443 type A card -> reader: each bit = two half-bits; '1' = subcarrier then off, '0' = off then on."""
    out = []
    for b in bits:
        out += [1, 0] if b else [0, 1]
    return out


def simulate(d=0.03, bits=(1, 0, 1, 1, 0), pause=(4e-6, 2.5e-6), card=CARD, reader=READER,
             steps_per_cycle=80):
    """Time-domain card circuit. Returns dict of arrays (t, I_reader, emf, V_card, V_reflected, ...).

    Timeline: carrier on, one reader pause (100% ASK, `pause` = (start, length) in s), then the card
    answers `bits` by load modulation (Manchester on the 847.5 kHz subcarrier)."""
    w = 2 * math.pi * FC
    M = mutual_inductance(reader["radius"], card["radius"], d, reader["turns"], card["turns"])
    L2, R2 = card["L"], card["R"]
    C2 = tuning_capacitor(L2)
    half = 64 / FC                                     # half-bit = 64 carrier cycles
    t_answer = pause[0] + pause[1] + 6e-6              # card answers after the field settles again
    T = t_answer + len(bits) * 2 * half + 2e-6
    dt = 1 / (FC * steps_per_cycle)
    n = int(T / dt)
    t = np.arange(n) * dt
    env = np.ones(n)
    env[(t >= pause[0]) & (t < pause[0] + pause[1])] = 0.0
    # smooth the pause edges over ~3 carrier cycles (the reader's own Q)
    tau = 3 / FC
    for k in range(1, n):
        env[k] = env[k - 1] + (env[k] - env[k - 1]) * min(1.0, dt / tau) if env[k] != env[k - 1] else env[k]
    I = reader["I_peak"] * env * np.sin(w * t)
    dIdt = np.gradient(I, dt)
    emf = -M * dIdt
    chips = manchester(bits)
    mod = np.zeros(n, bool)
    for j, c in enumerate(chips):
        if c:
            seg = (t >= t_answer + j * half) & (t < t_answer + (j + 1) * half)
            mod |= seg & (np.sin(2 * math.pi * F_SUB * (t - t_answer)) > 0)     # on/off at the subcarrier
    RL = np.where(mod, card["R_mod"] * card["R_load"] / (card["R_mod"] + card["R_load"]), card["R_load"])

    Vc, Rs = card["V_clamp"], card["R_shunt"]

    def deriv(i2, v, k):
        shunt = (abs(v) - Vc) / Rs * (1 if v > 0 else -1) if abs(v) > Vc else 0.0   # regulator
        return (emf[k] - R2 * i2 - v) / L2, (i2 - v / RL[k] - shunt) / C2

    i2 = v = 0.0
    I2, V = np.zeros(n), np.zeros(n)
    for k in range(n - 1):                              # RK4 with the source held over the step
        a1 = deriv(i2, v, k)
        a2 = deriv(i2 + dt / 2 * a1[0], v + dt / 2 * a1[1], k)
        a3 = deriv(i2 + dt / 2 * a2[0], v + dt / 2 * a2[1], k)
        a4 = deriv(i2 + dt * a3[0], v + dt * a3[1], k + 1)
        i2 += dt / 6 * (a1[0] + 2 * a2[0] + 2 * a3[0] + a4[0])
        v += dt / 6 * (a1[1] + 2 * a2[1] + 2 * a3[1] + a4[1])
        I2[k + 1], V[k + 1] = i2, v
    V_reflected = M * np.gradient(I2, dt)               # what the card's current induces back in the reader
    return dict(t=t, I_reader=I, emf=emf, V_card=V, I_card=I2, V_reflected=V_reflected, modulating=mod,
                bits=list(bits), t_answer=t_answer, half_bit=half, M=M, C2=C2, d=d)


def envelope(x, t, window_cycles=4):
    """Amplitude envelope of a 13.56 MHz signal: moving max of |x| over a few carrier cycles."""
    dt = t[1] - t[0]
    w = max(1, int(window_cycles / FC / dt))
    pad = np.pad(np.abs(x), (w, w), mode="edge")
    return np.array([pad[k:k + 2 * w].max() for k in range(len(x))])


def decode(sim):
    """Recover the card's bits from the reader-side voltage: subcarrier energy in each half-bit."""
    t, x = sim["t"], sim["V_reflected"]
    env = envelope(x, t)
    out = []
    for j in range(len(sim["bits"])):
        def ripple(h):
            seg = (t >= sim["t_answer"] + h * sim["half_bit"]) & (t < sim["t_answer"] + (h + 1) * sim["half_bit"])
            e = env[seg]
            return e.max() - e.min()
        out.append(1 if ripple(2 * j) > ripple(2 * j + 1) else 0)
    return out


def plot(sim, path=None):
    """Voltage vs. time figure (Okabe-Ito colours, line styles differ) plus a text description."""
    import matplotlib.pyplot as plt
    t_us = sim["t"] * 1e6
    fig, ax = plt.subplots(3, 1, figsize=(9, 7), sharex=True)
    ax[0].plot(t_us, sim["I_reader"], color="#0072B2", lw=0.6)
    ax[0].set_ylabel("reader coil\ncurrent (A)")
    ax[1].plot(t_us, sim["V_card"], color="#D55E00", lw=0.6)
    ax[1].plot(t_us, envelope(sim["V_card"], sim["t"]), color="black", ls="--", lw=1)
    ax[1].set_ylabel("card chip\nvoltage (V)")
    ax[2].plot(t_us, sim["V_reflected"] * 1e3, color="#009E73", lw=0.6)
    ax[2].set_ylabel("reader-side\nripple (mV)")
    ax[2].set_xlabel("time (µs)")
    for a in ax:
        a.axvspan(sim["t_answer"] * 1e6, sim["t"][-1] * 1e6, color="#F0E442", alpha=0.15)
    fig.suptitle(f"Contactless card at {sim['d'] * 100:.0f} cm: bits sent {sim['bits']}, decoded {decode(sim)}")
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=130)
    print("Figure: top, the reader's 13.56 MHz coil current with a 2.5 us pause (the reader 'talking'). "
          "Middle, the voltage on the card's chip (orange) and its envelope (black dashed): it rings up to "
          "a few volts (clamped at 3.3 V by the chip's regulator), sags during the pause, then dips in a Manchester pattern while the card load-modulates "
          "(yellow band). Bottom, the small voltage the card's current induces back in the reader coil, which "
          "carries those dips and is what the reader decodes.")
    return fig
