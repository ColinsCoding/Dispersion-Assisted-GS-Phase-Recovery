"""A qubit you can build on a bench: two coupled LC tanks on an oscilloscope behave like a two-level system.

In plain words: a real superconducting qubit needs a dilution refrigerator. But the *dynamics* of a
driven two-level system (Rabi oscillation, detuning, the avoided crossing seen in qubit spectroscopy)
are the same mathematics as two weakly coupled resonators. Ring tank 1 and its energy flows into tank
2 and back, at a rate set by the coupling, exactly as a resonant drive moves population from |0> to
|1> and back. Detune tank 2 and the transfer becomes incomplete and faster, the same
Omega^2 / (Omega^2 + Delta^2) law as a detuned Rabi oscillation. This module designs, simulates and
analyses that experiment for an ordinary teaching-lab scope (a CSUS bench), so the measured traces can
be compared with predictions.

The correspondence (and its limit):

    tank 1 energy fraction   <->  population of |0>
    tank 2 energy fraction   <->  population of |1>
    coupling k (mutual M)    <->  drive strength: Omega ~ k omega0
    tank detuning            <->  qubit-drive detuning Delta
    ring-down 2L/r           <->  T1-like decay
    beat seen on the scope   <->  Rabi oscillation
  It is CLASSICAL: amplitudes are voltages, not probability amplitudes. There is no superposition of
  outcomes, no measurement collapse, no entanglement. What carries over is the equation of motion: in
  the slowly-varying-envelope (rotating-wave) limit the coupled-mode equations for the two tank
  amplitudes are the same 2 x 2 Schrodinger equation as a driven qubit.

The pieces:
  * solenoid_coil(), coil_pair_coupling(): L and k of two air-core coils vs spacing, from the
    filament-loop formulas in dgs.mutual_inductance (so k is set by a ruler, then checked with an LCR
    meter using the series aiding/opposing recipe).
  * normal_modes(): exact lossless mode frequencies from det[[1/C1 - w^2 L1, -w^2 M], [-w^2 M,
    1/C2 - w^2 L2]] = 0; for identical tanks w = w0 / sqrt(1 +/- k).
  * simulate(): exact linear time evolution (matrix exponential of the 4-state circuit equations)
    with coil resistance, probe capacitance and a capacitive kick as the excitation.
  * rabi_prediction(): the coupled-mode (qubit) formula for the energy in tank 2.
  * k_from_mode_split(): the analysis step: two FFT peaks f+ < f- on the scope trace give
    k = (r^2 - 1) / (r^2 + 1), r = f- / f+ (exact for identical tanks).
  * probe_loading(): what a 10x probe (about 10 MOhm || 12 pF, typical, from memory) does to the tank.

Status: a protocol with simulated predictions. No measurement has been made yet; the lab write-up
(docs/csus_qubit_analog_lab.md) has empty tables for the measured values.

Connects to: dgs.mutual_inductance (L, M, k and the LCR recipe), dgs.circuit_rules (the coupled-tank
characteristic polynomial derived from KCL/KVL), dgs.qubit_scope (the real qubit's Rabi formula),
dgs.scope_bench (the simulated scope), dgs.qubit_microwave (the real qubit's microwave chain).
"""
from __future__ import annotations

import math

import numpy as np

from dgs import mutual_inductance as mi

ANALOGY = [
    ("tank 1 energy fraction", "population of |0>"),
    ("tank 2 energy fraction", "population of |1>"),
    ("coupling k (mutual inductance M = k L)", "drive strength, Omega ~ k omega0"),
    ("tank 2 tuned away (extra C)", "detuning Delta"),
    ("ring-down, energy time constant L / r", "T1-like energy decay"),
    ("energy beat between tanks on the scope", "Rabi oscillation"),
    ("mode splitting vs detuning", "avoided crossing in qubit spectroscopy"),
]

COPPER_RHO = 1.68e-8


# ---------------------------------------------------------------- coils
def solenoid_coil(radius, turns, length, wire_radius):
    """Air-core single-layer coil as `turns` coaxial filament loops. Returns (z positions, L in H)."""
    if turns < 1 or radius <= 0 or length < 0 or wire_radius <= 0:
        raise ValueError("need turns >= 1, radius > 0, length >= 0, wire_radius > 0")
    z = np.linspace(0.0, length, int(turns))
    L = turns * mi.loop_self_inductance(radius, wire_radius)
    for i in range(len(z)):
        for j in range(i + 1, len(z)):
            L += 2 * mi.coaxial_loops_mutual(radius, radius, z[j] - z[i])
    return z, L


def coil_pair_coupling(radius, turns, length, wire_radius, gap):
    """Two identical coaxial coils, end-to-end gap `gap` (m). Returns dict L, M, k."""
    if gap <= 0:
        raise ValueError("gap must be positive")
    z, L = solenoid_coil(radius, turns, length, wire_radius)
    M = sum(mi.coaxial_loops_mutual(radius, radius, length + gap + b - a) for a in z for b in z)
    return {"L": L, "M": M, "k": M / L}


def wire_resistance(radius, turns, wire_radius, f=None):
    """DC copper resistance, or a crude skin-effect estimate at f (round wire, ignores proximity effect)."""
    R_dc = COPPER_RHO * turns * 2 * math.pi * radius / (math.pi * wire_radius ** 2)
    if f is None:
        return R_dc
    delta = math.sqrt(COPPER_RHO / (math.pi * f * 4e-7 * math.pi))
    return R_dc * max(1.0, wire_radius / (2 * delta))


# ---------------------------------------------------------------- physics
def normal_modes(L1, L2, C1, C2, k):
    """Lossless mode angular frequencies (w_low, w_high) of two LC tanks coupled by M = k sqrt(L1 L2)."""
    M = mi.mutual_inductance(L1, L2, k)
    if k >= 1:
        raise ValueError("k must be < 1 for two distinct modes")
    # (1/C1 - x L1)(1/C2 - x L2) - x^2 M^2 = 0, x = w^2
    a = L1 * L2 - M * M
    b = -(L1 / C2 + L2 / C1)
    c = 1 / (C1 * C2)
    disc = math.sqrt(b * b - 4 * a * c)
    x1, x2 = (-b - disc) / (2 * a), (-b + disc) / (2 * a)
    return math.sqrt(x1), math.sqrt(x2)


def k_from_mode_split(f_low, f_high):
    """Identical tanks: (f_high / f_low)^2 = (1 + k) / (1 - k)  ->  k = (r^2 - 1) / (r^2 + 1)."""
    if not 0 < f_low < f_high:
        raise ValueError("need 0 < f_low < f_high")
    r2 = (f_high / f_low) ** 2
    return (r2 - 1) / (r2 + 1)


def rabi_prediction(t, k, w1, w2, decay_rate=0.0):
    """Coupled-mode (qubit) prediction for the fraction of energy in tank 2 after kicking tank 1.

    Omega = k w0 with w0 = sqrt(w1 w2), Delta = w2 - w1: P2 = Omega^2 / W^2 sin^2(W t / 2), W^2 = Omega^2 + Delta^2.
    Both tanks lose energy at decay_rate, which cancels in the FRACTION (equal-loss case).
    """
    Om = k * math.sqrt(w1 * w2)
    D = w2 - w1
    W = math.hypot(Om, D)
    return (Om / W) ** 2 * np.sin(W * np.asarray(t, float) / 2) ** 2


def simulate(t, L, C1, C2, r, k, v_kick=0.05, L2=None, r2=None):
    """Exact evolution of two series-loop LC tanks coupled by M. Tank 1 starts with v_kick on its capacitor.

    State x = [q1, q2, i1, i2]: dq/dt = i,  [[L1, M], [M, L2]] di/dt = -r i - q / C.
    Returns dict t, v1, v2 (capacitor voltages) and E1, E2 (energy in each tank, coupling energy split evenly).
    """
    L2 = L if L2 is None else L2
    r2 = r if r2 is None else r2
    M = mi.mutual_inductance(L, L2, k)
    Lm = np.array([[L, M], [M, L2]])
    Li = np.linalg.inv(Lm)
    A = np.zeros((4, 4))
    A[0, 2] = A[1, 3] = 1.0
    A[2:, :2] = -Li @ np.diag([1 / C1, 1 / C2])
    A[2:, 2:] = -Li @ np.diag([r, r2])
    lam, V = np.linalg.eig(A)
    x0 = np.array([C1 * v_kick, 0.0, 0.0, 0.0])
    c = np.linalg.solve(V, x0)
    t = np.asarray(t, float)
    X = (V @ (c[:, None] * np.exp(np.outer(lam, t)))).real
    q1, q2, i1, i2 = X
    Ecross = M * i1 * i2
    E1 = 0.5 * q1 ** 2 / C1 + 0.5 * L * i1 ** 2 + 0.5 * Ecross
    E2 = 0.5 * q2 ** 2 / C2 + 0.5 * L2 * i2 ** 2 + 0.5 * Ecross
    return {"t": t, "v1": q1 / C1, "v2": q2 / C2, "i1": i1, "i2": i2, "E1": E1, "E2": E2}


def probe_loading(L, C, C_probe=12e-12, R_probe=10e6):
    """Effect of a scope probe across a tank: fractional frequency shift and the Q the probe alone allows."""
    f0 = 1 / (2 * math.pi * math.sqrt(L * C))
    f_loaded = 1 / (2 * math.pi * math.sqrt(L * (C + C_probe)))
    Q_probe = R_probe * math.sqrt((C + C_probe) / L)          # parallel-RLC Q (dgs.circuit_rules rule)
    return {"f0": f0, "f_loaded": f_loaded, "shift_frac": f_loaded / f0 - 1, "Q_probe": Q_probe}


def capacitive_kick(V_step, C_couple, C_tank):
    """Voltage a generator step puts on the tank through a small coupling capacitor (charge sharing)."""
    return V_step * C_couple / (C_couple + C_tank)


def fft_peaks(t, v, n_peaks=2, f_min=None, f_max=None):
    """Frequencies of the strongest spectral peaks (Hann window, zero-padded, parabolic interpolation)."""
    t = np.asarray(t, float); v = np.asarray(v, float)
    dt = t[1] - t[0]
    n = 8 * len(v)
    spec = np.abs(np.fft.rfft((v - v.mean()) * np.hanning(len(v)), n))
    f = np.fft.rfftfreq(n, dt)
    m = np.ones_like(f, bool)
    if f_min is not None:
        m &= f >= f_min
    if f_max is not None:
        m &= f <= f_max
    s = np.where(m, spec, 0.0)
    peaks = [i for i in range(1, len(s) - 1) if s[i] > s[i - 1] and s[i] >= s[i + 1] and s[i] > 0]
    peaks = sorted(peaks, key=lambda i: -s[i])[:n_peaks]
    out = []
    for i in sorted(peaks):
        y0, y1, y2 = np.log(s[i - 1:i + 2] + 1e-300)
        den = y0 - 2 * y1 + y2
        out.append(f[i] + (0.5 * (y0 - y2) / den if den else 0.0) * (f[1] - f[0]))
    return out
