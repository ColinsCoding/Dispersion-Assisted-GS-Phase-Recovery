"""One-qubit experiments you can watch on an oscilloscope: Rabi and Ramsey, from drive volts to readout volts.

In plain words: a qubit in a lab is controlled by voltages and read out as a voltage. An arbitrary
waveform generator (AWG) sends a microwave burst; its amplitude (volts) sets how fast the qubit
rotates (the Rabi frequency, Omega = kappa * amplitude) and its length sets how far. A readout pulse
then returns a voltage that is a little higher if the qubit was in |1> than in |0>, buried in noise,
so you repeat the shot many times and average. Each experiment here sweeps ONE knob (a 1-D sweep):

  * Rabi:   sweep the pulse length at fixed amplitude. Excited population
            P1(t) = (Omega^2 / W^2) sin^2(W t / 2), W = sqrt(Omega^2 + Delta^2) (Delta = detuning).
            Fit it and you have calibrated the pi pulse: t_pi = pi / Omega.
  * Ramsey: two pi/2 pulses separated by a wait tau. P1 = (1 + exp(-tau / T2) cos(Delta tau)) / 2.
            The fringe frequency measures the detuning; the decay measures T2 (dephasing).

Everything is checked: the closed forms against direct integration of the Schrodinger equation
(2 x 2 unitaries), and the fit against the true parameters. The signals are functions of time in
volts, so dgs.scope_bench.Scope can display the drive burst (CH1) and a readout trace (CH2) with
real scope limits (bandwidth, ADC bits, noise).

Simplifications, stated plainly: the drive is in the rotating frame (rotating-wave approximation),
pulses are square, readout is modelled as two Gaussian voltage levels (dispersive readout after
demodulation), and T1 decay during the pulses is ignored. Numbers like kappa (MHz per volt) and the
readout separation are illustrative, not from a specific device.

Connects to: dgs.scope_bench (the instrument), dgs.paul_trap (a trapped-ion qubit and its drive),
dgs.qubits (Bloch-sphere basics), dgs.circuit_rules (the volts that make it happen).
"""
from __future__ import annotations

import math

import numpy as np

SX = np.array([[0, 1], [1, 0]], complex)
SY = np.array([[0, -1j], [1j, 0]], complex)
SZ = np.array([[1, 0], [0, -1]], complex)


def _check(name, x, allow_zero=False):
    x = float(x)
    if not math.isfinite(x) or x < 0 or (x == 0 and not allow_zero):
        raise ValueError(f"{name} must be {'non-negative' if allow_zero else 'positive'}; got {x!r}")
    return x


# ---------------------------------------------------------------- closed forms
def rabi_probability(t, Omega, Delta=0.0):
    """P(|1>) after a square pulse of length t (rad/s units for Omega and Delta)."""
    _check("Omega", Omega)
    W = math.sqrt(Omega ** 2 + Delta ** 2)
    return (Omega / W) ** 2 * np.sin(W * np.asarray(t, float) / 2) ** 2


def ramsey_probability(tau, Delta, T2=math.inf):
    """P(|1>) after pi/2 - wait tau - pi/2 with detuning Delta and coherence time T2."""
    tau = np.asarray(tau, float)
    decay = np.ones_like(tau) if math.isinf(T2) else np.exp(-tau / _check("T2", T2))
    return 0.5 * (1 + decay * np.cos(Delta * tau))


# ---------------------------------------------------------------- direct simulation
def unitary(H, t):
    """exp(-i H t) for a 2x2 Hermitian H (hbar = 1), via eigen-decomposition."""
    w, V = np.linalg.eigh(H)
    return V @ np.diag(np.exp(-1j * w * t)) @ V.conj().T


def drive_hamiltonian(Omega, Delta=0.0, phase=0.0):
    """Rotating-frame H = (Omega/2)(cos(phi) X + sin(phi) Y) - (Delta/2) Z (hbar = 1)."""
    return 0.5 * Omega * (math.cos(phase) * SX + math.sin(phase) * SY) - 0.5 * Delta * SZ


def simulate_rabi(t, Omega, Delta=0.0):
    psi0 = np.array([1, 0], complex)
    return np.array([abs((unitary(drive_hamiltonian(Omega, Delta), ti) @ psi0)[1]) ** 2 for ti in np.atleast_1d(t)])


def simulate_ramsey(tau, Delta, T2=math.inf):
    """Density-matrix version: ideal pi/2 pulses, free precession at Delta, coherences decay as exp(-tau/T2)."""
    half = unitary(drive_hamiltonian(1.0), math.pi / 2)              # pi/2 about x
    out = []
    for ta in np.atleast_1d(tau):
        rho = np.array([[1, 0], [0, 0]], complex)
        rho = half @ rho @ half.conj().T
        U = unitary(-0.5 * Delta * SZ, ta)
        rho = U @ rho @ U.conj().T
        if not math.isinf(T2):
            d = math.exp(-ta / T2)
            rho = np.array([[rho[0, 0], rho[0, 1] * d], [rho[1, 0] * d, rho[1, 1]]])
        rho = half @ rho @ half.conj().T
        out.append(rho[1, 1].real)
    return np.array(out)


# ---------------------------------------------------------------- voltages: the lab view
def drive_voltage(amplitude, duration, f_carrier, t_start=0.0):
    """AWG output f(t) -> volts: a square-envelope burst A cos(2 pi f t) from t_start for `duration`."""
    _check("amplitude", amplitude, allow_zero=True); _check("duration", duration); _check("f_carrier", f_carrier)

    def f(t):
        t = np.asarray(t, float)
        on = (t >= t_start) & (t < t_start + duration)
        return np.where(on, amplitude * np.cos(2 * np.pi * f_carrier * (t - t_start)), 0.0)
    return f


def single_shots(p1, n_shots, v0=0.0, v1=1e-3, sigma=2e-3, rng=None):
    """Demodulated readout voltages: each shot lands near v0 (|0>) or v1 (|1>) with Gaussian noise."""
    rng = np.random.default_rng(rng)
    if not 0 <= p1 <= 1:
        raise ValueError("p1 must be a probability")
    excited = rng.random(int(n_shots)) < p1
    return np.where(excited, v1, v0) + sigma * rng.standard_normal(int(n_shots))


def readout_trace(state_voltage, t_start, t_len, tau_ring=50e-9, noise=0.0, rng=None):
    """f(t) -> volts for one readout window: the cavity rings up to `state_voltage` with time constant tau_ring."""
    rng = np.random.default_rng(rng)

    def f(t):
        t = np.asarray(t, float)
        on = (t >= t_start) & (t < t_start + t_len)
        v = np.where(on, state_voltage * (1 - np.exp(-np.clip(t - t_start, 0, None) / tau_ring)), 0.0)
        return v + (noise * rng.standard_normal(t.shape) if noise else 0.0)
    return f


def rabi_experiment(durations, amplitude, kappa, Delta=0.0, n_shots=500, v0=0.0, v1=1e-3, sigma=2e-3, rng=0):
    """Sweep pulse length; return mean readout voltage per point and the true P1. kappa in rad/s per volt."""
    rng = np.random.default_rng(rng)
    Omega = kappa * amplitude
    p = rabi_probability(durations, Omega, Delta)
    means = np.array([single_shots(pi, n_shots, v0, v1, sigma, rng).mean() for pi in p])
    return {"durations": np.asarray(durations, float), "p1": p, "v_mean": means,
            "v_err": sigma / math.sqrt(n_shots) * np.ones_like(means), "Omega": Omega}


def fit_rabi(durations, v, Omega_grid=None):
    """Fit v = c0 + c1 sin^2(Omega t / 2) (on resonance) by grid search over Omega + linear least squares.

    Returns (Omega, c0, c1). Unknown voltage scale and offset are why c0, c1 are fitted too.
    """
    d = np.asarray(durations, float)
    v = np.asarray(v, float)
    if Omega_grid is None:
        span = d.max() - d.min()
        Omega_grid = np.linspace(0.2, 2 * math.pi * len(d) / (2 * span), 4000)
    best = (math.inf, None)
    for Om in Omega_grid:
        A = np.column_stack([np.ones_like(d), np.sin(Om * d / 2) ** 2])
        c, *_ = np.linalg.lstsq(A, v, rcond=None)
        r = np.sum((A @ c - v) ** 2)
        if r < best[0]:
            best = (r, (Om, c[0], c[1]))
    Om, c0, c1 = best[1]
    # refine around the best grid point
    step = Omega_grid[1] - Omega_grid[0]
    for _ in range(3):
        grid = np.linspace(Om - step, Om + step, 201)
        for g in grid:
            A = np.column_stack([np.ones_like(d), np.sin(g * d / 2) ** 2])
            c, *_ = np.linalg.lstsq(A, v, rcond=None)
            r = np.sum((A @ c - v) ** 2)
            if r < best[0]:
                best = (r, (g, c[0], c[1]))
        Om, c0, c1 = best[1]
        step = grid[1] - grid[0]
    return Om, c0, c1
