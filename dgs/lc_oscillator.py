"""The LC circuit is a harmonic oscillator, classically and quantum mechanically.

    mass-spring          LC circuit             quantum oscillator / superconducting qubit
    position x           charge Q               operator Q
    momentum p = m x'    flux Phi = L I         operator Phi, [Q, Phi] = i hbar
    mass m               inductance L
    spring k             1/C
    w = sqrt(k/m)        w = 1/sqrt(LC)         E_n = hbar w (n + 1/2)

Sections: parameters and the mapping, quantized levels and zero-point fluctuations, a symplectic
simulation that conserves energy (time-translation symmetry -> energy, Noether), when lumped
P = V I circuit theory breaks down (terahertz), photonic time stretch, and ODE residuals in torch
on CPU or CUDA.
"""
from __future__ import annotations

import math

import numpy as np

HBAR = 1.054571817e-34      # J s
KB = 1.380649e-23           # J/K
E_CHARGE = 1.602176634e-19  # C
C_LIGHT = 2.99792458e8      # m/s
PHI0 = 2.067833848e-15      # flux quantum h/2e, Wb

MAPPING = [
    ("position", "x", "charge Q", "C"),
    ("momentum", "p = m dx/dt", "flux Phi = L I", "Wb"),
    ("mass (inertia)", "m", "inductance L", "H"),
    ("stiffness", "k", "1/C", "1/F"),
    ("kinetic energy", "p^2 / 2m", "Phi^2 / 2L = L I^2 / 2", "J"),
    ("potential energy", "k x^2 / 2", "Q^2 / 2C", "J"),
    ("angular frequency", "sqrt(k/m)", "1/sqrt(LC)", "rad/s"),
    ("damping", "b dx/dt", "R I", ""),
]


def _pos(**kw):
    for k, v in kw.items():
        if not (isinstance(v, (int, float)) and v > 0):
            raise ValueError(f"{k} must be a positive number, got {v!r}")


def lc_params(L, C):
    """omega0 (rad/s), f0 (Hz), characteristic impedance Z = sqrt(L/C) (ohm)."""
    _pos(L=L, C=C)
    w = 1 / math.sqrt(L * C)
    return {"omega0": w, "f0": w / (2 * math.pi), "Z": math.sqrt(L / C)}


def quantum_levels(L, C, n_max=4):
    """E_n = hbar w (n + 1/2) for n = 0..n_max, in joules, micro-eV and kelvin (E / k_B)."""
    if not (isinstance(n_max, int) and n_max >= 0):
        raise ValueError("n_max must be an integer >= 0")
    w = lc_params(L, C)["omega0"]
    E = np.array([HBAR * w * (n + 0.5) for n in range(n_max + 1)])
    return {"n": np.arange(n_max + 1), "E_J": E, "E_ueV": E / E_CHARGE * 1e6, "E_K": E / KB,
            "spacing_GHz": w / (2 * math.pi) / 1e9}


def zero_point(L, C):
    """Ground-state rms fluctuations: dQ = sqrt(hbar/2Z), dPhi = sqrt(hbar Z/2), and dV = dQ/C, dI = dPhi/L.

    dQ * dPhi = hbar/2 exactly: the circuit sits at the uncertainty limit, like a Gaussian packet.
    """
    p = lc_params(L, C)
    dQ = math.sqrt(HBAR / (2 * p["Z"]))
    dP = math.sqrt(HBAR * p["Z"] / 2)
    return {"dQ_C": dQ, "dQ_electrons": dQ / E_CHARGE, "dPhi_Wb": dP, "dPhi_over_Phi0": dP / PHI0,
            "dV_V": dQ / C, "dI_A": dP / L, "product_over_hbar": dQ * dP / HBAR}


def thermal_occupation(L, C, T):
    """Bose-Einstein mean photon number 1/(exp(hbar w / kT) - 1): is the circuit 'quantum' at T?"""
    _pos(T=T)
    x = HBAR * lc_params(L, C)["omega0"] / (KB * T)
    return 1 / math.expm1(x)


def simulate_lc(L, C, Q0, I0=0.0, periods=50, steps_per_period=200, R=0.0):
    """Leapfrog (symplectic) integration of L Q'' + R Q' + Q/C = 0.

    Returns t, Q, I and the energy E = L I^2/2 + Q^2/2C. With R = 0 the energy error stays bounded
    (no drift) because leapfrog respects the time-translation symmetry of the Hamiltonian.
    """
    _pos(L=L, C=C, periods=periods, steps_per_period=steps_per_period)
    if R < 0:
        raise ValueError("R must be >= 0")
    w = 1 / math.sqrt(L * C)
    dt = 2 * math.pi / w / steps_per_period
    n = int(periods * steps_per_period)
    Q, I = np.empty(n + 1), np.empty(n + 1)
    Q[0], I[0] = Q0, I0
    for k in range(n):
        Ih = I[k] - 0.5 * dt * (Q[k] / C + R * I[k]) / L
        Q[k + 1] = Q[k] + dt * Ih
        I[k + 1] = Ih - 0.5 * dt * (Q[k + 1] / C + R * Ih) / L
    t = np.arange(n + 1) * dt
    return {"t": t, "Q": Q, "I": I, "E": 0.5 * L * I**2 + Q**2 / (2 * C)}


# ---- terahertz: where lumped P = V I stops working -------------------------------------------
def wavelength(f, n_index=1.0):
    """Wavelength in a medium of index n (m)."""
    _pos(f=f, n_index=n_index)
    return C_LIGHT / (n_index * f)


def lumped_ok(f, size_m, n_index=1.0, fraction=0.1):
    """Rule of thumb: lumped elements (and a single V, I per node) are fine if size < lambda/10."""
    _pos(size_m=size_m, fraction=fraction)
    lam = wavelength(f, n_index)
    return {"lambda_m": lam, "size_over_lambda": size_m / lam, "lumped_ok": size_m < fraction * lam}


def average_power(V0, I0, phase_rad=0.0):
    """Time-averaged P = (1/2) V0 I0 cos(phase) for sinusoids; P = V I is the instantaneous value."""
    return 0.5 * V0 * I0 * math.cos(phase_rad)


def time_stretch(adc_bandwidth_Hz, M, record_s):
    """Photonic time stretch (Jalali): stretch a signal by M so a slower ADC captures it.

    Effective bandwidth = M x ADC bandwidth; the record you can capture shrinks by M per frame.
    """
    _pos(adc_bandwidth_Hz=adc_bandwidth_Hz, M=M, record_s=record_s)
    return {"effective_bandwidth_Hz": M * adc_bandwidth_Hz, "captured_window_s": record_s / M}


# Rough 2026 price ranges in USD for orders of magnitude only. Not quotes: check with vendors.
COSTS = [
    ("simulation on your own laptop / RTX 4060 (this notebook)", 0, 0),
    ("university lab or shared-facility access", 0, 5_000),
    ("~20-33 GHz real-time oscilloscope", 100_000, 300_000),
    ("50-100 GHz sampling (equivalent-time) scope", 50_000, 200_000),
    ("~100+ GHz real-time oscilloscope (top of the market)", 1_000_000, 2_000_000),
    ("commercial THz time-domain spectroscopy system", 50_000, 300_000),
    ("DIY time-stretch bench: fiber laser, dispersive fiber, photodiode, GHz digitizer", 20_000, 100_000),
]


# ---- residuals in torch ------------------------------------------------------------------------
def torch_device(prefer="cuda"):
    """'cuda' if requested and available, else 'cpu'."""
    import torch
    if prefer not in ("cuda", "cpu"):
        raise ValueError("prefer must be 'cuda' or 'cpu'")
    return torch.device("cuda" if prefer == "cuda" and torch.cuda.is_available() else "cpu")


def lc_residual(Q_fn, L, C, t, device="cpu"):
    """Residual r(t) = L Q'' + Q/C via autograd, normalized by max|Q|/C so it is dimensionless.

    ``Q_fn`` maps a torch tensor of times (s) to charge. r = 0 everywhere means Q solves the LC equation.
    """
    import torch
    tt = torch.as_tensor(t, dtype=torch.float64, device=device).clone().requires_grad_(True)
    Q = Q_fn(tt)
    dQ, = torch.autograd.grad(Q.sum(), tt, create_graph=True)
    d2Q, = torch.autograd.grad(dQ.sum(), tt, create_graph=True)
    scale = Q.detach().abs().max() / C
    return (L * d2Q + Q / C) / scale


def fit_omega(L, C, guess_factor=0.6, steps=300, lr=0.05, device="cpu"):
    """Learn omega in Q = cos(omega t) by minimizing the mean squared residual (a one-parameter PINN).

    omega = s * omega_guess; the loss is minimized at s such that L C omega^2 = 1.
    """
    import torch
    _pos(guess_factor=guess_factor, steps=steps, lr=lr)
    w_true = 1 / math.sqrt(L * C)
    w_g = guess_factor * w_true
    s = torch.tensor(1.0, dtype=torch.float64, device=device, requires_grad=True)
    t = torch.linspace(0, 6 * math.pi / w_g, 400, dtype=torch.float64, device=device)
    opt = torch.optim.Adam([s], lr=lr)
    hist = []
    for _ in range(steps):
        opt.zero_grad()
        r = lc_residual(lambda tt: torch.cos(s * w_g * tt), L, C, t, device)
        loss = (r**2).mean()
        loss.backward()
        opt.step()
        hist.append(loss.item())
    return {"omega_fit": s.item() * w_g, "omega_true": w_true, "loss": np.array(hist)}
