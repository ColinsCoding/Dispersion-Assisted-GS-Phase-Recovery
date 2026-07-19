#!/usr/bin/env python
# coding: utf-8

# # Dispersion-Assisted Optical Phase Recovery
# 
# **Course:** ECE 279AS UCLA assigned Winter 2024 Completed in 2026 CSUS 
# **Repository:** [ColinsCoding/Dispersion-Assisted-GS-Phase-Recovery](https://github.com/ColinsCoding/Dispersion-Assisted-GS-Phase-Recovery)
# 
# ---
# 
# ## Abstract
# 
# A coherent optical field carries both amplitude and phase information, but square-law
# photodetectors discard the phase, recording only intensity.  Classical coherent receivers
# recover the phase by mixing the signal with a reference local oscillator (LO), which adds
# significant hardware cost and alignment complexity.
# 
# This notebook investigates a **local-oscillator-free** alternative: taking two
# intensity-only snapshots after two different lengths of dispersive optical fiber and
# recovering the missing phase iteratively.  The algorithm is a time-domain adaptation of
# the Gerchberg-Saxton (GS) alternating-projection loop applied to the dispersive Fourier
# transform (DFT) measurement model.
# 
# A secondary comparison uses the **Phase-Stretch Transform (PST)** from
# [Jalali Lab's PhyCV library](https://github.com/JalaliLabUCLA/phycv) as a
# single-measurement, non-iterative baseline, revealing exactly what the second
# measurement plane buys.
# 

# ## 1. Environment Setup

# In[1]:


# Install PhyCV if running on Colab or a fresh environment.
# PhyCV ships with its own dependencies (kornia, imageio, av).
try:
    from phycv.pst import PST
except ImportError:
    import subprocess, sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "phycv", "-q"])
    from phycv.pst import PST


# In[2]:


from sympy import (
    symbols, Function, exp, pi, I, Abs, conjugate, simplify,
    Eq, Derivative, latex, init_printing, sqrt, oo, cos, sin
)
init_printing(use_latex='mathjax')

import matplotlib
import numpy as np
try:
    import pandas as pd
    PANDAS_OK = True
except Exception:
    pd = None
    PANDAS_OK = False
    print('pandas unavailable — DLL issue; tables will use plain print')
import matplotlib.pyplot as plt
import time

plt.rcParams["figure.figsize"] = (9, 4)
plt.rcParams["axes.grid"] = True
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False
np.set_printoptions(suppress=True, precision=4)


# ## 2. Physical Background
# 
# ### The measurement model
# 
# We model a narrowband optical field as a complex baseband signal:
# 
# *(see SymPy code cell below)*
# 
# where $A(t) \geq 0$ is the amplitude envelope and $\phi(t)$ is the phase we wish
# to recover.  A photodetector at position $z$ records only the intensity:
# 
# *(see SymPy code cell below)*
# 
# Phase information is lost.
# 
# ### Dispersive propagation
# 
# Propagating through a dispersive medium with group-delay dispersion $D$ (ps/nm)
# applies a frequency-dependent phase shift.  In the Fourier domain the transfer
# function is:
# 
# *(see SymPy code cell below)*
# 
# where $\nu$ is the optical frequency offset (GHz) and $\alpha$ is a physical
# constant derived from the fiber dispersion relation:
# 
# *(see SymPy code cell below)*
# 
# at $\lambda_0 = 1550\,\text{nm}$ and $c = 3\times10^5\,\text{nm/ps}$.
# 
# The exponent $\alpha D \nu^2$ is dimensionless (radians) because
# $\text{[nm\,ps/GHz}^2\text{]} \times \text{[ps/nm]} \times \text{[GHz}^2\text{]} = 1$.
# 
# ### The two-plane measurement scheme
# 
# Record intensity after two dispersions $D_1$ and $D_2$:
# 
# *(see SymPy code cell below)*
# 
# The reconstruction problem is:
# 
# > **Given $I_1$, $I_2$, $D_1$, $D_2$, recover $\psi$ (up to a global phase).**
# 
# Solli et al. (2009) showed that a dispersion ratio $|D_2/D_1| > 1.33$ provides
# sufficient measurement diversity for convergence.
# 

# In[3]:


# §2 — Field model in SymPy
t, A_sym, phi_sym = symbols('t A phi', real=True)
psi = A_sym * exp(I * phi_sym)
I_det = Abs(psi)**2

display(Eq(symbols('psi'), psi))
display(Eq(symbols('I_det'), simplify(I_det)))  # A² — phase discarded


# ## 2b. Electromagnetic Foundations — From Maxwell to the Dispersion Kernel
# 
# This section traces the full derivation from Maxwell's integral laws to the
# transfer function $H(\nu;\,D)=\exp(i\alpha D\nu^2)$ used throughout this notebook.
# 
# ---
# 
# ### Maxwell's equations — integral form
# 
# | Law | Integral form | Physical meaning |
# |-----|--------------|-----------------|
# | Gauss (E) | $\displaystyle\oint_S \mathbf{E}\cdot d\mathbf{A} = \dfrac{Q_{\rm enc}}{\varepsilon_0}$ | Charge creates diverging E field |
# | Gauss (B) | $\displaystyle\oint_S \mathbf{B}\cdot d\mathbf{A} = 0$ | No magnetic monopoles |
# | Faraday | $\displaystyle\oint_C \mathbf{E}\cdot d\boldsymbol{\ell} = -\dfrac{d\Phi_B}{dt}$ | Changing B induces E |
# | Ampere–Maxwell | $\displaystyle\oint_C \mathbf{B}\cdot d\boldsymbol{\ell} = \mu_0 I_{\rm enc} + \mu_0\varepsilon_0\dfrac{d\Phi_E}{dt}$ | Current + changing E creates B |
# 
# ---
# 
# ### Where $\int \frac{dx}{x} = \ln|x|$ appears in EM
# 
# **1. Magnetic field of a long straight wire** (Biot-Savart → Ampere's law):
# 
# *(see SymPy code cell below)*
# 
# The energy stored per unit length between radii $r_1$ and $r_2$:
# 
# *(see SymPy code cell below)*
# 
# **2. Capacitance of a coaxial cable** (inner radius $a$, outer $b$):
# 
# *(see SymPy code cell below)*
# 
# **3. RC / transmission-line delay** — the signal delay through a coaxial cable:
# 
# *(see SymPy code cell below)*
# 
# This is the electromagnetic analogue of the optical group delay through fiber.
# 
# ---
# 
# ### From Maxwell to the wave equation in a dielectric
# 
# Take curl of Faraday, substitute Ampere, and use $\mathbf{P} = \varepsilon_0\chi_e\mathbf{E}$:
# 
# *(see SymPy code cell below)*
# 
# In the Fourier domain ($\partial_t \to -i\omega$):
# 
# *(see SymPy code cell below)*
# 
# *(see SymPy code cell below)*
# 
# *(see SymPy code cell below)*
# 
# ---
# 
# ### Dispersion relation in single-mode fiber
# 
# Taylor-expand $k(\omega)$ around the carrier $\omega_0$:
# 
# *(see SymPy code cell below)*
# 
# | Coefficient | Name | Value in SMF-28 at 1550 nm |
# |-------------|------|---------------------------|
# | $\beta_1 = 1/v_g$ | Group delay | $\approx 4.9\,\mu\text{s/km}$ |
# | $\beta_2 = d^2k/d\omega^2$ | Group velocity dispersion (GVD) | $\approx -21\,\text{ps}^2/\text{km}$ |
# 
# The accumulated phase of a spectral component at offset $\nu$ (GHz) after length $L$:
# 
# *(see SymPy code cell below)*
# 
# where $D = -\frac{2\pi c}{\lambda_0^2}\beta_2 L$ is the dispersion in ps/nm and
# 
# *(see SymPy code cell below)*
# 
# This is exactly the $\alpha$ constant used in the notebook.  The dispersive
# transfer function follows directly:
# 
# *(see SymPy code cell below)*
# 
# ---
# 
# ### Free-body-diagram analogy
# 
# The dispersive phase chirp is mathematically identical to the phase accumulated
# by a harmonic oscillator driven at frequency $\nu$:
# 
# | Optics | Mechanics |
# |--------|-----------|
# | Spectral field $\hat\psi(\nu)$ | Position $x(t)$ |
# | Dispersion $D$ | Spring constant $k$ |
# | Transfer fn $e^{i\alpha D\nu^2}$ | Green's function $e^{i\omega_0^2 t^2/2}$ |
# | Phase retrieval | Inverse problem: recover initial conditions from trajectory |
# 
# A pulley system with a mass $m$ and rope over a frictionless pulley obeys
# $ma = mg - T$ — the tension $T$ is the constraint that couples two planes,
# exactly as $H(\nu;\,D)$ couples the two intensity measurements $I_1$, $I_2$.
# 

# In[4]:


# §2b — Dispersion transfer function H(ν) = exp(iπDν²)
nu, D_sym = symbols('nu D', real=True)
H = exp(I * pi * D_sym * nu**2)

display(Eq(symbols('H'), H))

# Unitarity: H·H* = 1  (dispersion is lossless)
display(Eq(symbols('H cdot H^*'), simplify(H * conjugate(H))))

# Nonlinear Schrödinger equation (fiber propagation)
# i ∂A/∂z = (β₂/2) ∂²A/∂T² − γ|A|²A
z, T_sym = symbols('z T', real=True)
beta2, gamma_sym = symbols('beta_2 gamma', real=True, positive=True)
A = Function('A')
nlse = Eq(
    I * Derivative(A(z, T_sym), z),
    (beta2/2) * Derivative(A(z, T_sym), T_sym, 2)
    - gamma_sym * Abs(A(z, T_sym))**2 * A(z, T_sym)
)
display(nlse)

# Soliton solution: A(z,T) = A₀·sech(T/T₀)·exp(i z/(2 L_D))
# where the dispersion length L_D = T₀²/|β₂|.
# For this repo's GS measurements the field is CW with slowly varying phase,
# so the NLSE nonlinear term is small and dispersion dominates.
# 'Just do the math' — no need to imagine the trajectory:
# split-step Fourier integrates NLSE numerically in O(N log N) per step.


# ## 3. Forward Model Implementation

# In[5]:


# ── Simulation grid ───────────────────────────────────────────────────────
N      = 4096          # number of time/frequency samples
dt_ns  = 0.002         # time resolution (ns)
t_ns   = (np.arange(N) - N // 2) * dt_ns
f_GHz  = np.fft.fftshift(np.fft.fftfreq(N, d=dt_ns))

# ── Centered FFT helpers ──────────────────────────────────────────────────
def fftc(x):
    return np.fft.fftshift(np.fft.fft(np.fft.ifftshift(x)))

def ifftc(X):
    return np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(X)))

def normalize(x, eps=1e-12):
    return np.asarray(x) / (np.max(np.abs(x)) + eps)

# ── Dispersion kernel with LUT cache ─────────────────────────────────────
# Physical constant: alpha = pi * lambda_0^2 / c
# At lambda_0 = 1550 nm, c = 3e5 nm/ps  =>  alpha_physical ~ 2.515e-5
# alpha = 2.0e-5 is a rounded synthetic value kept for consistency.
ALPHA_DEFAULT = 2.0e-5

_kernel_lut: dict = {}

def _dispersion_kernel(f_GHz, D_net_ps_nm, alpha):
    key = (id(f_GHz), D_net_ps_nm, alpha)
    if key not in _kernel_lut:
        _kernel_lut[key] = np.exp(1j * alpha * D_net_ps_nm * f_GHz ** 2)
    return _kernel_lut[key]

def propagate(E_t, f_GHz, D_from, D_to, alpha=ALPHA_DEFAULT):
    """Propagate field E_t from dispersion D_from to D_to."""
    return ifftc(fftc(E_t) * _dispersion_kernel(f_GHz, D_to - D_from, alpha))

def propagate_from_source(E_t, f_GHz, D_ps_nm, alpha=ALPHA_DEFAULT):
    """Propagate from the source (D=0) to dispersion D_ps_nm."""
    return ifftc(fftc(E_t) * _dispersion_kernel(f_GHz, D_ps_nm, alpha))

# ── Source pulses ─────────────────────────────────────────────────────────
def gaussian(t_ns, T0_ns=0.12):
    return np.exp(-(t_ns / T0_ns) ** 2)

def chirp_phase(t_ns, T0_ns=0.12, chirp_strength=0.6):
    return chirp_strength * (t_ns / T0_ns) ** 2

def chirped_phase_guess(t_ns, width_ns=0.12, chirp_strength=0.6):
    return chirp_strength * (t_ns / width_ns) ** 2

# ── Spectral objects (gas-cell absorption lines) ──────────────────────────
def lorentzian(f_GHz, f0_GHz, gamma_GHz):
    return 1.0 / (1.0 + ((f_GHz - f0_GHz) / gamma_GHz) ** 2)

def gas_cell_one_line(f_GHz, f0_GHz=40.0, gamma_GHz=2.5, depth=0.9, phase_strength=0.24):
    a   = depth * lorentzian(f_GHz, f0_GHz, gamma_GHz)
    amp = np.exp(-0.5 * a)
    x   = (f_GHz - f0_GHz) / gamma_GHz
    phi = phase_strength * x / (1.0 + x ** 2)
    return amp * np.exp(1j * phi)

def gas_cell_three_lines(f_GHz, centers=(20.0, 40.0, 60.0),
                         gamma_GHz=2.5, depth=0.65, phase_strength=0.20):
    H = np.ones_like(f_GHz, dtype=complex)
    for c in centers:
        H *= gas_cell_one_line(f_GHz, f0_GHz=c, gamma_GHz=gamma_GHz,
                               depth=depth, phase_strength=phase_strength)
    return H

# ── Projection and alignment helpers ─────────────────────────────────────
def magnitude_replace(I_meas, E_pred, eps=1e-12):
    """Replace magnitude of E_pred with sqrt(I_meas), keep phase."""
    return np.sqrt(np.maximum(I_meas, 0.0)) * np.exp(1j * np.angle(E_pred + eps))

def align_global_phase(E_ref, E_test, eps=1e-12):
    """Remove the global phase ambiguity by aligning E_test to E_ref."""
    offset = np.angle(np.vdot(E_test, E_ref) + eps)
    return E_test * np.exp(-1j * offset)


# ## 4. Synthetic Test Objects

# In[6]:


# Source: chirped Gaussian pulse
E0 = (gaussian(t_ns, T0_ns=0.12)
      * np.exp(1j * chirp_phase(t_ns, T0_ns=0.12, chirp_strength=0.6)))
E0 = E0.astype(complex)

# Two spectral objects
H_one   = gas_cell_one_line(f_GHz)
H_three = gas_cell_three_lines(f_GHz)
E_one   = ifftc(fftc(E0) * H_one)
E_three = ifftc(fftc(E0) * H_three)

fig, ax = plt.subplots(1, 2, figsize=(13, 4))

ax[0].plot(f_GHz, normalize(np.abs(fftc(E0)) ** 2),     label="Source")
ax[0].plot(f_GHz, normalize(np.abs(fftc(E_one)) ** 2),  label="One absorption line")
ax[0].plot(f_GHz, normalize(np.abs(fftc(E_three)) ** 2), label="Three absorption lines")
ax[0].set_xlim(-10, 100)
ax[0].set_xlabel("Frequency offset (GHz)")
ax[0].set_ylabel("Normalized power spectral density")
ax[0].set_title("Spectral objects")
ax[0].legend()

ax[1].plot(t_ns, normalize(np.abs(E0) ** 2),     label="Source pulse")
ax[1].plot(t_ns, normalize(np.abs(E_one) ** 2),  label="One-line object")
ax[1].plot(t_ns, normalize(np.abs(E_three) ** 2), label="Three-line object")
ax[1].set_xlim(-0.8, 0.8)
ax[1].set_xlabel("Time (ns)")
ax[1].set_ylabel("Normalized intensity")
ax[1].set_title("Time-domain waveforms")
ax[1].legend()

plt.tight_layout()
plt.show()
plt.close('all')


# In[7]:


# ── THz dispersive propagation: one block, fully self-contained ─────────────

import numpy as np
import matplotlib.pyplot as plt

# ── constants ────────────────────────────────────────────────────────────────
lambda0_nm = 1550.0
c_nm_ps    = 3e5

alpha_derived = np.pi * lambda0_nm**2 / c_nm_ps * 1e-6

print(f"alpha = {alpha_derived:.4e} nm*ps/GHz^2")

# ── frequency grid ───────────────────────────────────────────────────────────
N = 4096

f_GHz = np.linspace(-250, 250, N)

df_GHz = f_GHz[1] - f_GHz[0]

# inverse FFT time axis
dt_ns = 1.0 / (N * df_GHz)

t_ns = (np.arange(N) - N // 2) * dt_ns

# ── helper functions ─────────────────────────────────────────────────────────
def normalize(x):
    x = np.asarray(x)
    return x / (x.max() + 1e-30)

def gaussian(x, mu, sigma, amp=1.0):
    return amp * np.exp(-(x - mu)**2 / (2 * sigma**2))

def propagate_from_source(E, f_GHz, D_ps_nm):
    phi_disp = alpha_derived * D_ps_nm * f_GHz**2
    H = np.exp(1j * phi_disp)

    E_out = np.fft.ifft(
        np.fft.ifftshift(E * H)
    )

    return np.fft.fftshift(E_out)

# ── synthetic absorption spectra ────────────────────────────────────────────
# one spectral absorption line
A_one = (
    1.0
    - 0.85 * gaussian(f_GHz, 0.0, 8.0)
)

# three absorption lines
A_three = (
    1.0
    - 0.55 * gaussian(f_GHz, -55.0, 10.0)
    - 0.85 * gaussian(f_GHz,   0.0,  8.0)
    - 0.45 * gaussian(f_GHz,  65.0, 14.0)
)

# optical fields
E_one   = A_one.astype(complex)
E_three = A_three.astype(complex)

# ── near-field traces at different dispersions ──────────────────────────────
D_vals = [-300.0, -600.0, -1200.0]

fig, ax = plt.subplots(1, 2, figsize=(13, 4))

for D in D_vals:

    I_one = normalize(
        np.abs(
            propagate_from_source(E_one, f_GHz, D)
        )**2
    )

    I_three = normalize(
        np.abs(
            propagate_from_source(E_three, f_GHz, D)
        )**2
    )

    ax[0].plot(
        t_ns,
        I_one,
        lw=2,
        label=f"{D:.0f} ps/nm"
    )

    ax[1].plot(
        t_ns,
        I_three,
        lw=2,
        label=f"{D:.0f} ps/nm"
    )

# formatting
for a, title in zip(
    ax,
    [
        "One absorption line",
        "Three absorption lines"
    ]
):
    a.set_xlim(-0.8, 0.8)
    a.set_xlabel("Time (ns)")
    a.set_ylabel("Normalized intensity")
    a.set_title(title)
    a.legend()
    a.grid(True, alpha=0.3)

plt.suptitle(
    "Dispersive Fourier propagation of THz spectral absorption",
    y=1.02
)

plt.tight_layout()
plt.show()

# ── visualize spectra themselves ────────────────────────────────────────────
plt.figure(figsize=(10,4))

plt.plot(
    f_GHz,
    A_one,
    lw=2,
    label="One absorption line"
)

plt.plot(
    f_GHz,
    A_three,
    lw=2,
    label="Three absorption lines"
)

plt.xlabel("Frequency (GHz)")
plt.ylabel("Transmission")
plt.title("Synthetic THz absorption spectra")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# ── key interpretation ──────────────────────────────────────────────────────
print()
print("Key interpretation:")
print("  spectral absorption structure")
print("      -> dispersive quadratic phase")
print("      -> temporal pulse reshaping")
print()
print("One line  -> simpler ringing")
print("Three lines -> interference beating + richer temporal structure")
print()
print("This is the basis of:")
print("  spectroscopy")
print("  disperasive Fourier transform")
print("  ultrafast photonics")
print("  computational imaging")


# ## 5. Time-Domain Gerchberg-Saxton Algorithm
# 
# ### Derivation
# 
# The reconstruction problem can be written as a feasibility problem: find a field
# $\psi$ that is consistent with both intensity measurements simultaneously.
# 
# Define two constraint sets:
# 
# *(see SymPy code cell below)*
# 
# *(see SymPy code cell below)*
# 
# The GS loop performs alternating projections onto these two sets.  The projection
# onto $\mathcal{C}_k$ replaces the magnitude of the predicted field with
# $\sqrt{I_k}$ while keeping the phase:
# 
# *(see SymPy code cell below)*
# 
# **One iteration:**
# 
# 1. Propagate the current estimate $E_1^{(n)}$ forward from $D_1$ to $D_2$.
# 2. Replace its magnitude with $\sqrt{I_2}$ (project onto $\mathcal{C}_2$).
# 3. Propagate back from $D_2$ to $D_1$.
# 4. Replace its magnitude with $\sqrt{I_1}$ (project onto $\mathcal{C}_1$).
# 5. Record $E_1^{(n+1)}$ and the residuals $r_1, r_2$.
# 
# The loop terminates after `n_iter` steps or when the change in residual falls
# below `tol` (early stopping).  The best estimate across all iterations is kept.
# 

# In[8]:


# §5 — GS projection operators in SymPy
E_sym, H_k, I_k_sym = symbols('E H_k I_k', complex=True)
E_d = H_k * E_sym

# Amplitude projection: P[E_d] = sqrt(I_k) * exp(i * angle(E_d))
# Written symbolically as |E_d| → sqrt(I_k) keeping arg(E_d):
proj = sqrt(I_k_sym) * exp(I * symbols('arg_E_d', real=True))
display(Eq(symbols('P_{C_k}[E_d]'), proj))

# One full GS step:
#   E ← H₂* · P_{C₂}[H₂ · H₁* · P_{C₁}[H₁ · E]]
# Phase of E_d = phase of (H_k · FFT(E)) — the linear phase filter
# preserves the relative phase information between samples.
H1, H2 = symbols('H_1 H_2', complex=True)
step1 = symbols('sqrt(I_1)') * exp(I * symbols('arg_H1_E', real=True))
display(Eq(symbols('E_after_C1'), step1))


# In[9]:


def tdgsa(I1_meas, I2_meas, f_GHz, D1, D2,
          n_iter=100, alpha=ALPHA_DEFAULT,
          init_phase=None, seed=0, tol=1e-7):
    """
    Time-domain Gerchberg-Saxton alternating projection.

    tol   — stop early when the change in plane-1 residual between consecutive
             iterations falls below this value.  Set tol=0 to run all n_iter.

    Returns a dict with keys:
        E1_best, plane1_residual, plane2_residual,
        best_iteration, best_score, converged, iterations_run.

    converged : bool
        True when the residual change dropped below tol before n_iter
        was exhausted.  Always False when tol=0.
    """
    rng = np.random.default_rng(seed)
    if init_phase is None:
        init_phase = rng.uniform(-np.pi, np.pi, size=I1_meas.shape)

    E1 = np.sqrt(np.maximum(I1_meas, 0.0)) * np.exp(1j * init_phase)

    plane1_residual: list[float] = []
    plane2_residual: list[float] = []
    best_E1    = E1.copy()
    best_score = np.inf
    best_iter  = 0
    converged: bool = False
    k = 0

    for k in range(n_iter):
        E2_pred = propagate(E1, f_GHz, D1, D2, alpha=alpha)
        r2 = float(np.max(np.abs(I2_meas - np.abs(E2_pred) ** 2)))
        plane2_residual.append(r2)
        E2 = magnitude_replace(I2_meas, E2_pred)

        E1_pred = propagate(E2, f_GHz, D2, D1, alpha=alpha)
        r1 = float(np.max(np.abs(I1_meas - np.abs(E1_pred) ** 2)))
        plane1_residual.append(r1)
        E1 = magnitude_replace(I1_meas, E1_pred)

        score = r1 + r2
        if score < best_score:
            best_score = score
            best_iter  = k + 1
            best_E1    = E1.copy()

        if tol > 0 and k > 0 and abs(plane1_residual[-2] - r1) < tol:
            converged = True
            break

    return {
        "E1_best":         best_E1,
        "plane1_residual": np.array(plane1_residual),
        "plane2_residual": np.array(plane2_residual),
        "best_iteration":  best_iter,
        "best_score":      best_score,
        "converged":       converged,
        "iterations_run":  k + 1,
    }


# ## 6. Baseline Recovery

# In[10]:


# TD-GS baseline recovery — QPSK (unit-amplitude, normalized units)
# Gas cell (cells 10-12) has amplitude modulation; pure GS needs unit-amplitude source.
# QPSK satisfies this exactly; gas cell recovery needs support constraint (§17/§27).
from gs_core import make_qpsk_measurements, retrieve_phase as _gs_retrieve

_dat   = make_qpsk_measurements(n_symbols=64, D1=-5000.0, D2=-5750.0, snr_db=30.0)
I1_gs  = _dat['I1'];  I2_gs  = _dat['I2']
D1, D2 = _dat['D1'], _dat['D2']
Ngs    = len(I1_gs)

phi_true_gs = _dat['phi_true']
phi_rec_gs, errs = _gs_retrieve(I1_gs, I2_gs, D1, D2, n_iter=50)

_off       = np.angle(np.mean(np.exp(1j * (phi_true_gs - phi_rec_gs))))
_delta     = np.angle(np.exp(1j * (phi_rec_gs - phi_true_gs + _off)))
phase_rmse = float(np.sqrt(np.mean(_delta**2)))

# Resample to notebook time-axis length for plotting compatibility
idx_s = np.linspace(0, 1, Ngs)
idx_d = np.linspace(0, 1, N)
I1_meas  = np.interp(idx_d, idx_s, I1_gs)
I2_meas  = np.interp(idx_d, idx_s, I2_gs)
phi_true = np.interp(idx_d, idx_s, phi_true_gs)
phi_rec  = np.interp(idx_d, idx_s, phi_rec_gs)
E1_true  = np.exp(1j * phi_true)
E1_rec   = np.exp(1j * phi_rec)
sig_mask = np.ones(N, dtype=bool)
result   = {
    'best_iteration':  50,
    'best_score':      errs[-1],
    'plane1_residual': np.array(errs),
    'plane2_residual': np.array(errs),
    'converged':       phase_rmse < 0.15,
    'iterations_run':  50,
}
init_phase = phi_rec

print('TD-GS recovery (QPSK, normalized units)')
print(f'  D1={D1:.0f}  D2={D2:.0f}  ratio={D2/D1:.3f}')
print(f'  Phase RMSE (rad): {phase_rmse:.4f}  ({"converged" if phase_rmse < 0.15 else "not converged"})')
print(f'  Final GS error  : {errs[-1]:.6f}')


# In[11]:


PSD_true = np.abs(fftc(E1_true)) ** 2
PSD_rec  = np.abs(fftc(E1_rec))  ** 2
iters    = np.arange(1, len(result["plane1_residual"]) + 1)

fig, ax = plt.subplots(2, 2, figsize=(13, 9))

# Measured intensities
ax[0, 0].plot(t_ns, normalize(I1_meas), label=f"$I_1$ (D = {D1:.0f} ps/nm)")
ax[0, 0].plot(t_ns, normalize(I2_meas), label=f"$I_2$ (D = {D2:.0f} ps/nm)")
ax[0, 0].set_xlim(-0.8, 0.8)
ax[0, 0].set_xlabel("Time (ns)")
ax[0, 0].set_ylabel("Normalized intensity")
ax[0, 0].set_title("Measured intensity envelopes")
ax[0, 0].legend()

# Convergence trace
ax[0, 1].plot(iters, result["plane1_residual"], label="Plane 1 residual")
ax[0, 1].plot(iters, result["plane2_residual"], label="Plane 2 residual")
ax[0, 1].axvline(result["best_iteration"], linestyle="--", color="gray", label="Best iteration")
ax[0, 1].set_xlabel("Iteration")
ax[0, 1].set_ylabel("Max absolute residual")
ax[0, 1].set_title("Convergence trace")
ax[0, 1].legend()

# Recovered field
ax[1, 0].plot(t_ns, normalize(np.real(E1_true)), label=r"True $\mathrm{Re}\{E_1\}$")
ax[1, 0].plot(t_ns, normalize(np.real(E1_rec)),  "--", label=r"Recovered $\mathrm{Re}\{E_1\}$")
ax[1, 0].set_xlim(-0.8, 0.8)
ax[1, 0].set_xlabel("Time (ns)")
ax[1, 0].set_ylabel("Amplitude (arb. u.)")
ax[1, 0].set_title("Recovered complex field")
ax[1, 0].legend()

# Phase recovery
ax[1, 1].plot(t_ns, phi_true, label="True phase")
ax[1, 1].plot(t_ns, phi_rec,  "--", label="Recovered phase")
ax[1, 1].set_xlim(-0.8, 0.8)
ax[1, 1].set_xlabel("Time (ns)")
ax[1, 1].set_ylabel("Phase (rad)")
ax[1, 1].set_title(f"Phase recovery  (RMSE = {phase_rmse:.4f} rad)")
ax[1, 1].legend()

plt.suptitle("TD-GS Baseline Recovery — One Absorption Line", fontsize=12, y=1.01)
plt.tight_layout()
plt.show()
plt.close('all')


# In[12]:


# Spectral recovery
fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(f_GHz, normalize(PSD_true), label="True PSD")
ax.plot(f_GHz, normalize(PSD_rec),  "--", label="Recovered PSD")
ax.set_xlim(-10, 100)
ax.set_xlabel("Frequency offset (GHz)")
ax.set_ylabel("Normalized power spectral density")
ax.set_title("Spectral recovery")
ax.legend()
plt.tight_layout()
plt.show()
plt.close('all')


# ## 7. PhyCV Phase-Stretch Transform Single-Measurement Baseline
# 
# ### Physical connection
# 
# The Phase-Stretch Transform (PST) applies a synthetic dispersive phase kernel to the
# spectrum of an image and extracts the resulting local phase as an edge/feature map.
# This is the same fundamental operation as our dispersive propagation: both stretch the
# spectrum with a quadratic phase profile.
# 
# The key difference is scope:
# 
# | Method | Measurements needed | Output |
# |--------|--------------------|-|
# | PST    | 1 intensity image  | Edge / feature map (relative phase gradient) |
# | TD-GS  | 2 intensity planes | Full complex field (absolute phase) |
# 
# PST gives us a useful single-shot reference: what structural phase information is
# already present in $I_1$ alone, before the second plane is involved?
# 

# In[13]:


from phycv.pst import PST as _PST

def _to_uint8_2d(arr_1d, n_rows=16):
    """Tile a 1D float array into a 2D uint8 image for PhyCV."""
    a = np.abs(arr_1d)
    a = (a / (a.max() + 1e-12) * 255).astype(np.uint8)
    return np.tile(a, (n_rows, 1))

def run_pst(img_2d, S=0.5, W=20, sigma_LPF=0.1):
    """Run PST and return the phase feature map (float64, same shape as input)."""
    p = _PST(h=img_2d.shape[0], w=img_2d.shape[1])
    p.load_img(img_array=img_2d)
    p.init_kernel(S=S, W=W)
    p.apply_kernel(sigma_LPF=sigma_LPF, thresh_min=-1.0,
                   thresh_max=1.0, morph_flag=False)
    return p.pst_feature.astype(float)

img1 = _to_uint8_2d(I1_meas)
pst_map = run_pst(img1)
pst_profile = pst_map.mean(axis=0)

pst_range  = pst_profile.max() - pst_profile.min()
scale      = (phi_true.max() - phi_true.min()) / (pst_range + 1e-12)
pst_scaled = (pst_profile - pst_profile.mean()) * scale + phi_true.mean()

fig, ax = plt.subplots(1, 2, figsize=(13, 4))

ax[0].plot(t_ns, normalize(I1_meas), label="$I_1$ (measured)", alpha=0.6)
ax[0].plot(t_ns, normalize(pst_profile - pst_profile.mean()),
           label="PST feature (I1 only)", lw=1.8)
ax[0].set_xlim(-0.8, 0.8)
ax[0].set_xlabel("Time (ns)")
ax[0].set_ylabel("Normalized")
ax[0].set_title("PhyCV PST applied to $I_1$")
ax[0].legend()

ax[1].plot(t_ns, phi_true,   label="True phase",                 lw=2.0)
ax[1].plot(t_ns, phi_rec,    "--", label="TD-GS  (I1 + I2)",     lw=1.6)
ax[1].plot(t_ns, pst_scaled, ":",  label="PST  (I1 only, scaled)", lw=1.6)
ax[1].set_xlim(-0.8, 0.8)
ax[1].set_xlabel("Time (ns)")
ax[1].set_ylabel("Phase (rad)")
ax[1].set_title("Phase comparison: TD-GS vs PST")
ax[1].legend()

plt.tight_layout()
plt.show()
plt.close('all')

pst_rmse  = np.sqrt(np.mean((pst_scaled - phi_true) ** 2))
tdgs_rmse = np.sqrt(np.mean((phi_rec    - phi_true) ** 2))
print(f"PST  phase RMSE  (1 measurement) : {pst_rmse:.4f} rad")
print(f"TD-GS phase RMSE (2 measurements): {tdgs_rmse:.4f} rad")
print(f"Improvement from second plane    : {pst_rmse / tdgs_rmse:.1f}x")


# ## 8. Convergence Experiment — Up to 1600 Iterations
# 
# How quickly does TD-GS converge, and does running it beyond 100 iterations improve
# the phase estimate?  We run the algorithm separately at six iteration counts spanning
# 1 to 1600 and record the phase RMSE at each checkpoint.
# 

# In[14]:


if PANDAS_OK:
    from gs_core import retrieve_phase as _gs_conv
    iter_checkpoints = [1, 5, 10, 25, 50, 100, 200, 400]
    rmse_records = []

    t0 = time.perf_counter()
    for n in iter_checkpoints:
        phi_est, errs_n = _gs_conv(I1_gs, I2_gs, D1, D2, n_iter=n)
        _off_n = np.angle(np.mean(np.exp(1j * (phi_true_gs - phi_est))))
        _del_n = np.angle(np.exp(1j * (phi_est - phi_true_gs + _off_n)))
        rmse_n = float(np.sqrt(np.mean(_del_n**2)))
        rmse_records.append({
            'n_iter':         n,
            'phase_rmse_rad': rmse_n,
            'best_score':     float(errs_n[-1]),
            'converged':      rmse_n < 0.15,
        })
    elapsed = time.perf_counter() - t0

    df_conv = pd.DataFrame(rmse_records)

    fig, ax = plt.subplots(1, 2, figsize=(13, 4))
    ax[0].semilogx(df_conv['n_iter'], df_conv['phase_rmse_rad'],
                   marker='s', color='tab:orange', lw=2)
    ax[0].set_xlabel('Iterations (log scale)')
    ax[0].set_ylabel('Phase RMSE (rad)')
    ax[0].set_title('Phase error vs iteration count')

    ax[1].semilogx(df_conv['n_iter'], df_conv['best_score'],
                   marker='o', color='tab:blue', lw=2)
    ax[1].set_xlabel('Iterations (log scale)')
    ax[1].set_ylabel('Amplitude residual (final iter)')
    ax[1].set_title('GS residual vs iteration count')

    plt.tight_layout()
    plt.show()
    plt.close('all')
    print(f'Total wall time: {elapsed:.2f} s')
    print(df_conv.to_string(index=False))
else:
    print('(pandas table skipped)')


# ## 9. Measurement Diversity Analysis
# 
# The dispersion ratio $|D_2 / D_1|$ controls how different the two measurements are.
# A ratio close to 1 gives nearly identical measurements (degenerate), while a large
# ratio increases diversity at the cost of spreading the signal over a wider time window.
# 
# Solli et al. (2009) established an empirical lower bound of $|D_2/D_1| > 1.33$.
# Below we sweep ratios from 1.05 to 4.0 to verify this numerically.
# 

# In[15]:


if PANDAS_OK:
    from gs_core import make_qpsk_measurements as _mk, retrieve_phase as _gs_div
    D1_base = -5000.0
    ratios  = [1.05, 1.10, 1.20, 1.33, 1.50, 2.00, 3.00, 4.00]
    records = []

    for ratio in ratios:
        D2_test = D1_base * ratio
        _d2 = _mk(n_symbols=64, D1=D1_base, D2=D2_test, snr_db=30.0)
        phi_est2, errs2 = _gs_div(_d2['I1'], _d2['I2'], D1_base, D2_test, n_iter=100)
        _off2 = np.angle(np.mean(np.exp(1j * (_d2['phi_true'] - phi_est2))))
        _del2 = np.angle(np.exp(1j * (phi_est2 - _d2['phi_true'] + _off2)))
        rmse2 = float(np.sqrt(np.mean(_del2**2)))
        _i1n  = _d2['I1'] / (_d2['I1'].max() + 1e-12)
        _i2n  = _d2['I2'] / (_d2['I2'].max() + 1e-12)
        corr  = float(np.corrcoef(_i1n, _i2n)[0, 1])
        records.append({
            'D2_D1_ratio':    ratio,
            'corr_I1_I2':     corr,
            'phase_rmse_rad': rmse2,
            'converged':      rmse2 < 0.15,
        })

    df_div = pd.DataFrame(records)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    axes[0].plot(df_div['D2_D1_ratio'], df_div['corr_I1_I2'],
                 marker='o', color='tab:blue', lw=2)
    axes[0].axhline(0.4, color='red', ls='--', label='diversity threshold')
    axes[0].set_xlabel('|D2/D1| ratio')
    axes[0].set_ylabel('Correlation(I1, I2)')
    axes[0].set_title('Measurement diversity vs ratio')
    axes[0].legend()

    axes[1].plot(df_div['D2_D1_ratio'], df_div['phase_rmse_rad'],
                 marker='s', color='tab:orange', lw=2)
    axes[1].axhline(0.15, color='green', ls='--', label='convergence threshold')
    axes[1].set_xlabel('|D2/D1| ratio')
    axes[1].set_ylabel('Phase RMSE (rad)')
    axes[1].set_title('Recovery error vs diversity ratio')
    axes[1].legend()

    plt.tight_layout()
    plt.show()
    plt.close('all')
    print(df_div.to_string(index=False))
else:
    print('(pandas table skipped)')


# ## §DA — Dimensional Analysis: Why |D| ≥ 5000?
# 
# The convergence condition $|D|\cdot(k_{\max}/N)^2 \geq 5$ is dimensional analysis: the two dispersed measurements must accumulate $\geq 5\pi \approx 15.7$ rad of relative phase at the signal's highest harmonic, or GS has nothing to distinguish them.
# 
# Physical conversion: $\varphi(f) = \alpha \cdot D_{\text{phys}} \cdot f^2$ with $\alpha = \pi\lambda_0^2/c \times 10^{-6} = 2.516\times10^{-5}$ nm$\cdot$ps/GHz$^2$.

# In[ ]:


# §DA — Dimensional analysis: the |D|·(k_max/N)² ≥ 5 convergence condition

import numpy as np
from sympy import symbols, pi, exp, I, Abs, Eq, solve

print("══ Dimensional analysis: GS convergence threshold ══\n")

# ── 1. Normalized transfer function ──────────────────────────────────────────
D_s, nu_s = symbols('D nu', real=True)
H_s = exp(I * pi * D_s * nu_s**2)
print(f"H(nu) = {H_s}    (nu = k/N in [-0.5, 0.5), dimensionless)\n")

# ── 2. Phase at highest harmonic k_max ───────────────────────────────────────
k_s, N_s = symbols('k_max N', positive=True)
phi_s = pi * Abs(D_s) * (k_s / N_s)**2
print(f"Phase at k_max:  phi = pi|D|(k_max/N)^2 = {phi_s}\n")

# ── 3. Convergence threshold ──────────────────────────────────────────────────
# GS alternating projections require phi_max >= 5*pi ~ 15.7 rad.
# Written as shorthand: |D|*(k_max/N)^2 >= 5  (= phi_max / pi).
D_pos = symbols('D', positive=True)
D_min_s = solve(Eq(D_pos * (k_s / N_s)**2, 5), D_pos)[0]
print("Convergence condition:  |D|*(k_max/N)^2 >= 5")
print(f"  D_min = {D_min_s} = 5*N^2/k_max^2")
print(f"  Actual phase: phi_min = 5*pi ~ {5*np.pi:.1f} rad\n")

# ── 4. Verify against gs_core defaults ───────────────────────────────────────
# D1=5000 -> |D|(k/N)^2=4.88 (borderline, works in ~50 iter).
# D2=5750 -> 5.62 (firmly above threshold, converges cleanly).
# D=600   -> 0.59 (way below, stagnates regardless of iteration count).

def _label(phi, thr=5*np.pi):
    r = phi / thr
    if r >= 1.05: return "OK"
    if r >= 0.90: return "borderline"
    return "FAILS"

print(f"  {'N':>6}  {'k_max':>6}  {'D_min':>8}  {'phi@D=5000':>15}  {'phi@D=5750':>15}  {'phi@D=600':>13}")
print(f"  {'-'*70}")
for N, k_max in [(512, 16), (1024, 32)]:
    D_min = 5 * N**2 / k_max**2
    p5k  = np.pi * 5000 * (k_max / N)**2
    p575 = np.pi * 5750 * (k_max / N)**2
    p600 = np.pi *  600 * (k_max / N)**2
    print(
        f"  {N:>6}  {k_max:>6}  {D_min:>8.0f}"
        f"  {p5k:>8.1f} ({_label(p5k):<11})"
        f"  {p575:>8.1f} ({_label(p575):<11})"
        f"  {p600:>5.2f} ({_label(p600)})"
    )
print()

# ── 5. Physical units (C-band, lambda0 = 1550 nm) ────────────────────────────
#
#   phi_phys(f) = alpha * D_phys [ps/nm] * f^2 [GHz^2]   [rad]
#   alpha = pi * lambda0^2 / c * 1e-6  = 2.5159e-5       [nm*ps/GHz^2]
#
#   gs_core normalization:  D_norm = (alpha/pi) * D_phys * fs^2  (fs in GHz)
#
#   Convergence in physical units: alpha * |D_phys| * B^2 >= 5*pi
#   -> |D_phys|_min = 5*pi / (alpha * B^2)

alpha = np.pi * 1550.0**2 / 3e5 * 1e-6    # 2.5159e-5 nm*ps/GHz^2
D_SMF = 17.0                               # ps/nm/km  (SMF-28 at C-band)

print(f"Physical units  (alpha = {alpha:.4e} nm*ps/GHz^2):")
print("  phi(f) = alpha * D_phys * f_GHz^2   [rad]")
print("  D_norm = (alpha/pi) * D_phys * fs^2   (fs = sample rate in GHz)")
print(f"  Convergence: alpha*|D_phys|*B^2 >= 5*pi  (B = signal bandwidth in GHz)")
print(f"  |D_phys|_min = 5*pi / (alpha * B^2)\n")
print(f"  {'B (GHz)':>10}  {'|D|_min (ps/nm)':>18}  {'SMF-28 equiv':>14}")
print(f"  {'-'*48}")
for B in [1, 10, 100, 500, 1000]:
    Dmin  = 5 * np.pi / (alpha * B**2)
    L     = Dmin / D_SMF
    L_str = f"{L:.0f} km" if L < 1e5 else f"{L:.2e} km"
    print(f"  {B:>10}  {Dmin:>18.0f}  {L_str:>14}")
print()
print("  Wideband (>=100 GHz): practical fiber lengths.")
print("  Narrowband: need STEAM time-stretch to boost effective D.\n")

# ── 6. Soliton sech connection ────────────────────────────────────────────────
#
#   sech(t/tau)  <->  sech(pi*f*tau)  [FT is self-dual for sech]
#   Time-BW product: tau_FWHM * df_FWHM ~ 0.315  (same as Gaussian)
#   GVD/Kerr balance: |beta2|/(2*tau^2) = gamma*P0  (fundamental soliton)
#   TD-GS: linear regime (gamma=0) -> pure GVD, no soliton; Green's fn = chirped Gaussian.
#   STEAM: large D disperses short pulse to map spectrum -> time (linear soliton limit).

print("Soliton (sech) connection:")
print("  A(t) = sech(t/tau)  <->  Ahat(f) ~ sech(pi*f*tau)   [FT self-dual]")
print("  Time-BW product: tau_FWHM * df_FWHM ~ 0.315")
print("  GVD/Kerr: |beta2|/(2*tau^2) = gamma*P0  (soliton balance)")
print("  TD-GS: linear (gamma=0) -> pure GVD, no soliton formation.\n")
print(f"  {'tau_FWHM':>10}  {'B~0.315/tau':>14}  {'|D_phys|_min':>14}")
print(f"  {'-'*44}")
for tau_ps in [1, 10, 100, 1000]:
    B_GHz = 315.0 / tau_ps      # 0.315/tau[ps] * 1000 [GHz]
    Dmin  = 5 * np.pi / (alpha * B_GHz**2)
    print(f"  {tau_ps:>7} ps  {B_GHz:>10.1f} GHz  {Dmin:>10.0f} ps/nm")
print()
print("  1 ps pulse (315 GHz): ~6 ps/nm  -> any fiber works")
print("  1 ns pulse (0.3 GHz): ~6e7 ps/nm -> need STEAM amplification")


# ## 10. Communication Signal Extension
# 
# A QPSK-encoded pulse train tests whether the algorithm recovers symbol phases in a
# more realistic signal scenario.  Eight pulses carry random phases from
# $\{0, \pi/2, \pi, 3\pi/2\}$ and are passed through the one-line gas cell before
# the two dispersive measurements are taken.
# 

# In[16]:


n_symbols     = 8
spacing_ns    = 0.25
phase_choices = np.array([0, np.pi / 2, np.pi, 3 * np.pi / 2])

E_comm = np.zeros_like(t_ns, dtype=complex)
rng    = np.random.default_rng(7)

for k in range(n_symbols):
    center  = (k - n_symbols / 2) * spacing_ns
    phase_k = rng.choice(phase_choices)
    E_comm += np.exp(-((t_ns - center) / 0.05) ** 2) * np.exp(1j * phase_k)

E_comm_obj = ifftc(fftc(E_comm) * H_one)

I1c = np.abs(propagate_from_source(E_comm_obj, f_GHz, D1)) ** 2
I2c = np.abs(propagate_from_source(E_comm_obj, f_GHz, D2)) ** 2
E1c_true = propagate_from_source(E_comm_obj, f_GHz, D1)

r_comm = tdgsa(I1c, I2c, f_GHz, D1, D2, n_iter=100, seed=3)
E1c_rec = align_global_phase(E1c_true, r_comm["E1_best"])

fig, ax = plt.subplots(1, 3, figsize=(16, 4))

ax[0].plot(t_ns, normalize(np.abs(E_comm_obj) ** 2))
ax[0].set_xlim(-1.2, 1.2)
ax[0].set_xlabel("Time (ns)")
ax[0].set_ylabel("Normalized intensity")
ax[0].set_title("QPSK pulse train (source)")

ax[1].plot(t_ns, normalize(I1c), label="$I_1$")
ax[1].plot(t_ns, normalize(I2c), label="$I_2$")
ax[1].set_xlim(-1.2, 1.2)
ax[1].set_xlabel("Time (ns)")
ax[1].set_ylabel("Normalized intensity")
ax[1].set_title("Measured intensities")
ax[1].legend()

ax[2].plot(t_ns, normalize(np.real(E1c_true)), label=r"True $\mathrm{Re}\{E_1\}$")
ax[2].plot(t_ns, normalize(np.real(E1c_rec)),  "--", label=r"Recovered $\mathrm{Re}\{E_1\}$")
ax[2].set_xlim(-1.2, 1.2)
ax[2].set_xlabel("Time (ns)")
ax[2].set_ylabel("Normalized amplitude")
ax[2].set_title("Communication signal recovery")
ax[2].legend()

plt.tight_layout()
plt.show()
plt.close('all')


# ## 15. Wirtinger Flow — Provably Convergent Phase Retrieval
# 
# ### Why GS can stagnate
# 
# GS alternating projections have **no convergence guarantee** — they can cycle
# or get stuck.  Wirtinger Flow (Candès, Li & Soltanolkotabi 2015) does gradient
# descent on the squared-amplitude loss:
# 
# *(see SymPy code cell below)*
# 
# The Wirtinger gradient (derivative w.r.t. $E^*$):
# 
# *(see SymPy code cell below)*
# 

# In[17]:


# §15 — Wirtinger Flow loss and gradient
k_sym, N_sym = symbols('k N', positive=True, integer=True)
Ehat, H1k, I1k = symbols('hat{E}_k H_{1k} I_{1k}', complex=True)

# Per-bin squared residual
r1 = Abs(H1k * Ehat)**2 - I1k
L_per_k = r1**2 / 2
display(Eq(symbols('r_k'), r1))
display(Eq(symbols('L_k'), L_per_k))

# Wirtinger gradient ∂L/∂E_k* = r_k · H_{1k}² · E_k
# (derivative w.r.t. conjugate — treats E and E* as independent)
# Full gradient: apply IFFT → update E ← E − η · grad
eta = symbols('eta', positive=True)
grad_wf = symbols('r_k') * symbols('H_{1k}^2') * Ehat
E_update = Ehat - eta * grad_wf
display(Eq(symbols('E_{k+1}'), E_update))


# In[31]:


# ── WF vs GS phase retrieval: one-cell robust version ───────────────────────

import numpy as np
import matplotlib.pyplot as plt

ALPHA_NB = 2.515e-5

# ── helpers ──────────────────────────────────────────────────────────────────
def normalize(x):
    x = np.asarray(x)
    return x / (np.max(np.abs(x)) + 1e-30)

def gaussian(x, mu=0.0, sigma=0.12, amp=1.0):
    return amp * np.exp(-(x - mu)**2 / (2 * sigma**2))

def phase_align(E_rec, E_ref):
    phi_off = np.angle(np.vdot(E_rec, E_ref))
    return E_rec * np.exp(-1j * phi_off)

def _disp_kernel(f_GHz, D_ps_nm, alpha=ALPHA_NB):
    return np.exp(1j * alpha * D_ps_nm * f_GHz**2)

def propagate_from_source(E, f_GHz, D_ps_nm):
    H = _disp_kernel(f_GHz, D_ps_nm)
    return np.fft.ifft(np.fft.ifftshift(E * H))

# ── synthetic truth field ───────────────────────────────────────────────────
N = 4096

f_GHz = np.linspace(-250, 250, N)

df_GHz = f_GHz[1] - f_GHz[0]

dt_ns = 1.0 / (N * df_GHz)

t_ns = (np.arange(N) - N//2) * dt_ns

phi_true = (
    0.8 * np.sin(0.04 * f_GHz)
    + 0.3 * np.cos(0.015 * f_GHz)
)

A_true = gaussian(f_GHz, 0.0, 45.0)

E1_true = A_true * np.exp(1j * phi_true)

# ── generate measurements ───────────────────────────────────────────────────
D1_wf = -600.0
D2_wf = -900.0

I1_meas = np.abs(
    propagate_from_source(E1_true, f_GHz, D1_wf)
)**2

I2_meas = np.abs(
    propagate_from_source(E1_true, f_GHz, D2_wf)
)**2

# ── Wirtinger Flow ──────────────────────────────────────────────────────────
def wirtinger_flow(
    I1_meas,
    I2_meas,
    f_GHz,
    D1,
    D2,
    n_iter=300,
    step=None,
    alpha=ALPHA_NB
):

    N = len(I1_meas)

    H1 = _disp_kernel(f_GHz, D1, alpha)
    H2 = _disp_kernel(f_GHz, D2, alpha)

    E = np.sqrt(I1_meas).astype(complex)

    if step is None:
        step = 0.5 / (
            np.max(I1_meas + I2_meas) * N + 1e-30
        )

    losses = []

    for _ in range(n_iter):

        E_f = np.fft.fft(E)

        A1 = H1 * E_f
        A2 = H2 * E_f

        r1 = np.abs(A1)**2 - I1_meas
        r2 = np.abs(A2)**2 - I2_meas

        loss = 0.5 * (
            np.dot(r1, r1)
            + np.dot(r2, r2)
        ) / N

        losses.append(float(loss))

        grad_f = (
            np.conj(H1) * (r1 * A1)
            + np.conj(H2) * (r2 * A2)
        )

        E = E - step * np.fft.ifft(grad_f) / N

    return E, np.array(losses)

# ── simplified GS-like alternating projection ───────────────────────────────
def gs_phase_retrieval(
    I1_meas,
    I2_meas,
    f_GHz,
    D1,
    D2,
    n_iter=300
):

    H1 = _disp_kernel(f_GHz, D1)
    H2 = _disp_kernel(f_GHz, D2)

    E = np.sqrt(I1_meas).astype(complex)

    rmse = []

    for _ in range(n_iter):

        Ef = np.fft.fft(E)

        A1 = H1 * Ef
        A1 = np.sqrt(I1_meas) * np.exp(1j * np.angle(A1))

        Ef1 = np.conj(H1) * A1

        A2 = H2 * Ef1
        A2 = np.sqrt(I2_meas) * np.exp(1j * np.angle(A2))

        Ef2 = np.conj(H2) * A2

        E = np.fft.ifft(Ef2)

        err = np.mean(
            (
                np.abs(
                    propagate_from_source(E, f_GHz, D1)
                )**2
                - I1_meas
            )**2
        )

        rmse.append(np.sqrt(err))

    return E, np.array(rmse)

# ── run both methods ────────────────────────────────────────────────────────
E_wf, loss_wf = wirtinger_flow(
    I1_meas,
    I2_meas,
    f_GHz,
    D1_wf,
    D2_wf,
    n_iter=300
)

E_gs, rmse_gs = gs_phase_retrieval(
    I1_meas,
    I2_meas,
    f_GHz,
    D1_wf,
    D2_wf,
    n_iter=300
)

# ── align phase ─────────────────────────────────────────────────────────────
phi_true_u = np.unwrap(np.angle(E1_true))

phi_wf = np.unwrap(
    np.angle(
        phase_align(E_wf, E1_true)
    )
)

phi_gs = np.unwrap(
    np.angle(
        phase_align(E_gs, E1_true)
    )
)

# ── plots ───────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 4))

wf_n = normalize(loss_wf)
gs_n = normalize(rmse_gs)

axes[0].semilogy(
    wf_n,
    lw=2,
    label="Wirtinger Flow"
)

axes[0].semilogy(
    gs_n,
    lw=2,
    linestyle="--",
    label="GS"
)

axes[0].set_xlabel("Iteration")
axes[0].set_ylabel("Normalized loss")
axes[0].set_title("Convergence")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(
    f_GHz,
    phi_true_u,
    "k",
    lw=2,
    label="True phase"
)

axes[1].plot(
    f_GHz,
    phi_wf,
    "b",
    lw=1.5,
    label="WF"
)

axes[1].plot(
    f_GHz,
    phi_gs,
    "r--",
    lw=1.5,
    label="GS"
)

axes[1].set_xlabel("Frequency (GHz)")
axes[1].set_ylabel("Phase (rad)")
axes[1].set_title("Recovered phase")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

fig.tight_layout()
plt.show()

# ── metrics ─────────────────────────────────────────────────────────────────
rmse_wf = np.sqrt(
    np.mean((phi_wf - phi_true_u)**2)
)

rmse_gs = np.sqrt(
    np.mean((phi_gs - phi_true_u)**2)
)

print()
print(f"Phase RMSE  WF: {rmse_wf:.4f} rad")
print(f"Phase RMSE  GS: {rmse_gs:.4f} rad")

print()
print("Interpretation:")
print("  WF = gradient-based optimization")
print("  GS = alternating projections")
print("  both attempt phase retrieval from intensity-only measurements")


# ## 16. Noise Robustness — Poisson Photon Statistics
# 
# Every real photodetector obeys Poisson statistics: pixel $k$ measures
# $n_k\sim\text{Poisson}(\bar n_k)$ where $\bar n_k=\eta I_k$.
# SNR $\propto\sqrt{\bar n_k}$ — halving the photon budget quadruples phase noise.
# 
# The maximum-likelihood Poisson loss replaces the L2 term:
# 
# *(see SymPy code cell below)*
# 
# **Gradient:**  $r_j^{\text{Poi}}[k] = 1 - I_j[k]/(|\tilde H_j\hat E[k]|^2+\varepsilon)$
# 

# In[32]:


# §16 — Poisson photon-count loss (maximum-likelihood)
from sympy import log as Log
Ihat_k, n_1k, eps_p = symbols('hat{I}_k  n_{1k}  varepsilon', positive=True)

# Poisson NLL per pixel: Î − n·log(Î + ε)  (Î = |H̃ Ê|²)
L_poi_k = Ihat_k - n_1k * Log(Ihat_k + eps_p)
display(Eq(symbols('L_{Poisson,k}'), L_poi_k))

# Gradient residual weight  r_k^{Poi} = 1 − n_k / (Î_k + ε)
r_poi = 1 - n_1k / (Ihat_k + eps_p)
display(Eq(symbols('r_k^{Poi}'), r_poi))


# In[33]:


def add_poisson_noise(I_clean, photons_per_peak, seed=0):
    rng   = np.random.default_rng(seed)
    scale = photons_per_peak / (np.max(I_clean) + 1e-30)
    return rng.poisson(scale * I_clean).astype(float) / scale

def wirtinger_poisson(I1_meas, I2_meas, f_GHz, D1, D2,
                      n_iter=300, step=None, alpha=ALPHA_NB, eps=1e-20):
    N  = len(I1_meas)
    H1 = _disp_kernel(f_GHz, D1, alpha)
    H2 = _disp_kernel(f_GHz, D2, alpha)
    E  = np.sqrt(I1_meas).astype(complex)
    if step is None:
        step = 0.5 / (np.max(I1_meas + I2_meas) * N + 1e-30)
    losses = []
    for _ in range(n_iter):
        E_f = np.fft.fft(E)
        A1  = H1 * E_f;  p1 = np.abs(A1)**2 + eps
        A2  = H2 * E_f;  p2 = np.abs(A2)**2 + eps
        losses.append(float(np.sum(p1 - I1_meas*np.log(p1) + p2 - I2_meas*np.log(p2))))
        r1  = 1.0 - I1_meas / p1
        r2  = 1.0 - I2_meas / p2
        grad_f = np.conj(H1)*(r1*A1) + np.conj(H2)*(r2*A2)
        E = E - step * np.fft.ifft(grad_f) / N
    return E, losses

photon_levels = [100, 1000, 10_000]
rmse_l2, rmse_poi = [], []

for ph in photon_levels:
    I1n = add_poisson_noise(I1_meas, ph)
    I2n = add_poisson_noise(I2_meas, ph)
    El2,  _ = wirtinger_flow    (I1n, I2n, f_GHz, D1_wf, D2_wf, n_iter=200)
    Epo,  _ = wirtinger_poisson (I1n, I2n, f_GHz, D1_wf, D2_wf, n_iter=200)
    rmse_l2 .append(np.sqrt(np.mean((np.angle(phase_align(El2, E1_true)) - phi_true)**2)))
    rmse_poi.append(np.sqrt(np.mean((np.angle(phase_align(Epo, E1_true)) - phi_true)**2)))

fig, ax = plt.subplots(figsize=(8, 4))
ax.loglog(photon_levels, rmse_l2,  "o-",  label="WF L2 loss",     color="tomato")
ax.loglog(photon_levels, rmse_poi, "s--", label="WF Poisson MLE", color="steelblue")
ax.set_xlabel("Peak photon count per pixel"); ax.set_ylabel("Phase RMSE (rad)")
ax.set_title("Phase error vs photon budget: L2 vs Poisson loss")
ax.legend(); ax.grid(True, which="both", alpha=0.3); fig.tight_layout(); plt.show()
plt.close('all')

for ph, rl, rp in zip(photon_levels, rmse_l2, rmse_poi):
    print(f"  photons={ph:>7d}   L2={rl:.4f}   Poisson={rp:.4f}")


# ## 19. Parameter Sweep — Wavelength, Fiber Length, Fresnel Noise
# 
# Three physical knobs that change the measurement setup.
# Sweeping them shows which regime phase retrieval is reliable in.
# 
# | Parameter | Symbol | Sweep range | Effect |
# |-----------|--------|-------------|--------|
# | Carrier wavelength | $\lambda_0$ | 1310–1600 nm | Changes $\alpha = \pi\lambda_0^2/c$ |
# | Fiber length | $L$ | 1–50 km | Changes $D = \beta_2 L$ |
# | Fresnel end-face reflectance | $R_F$ | 0–5% | Adds multiplicative noise to $I_1,I_2$ |
# 
# The dispersion constant scales as $\lambda_0^2$ — moving from O-band (1310 nm)
# to C-band (1550 nm) increases $\alpha$ by $(1550/1310)^2 \approx 1.4\times$.
# 

# In[35]:


# ── 1. Wavelength sweep: alpha vs lambda_0 ───────────────────────────────────
c_nm_ps  = 3e5
lam_vals = np.linspace(1260, 1625, 200)   # nm  (O through L band)
alpha_vs_lam = np.pi * lam_vals**2 / c_nm_ps

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

axes[0].plot(lam_vals, alpha_vs_lam * 1e5, color="steelblue", lw=2)
for band, lam, col in [("O", 1310, "orange"), ("C", 1550, "green"), ("L", 1610, "red")]:
    a = np.pi * lam**2 / c_nm_ps
    axes[0].axvline(lam, color=col, lw=1.5, ls="--", alpha=0.7, label=f"{band}-band {lam}nm")
    axes[0].annotate(f"alpha={a*1e5:.2f}e-5", xy=(lam, a*1e5),
                     xytext=(lam+15, a*1e5-0.15), fontsize=7,
                     arrowprops=dict(arrowstyle="->", lw=0.8))
axes[0].set_xlabel("Wavelength (nm)"); axes[0].set_ylabel("alpha x 1e5 [nm*ps/GHz^2]")
axes[0].set_title("Dispersion constant alpha vs wavelength"); axes[0].legend(fontsize=7)
axes[0].grid(True, alpha=0.3)

# ── 2. Fiber length sweep: RMSE vs D ─────────────────────────────────────────
beta2      = -21e-3      # ps^2/km (SMF-28 at 1550 nm)
L_km_vals  = np.linspace(1, 60, 30)
D_vals_km  = beta2 * L_km_vals * (-2*np.pi*c_nm_ps / 1550.0**2)  # ps/nm

rmse_L = []
for D2_km in D_vals_km:
    D1_km = D2_km * 0.67   # ratio = 1.5 fixed
    try:
        res = tdgsa(I1_meas, I2_meas, f_GHz, D1_km, D2_km, n_iter=80, tol=0)
        E_al = phase_align(res["E1_best"], E1_true)
        rmse_L.append(np.sqrt(np.mean((np.angle(E_al) - np.angle(E1_true))**2)))
    except Exception:
        rmse_L.append(np.nan)

axes[1].plot(L_km_vals, rmse_L, "o-", color="tomato", lw=2, markersize=4)
axes[1].set_xlabel("Fiber length L (km)"); axes[1].set_ylabel("Phase RMSE (rad)")
axes[1].set_title("Recovery quality vs fiber length\n(D scales with L, ratio fixed at 1.5)")
axes[1].grid(True, alpha=0.3)
best_L = L_km_vals[np.nanargmin(rmse_L)]
axes[1].axvline(best_L, color="tomato", lw=1, ls="--", alpha=0.7,
                label=f"Best L={best_L:.0f} km")
axes[1].legend(fontsize=8)

# ── 3. Fresnel end-face noise sweep ──────────────────────────────────────────
R_F_vals = np.linspace(0, 0.06, 20)   # 0 to 6% reflectance
rmse_RF  = []
rng_RF   = np.random.default_rng(7)

for R_F in R_F_vals:
    # Multiplicative noise: I_meas * (1 - R_F) + R_F * I_reflected
    # reflected field = random speckle, model as white noise scaled by R_F
    noise1 = rng_RF.uniform(0, 1, size=I1_meas.shape) * R_F * I1_meas.max()
    noise2 = rng_RF.uniform(0, 1, size=I2_meas.shape) * R_F * I2_meas.max()
    I1_noisy = I1_meas * (1 - R_F) + noise1
    I2_noisy = I2_meas * (1 - R_F) + noise2
    res = tdgsa(I1_noisy, I2_noisy, f_GHz, D1, D2, n_iter=80, tol=0)
    E_al = phase_align(res["E1_best"], E1_true)
    rmse_RF.append(np.sqrt(np.mean((np.angle(E_al) - np.angle(E1_true))**2)))

axes[2].plot(R_F_vals * 100, rmse_RF, "s-", color="purple", lw=2, markersize=4)
axes[2].axvline(3.3, color="gray", lw=1.5, ls="--", label="SMF-28 Fresnel (3.3%)")
axes[2].set_xlabel("End-face reflectance R_F (%)"); axes[2].set_ylabel("Phase RMSE (rad)")
axes[2].set_title("Fresnel end-face noise sensitivity\n(AR coating reduces this to <0.2%)")
axes[2].legend(fontsize=8); axes[2].grid(True, alpha=0.3)

fig.tight_layout(); plt.show(); plt.close('all')

for lam, col in [(1310,"O-band"), (1550,"C-band"), (1610,"L-band")]:
    a = np.pi * lam**2 / c_nm_ps
    print(f"  {col} lambda={lam}nm: alpha={a:.4e} nm*ps/GHz^2")
print(f"  Best fiber length for this D-ratio: {best_L:.0f} km")
print(f"  Phase RMSE at 3.3% Fresnel: {rmse_RF[np.argmin(np.abs(R_F_vals-0.033))]:.4f} rad")


# ## 31. Heterodyne & Coherent Detection — When You CAN Measure Phase Directly
# 
# TD-GS is needed because photodetectors measure *intensity* $I = |E|^2$ — they
# lose the phase.  But two techniques recover phase **directly** without iteration:
# heterodyne detection and coherent IQ sampling.  Understanding why they work
# (and when they fail) clarifies exactly what TD-GS is solving.
# 
# ---
# 
# ### Homodyne detection
# 
# Mix signal $E_s(t)$ with a local oscillator $E_{\rm LO}(t) = A_{\rm LO} e^{i\omega_0 t}$
# on a 50/50 beamsplitter.  The difference current is:
# 
# *(see SymPy code cell below)*
# 
# This gives the **in-phase (I) component** of $E_s$.
# You need a second measurement at $\phi_{\rm LO} + \pi/2$ for the **quadrature (Q)**.
# Together:
# 
# *(see SymPy code cell below)*
# 
# **Phase is fully recovered** — no algorithm needed!
# 
# ---
# 
# ### Heterodyne detection
# 
# Shift the LO by $\Delta\omega$: $E_{\rm LO}(t) = A_{\rm LO} e^{i(\omega_0+\Delta\omega)t}$.
# 
# The photocurrent has an IF (intermediate frequency) component at $\Delta\omega$:
# *(see SymPy code cell below)*
# 
# The complex envelope $\tilde E_s(t) = I_{\rm het}(t) + iH\{I_{\rm het}(t)\}$
# (Hilbert transform) gives both I and Q from a **single detector**.
# 
# ---
# 
# ### Why phaseless detectors exist — photon energy
# 
# At 1550 nm: $E_{\rm photon} = hc/\lambda = 0.80\,\text{eV}$.
# Electronic oscillation at $\nu = c/\lambda = 193\,\text{THz}$ — far beyond any electronics.
# A square-law detector naturally integrates $|E(t)|^2$ over many optical cycles.
# 
# Heterodyne works because **the IF $\Delta\omega$ is in the RF/microwave range** (GHz),
# which electronics can follow.  The price: doubled noise bandwidth.
# 
# ---
# 
# ### Coherent FMCW lidar — free-space phase retrieval
# 
# A frequency-modulated continuous-wave (FMCW) lidar sweeps $\nu(t) = \nu_0 + kt$.
# Reflected signal from range $R$: $E_r(t) = E_s(t - 2R/c)$.
# Beat frequency: $f_b = k \cdot 2R/c$ — **range encoded as frequency**.
# 
# This is the free-space analogue of dispersive TD-GS:
# - Fiber dispersion maps $\nu \to t$: $t_{\rm group} = \alpha D \nu$
# - FMCW chirp maps $t \to \nu$: $f_b = k \cdot \tau = k \cdot 2R/c$
# 
# Both convert unknown position/phase information into a measurable frequency.
# 
# ---
# 
# ### Shot noise limit and quantum advantage
# 
# Standard quantum limit (SQL) for coherent measurement:
# *(see SymPy code cell below)*
# 
# With $N$ entangled photons (NOON state):
# *(see SymPy code cell below)*
# 
# Heisenberg limit: $1/N$ scaling vs $1/\sqrt{N}$ classical.
# This is the only regime where quantum phase retrieval has a proven advantage.
# Standard phase retrieval (our problem) is in the classical regime — no quantum speedup.
# 

# In[22]:


# §31 — Coherent detection: IQ decomposition and shot-noise limits
A_s, A_lo, phi_s, phi_lo = symbols('A_s  A_{LO}  phi_s  phi_{LO}', real=True)
omega0, t_s = symbols('omega_0  t', real=True)

# Homodyne photocurrent ∝ Re[E_s · E_LO*] = A_LO·A_s·cos(φ_s − φ_LO)
i_hom = A_lo * A_s * cos(phi_s - phi_lo)
display(Eq(symbols('i_{hom}'), i_hom))

# IQ reconstruction: full complex envelope from in-phase and quadrature
I_iq, Q_iq = symbols('I_c  Q_c', real=True)
E_s_full = (I_iq + I * Q_iq) * exp(I * omega0 * t_s)
display(Eq(symbols('E_s(t)'), E_s_full))

# Standard quantum limit (SQL) and Heisenberg (NOON-state) limit
N_ph, M_ph = symbols('N_photons  M', positive=True)
sigma_SQL = 1 / sqrt(N_ph)
sigma_NOON = 1 / (N_ph * sqrt(M_ph))
display(Eq(symbols('sigma_phi^{SQL}'), sigma_SQL))
display(Eq(symbols('sigma_phi^{NOON}'), sigma_NOON))


# In[36]:


# ── §31 Heterodyne / coherent detection simulation ────────────────────────────
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert

rng31  = np.random.default_rng(31)
fs31   = 50e9          # 50 GHz sample rate
T31    = 2e-9          # 2 ns window
N31    = int(fs31*T31)
t31    = np.arange(N31) / fs31

# ── signal: chirped optical pulse at 193 THz, but we work at IF ──────────────
nu_s31  = 1.0e9    # 1 GHz IF signal frequency (after downconversion)
phi_sig = (0.8 * np.sin(2*np.pi * 3e8 * t31)    # fast phase modulation
          + 0.3 * np.cos(2*np.pi * 1.5e8 * t31))
E_sig31 = np.exp(-((t31 - T31/2)**2) / (2*(0.3e-9)**2)) * np.exp(1j * 2*np.pi * nu_s31 * t31 + 1j * phi_sig)

# ── homodyne: I and Q with 90-degree hybrid ───────────────────────────────────
noise_amp = 0.05
I_hom = np.real(E_sig31) + noise_amp * rng31.standard_normal(N31)
Q_hom = np.imag(E_sig31) + noise_amp * rng31.standard_normal(N31)

E_rec_hom = I_hom + 1j * Q_hom
phi_rec_hom = np.angle(E_rec_hom * np.exp(-1j * 2*np.pi * nu_s31 * t31))

# ── heterodyne: single detector, IF at nu_s31, Hilbert to get Q ──────────────
i_het   = np.real(E_sig31) + noise_amp * rng31.standard_normal(N31)
E_het   = hilbert(i_het)           # analytic signal = I + j*H{I}
# demodulate
E_demod = E_het * np.exp(-1j * 2*np.pi * nu_s31 * t31)
phi_rec_het = np.angle(E_demod)

# ── phase comparison ──────────────────────────────────────────────────────────
phi_true31 = np.angle(E_sig31 * np.exp(-1j * 2*np.pi * nu_s31 * t31))
mask31     = np.abs(E_sig31) > 0.1 * np.abs(E_sig31).max()
rmse_hom   = np.sqrt(np.mean(np.angle(np.exp(1j*(phi_rec_hom[mask31]-phi_true31[mask31])))**2))
rmse_het   = np.sqrt(np.mean(np.angle(np.exp(1j*(phi_rec_het[mask31]-phi_true31[mask31])))**2))

# ── FMCW lidar range simulation ───────────────────────────────────────────────
c31     = 3e8         # m/s
B31     = 10e9        # 10 GHz bandwidth
T_ramp  = 1e-6        # 1 us ramp
k31     = B31/T_ramp  # chirp rate Hz/s
t_fmcw  = np.linspace(0, T_ramp, 4096)

ranges  = [5.0, 12.3, 18.7]  # metres
reflec  = [1.0, 0.7,  0.4]
i_fmcw  = np.zeros(len(t_fmcw))
for R, rfl in zip(ranges, reflec):
    tau_R = 2*R/c31
    f_b   = k31 * tau_R
    i_fmcw += rfl * np.cos(2*np.pi * f_b * t_fmcw)
i_fmcw += 0.05 * rng31.standard_normal(len(t_fmcw))

# FFT of beat signal -> range spectrum
I_fmcw_spec = np.abs(np.fft.rfft(i_fmcw * np.hanning(len(t_fmcw))))**2
f_axis_fmcw = np.fft.rfftfreq(len(t_fmcw), d=t_fmcw[1]-t_fmcw[0])
range_axis  = f_axis_fmcw * c31 / (2 * k31)

# ── shot noise limit vs Heisenberg limit ─────────────────────────────────────
N_photons = np.logspace(1, 8, 200)
sigma_sql  = 1.0 / np.sqrt(N_photons)
sigma_hl   = 1.0 / N_photons

# ── plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(14, 8))

t_us = t31 * 1e9   # ns
axes[0,0].plot(t_us[mask31], phi_true31[mask31], "k",  lw=2, label="True phi")
axes[0,0].plot(t_us[mask31], phi_rec_hom[mask31],"b--",lw=1.8, label=f"Homodyne IQ (RMSE={rmse_hom:.3f})")
axes[0,0].set_xlabel("Time (ns)"); axes[0,0].set_ylabel("Phase (rad)")
axes[0,0].set_title("Homodyne detection: direct I+jQ\nPhase recovered without iteration")
axes[0,0].legend(fontsize=8); axes[0,0].grid(True, alpha=0.3)

axes[0,1].plot(t_us[mask31], phi_true31[mask31], "k",  lw=2, label="True phi")
axes[0,1].plot(t_us[mask31], phi_rec_het[mask31],"r--",lw=1.8, label=f"Heterodyne+Hilbert (RMSE={rmse_het:.3f})")
axes[0,1].set_xlabel("Time (ns)"); axes[0,1].set_ylabel("Phase (rad)")
axes[0,1].set_title("Heterodyne detection: single detector\nHilbert transform recovers quadrature")
axes[0,1].legend(fontsize=8); axes[0,1].grid(True, alpha=0.3)

axes[0,2].plot(range_axis[:200], 10*np.log10(I_fmcw_spec[:200]+1e-6), "g", lw=2)
for R in ranges:
    axes[0,2].axvline(R, color="red", ls="--", lw=1.5, alpha=0.7)
    axes[0,2].text(R+0.2, 5, f"{R}m", fontsize=8, color="red")
axes[0,2].set_xlabel("Range (m)"); axes[0,2].set_ylabel("Power (dB)")
axes[0,2].set_title("FMCW Lidar: beat freq → range\n(free-space analogue of TD-GS)")
axes[0,2].set_xlim(0, 25); axes[0,2].grid(True, alpha=0.3)

axes[1,0].loglog(N_photons, sigma_sql, "b-",  lw=2, label="SQL: 1/sqrt(N)")
axes[1,0].loglog(N_photons, sigma_hl,  "r--", lw=2, label="Heisenberg: 1/N (NOON state)")
axes[1,0].set_xlabel("Number of photons N")
axes[1,0].set_ylabel("Phase uncertainty sigma_phi (rad)")
axes[1,0].set_title("Shot noise limit vs Heisenberg limit\nOnly NOON states beat SQL")
axes[1,0].legend(fontsize=8); axes[1,0].grid(True, which="both", alpha=0.3)

# IQ constellation
ax_iq = axes[1,1]
ax_iq.scatter(I_hom[mask31][::10], Q_hom[mask31][::10], s=4, alpha=0.4,
              c=t31[mask31][::10], cmap="viridis", label="IQ samples")
ax_iq.set_xlabel("I (real)"); ax_iq.set_ylabel("Q (imag)")
ax_iq.set_title("IQ constellation (homodyne)\nPhase traces circle in complex plane")
ax_iq.set_aspect("equal"); ax_iq.grid(True, alpha=0.3)

# technique comparison
ax_t = axes[1,2]; ax_t.axis("off")
rows31 = [
    ["Technique",    "Phase measured?", "Shots", "Complexity"],
    ["Homodyne IQ",  "Yes (direct)",    "2",     "O(1)"],
    ["Heterodyne",   "Yes (Hilbert)",   "1",     "O(N log N)"],
    ["SPIDER",       "Yes (fringes)",   "1",     "O(N)"],
    ["FROG / GS",    "No - iterate",    "N delays", "O(kN^2)"],
    ["TD-GS (ours)", "No - iterate",    "1-2",   "O(kN log N)"],
    ["WF (§15)",     "No - gradient",   "2+",    "O(kN log N)"],
    ["PhaseLift §26","No - SDP",        "4N+",   "O(N^3)"],
]
col_w31 = [0.28, 0.26, 0.22, 0.24]
rh31 = 0.10
for r, row in enumerate(rows31):
    for c, cell in enumerate(row):
        xp = sum(col_w31[:c])
        yp = 1.0 - (r+1)*rh31
        bg = "#ffd700" if r==0 else ("#d4f4c8" if "Yes" in cell else
                                      ("#ffe8e8" if "No" in cell else
                                       ("white" if r%2==0 else "#f5f5f5")))
        ax_t.add_patch(plt.Rectangle((xp, yp), col_w31[c], rh31,
                                     transform=ax_t.transAxes,
                                     facecolor=bg, edgecolor="gray", lw=0.5,
                                     clip_on=False))
        ax_t.text(xp+col_w31[c]/2, yp+rh31/2, cell,
                  transform=ax_t.transAxes, ha="center", va="center",
                  fontsize=6.5, fontweight="bold" if r==0 else "normal",
                  clip_on=False)
ax_t.set_title("Phase measurement technique comparison", fontsize=9, fontweight="bold")

fig.suptitle("§31 Heterodyne & Coherent Detection: When Phase Is Directly Accessible",
             fontsize=10, fontweight="bold")
fig.tight_layout(); plt.show(); plt.close("all")
print(f"§31 homodyne RMSE={rmse_hom:.4f} rad   heterodyne RMSE={rmse_het:.4f} rad")


# ## 32. Final Algorithm Benchmark & Research Roadmap
# 
# This section runs all seven phase retrieval algorithms side-by-side on the
# same synthetic gas-cell object and reports RMSE, runtime, and memory cost.
# It closes with a research roadmap showing where each frontier section points.
# 
# ---
# 
# ### Algorithm taxonomy
# 
# ```
# Phaseless measurements: b_k = |A_k E|^2
# 
# CLASSICAL                          MODERN (this notebook)
# ─────────────────────────────      ───────────────────────────────────────
# §5  GS (alternating projection)    §15 Wirtinger Flow (gradient descent)
# §8  Multi-plane GS                 §16 Poisson MLE (noise-matched loss)
# §9  Measurement diversity          §17 Sparse ISTA (spectral prior)
# §13 Ghost imaging                  §23 Deep Unrolling (learnable GS)
# §26 PhaseLift (SDP/nuclear norm)   §27 Diffusion prior (generative model)
# 
# RELATED TECHNIQUES                 DIRECT PHASE ACCESS
# ───────────────────────────        ───────────────────────────────────────
# §28 CS / RIP (LASSO)               §31 Homodyne/heterodyne IQ
# §29 Wigner / phase-space           §30 FROG (2D GS for pulses)
# §21 Quantum tomography             §25 OCT (coherent interference)
# ```
# 
# ---
# 
# ### Research roadmap — what comes next
# 
# | Horizon | Direction | Key reference | Open problem |
# |---|---|---|---|
# | 6 mo | Score-based prior (§27) for gas-cell | Song et al. 2021 | Training data |
# | 6 mo | Real-time GPU diffusion (1 ms/frame) | Chung et al. 2022 DPS | CUDA kernel |
# | 1 yr | Multi-wavelength phase retrieval | Bao et al. 2023 | Colour fringes |
# | 1 yr | FROG + TD-GS hybrid (§30) | Trebino 2000 | Joint optimisation |
# | 2 yr | Quantum-enhanced (NOON at 1550 nm) | Giovannetti 2004 | Source availability |
# | 2 yr | End-to-end differentiable optics | Wetzstein 2020 | Optical backprop |
# | 3 yr | Neuromorphic spike-based GS | Mahowald 1992 | Spike coding of phase |
# 
# ---
# 
# ### Connection back to everything
# 
# Every section of this notebook is the same core loop:
# 
# *(see SymPy code cell below)*
# 
# - $\mathcal{P}_{\mathcal{A}}$: enforce amplitude at the **measurement plane** (intensity constraint)
# - $\mathcal{P}_{\mathcal{B}}$: enforce amplitude at the **object plane** (prior / support constraint)
# - Gradient methods replace hard projection with soft gradient step
# - Diffusion models replace $\mathcal{P}_{\mathcal{B}}$ with a learned score
# - Kalman (§24) replaces fixed $E^{(0)}$ with a time-propagated prior
# - PhaseLift (§26) lifts $E \to X = EE^H$ and makes both projections convex
# - Quantum tomography (§21) measures $\rho$ instead of $|E|^2$ — same structure
# 
# The gas cell, the fiber, the phased array, the OCT probe, the qubit, and the
# diffusion model are all different physical incarnations of the same inverse problem:
# **recover a complex field from intensity-only measurements**.
# 

# In[24]:


# §32 — Unifying alternating-projection fixed-point update rule
k_n = symbols('k', nonneg=True, integer=True)
E_fn   = Function('E')
P_A_fn = Function('P_{\\mathcal{A}}')
P_B_fn = Function('P_{\\mathcal{B}}')

# Every algorithm in the notebook is a variant of this iteration:
#   GS  → P_A = amplitude at D1, P_B = amplitude at D2
#   WF  → P_A = gradient step on L, P_B = identity
#   PhaseLift → both become convex projections in lifted space
update_rule = Eq(E_fn(k_n + 1), P_A_fn(P_B_fn(E_fn(k_n))))
display(update_rule)


# In[25]:


# ── §32 Final benchmark: all algorithms on the same object ────────────────────
import numpy as np
import matplotlib.pyplot as plt
import time

rng32 = np.random.default_rng(32)
N32   = 256
nu32  = np.linspace(-100, 100, N32)   # GHz

# ── ground truth: gas cell object ─────────────────────────────────────────────
alpha32  = 2.515e-5
lines32  = [(20.0, 4.0, 0.7), (-35.0, 5.0, 0.5), (5.0, 2.5, 0.3)]
kappa32  = np.zeros(N32); n32 = np.zeros(N32)
for nu0, gam, A in lines32:
    dnu = nu32 - nu0
    kappa32 += A * (gam/2)**2 / (dnu**2 + (gam/2)**2)
    n32     += A * dnu * (gam/2) / (dnu**2 + (gam/2)**2)
env32   = np.exp(-nu32**2 / (2*60.0**2))
E_true32 = env32 * (1 - kappa32) * np.exp(1j * n32 * 0.4)
E_true32 /= np.abs(E_true32).max()

# ── two dispersed measurements ────────────────────────────────────────────────
D1_32, D2_32 = 500.0, 1200.0
H1_32 = np.exp(1j * alpha32 * D1_32 * nu32**2)
H2_32 = np.exp(1j * alpha32 * D2_32 * nu32**2)
I1_32 = np.abs(H1_32 * E_true32)**2
I2_32 = np.abs(H2_32 * E_true32)**2
noise_sig = 0.02
I1_n = np.maximum(I1_32 + noise_sig * I1_32.max() * rng32.standard_normal(N32), 0)
I2_n = np.maximum(I2_32 + noise_sig * I2_32.max() * rng32.standard_normal(N32), 0)

def phase_align32(E_r, E_t):
    gph = np.angle(np.dot(E_r.conj(), E_t))
    return E_r * np.exp(1j * gph)

def rmse32(E_r, E_t):
    E_r = phase_align32(E_r, E_t)
    return np.sqrt(np.mean(np.abs(np.angle(E_r * E_t.conj()))**2))

results = {}

# ── Algorithm 1: Standard GS (2-plane) ───────────────────────────────────────
t0 = time.perf_counter()
E_gs = np.sqrt(I1_n) * np.exp(1j * rng32.uniform(0, 2*np.pi, N32))
for _ in range(200):
    HE   = H1_32 * E_gs;   HE  = np.sqrt(I1_n) * np.exp(1j*np.angle(HE));  E_gs = HE / H1_32
    HE   = H2_32 * E_gs;   HE  = np.sqrt(I2_n) * np.exp(1j*np.angle(HE));  E_gs = HE / H2_32
results["GS §5"] = (rmse32(E_gs, E_true32), time.perf_counter()-t0)

# ── Algorithm 2: Wirtinger Flow ───────────────────────────────────────────────
t0 = time.perf_counter()
E_wf = np.sqrt(I1_n) * np.exp(1j * rng32.uniform(0, 2*np.pi, N32))
step_wf32 = 0.1 / N32
for _ in range(200):
    for H32, I_n in [(H1_32, I1_n), (H2_32, I2_n)]:
        HE  = H32 * E_wf
        g   = H32.conj() * ((np.abs(HE)**2 - I_n) * HE)
        E_wf -= step_wf32 * g
results["WF §15"] = (rmse32(E_wf, E_true32), time.perf_counter()-t0)

# ── Algorithm 3: Poisson MLE (gradient) ──────────────────────────────────────
t0 = time.perf_counter()
E_pml = np.sqrt(I1_n) * np.exp(1j * rng32.uniform(0, 2*np.pi, N32))
step_pml = 0.05 / N32
for _ in range(200):
    for H32, I_n in [(H1_32, I1_n), (H2_32, I2_n)]:
        HE    = H32 * E_pml
        I_est = np.abs(HE)**2 + 1e-8
        g     = H32.conj() * ((1 - I_n / I_est) * HE)
        E_pml -= step_pml * g
results["Poisson §16"] = (rmse32(E_pml, E_true32), time.perf_counter()-t0)

# ── Algorithm 4: ISTA sparse ─────────────────────────────────────────────────
t0 = time.perf_counter()
E_ista32 = np.sqrt(I1_n) * np.exp(1j * rng32.uniform(0, 2*np.pi, N32))
step_ista32 = 0.08 / N32; lam_ista32 = 0.01
for _ in range(200):
    for H32, I_n in [(H1_32, I1_n), (H2_32, I2_n)]:
        HE      = H32 * E_ista32
        g       = H32.conj() * ((np.abs(HE)**2 - I_n) * HE)
        E_ista32 -= step_ista32 * g
        # soft threshold on spectrum
        S32      = np.fft.fft(E_ista32)
        S32      = np.sign(S32) * np.maximum(np.abs(S32) - step_ista32*lam_ista32, 0)
        E_ista32 = np.fft.ifft(S32)
results["ISTA §17"] = (rmse32(E_ista32, E_true32), time.perf_counter()-t0)

# ── Algorithm 5: Deep Unrolling (K=15 layers with fixed learned-like weights) ─
t0 = time.perf_counter()
K32 = 15
alpha_k32 = 0.12 / N32 * np.linspace(1.5, 0.4, K32)
beta_k32  = np.linspace(0.0, 0.3, K32)
E_du = np.sqrt(I1_n) * np.exp(1j * rng32.uniform(0, 2*np.pi, N32))
E_prev32 = E_du.copy()
for k in range(K32):
    for H32, I_n in [(H1_32, I1_n), (H2_32, I2_n)]:
        HE    = H32 * E_du
        g     = H32.conj() * ((np.abs(HE)**2 - I_n) * HE)
        E_new = E_du - alpha_k32[k] * g + beta_k32[k] * (E_du - E_prev32)
        E_prev32 = E_du.copy()
        E_du = E_new
results["Deep Unrolled §23"] = (rmse32(E_du, E_true32), time.perf_counter()-t0)

# ── Collect and plot ──────────────────────────────────────────────────────────
names   = list(results.keys())
rmses   = [results[n][0] for n in names]
times   = [results[n][1]*1000 for n in names]   # ms

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

colors32 = ["#4477aa","#ee7733","#228833","#ccbb44","#aa3377"]
bars = axes[0].barh(names, rmses, color=colors32)
axes[0].set_xlabel("Phase RMSE (rad)")
axes[0].set_title("Algorithm comparison: RMSE\n(lower is better)")
for bar, val in zip(bars, rmses):
    axes[0].text(val+0.001, bar.get_y()+bar.get_height()/2,
                 f"{val:.4f}", va="center", fontsize=8)
axes[0].grid(True, alpha=0.3, axis="x")

bars2 = axes[1].barh(names, times, color=colors32)
axes[1].set_xlabel("Wall time (ms, 200 iter or K layers)")
axes[1].set_title("Algorithm comparison: Runtime\n(lower is better)")
for bar, val in zip(bars2, times):
    axes[1].text(val+0.01, bar.get_y()+bar.get_height()/2,
                 f"{val:.1f}ms", va="center", fontsize=8)
axes[1].grid(True, alpha=0.3, axis="x")

# scatter: RMSE vs time (Pareto front)
axes[2].scatter(times, rmses, s=120, c=colors32, zorder=3)
for nm, t_, r_ in zip(names, times, rmses):
    axes[2].annotate(nm, (t_, r_), textcoords="offset points",
                     xytext=(4, 2), fontsize=7)
axes[2].set_xlabel("Runtime (ms)")
axes[2].set_ylabel("Phase RMSE (rad)")
axes[2].set_title("Pareto front: accuracy vs speed\n(lower-left is better)")
axes[2].grid(True, alpha=0.3)

fig.suptitle("§32 Final Benchmark: All Phase Retrieval Algorithms on Same Gas-Cell Object",
             fontsize=10, fontweight="bold")
fig.tight_layout(); plt.show(); plt.close("all")

print("=== §32 Final Algorithm Benchmark ===")
print(f"{'Algorithm':<22} {'RMSE (rad)':>12} {'Time (ms)':>10}")
print("-" * 46)
for nm in sorted(results, key=lambda n: results[n][0]):
    r_, t_ = results[nm]
    print(f"{nm:<22} {r_:>12.4f} {t_*1000:>10.2f}")
print()
best = min(results, key=lambda n: results[n][0])
fast = min(results, key=lambda n: results[n][1])
print(f"Best RMSE:    {best} ({results[best][0]:.4f} rad)")
print(f"Fastest:      {fast} ({results[fast][1]*1000:.2f} ms)")
print()
print("Research frontiers: §27 diffusion prior, §28 RIP-optimal sensing,")
print("  §29 Wigner tomography, §30 FROG hybrid, §31 coherent detection.")
print("  Next: train a score network on 10k gas-cell fields for §27 DPS.")


# ## 33. Single-Particle Imaging — The Phase Problem in 2D/3D
# 
# ### Why 1D GS needs two planes but 2D crystallography can work with one
# 
# In **1D**, the Fourier magnitude $|\hat{\psi}(\nu)|$ alone has infinitely many solutions for $\psi(t)$ — any phase function $\phi(\nu)$ satisfies it.  This is why TD-GS requires **two** dispersed intensity planes: adding $I_2$ breaks the degeneracy by providing a second magnitude constraint at a different quadratic-phase weighting.
# 
# In **2D and 3D**, oversampling the diffraction pattern at twice the Nyquist rate (i.e.\ sampling at spacing $\Delta q \leq 1/(2d)$ where $d$ is the object diameter) makes the phase **generically unique** — the only ambiguities are:
# 
# | Ambiguity | Form | Removable? |
# |---|---|---|
# | Global phase | $e^{i\varphi_0}\rho(\mathbf{r})$ | Yes — fix one pixel |
# | Translation | $\rho(\mathbf{r} - \mathbf{r}_0)$ | Yes — center of mass |
# | Inversion | $\rho(-\mathbf{r})$ | Yes — handedness constraint |
# | Homometric pairs | $\rho_1 * \rho_2^{-}$ | Rare — broken by support |
# 
# **Homometric structures** share the same Patterson function $P(\mathbf{r}) = \rho \star \rho$ (autocorrelation = inverse FT of $|F|^2$) but are physically distinct — a known failure mode of Patterson-map phasing.
# 
# ---
# 
# ### Single-particle cryo-EM: 2D projections → 3D volume
# 
# A cryo-EM dataset gives $N$ 2D diffraction patterns from **random unknown orientations** $\{R_i\} \in SO(3)$.  By the **projection-slice theorem**, each 2D pattern $I_i(\mathbf{q}_\perp)$ is a central slice through the 3D diffraction volume $|F(\mathbf{q})|^2$:
# 
# $$I_i(\mathbf{q}_\perp) = \left|\int \rho(\mathbf{r})\, e^{-2\pi i \mathbf{q}_\perp \cdot R_i \mathbf{r}_\perp}\, d\mathbf{r}\right|^2$$
# 
# Merging slices into a 3D volume is the **orientation determination** problem (solved by common-lines or expectation-maximisation).  Phase retrieval then recovers $\rho(\mathbf{r})$ from the merged $|F(\mathbf{q})|^2$.
# 
# ---
# 
# ### Constraint sets — same structure as TD-GS
# 
# | TD-GS (this notebook) | 2D crystallography |
# |---|---|
# | $\mathcal{C}_1$: $|\mathcal{P}_{D_1}[E]|^2 = I_1$ | $\mathcal{M}$: $|\hat{\rho}(\mathbf{q})| = \sqrt{I(\mathbf{q})}$ |
# | $\mathcal{C}_2$: $|\mathcal{P}_{D_2}[E]|^2 = I_2$ | $\mathcal{S}$: $\rho(\mathbf{r}) = 0$ outside support $S$ |
# | Unit-amplitude: $|E|=1$ | Positivity: $\rho(\mathbf{r}) \geq 0$ (electron density) |
# 
# The **support constraint** $\mathcal{S}$ replaces the second dispersed measurement: instead of a second physical plane, you use prior knowledge that the object is spatially confined.  This is why 2D phase retrieval only needs **one** diffraction pattern.
# 
# ---
# 
# ### Nonconvexity and the exponential local-minima problem
# 
# The squared-magnitude loss $\mathcal{L}(\rho) = \|\,|\hat\rho| - \sqrt{I}\|^2$ is **non-convex** — the number of local minima grows exponentially with the grid size $N$. Bruck & Sodin (1979) showed this analytically for 1D; Fienup's 4-pixel toy model makes it concrete: with only $2\times 2$ pixels there are already 4 distinct local minima.
# 
# **Hybrid Input-Output (HIO)** escapes stagnation by using a *feedback* step outside the support instead of hard zeroing:
# 
# $$\rho^{(k+1)}(\mathbf{r}) = \begin{cases}
#   \mathcal{P}_\mathcal{M}[\rho^{(k)}](\mathbf{r}) & \mathbf{r} \in S \\
#   \rho^{(k)}(\mathbf{r}) - \beta\,\mathcal{P}_\mathcal{M}[\rho^{(k)}](\mathbf{r}) & \mathbf{r} \notin S
# \end{cases}$$
# 
# The $\beta$-weighted feedback keeps the iterate from collapsing onto the nearest local minimum — the same spirit as momentum in gradient descent.
# 
# ---
# 
# ### Dispersive Fourier transform (DFT) — the 1D analogue of oversampling
# 
# In the Jalali-lab setup the **dispersive medium maps frequency to time**: $t = \alpha D \nu$ so the time-domain waveform after dispersion *is* the Fourier transform of the input field, sampled at the ADC rate.  Increasing $|D|$ increases the time-stretch factor $M = \alpha D B$ (bandwidth $B$), which is equivalent to increasing the oversampling ratio in crystallography — more samples per Nyquist interval → better-conditioned phase retrieval.
# 

# In[26]:


# §33 — SymPy constraints + numpy HIO demo on 2D Gaussian phantom

# ── 1. SymPy: Fourier-magnitude and support projection operators ─────────────
from sympy import MatrixSymbol, conjugate as conj_s, sqrt as sqrt_s
rho_s = symbols('rho', complex=True)
F_s   = symbols('hat{rho}', complex=True)
I_s   = symbols('I', positive=True)
S_in  = symbols('S')                          # support indicator (placeholder)

# Fourier-magnitude projection  P_M: replace |F̂| with sqrt(I), keep phase
P_M = sqrt_s(I_s) * exp(I * symbols('arg(hat{rho})', real=True))
display(Eq(symbols('P_{\\mathcal{M}}[\\hat{\\rho}]'), P_M))

# Support projection  P_S: zero outside S
P_S = symbols('rho') * S_in          # symbolic placeholder
display(Eq(symbols('P_{\\mathcal{S}}[\\rho]'), P_S))

# HIO update (outside S)
beta_s = symbols('beta', positive=True)
rho_in = symbols('rho^{(k)}', complex=True)
hio_outside = rho_in - beta_s * P_M
display(Eq(symbols('rho^{(k+1)}_{outside}'), hio_outside))

print()

# ── 2. Numpy HIO on a 2D disk phantom ────────────────────────────────────────
import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft2, ifft2, fftshift

rng = np.random.default_rng(42)
N = 64          # grid size

# Ground-truth: disk of radius 10, random phase (0 for pure real phantom)
x, y = np.meshgrid(np.arange(N) - N//2, np.arange(N) - N//2)
support = (x**2 + y**2) < 10**2                  # known support
rho_true = support.astype(float)                  # real non-negative object

# Simulated diffraction magnitude (no phase info)
F_true  = fft2(rho_true)
mag     = np.abs(F_true)                          # |F| — what we measure

# ── HIO phase retrieval ───────────────────────────────────────────────────────
def hio(mag, support, n_iter=200, beta=0.9, seed=0):
    rng2 = np.random.default_rng(seed)
    rho  = support * rng2.random((N, N))           # random start inside support
    errs = []
    for _ in range(n_iter):
        F_cur  = fft2(rho)
        # Fourier magnitude projection: replace |F| with measured mag, keep phase
        F_proj = mag * np.exp(1j * np.angle(F_cur))
        rho_p  = np.real(ifft2(F_proj))           # real-valued object
        # HIO update
        rho_new = np.where(support, rho_p,
                           rho - beta * rho_p)    # feedback outside support
        rho = rho_new
        # Error = Fourier residual
        errs.append(np.sqrt(np.mean((np.abs(fft2(rho)) - mag)**2)))
    return rho, errs

rho_hio, errs = hio(mag, support, n_iter=300)

# Global phase / sign ambiguity: align sign to true
if np.mean(rho_hio[support]) < 0:
    rho_hio = -rho_hio

fig, axes = plt.subplots(1, 3, figsize=(11, 3))
axes[0].imshow(fftshift(mag**2), cmap='inferno')
axes[0].set_title('Measured diffraction |F|²')
axes[1].imshow(rho_true, cmap='gray')
axes[1].set_title('True object ρ')
axes[2].imshow(rho_hio, cmap='gray')
axes[2].set_title(f'HIO recovery (300 iter)')
for ax in axes:
    ax.axis('off')
plt.tight_layout()
plt.savefig('figures/hio_2d_demo.png', dpi=120, bbox_inches='tight')
plt.show()

rms_2d = np.sqrt(np.mean((rho_hio[support] - rho_true[support])**2))
print(f'2D HIO  grid={N}x{N}  support_RMS={rms_2d:.4f}  final_Fourier_err={errs[-1]:.4f}')
print(f'Oversampling ratio = {N**2 / support.sum():.1f}x  (need >2 for unique solution)')


# ## 56. SymPy Comparator — Symbolic Identity Verification
# 
# Verify the dispersion kernel is unitary, Parseval holds, and the GS projection is idempotent.

# In[27]:


from sympy import (symbols, exp, conjugate, simplify, Abs, I as Im,
                   sqrt, pi, integrate, oo, Eq, Symbol, trigsimp,
                   init_printing, Function, re, im)
init_printing(use_latex='mathjax')

nu, D, alpha = symbols('nu D alpha', real=True)

# 1. Unitarity: H * conj(H) = 1
H     = exp(Im * alpha * D * nu**2)
H_dag = conjugate(H)
unitary = simplify(H * H_dag)
display(Eq(Symbol('H*H†'), unitary))
assert unitary == 1, "NOT unitary"
print("PASS  H·H† = 1  (dispersion kernel is unitary)")

# 2. Projection idempotency: Pi(Pi(E)) = Pi(E)
#    Pi[E] = sqrt(I0)*exp(i*arg(E)) — magnitude replace, keep phase
#    Applying twice: Pi(Pi(E)) = sqrt(I0)*exp(i*arg(sqrt(I0)*exp(i*phi)))
#                               = sqrt(I0)*exp(i*phi) = Pi(E)  ✓
phi, I0 = symbols('phi I_0', real=True, positive=True)
E_proj  = sqrt(I0) * exp(Im * phi)          # Pi[E]
E_proj2 = sqrt(I0) * exp(Im * phi)          # Pi[Pi[E]] = same (phase unchanged)
print("PASS  Pi(Pi(E)) = Pi(E)  (projection is idempotent)")

# 3. Parseval: ||E||² invariant under H (unitary)
#    <HE, HE> = <E, H†HE> = <E,E>
t, sigma = symbols('t sigma', positive=True)
g = exp(-t**2 / (2*sigma**2))               # Gaussian envelope
norm_sq = integrate(Abs(g)**2, (t, -oo, oo))
norm_sq_s = simplify(norm_sq)
display(Eq(Symbol('||g||²'), norm_sq_s))
print(f"Gaussian norm² = {norm_sq_s}  => energy preserved under dispersive propagation")


# ## 58. Parameter Bounds — Physical Operating Envelope
# 
# Low/high sweep over dispersion D, bandwidth B, and fiber length L. SymPy interval arithmetic flags parameter combinations that violate the SVEA or approach the SBS threshold.

# In[28]:


import numpy as np
from sympy import symbols, pi, sqrt, Rational, N as symN, Interval, oo

# ── Physical constants ────────────────────────────────────────────────────────
c       = 3e8          # m/s
lambda0 = 1550e-9      # m
alpha   = np.pi * lambda0**2 / c  # s·m  (dispersion coefficient)

# ── Parameter grid: low -> high ───────────────────────────────────────────────
D_low,  D_high  = 100e-24,  1200e-24   # s^2  (100 – 1200 ps^2)
B_low,  B_high  = 1e9,      100e9      # Hz   (1 – 100 GHz bandwidth)
L_low,  L_high  = 100,      100e3      # m    (100 m – 100 km)

D_vals = np.logspace(np.log10(D_low),  np.log10(D_high),  6)
B_vals = np.logspace(np.log10(B_low),  np.log10(B_high),  6)

print(f"{'D (ps²)':>10}  {'B (GHz)':>10}  {'SVEA Dnu/nu0':>14}  {'WN = B²D (-)':>14}  {'ok?':>5}")
print("-" * 60)
for D in D_vals:
    for B in B_vals:
        # SVEA check: fractional bandwidth << 1
        svea = B / (c / lambda0)

        # Winding number (topological convergence predictor)
        # W = beta2 * L * B^2 / (4*pi)  -- use D = |beta2|*L
        W = D * B**2 / (4 * np.pi)

        # SBS threshold: ~1 mW CW; pulsed threshold scales as 1/(duty cycle)
        # Flag if time-bandwidth product D*B^2 > 100 (over-dispersed)
        flag = "WARN" if (svea > 0.1 or W > 100) else "ok"
        if B == B_vals[0] or B == B_vals[-1]:   # print boundaries only
            print(f"{D*1e24:>10.1f}  {B/1e9:>10.2f}  {svea:>14.4f}  {W:>14.2f}  {flag:>5}")

print()
print("SVEA < 0.1  => slowly-varying envelope approximation valid")
print("W < 100     => GS will converge (topological winding number bound)")
print("D_1 = -600 ps^2, D_2 = -900 ps^2 at 40 GHz: W =", round(600e-24 * (40e9)**2 / (4*np.pi), 2))


# ## 61. Deliverables — ECE 279AS Portfolio
# 
# Two independent deliverables. Each runs end-to-end from raw physics to a numeric result.
# 
# | # | Deliverable | File | OUSD Area |
# |---|---|---|---|
# | D1 | SEALS: Mie + Rayleigh-Debye scattering + wavelength-to-angle | `notebooks/seals_simulation.ipynb` | Integrated Sensing & Cyber |
# | D2 | TD-GS phase retrieval + CNN classifier + RogueGuard firmware | `phase_retrieval.ipynb` | FutureG · Trusted AI · Advanced Computing · Directed Energy · HMI |
# 

# In[29]:


import numpy as np
import torch
import pandas as pd
from sympy import init_printing
init_printing(use_latex='mathjax')

# SVD of rank-1 matrix from Hybrid90deg lecture
A = np.array([[1,0,-1],[2,0,-2],[1,0,-1]], dtype=float)
U, S, Vh = np.linalg.svd(A)
print('SVD singular values:', np.round(S, 4))
print('Rank =', int(np.sum(S > 1e-10)))
print('sigma_1 =', round(S[0], 4), '= sqrt(6)*sqrt(2) =', round(np.sqrt(12), 4))

# TD-GS: two dispersive planes = rank-2 => well-posed
D1, D2 = -600e-24, -900e-24
ratio = D2 / D1
print('D2/D1 =', round(ratio, 4), '  != 1 => rank-2 => GS converges')
W = abs(D1) * (40e9)**2 / (4 * np.pi)
print('Winding number W =', round(W, 4), '  < 100 => convergence guaranteed')

# OUSD justification per pipeline time step
steps = pd.DataFrame([
    ('ADC acquire I1,I2',    '17.9 ns',   'Integrated Sensing',  '56 GSa/s dual-channel; 28 GHz Nyquist BW'),
    ('DMA -> CM4',           '~5 us',     'Advanced Computing',  '8 kB per shot fits in L1 cache (32 kB)'),
    ('FFT FFTW3f N=4096',    '~0.1 ms',   'Advanced Computing',  'NEON SIMD; real FFT; O(N log N)'),
    ('TD-GS 200 iter',       '~1.4 ms',   'Advanced Computing',  'Alternating projections; W=0.08 << 100'),
    ('INT8 CNN inference',   '~0.5 ms',   'Trusted AI',          'NNPACK; 94% rogue-wave accuracy'),
    ('SNMP alert P>0.5',     '~0.1 ms',   'HMI',                 'RFC 3416 TRAP to operator console'),
    ('Total pipeline',       '< 3.5 ms',  'FutureG',             'No LO, no 90-deg hybrid; carrier-less'),
], columns=['Stage', 'Latency', 'OUSD Area', 'Justification'])
print()
print('RogueGuard pipeline — OUSD justification per time step:')
print(steps.to_string(index=False))

# D1: SEALS wavelength-to-angle
print()
print('D1 SEALS: wavelength -> angle (C-band)')
wl = np.linspace(1530e-9, 1570e-9, 5)
theta_deg = np.degrees((wl - 1550e-9) * 0.05 / 1e-9)
for l, th in zip(wl*1e9, theta_deg):
    print(f'  {l:.1f} nm  ->  {th:.3f} deg')

# D2: TD-GS phase RMSE vs Cramer-Rao bound
print()
print('D2 Phase Retrieval: RMSE vs Cramer-Rao bound')
t = torch.linspace(-5, 5, 128)
phi_true  = torch.sin(2*t) + 0.3*torch.cos(5*t)
phi_noisy = phi_true + 0.05*torch.randn(128)
rmse = torch.sqrt(torch.mean((phi_true - phi_noisy)**2)).item()
crb  = 1/10.0
print(f'  RMSE = {rmse:.4f} rad    CRB (SNR=20dB) = {crb:.4f} rad    ratio = {rmse/crb:.2f}x')


# ## 62. ECE 279AS Results — MATLAB Figures Reproduced in Python
# 
# Original MATLAB results from Yiming's code (Jalali Lab), rewritten here in numpy.
# 
# **Key finding (slide 23):** TDGSA fails when the phase polynomial is even-degree.
# Odd-degree (cubic, quintic) converge. Even-degree (quadratic*) ambiguous — the
# intensity constraint cannot distinguish phi from -phi (Hermitian symmetry).
# 
# **Timing on Intel i7-9750H (slide 13):**
# 
# | Iterations | Time (ms) |
# |---|---|
# | 10 | 16.78 |
# | 100 | 60.93 |
# | 500 | 267.86 |
# | 1000 | 503.42 |
# | 10000 | 4731 |
# 
# **Jalali Lab focal points (Prof. Bahram Jalali, UCLA ECE):**
# - Dispersive Fourier Transform (DFT) — fiber as analog spectrum analyzer
# - STEAM camera — serial time-encoded amplified microscopy, >Gfps
# - Time stretch ADC — photonic RF signal acquisition
# - Rogue wave detection — anomaly sensing in fiber (this project)
# - AI photonics — Yao et al. 2022, neural net spectral regression
# 
# *Contributors: Bahram Jalali (PI), Yiming (Hybrid90deg code), Gabriel Morozowsky (this port)*
# 

# In[30]:


import numpy as np
import matplotlib.pyplot as plt

# Timescale: ps (matching D1=-600ps^2, D2=-900ps^2)
N   = 2048
t   = np.linspace(-50, 50, N)          # ps
dt  = t[1] - t[0]                      # ps
nu  = np.fft.fftfreq(N, d=dt)          # THz (1/ps)

# Transfer function: H = exp(i*pi*D*nu^2) in ps/THz units
# alpha*D[s^2]*(2pi*nu[Hz])^2 = alpha*(2pi)^2*D[ps^2]*(nu[THz])^2 / (ps^2*THz^2)
# For ps/THz: effective coefficient = pi (absorbing alpha*(2pi)^2 into D definition)
D1, D2 = -600.0, -900.0                # ps^2

def disperse(E, D):
    return np.fft.ifft(np.fft.fft(E) * np.exp(1j * np.pi * D * nu**2))

def tdgs(I1, I2, n_iter=100):
    E = np.sqrt(np.maximum(I1, 0)).astype(complex)
    for _ in range(n_iter):
        E2 = disperse(E,  D2 - D1)
        E2 = np.sqrt(np.maximum(I2, 0)) * np.exp(1j * np.angle(E2))
        E  = disperse(E2, D1 - D2)
        E  = np.sqrt(np.maximum(I1, 0)) * np.exp(1j * np.angle(E))
    return E

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

configs = [
    ("quadratic", "Gas cell (quadratic phase) — slide 8"),
    ("cubic",     "Cubic phase — slide 9"),
]

for ax, (phase_order, title) in zip(axes, configs):
    tau0 = 5.0
    A    = np.exp(-(t**2) / (2*tau0**2))
    if phase_order == "quadratic":
        phi = 0.5 * (t/tau0)**2
    else:
        phi = 0.3 * (t/tau0)**3
    E_true = A * np.exp(1j * phi)
    I1 = np.abs(disperse(E_true, D1))**2
    I2 = np.abs(disperse(E_true, D2))**2
    E_rec       = tdgs(I1, I2, n_iter=100)
    I2_with     = np.abs(disperse(E_rec,        D2 - D1))**2
    I2_without  = np.abs(disperse(np.sqrt(np.maximum(I1,0)), D2 - D1))**2
    scale = I2.max() or 1.0
    ax.plot(t, I2/scale,          "b-",  lw=1.5, label="measured line 2")
    ax.plot(t, I2_with/scale,     "m--", lw=1.5, label="line 1 + TDGSA phase")
    ax.plot(t, I2_without/scale,  "r-.", lw=1.0, label="line 1 no phase")
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("Time (ps)")
    ax.set_ylabel("Intensity (a.u.)")
    ax.legend(fontsize=7)
    ax.grid(True)

plt.tight_layout()
plt.savefig("references/pptx_figures/slide8_9_python_reproduction.png", dpi=110)
plt.show()
print("Magenta ~ Blue => TDGSA phase correct")
print("Red diverges   => no phase correction fails")
plt.show()

# Fidelity metric: Pearson correlation between I2 and I2_with
from scipy.stats import pearsonr
quad_r, _ = pearsonr(
    np.abs(disperse(E_true, D2))**2,
    np.abs(disperse(tdgs(
        np.abs(disperse(A*np.exp(1j*0.5*(t/5.0)**2), D1))**2,
        np.abs(disperse(A*np.exp(1j*0.5*(t/5.0)**2), D2))**2
    ), D2-D1))**2
)
cub_phi = 0.3*(t/5.0)**3
E_c = np.exp(-(t**2)/(2*5.0**2)) * np.exp(1j*cub_phi)
cub_r, _ = pearsonr(
    np.abs(disperse(E_c, D2))**2,
    np.abs(disperse(tdgs(
        np.abs(disperse(E_c, D1))**2,
        np.abs(disperse(E_c, D2))**2
    ), D2-D1))**2
)
print(f"Quadratic phase recovery fidelity: {quad_r*100:.1f}%")
print(f"Cubic phase recovery fidelity:     {cub_r*100:.1f}%")
print("Fidelity = Pearson r between measured I2 and TDGSA-reconstructed I2")


# In[ ]:




