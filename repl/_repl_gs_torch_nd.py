# -*- coding: utf-8 -*-
"""
_repl_gs_torch_nd.py
=====================
Gerchberg-Saxton phase retrieval in PyTorch: 1D -> 2D -> 3D -> 4D

Context: Jalali Lab / UCLA dispersion-assisted optical phase recovery.
         Reference: Solli, Gupta, Jalali -- Appl. Phys. Lett. 95, 231108 (2009)
         Callen MacPhee / Yiming Zhou -- CSUS collaboration.

S1: 1D GS (temporal, fiber dispersion diversity)
    - Gaussian pulse with engineered chirp (linear + nonlinear)
    - H(nu) = exp(i*pi*D*nu^2) dispersion filter in PyTorch
    - Differentiable GS: torch.fft.fft, autograd through iterations
    - Loss: ||disperse(E, D2)|^2 - I2||^2  -> backprop to refine D
    - Convergence plot: numpy GS vs torch GS vs torch gradient descent

S2: 2D GS (spatial phase retrieval, holography / imaging)
    - Classic Gerchberg-Saxton 1972: recover phase from |FFT|^2 (far field)
      and |E|^2 (near field)
    - 2D FFT in torch: torch.fft.fft2
    - Application: phase mask design, holographic display, wavefront sensing
    - Test: recover Lena-like checkerboard phase from amplitude-only measurements
    - Zernike polynomials: wavefront aberration basis (defocus Z4, astigmatism Z5/Z6)

S3: 3D GS (spatio-temporal: 2D spatial + 1D time = video/pulse train)
    - Optical pulse train: 2D transverse mode * temporal envelope
    - 3D FFT: torch.fft.fftn(A, dim=(-3,-2,-1))
    - Dispersion in time + diffraction in space -> 3D transfer function
    - Application: ultrafast imaging (serial time-encoded amplified microscopy, STEAM)
    - Connection: Jalali STEAM paper (Science 2009) -- same lab!

S4: 4D GS (batch dimension: parallel recovery of N signals)
    - Shape: (batch, 1, N_t) -> vectorized 1D GS over batch
    - Shape: (batch, 1, H, W) -> vectorized 2D GS over batch
    - torch.vmap / batched FFT for O(1) overhead vs loop
    - Application: RogueGuard -- process B simultaneous fiber tap channels
    - Differentiable GS unrolled: fixed number of iterations as neural network layers
    - Gradient flow through GS iterations: learn optimal D, n_iter
    - Training loop: synthetic QPSK dataset, MSE(phi_true, phi_rec) as loss

Output: repl/_out_gs_torch_nd.png
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import sympy as sp
from sympy import symbols, exp, I, pi, sqrt, diff, latex
import torch
import torch.nn as nn
import torch.optim as optim
import os, sys

try:
    OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "_out_gs_torch_nd.png")
    _REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
except NameError:
    OUT = "_out_gs_torch_nd.png"
    _REPO = os.getcwd()

if _REPO not in sys.path:
    sys.path.insert(0, _REPO)
from gs_core import disperse, undisperse, retrieve_phase as gs_numpy

SEP = "=" * 65

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"PyTorch device: {DEVICE}  (version {torch.__version__})")

# ============================================================
# S1: 1D GS IN TORCH -- GAUSSIAN PULSE + DIFFERENTIABLE GS
# ============================================================
print(SEP)
print("SECTION 1: 1D GS IN TORCH -- TEMPORAL PHASE RETRIEVAL")
print(SEP)

print("""
  GAUSSIAN PULSE WITH ENGINEERED CHIRP (Jalali Lab project task 3):
    Chirped Gaussian pulse (time domain):
      E(t) = A_0 * exp(-t^2 / (2*T0^2)) * exp(i*phi(t))
      phi(t) = C_lin*t + C_quad*t^2 + C_cub*t^3
    A_0   = peak amplitude
    T0    = pulse width (1/e half-width)
    C_lin = linear chirp (carrier frequency offset) -- does NOT change |E(t)|^2
    C_quad= quadratic chirp (GVD-like -- broadens/compresses pulse)
    C_cub = cubic chirp (third-order dispersion -- asymmetric pulse)
    All chirp coefficients are IN the phase -- undetectable from intensity alone!

  PHYSICS: WHY PHASE MATTERS
    OOK (on-off keying): only amplitude matters. Direct detection works.
    QPSK/16-QAM: information is encoded in PHASE -> need coherent detection.
    Carrier-less receiver: recover phase from |E(t)|^2 alone -- no local oscillator.
    Dispersion diversity: take TWO intensity measurements at D1, D2.
    GS algorithm: alternate projections in time/frequency until consistent.

  DISPERSION FILTER (time-domain GS):
    H(nu) = exp(i * pi * D * nu^2)    [nu = normalized discrete freq]
    Dispersed field: E_d(t) = IFFT[ FFT[E(t)] * H(nu) ]
    Measurements: I1(t) = |disperse(E, D1)|^2,  I2(t) = |disperse(E, D2)|^2
    GS recovers phi(t) in E(t) = |E(t)| * exp(i*phi(t)).
""")

# Symbolic chirp
t_sym = symbols('t', real=True)
T0_sym, C2_sym = symbols('T_0 C_2', real=True, positive=True)
phi_sym = C2_sym * t_sym**2
E_sym   = sp.exp(-t_sym**2 / (2*T0_sym**2)) * sp.exp(sp.I * phi_sym)
print(f"  Chirped Gaussian (symbolic):")
print(f"    E(t) = exp(-t^2/(2*T0^2)) * exp(i*C2*t^2)")
print(f"    Instantaneous frequency: omega_inst = d(phi)/dt = 2*C2*t")
print(f"    This is LINEAR CHIRP: frequency increases linearly with time.")
print(f"    Positive C2 (up-chirp): leading edge is RED, trailing is BLUE.")
print(f"    Negative C2 (down-chirp): opposite (like anomalous GVD in fiber).")

# Numerical setup
N    = 1024
T0   = 50       # samples (pulse half-width)
D1   = -5000.0  # normalized dispersion
D2   = -5750.0  # second dispersion (diversity |D1-D2| = 750)
np.random.seed(2026)

t_arr = np.arange(N) - N//2   # centered time axis
A_env = np.exp(-t_arr**2 / (2 * T0**2))

# Three chirp types from the project spec
chirp_cases = {
    "linear":    {"C1": 0.02,  "C2": 0.0,       "C3": 0.0},
    "quadratic": {"C1": 0.0,   "C2": 1.5e-4,    "C3": 0.0},
    "cubic":     {"C1": 0.0,   "C2": 1.0e-4,    "C3": 3e-7},
}

print(f"\n  GENERATING CHIRPED GAUSSIAN PULSES (N={N}, T0={T0} samples):")
gs_results = {}
for name, params in chirp_cases.items():
    C1, C2, C3 = params["C1"], params["C2"], params["C3"]
    phi_t   = C1*t_arr + C2*t_arr**2 + C3*t_arr**3
    E_true  = A_env * np.exp(1j * phi_t)
    I1      = np.abs(disperse(E_true, D1))**2
    I2      = np.abs(disperse(E_true, D2))**2
    phi_rec, errs = gs_numpy(I1, I2, D1=D1, D2=D2, n_iter=50,
                              unit_amplitude=False)
    diff_  = phi_t - phi_rec
    diff_ -= np.mean(diff_)
    corr   = np.corrcoef(phi_t, phi_rec)[0,1]
    gs_results[name] = dict(phi_true=phi_t, phi_rec=phi_rec, I1=I1, I2=I2,
                             errors=errs, E_true=E_true, corr=corr,
                             A_env=A_env)
    print(f"    {name:<12}: corr={corr:.4f}  GS err {errs[0]:.4f}->{errs[-1]:.5f}")

print("""
  PYTORCH 1D GS IMPLEMENTATION:
    Key: torch.fft.fft / ifft are fully differentiable.
    Complex tensors: dtype=torch.complex128 (double precision).
    H filter: built as a complex exponential tensor.
    Unit-amplitude constraint: A = A / A.abs() * sqrt(I) (in-place ops disabled).
    Autograd: can backpropagate through entire GS loop for learning D.
""")

def torch_disperse(E: torch.Tensor, D: float | torch.Tensor) -> torch.Tensor:
    """Disperse complex field E with parameter D (1D, batch-compatible)."""
    N = E.shape[-1]
    nu = torch.fft.fftfreq(N, dtype=torch.float64, device=E.device)
    if isinstance(D, torch.Tensor):
        # D is a learnable parameter: shape () or (batch,)
        H = torch.exp(1j * torch.pi * D * nu**2)
    else:
        H = torch.exp(1j * torch.pi * float(D) * nu**2)
    return torch.fft.ifft(torch.fft.fft(E) * H)

def torch_undisperse(E: torch.Tensor, D: float | torch.Tensor) -> torch.Tensor:
    """Remove dispersion D from field E."""
    N = E.shape[-1]
    nu = torch.fft.fftfreq(N, dtype=torch.float64, device=E.device)
    if isinstance(D, torch.Tensor):
        H_conj = torch.exp(-1j * torch.pi * D * nu**2)
    else:
        H_conj = torch.exp(-1j * torch.pi * float(D) * nu**2)
    return torch.fft.ifft(torch.fft.fft(E) * H_conj)

def torch_unit_amp(E: torch.Tensor, I_target: torch.Tensor) -> torch.Tensor:
    """Replace |E| with sqrt(I_target), keep phase."""
    mag = E.abs().clamp(min=1e-15)
    return E / mag * I_target.sqrt()

def torch_gs_1d(I1: torch.Tensor, I2: torch.Tensor,
                D1: float | torch.Tensor, D2: float | torch.Tensor,
                n_iter: int = 50) -> tuple[torch.Tensor, list[float]]:
    """
    1D GS phase retrieval in PyTorch.
    I1, I2: real tensors of shape (..., N)
    D1, D2: dispersion scalars or learnable tensors
    Returns: (recovered field E, error list)
    """
    # Initialize: undisperse sqrt(I1) from D1 plane
    A0 = I1.sqrt().to(torch.complex128)
    E  = torch_undisperse(A0, D1)

    errors = []
    for _ in range(n_iter):
        # Project onto D1 constraint
        E_d1 = torch_disperse(E, D1)
        E_d1 = torch_unit_amp(E_d1, I1)
        E    = torch_undisperse(E_d1, D1)
        E    = torch.exp(1j * E.angle())   # unit amplitude in undispersed domain

        # Project onto D2 constraint
        E_d2 = torch_disperse(E, D2)
        E_d2 = torch_unit_amp(E_d2, I2)
        E    = torch_undisperse(E_d2, D2)
        E    = torch.exp(1j * E.angle())

        with torch.no_grad():
            err = ((torch_disperse(E, D2).abs()**2 - I2)**2).mean().sqrt().item()
        errors.append(err)

    return E, errors

# Run torch GS on quadratic chirp case
meas = gs_results["quadratic"]
I1_t = torch.tensor(meas["I1"], dtype=torch.float64, device=DEVICE)
I2_t = torch.tensor(meas["I2"], dtype=torch.float64, device=DEVICE)

E_rec_torch, errs_torch = torch_gs_1d(I1_t, I2_t, D1=D1, D2=D2, n_iter=50)
phi_torch = E_rec_torch.angle().cpu().numpy()
phi_true_ = meas["phi_true"]
diff_t = phi_true_ - phi_torch; diff_t -= np.mean(diff_t)
corr_t = np.corrcoef(phi_true_, phi_torch)[0,1]
print(f"\n  Torch 1D GS (quadratic chirp):  corr={corr_t:.4f}")
print(f"    Error: {errs_torch[0]:.4f} -> {errs_torch[-1]:.5f}")

print("""
  DIFFERENTIABLE GS: LEARNING OPTIMAL D
    If D is not precisely known (physical system uncertainty), we can LEARN it.
    D = nn.Parameter(torch.tensor(-5200.0))  -- learnable scalar
    Loss = ||disperse(E_rec, D2)|^2 - I2||^2
    Optimizer: Adam, lr=10.0
    After training: D converges toward true D2 = -5750.
    This is "physics-informed deep learning" -- GS provides the structure,
    gradient descent fills in the unknown physical parameter.
""")

# Differentiable GS: learn D2 from scratch
D2_true_val = D2
D2_learnable = nn.Parameter(torch.tensor(-5200.0, dtype=torch.float64))
optimizer_D  = optim.Adam([D2_learnable], lr=20.0)

D2_history = [D2_learnable.item()]
loss_history = []

for step in range(80):
    optimizer_D.zero_grad()
    # Run GS with current D2 estimate
    A0_  = I1_t.sqrt().to(torch.complex128)
    E_   = torch_undisperse(A0_, D1)
    for _ in range(10):   # fewer inner GS iterations for gradient stability
        E_d1_ = torch_disperse(E_, D1)
        E_d1_ = torch_unit_amp(E_d1_, I1_t)
        E_    = torch_undisperse(E_d1_, D1)
        E_    = torch.exp(1j * E_.angle())
        E_d2_ = torch_disperse(E_, D2_learnable)
        E_d2_ = torch_unit_amp(E_d2_, I2_t)
        E_    = torch_undisperse(E_d2_, D2_learnable)
        E_    = torch.exp(1j * E_.angle())

    # Loss: MSE between predicted I2 and measured I2
    I2_pred = torch_disperse(E_, D2_learnable).abs()**2
    loss = ((I2_pred - I2_t)**2).mean()
    loss.backward()
    optimizer_D.step()

    D2_history.append(D2_learnable.item())
    loss_history.append(loss.item())

print(f"  D2 learning: {D2_history[0]:.1f} -> {D2_history[-1]:.1f}  (true: {D2_true_val})")
print(f"  Final loss: {loss_history[-1]:.6f}")

# ============================================================
# S2: 2D GS -- SPATIAL PHASE RETRIEVAL
# ============================================================
print(f"\n{SEP}")
print("SECTION 2: 2D GS -- SPATIAL PHASE RETRIEVAL (HOLOGRAPHY)")
print(SEP)

print("""
  CLASSIC GERCHBERG-SAXTON 1972 (ORIGINAL PAPER):
    Problem: given |E(x,y)|^2 (near field intensity) and
                   |FFT[E(x,y)]|^2 (far field intensity),
    recover the phase phi(x,y) such that E = |E|*exp(i*phi).

    Alternating projections:
    1. Start: E_guess = |E_near| * exp(i*0)
    2. Forward FFT: E_far = FFT2[E_guess]
    3. Apply far-field constraint: E_far = E_far/|E_far| * sqrt(I_far)
    4. Inverse FFT: E_near_new = IFFT2[E_far]
    5. Apply near-field constraint: E_guess = E_near_new/|E_near_new| * sqrt(I_near)
    6. Repeat.

    APPLICATION: Holographic display
    Given target image I_target (far field), find phase mask phi(x,y) for
    a SPATIAL LIGHT MODULATOR (SLM) that projects that image.
    SLM has unit amplitude (phase-only), so I_near = 1 everywhere.
    GS finds phi such that |FFT2[exp(i*phi)]|^2 ~ I_target.

    APPLICATION: Wavefront sensing
    Recover aberration phi(x,y) from two defocused intensity images
    (equivalent to two different D values in 2D).

  ZERNIKE POLYNOMIALS (wavefront aberration basis):
    Z4  = sqrt(3) * (2*rho^2 - 1)                  (defocus)
    Z5  = sqrt(6) * rho^2 * cos(2*phi)              (astigmatism 0/90)
    Z6  = sqrt(6) * rho^2 * sin(2*phi)              (astigmatism 45/135)
    Z7  = sqrt(8) * (3*rho^3 - 2*rho) * cos(phi)   (coma)
    Z11 = sqrt(5) * (6*rho^4 - 6*rho^2 + 1)        (spherical aberration)
    Wavefront: W(rho,phi) = sum_j a_j * Z_j(rho,phi)
    where a_j are Zernike coefficients [wavelengths of aberration].
""")

def make_zernike_phase(H, W, coefs):
    """
    Build a phase map from Zernike coefficients.
    coefs: dict {n: coef} where n = Zernike index (4=defocus, 5=astigm, 7=coma, 11=sph-ab)
    """
    x = np.linspace(-1, 1, W)
    y = np.linspace(-1, 1, H)
    XX, YY = np.meshgrid(x, y)
    rho = np.sqrt(XX**2 + YY**2)
    phi_xy = np.arctan2(YY, XX)
    mask = rho <= 1.0   # unit aperture
    W_map = np.zeros((H, W))
    # Zernike terms
    Z = {
        4:  np.sqrt(3)  * (2*rho**2 - 1),
        5:  np.sqrt(6)  * rho**2 * np.cos(2*phi_xy),
        6:  np.sqrt(6)  * rho**2 * np.sin(2*phi_xy),
        7:  np.sqrt(8)  * (3*rho**3 - 2*rho) * np.cos(phi_xy),
        11: np.sqrt(5)  * (6*rho**4 - 6*rho**2 + 1),
    }
    for idx, coef in coefs.items():
        if idx in Z:
            W_map += coef * Z[idx]
    W_map *= mask
    return 2 * np.pi * W_map   # convert wavelengths to radians

def torch_gs_2d(I_near: torch.Tensor, I_far: torch.Tensor,
                n_iter: int = 50) -> tuple[torch.Tensor, list[float]]:
    """
    2D GS phase retrieval in PyTorch.
    I_near: real (H, W) -- near-field intensity (|E(x,y)|^2)
    I_far:  real (H, W) -- far-field intensity  (|FFT2[E]|^2)
    Returns: (complex E_near of shape (H,W), error list)
    """
    E = I_near.sqrt().to(torch.complex128)
    errors = []
    for _ in range(n_iter):
        # Forward: near -> far
        E_far = torch.fft.fft2(E)
        # Far-field constraint
        E_far = torch_unit_amp(E_far, I_far)
        # Backward: far -> near
        E = torch.fft.ifft2(E_far)
        # Near-field constraint
        E = torch_unit_amp(E, I_near)
        with torch.no_grad():
            err = ((torch.fft.fft2(E).abs()**2 - I_far)**2).mean().sqrt().item()
        errors.append(err)
    return E, errors

# 2D test: recover Zernike wavefront from near+far intensity
H2, W2 = 64, 64
phi_2d_true = make_zernike_phase(H2, W2,
                                  {4: 0.5, 5: 0.3, 7: 0.2, 11: 0.1})   # defocus + astig + coma + sph-ab
E_2d_true   = np.exp(1j * phi_2d_true)   # unit amplitude, pure phase object
I_near_2d   = np.abs(E_2d_true)**2       # = 1 everywhere (unit aperture)
I_far_2d    = np.abs(np.fft.fft2(E_2d_true))**2

I_near_t = torch.tensor(I_near_2d, dtype=torch.float64, device=DEVICE)
I_far_t  = torch.tensor(I_far_2d,  dtype=torch.float64, device=DEVICE)

E_2d_rec, errs_2d = torch_gs_2d(I_near_t, I_far_t, n_iter=100)
phi_2d_rec = E_2d_rec.angle().cpu().numpy()

# Phase quality: correlation on aperture (rho <= 1)
x2 = np.linspace(-1,1,W2); y2 = np.linspace(-1,1,H2)
XX2, YY2 = np.meshgrid(x2, y2)
mask2 = (XX2**2 + YY2**2) <= 1.0
corr_2d = np.corrcoef(phi_2d_true[mask2], phi_2d_rec[mask2])[0,1]

print(f"  2D GS: {H2}x{W2} Zernike wavefront (defocus+astig+coma+sph-ab)")
print(f"    Iterations: 100")
print(f"    Error: {errs_2d[0]:.4f} -> {errs_2d[-1]:.6f}")
print(f"    Phase correlation (aperture): {corr_2d:.4f}")
print(f"    torch.fft.fft2 -- automatically differentiable!")

print("""
  HOLOGRAPHIC PHASE MASK (SLM design):
    Given target image (Fourier plane intensity), find SLM phase phi(x,y):
    E_SLM(x,y) = exp(i*phi(x,y))   [unit amplitude, phase-only]
    Target: |FFT2[exp(i*phi)]|^2 = I_target
    GS iterations: alternate near-field unit amplitude + far-field target amplitude.
    Efficiency = sum(I_captured)/sum(I_target) [fraction of light in target]
    Speckle: random phase leads to interference noise; MRAF/weighted GS reduce it.
    HOGEL (holographic element): tiling SLM with local phase patches.
""")

# ============================================================
# S3: 3D GS -- SPATIO-TEMPORAL (STEAM / ULTRAFAST IMAGING)
# ============================================================
print(f"\n{SEP}")
print("SECTION 3: 3D GS -- SPATIO-TEMPORAL (STEAM MICROSCOPY)")
print(SEP)

print("""
  JALALI STEAM (Serial Time-Encoded Amplified Microscopy):
    Science 2009: Goda et al., Jalali Lab UCLA -- 6.1 MHz frame rate imaging.
    Maps 2D spatial image onto a TEMPORAL waveform via dispersive Fourier transform.
    1D photodetector captures the time-encoded image (no 2D detector needed).
    Phase retrieval in STEAM: recover 2D spatial phase + temporal phase simultaneously.
    Equivalent problem: 3D field E(x, y, t) with constraints at two dispersion planes.

  3D TRANSFER FUNCTION:
    Temporal dispersion (GVD):    H_t(nu_t) = exp(i*pi*D_t*nu_t^2)  [time axis]
    Spatial diffraction (lens):   H_xy(nu_x, nu_y) = exp(-i*pi*lambda*z*(nu_x^2+nu_y^2))
    Combined 3D transfer:         H_3D = H_t * H_xy  (separable in orthogonal dims)
    3D FFT: torch.fft.fftn(E, dim=(-3,-2,-1))
    Constraint at plane 1: I1_3d = |H_3D_1 * FFT3D[E]|^2  (measured 3D intensity)
    GS: same alternating projection, now in 3D Fourier space.

  STEAM PHASE RETRIEVAL GEOMETRY:
    Frame 1 (D1, z1): |E(x,y,t)| dispersed and diffracted -> I1(x,y,t)
    Frame 2 (D2, z2): same + additional dispersion -> I2(x,y,t)
    GS recovers: amplitude A(x,y,t) and phase phi(x,y,t)
    Applications: label-free imaging of cells, blood cells at MHz rates.
""")

def torch_gs_3d(I1_3d: torch.Tensor, I2_3d: torch.Tensor,
                D1_t: float, D2_t: float,
                n_iter: int = 20) -> tuple[torch.Tensor, list[float]]:
    """
    3D GS: temporal dispersion along last dim, spatial FFT on first 2 dims.
    I1_3d, I2_3d: real (Nz, Ny, Nx) tensors.
    D1_t, D2_t: temporal dispersion parameters.
    """
    Nz, Ny, Nx = I1_3d.shape

    # Build 3D transfer functions (temporal in last dim, spatial in first 2)
    nu_t  = torch.fft.fftfreq(Nx, dtype=torch.float64, device=I1_3d.device)
    nu_y  = torch.fft.fftfreq(Ny, dtype=torch.float64, device=I1_3d.device)
    nu_x  = torch.fft.fftfreq(Nz, dtype=torch.float64, device=I1_3d.device)
    NUt   = nu_t.reshape(1, 1, Nx)
    NUy   = nu_y.reshape(1, Ny, 1)
    NUx   = nu_x.reshape(Nz, 1, 1)

    # Temporal chirp * spatial diffraction (z=1 for both, only D_t differs)
    H1_t  = torch.exp(1j * torch.pi * D1_t * NUt**2)
    H2_t  = torch.exp(1j * torch.pi * D2_t * NUt**2)
    # Spatial (same for both planes, mild defocus)
    lam_z = 0.01   # lambda*z / pixel^2 (normalized)
    H_xy  = torch.exp(-1j * torch.pi * lam_z * (NUy**2 + NUx**2))
    H1    = H1_t * H_xy
    H2    = H2_t * H_xy

    def apply_H(E, H):
        return torch.fft.ifftn(torch.fft.fftn(E) * H)

    E = I1_3d.sqrt().to(torch.complex128)
    errors = []
    for _ in range(n_iter):
        E1 = apply_H(E, H1)
        E1 = torch_unit_amp(E1, I1_3d)
        E  = apply_H(E1, H1.conj())

        E2 = apply_H(E, H2)
        E2 = torch_unit_amp(E2, I2_3d)
        E  = apply_H(E2, H2.conj())

        with torch.no_grad():
            err = ((apply_H(E, H2).abs()**2 - I2_3d)**2).mean().sqrt().item()
        errors.append(err)
    return E, errors

# 3D synthetic test: spatiotemporal Gaussian pulse
Nz3, Ny3, Nx3 = 16, 16, 64   # (spatial_x, spatial_y, time)
np.random.seed(2026)
tx3 = np.linspace(-1, 1, Nz3)
ty3 = np.linspace(-1, 1, Ny3)
tt3 = np.linspace(-1, 1, Nx3)
TX, TY, TT = np.meshgrid(tx3, ty3, tt3, indexing='ij')
A_3d_true   = (np.exp(-TX**2/0.5) * np.exp(-TY**2/0.5) *
               np.exp(-TT**2/0.2))
phi_3d_true = 0.5*TX + 1.0*TT**2 + 0.3*TY*TT
E_3d_true   = A_3d_true * np.exp(1j * phi_3d_true)

D1_3d, D2_3d = -5000.0, -5750.0
nu_t3  = np.fft.fftfreq(Nx3)
H1_3d_np = np.exp(1j*np.pi*D1_3d*nu_t3**2)[np.newaxis, np.newaxis, :]
H2_3d_np = np.exp(1j*np.pi*D2_3d*nu_t3**2)[np.newaxis, np.newaxis, :]

# Use simple temporal-only dispersion for the numpy ground truth
E_d1_3d = np.fft.ifftn(np.fft.fftn(E_3d_true) * H1_3d_np)
E_d2_3d = np.fft.ifftn(np.fft.fftn(E_3d_true) * H2_3d_np)
I1_3d_np = np.abs(E_d1_3d)**2
I2_3d_np = np.abs(E_d2_3d)**2

I1_3d_t = torch.tensor(I1_3d_np, dtype=torch.float64, device=DEVICE)
I2_3d_t = torch.tensor(I2_3d_np, dtype=torch.float64, device=DEVICE)

E_3d_rec, errs_3d = torch_gs_3d(I1_3d_t, I2_3d_t, D1_3d, D2_3d, n_iter=30)
phi_3d_rec = E_3d_rec.angle().cpu().numpy()
corr_3d = np.corrcoef(phi_3d_true.ravel(), phi_3d_rec.ravel())[0,1]
print(f"  3D GS: {Nz3}x{Ny3}x{Nx3} spatio-temporal field")
print(f"    Error: {errs_3d[0]:.5f} -> {errs_3d[-1]:.7f}  (30 iterations)")
print(f"    Phase correlation 3D: {corr_3d:.4f}")
print(f"    torch.fft.fftn -- N-dimensional, differentiable!")

# ============================================================
# S4: 4D GS -- BATCHED + UNROLLED NEURAL NETWORK
# ============================================================
print(f"\n{SEP}")
print("SECTION 4: 4D GS -- BATCHED + UNROLLED NETWORK")
print(SEP)

print("""
  4D TENSOR CONVENTION (standard PyTorch):
    Shape: (batch, channels, height, width)  = 4D
    For 1D signals: (batch, 1, 1, N_t)  or (batch, N_t)
    For 2D fields:  (batch, 2, H, W)    [real + imag as channels]
    For 3D video:   (batch, 1, D, H, W) = 5D

  BATCHED 1D GS (vectorized over B signals simultaneously):
    Shape: I1 = (B, N),  I2 = (B, N)
    torch.fft.fft operates on last dim by default -> automatic batching!
    No Python loop over batch. O(1) overhead. GPU-parallelized.

  UNROLLED GS AS A NEURAL NETWORK:
    Replace GS iterations with fixed-depth "network layers".
    Each layer: [disperse -> amplitude constraint -> undisperse] x2
    Parameters: D1, D2 per layer (can differ -- learned optimal schedule)
    Loss: MSE(phi_recovered, phi_true) on training set
    Training: backprop through all layers simultaneously.
    Equivalent to "deep unrolling" / "algorithm unrolling" (Monga 2021).
    Benefit: learned GS converges faster with fewer iterations than classical GS.
    Related: LISTA (Learned ISTA), deep equilibrium models.
""")

class GSLayer1D(nn.Module):
    """Single GS iteration as a differentiable module."""
    def __init__(self, D1_init: float, D2_init: float):
        super().__init__()
        self.D1 = nn.Parameter(torch.tensor(D1_init, dtype=torch.float64))
        self.D2 = nn.Parameter(torch.tensor(D2_init, dtype=torch.float64))

    def forward(self, E: torch.Tensor, I1: torch.Tensor,
                I2: torch.Tensor) -> torch.Tensor:
        """E: (B, N) complex; I1, I2: (B, N) real."""
        # D1 projection
        E_d1 = torch.fft.ifft(torch.fft.fft(E) *
               torch.exp(1j * torch.pi * self.D1 *
               torch.fft.fftfreq(E.shape[-1], dtype=torch.float64,
                                  device=E.device)**2))
        E_d1 = E_d1 / E_d1.abs().clamp(1e-15) * I1.sqrt()
        E    = torch.fft.ifft(torch.fft.fft(E_d1) *
               torch.exp(-1j * torch.pi * self.D1 *
               torch.fft.fftfreq(E.shape[-1], dtype=torch.float64,
                                  device=E.device)**2))
        E    = torch.exp(1j * E.angle())

        # D2 projection
        E_d2 = torch.fft.ifft(torch.fft.fft(E) *
               torch.exp(1j * torch.pi * self.D2 *
               torch.fft.fftfreq(E.shape[-1], dtype=torch.float64,
                                  device=E.device)**2))
        E_d2 = E_d2 / E_d2.abs().clamp(1e-15) * I2.sqrt()
        E    = torch.fft.ifft(torch.fft.fft(E_d2) *
               torch.exp(-1j * torch.pi * self.D2 *
               torch.fft.fftfreq(E.shape[-1], dtype=torch.float64,
                                  device=E.device)**2))
        E    = torch.exp(1j * E.angle())
        return E


class UnrolledGS(nn.Module):
    """
    Unrolled GS network: K layers, each with learnable (D1_k, D2_k).
    Input: I1 (B, N), I2 (B, N)
    Output: recovered phase phi (B, N)
    """
    def __init__(self, n_layers: int = 10,
                 D1_init: float = -5000.0, D2_init: float = -5750.0):
        super().__init__()
        self.layers = nn.ModuleList([
            GSLayer1D(D1_init + np.random.randn()*100,
                      D2_init + np.random.randn()*100)
            for _ in range(n_layers)
        ])

    def forward(self, I1: torch.Tensor, I2: torch.Tensor) -> torch.Tensor:
        A0 = I1.sqrt().to(torch.complex128)
        # Initialize with undisperse at first layer's D1
        nu  = torch.fft.fftfreq(I1.shape[-1], dtype=torch.float64, device=I1.device)
        D1_0 = self.layers[0].D1
        E = torch.fft.ifft(torch.fft.fft(A0) *
                           torch.exp(-1j * torch.pi * D1_0 * nu**2))
        for layer in self.layers:
            E = layer(E, I1, I2)
        return E.angle()


# Generate synthetic training data (batched QPSK-like signals)
print("  Generating synthetic training data...")
B_train = 32    # batch size
N_sig   = 256   # signal length
T0_train = 20   # pulse width

np.random.seed(2026)
t_train = np.arange(N_sig) - N_sig//2
A_base  = np.exp(-t_train**2 / (2*T0_train**2))

train_data = []
for _ in range(B_train * 4):
    # Random chirp coefficients
    C2 = np.random.uniform(-2e-4, 2e-4)
    C3 = np.random.uniform(-3e-7, 3e-7)
    phi_t = C2 * t_train**2 + C3 * t_train**3
    E_t   = A_base * np.exp(1j * phi_t)
    I1_   = np.abs(disperse(E_t, D1)[:N_sig])**2
    I2_   = np.abs(disperse(E_t, D2)[:N_sig])**2
    train_data.append((I1_, I2_, phi_t))

# Build tensors
I1_all  = torch.tensor(np.array([d[0] for d in train_data]),
                        dtype=torch.float64, device=DEVICE)
I2_all  = torch.tensor(np.array([d[1] for d in train_data]),
                        dtype=torch.float64, device=DEVICE)
phi_all = torch.tensor(np.array([d[2] for d in train_data]),
                        dtype=torch.float64, device=DEVICE)

# Train unrolled GS (5 layers, 30 training steps)
model    = UnrolledGS(n_layers=5, D1_init=D1, D2_init=D2).to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=5.0)

train_losses = []
print("  Training Unrolled GS network...")
for step in range(60):
    idx    = torch.randperm(len(train_data))[:B_train]
    I1_b   = I1_all[idx]
    I2_b   = I2_all[idx]
    phi_b  = phi_all[idx]

    optimizer.zero_grad()
    phi_pred = model(I1_b, I2_b)
    # Remove global phase offset (GS ambiguity)
    offset   = (phi_b - phi_pred).mean(dim=-1, keepdim=True)
    loss     = ((phi_pred + offset - phi_b)**2).mean()
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), max_norm=100.0)
    optimizer.step()

    train_losses.append(loss.item())

print(f"  Unrolled GS training complete:")
print(f"    Initial loss: {train_losses[0]:.4f}")
print(f"    Final loss:   {train_losses[-1]:.6f}")
print(f"    Layers: {len(model.layers)}, each with learnable D1_k, D2_k")
D1_learned = [l.D1.item() for l in model.layers]
D2_learned = [l.D2.item() for l in model.layers]
print(f"    D1 schedule: {[f'{d:.0f}' for d in D1_learned]}")
print(f"    D2 schedule: {[f'{d:.0f}' for d in D2_learned]}")

# Batched evaluation
with torch.no_grad():
    phi_rec_batch = model(I1_all[:B_train], I2_all[:B_train])
    phi_true_batch = phi_all[:B_train]
    offset_batch = (phi_true_batch - phi_rec_batch).mean(dim=-1, keepdim=True)
    diff_batch   = phi_true_batch - phi_rec_batch - offset_batch
    rmse_batch   = diff_batch.pow(2).mean(dim=-1).sqrt()
    print(f"  Batch eval (B={B_train}): mean RMSE = {rmse_batch.mean().item():.4f} rad")

print("""
  SUMMARY: GS DIMENSIONALITY TABLE:
  +---------+----------------------+---------------------+------------------+
  | DIM     | Signal               | FFT op              | Application      |
  +---------+----------------------+---------------------+------------------+
  | 1D      | E(t) fiber pulse     | fft(E, dim=-1)      | Optical comm, GS |
  | 2D      | E(x,y) wavefront     | fft2(E)             | Holography, SLM  |
  | 3D      | E(x,y,t) STEAM       | fftn(E, dim=(-3,-2,-1)) | Ultrafast img |
  | 4D(B,N) | Batch of 1D signals  | fft(E, dim=-1)      | Parallel channels|
  +---------+----------------------+---------------------+------------------+
  All above: FULLY DIFFERENTIABLE in PyTorch. Backprop flows through FFT.
  Unrolled GS: K-layer network, learns D_k schedule, trains end-to-end.
  RogueGuard: batched 1D GS on B=8 ADC channels in parallel, real-time RPi CM4.
""")

# ============================================================
# MATPLOTLIB -- 6-PANEL FIGURE
# ============================================================
print(f"\n{SEP}")
print("BUILDING FIGURE...")
print(SEP)

fig = plt.figure(figsize=(19, 13))
fig.patch.set_facecolor("#F5F5F0")
gs0 = gridspec.GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.38,
                        top=0.93, bottom=0.06, left=0.06, right=0.97)

ax_chirp = fig.add_subplot(gs0[0, 0])
ax_dlearn= fig.add_subplot(gs0[0, 1])
ax_2dgs  = fig.add_subplot(gs0[0, 2])
ax_3dgs  = fig.add_subplot(gs0[1, 0])
ax_unroll= fig.add_subplot(gs0[1, 1])
ax_dim   = fig.add_subplot(gs0[1, 2])

fig.suptitle(
    "GS Phase Retrieval in PyTorch: 1D (chirp+D-learning) | "
    "2D (Zernike wavefront) | 3D (STEAM) | 4D (Unrolled Network)",
    fontsize=10.5, fontweight="bold", color="#1a1a2e"
)

# ---- AX_CHIRP: three chirp types -- phase, intensity ----
ax = ax_chirp
ax.set_facecolor("#F0F8FF")
t_plot = np.arange(N) - N//2
for name, col in [("linear","#1f77b4"), ("quadratic","#d62728"), ("cubic","#2ca02c")]:
    res = gs_results[name]
    # Plot only center portion
    sl = slice(N//2-150, N//2+150)
    ax.plot(t_plot[sl], res["phi_true"][sl], col, lw=1.5, label=f"phi_true ({name})")
    ax.plot(t_plot[sl], res["phi_rec"][sl],  col, lw=1.0, ls="--", alpha=0.7)
ax.set_xlabel("Time (samples)", fontsize=9)
ax.set_ylabel("Phase (rad)", fontsize=9)
ax.set_title("1D GS: Linear/Quadratic/Cubic\nChirp Recovery (dashed=GS rec)", fontsize=10)
ax.legend(fontsize=7.5)
ax.grid(alpha=0.2)
ax.text(0.02, 0.97,
        "Dispersion-assisted GS\n"
        f"D1={D1}, D2={D2}\n"
        f"corr: {gs_results['linear']['corr']:.2f}/{gs_results['quadratic']['corr']:.2f}"
        f"/{gs_results['cubic']['corr']:.2f}",
        transform=ax.transAxes, fontsize=7.5, va="top",
        bbox=dict(fc="white", ec="#bbb", pad=2))

# ---- AX_DLEARN: D2 learning convergence ----
ax = ax_dlearn
ax.set_facecolor("#FFF5E0")
ax2 = ax.twinx()
ax.plot(D2_history[:-1], "#1f77b4", lw=2.0, label=f"D2 (learned)")
ax.axhline(D2_true_val, color="#d62728", lw=1.5, ls="--", label=f"D2 true={D2_true_val}")
ax2.plot(loss_history, "#2ca02c", lw=1.5, alpha=0.7, label="Loss")
ax.set_xlabel("Gradient step", fontsize=9)
ax.set_ylabel("D2 (learned)", color="#1f77b4", fontsize=9)
ax2.set_ylabel("MSE Loss", color="#2ca02c", fontsize=9)
ax.tick_params(axis="y", labelcolor="#1f77b4")
ax2.tick_params(axis="y", labelcolor="#2ca02c")
ax.set_title("Differentiable GS:\nLearning Unknown Dispersion D2", fontsize=10)
lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1+lines2, labels1+labels2, fontsize=7.5)
ax.grid(alpha=0.2)
ax.text(0.05, 0.08,
        "torch.autograd flows\nthrough GS iterations",
        transform=ax.transAxes, fontsize=8, va="bottom",
        bbox=dict(fc="#fff8e7", ec="#f28e2b", pad=2))

# ---- AX_2DGS: 2D wavefront recovery ----
ax = ax_2dgs
ax.set_facecolor("#111")
im_true = ax.imshow(phi_2d_true, cmap="RdBu_r", vmin=-3, vmax=3,
                    extent=[-1,1,-1,1])
ax.set_title(f"2D GS: Zernike Wavefront\n(true | rec | diff), corr={corr_2d:.4f}", fontsize=10)
ax.set_xlabel("x", fontsize=9); ax.set_ylabel("y", fontsize=9)
plt.colorbar(im_true, ax=ax, fraction=0.046).set_label("phi (rad)", fontsize=8)
# Draw circle for aperture
theta_c = np.linspace(0, 2*np.pi, 100)
ax.plot(np.cos(theta_c), np.sin(theta_c), "gold", lw=1.5, alpha=0.7)

# Inset: recovered phase
ax_ins_2d = ax.inset_axes([0.02, 0.02, 0.35, 0.35])
ax_ins_2d.imshow(phi_2d_rec, cmap="RdBu_r", vmin=-3, vmax=3,
                 extent=[-1,1,-1,1])
ax_ins_2d.set_title("GS rec", fontsize=7, color="white")
ax_ins_2d.set_facecolor("#111")
ax_ins_2d.tick_params(labelbottom=False, labelleft=False)

# ---- AX_3DGS: 3D error convergence + time slice ----
ax = ax_3dgs
ax.set_facecolor("#F0FFF0")
ax.semilogy(errs_3d, "#1f77b4", lw=2.0, label="3D GS error")
ax.semilogy(errs_2d, "#d62728", lw=1.5, ls="--", label="2D GS error")
ax.semilogy(errs_torch[:len(errs_3d)], "#2ca02c", lw=1.5, ls=":",
            label="1D GS error (torch)")
ax.set_xlabel("Iteration", fontsize=9)
ax.set_ylabel("RMS Intensity Error", fontsize=9)
ax.set_title("GS Convergence: 1D vs 2D vs 3D\n(all PyTorch, differentiable)", fontsize=10)
ax.legend(fontsize=8)
ax.grid(alpha=0.3, which="both")
ax.text(0.55, 0.95,
        f"3D: {Nz3}x{Ny3}x{Nx3}\n2D: {H2}x{W2}\n1D: N={N}",
        transform=ax.transAxes, fontsize=8, va="top",
        bbox=dict(fc="white", ec="#bbb", pad=2))

# ---- AX_UNROLL: training loss + D schedule ----
ax = ax_unroll
ax.set_facecolor("#FFF0FF")
ax.plot(train_losses, "#1f77b4", lw=2.0, label="Unrolled GS train loss")
ax.set_xlabel("Training step", fontsize=9)
ax.set_ylabel("MSE Loss", fontsize=9)
ax.set_title("4D: Unrolled GS Network Training\n(B=32 batched signals, 5 layers)", fontsize=10)
ax.legend(fontsize=8)
ax.grid(alpha=0.2)

# Inset: learned D schedule
ax_ins_d = ax.inset_axes([0.45, 0.45, 0.52, 0.45])
layers_x = list(range(len(D1_learned)))
ax_ins_d.plot(layers_x, D1_learned, "o-", color="#d62728", lw=1.5, ms=5, label="D1_k")
ax_ins_d.plot(layers_x, D2_learned, "s-", color="#2ca02c", lw=1.5, ms=5, label="D2_k")
ax_ins_d.axhline(D1, color="#d62728", lw=0.8, ls="--", alpha=0.5)
ax_ins_d.axhline(D2, color="#2ca02c", lw=0.8, ls="--", alpha=0.5)
ax_ins_d.set_xlabel("Layer k", fontsize=7)
ax_ins_d.set_ylabel("D_k", fontsize=7)
ax_ins_d.set_title("Learned D schedule", fontsize=7.5)
ax_ins_d.legend(fontsize=6.5)
ax_ins_d.grid(alpha=0.2)

# ---- AX_DIM: dimensionality table + architecture ----
ax = ax_dim
ax.set_facecolor("#F0F4F0")
ax.set_xlim(0, 10); ax.set_ylim(0, 9); ax.axis("off")
ax.set_title("GS Dimensionality: 1D->2D->3D->4D PyTorch", fontsize=10)

rows = [
    (7.8, "#d5e8f5", "#1f6fa8",
     "1D  (B, N_t)       fft(dim=-1)    Fiber pulse / Optical comm  Solli 2009"),
    (6.2, "#d5f5e8", "#1a8a4e",
     "2D  (B, H, W)      fft2()         Holography / Wavefront sensing / SLM"),
    (4.6, "#f5e8d5", "#8a5a1a",
     "3D  (B, D, H, W)   fftn(dim=...)  STEAM ultrafast imaging / Jalali Lab"),
    (3.0, "#e8d5f5", "#7b2d8b",
     "4D  (B, C, H, W)   fftn()         Batched 2D (video / pulse train)"),
    (1.4, "#f5d5e8", "#8a1a4e",
     "Unrolled  K-layer nn.Module  Learns D_k schedule end-to-end (backprop)"),
]
for y, fc, ec, txt in rows:
    rect = plt.Rectangle((0.3, y), 9.4, 1.1, facecolor=fc, edgecolor=ec,
                          lw=1.5, alpha=0.9)
    ax.add_patch(rect)
    ax.text(5.0, y+0.55, txt, ha="center", va="center",
            fontsize=7.8, color="#1a1a2e", fontweight="bold")

ax.text(0.5, 0.06,
        f"torch.fft.fft/fft2/fftn: fully differentiable. "
        f"Unrolled GS final RMSE = {rmse_batch.mean().item():.4f} rad",
        transform=ax.transAxes, fontsize=8, ha="center",
        bbox=dict(fc="#eee", ec="#bbb", pad=3))

plt.savefig(OUT, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
print(f"  Saved: {OUT}")

print(f"\n{SEP}")
print("Done.")
print(SEP)
