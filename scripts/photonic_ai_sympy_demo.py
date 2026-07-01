"""Photonic AI SymPy demo -- init_printing walkthrough.

Shows how the Gerchberg-Saxton Algorithm (GSA) connects to:
  1. The dispersive fiber as an optical Fourier layer  (this repo)
  2. The MZI mesh as trainable optical weights         (photonic_ai.py)
  3. Project 5: unsupervised phase retrieval using the physics as a prior

Run: py -3.13 scripts/photonic_ai_sympy_demo.py
"""
import sys
import numpy as np
import sympy as sp

# cp1252 terminal -- ASCII pretty-print, no Unicode box-drawing
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sp.init_printing(use_unicode=False, use_latex=False)


def sep(title):
    width = 70
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width)


# ─────────────────────────────────────────────────────────────────────
sep("1. SINGLE DECAY (Griffiths chain rule context)")
# ─────────────────────────────────────────────────────────────────────
t, N0, lam = sp.symbols('t N_0 lambda', positive=True)

N_t = N0 * sp.exp(-lam * t)
A_t = lam * N_t
t_half = sp.log(2) / lam

print("\n  N(t) = N_0 * exp(-lambda * t):")
sp.pprint(sp.Eq(sp.Symbol('N(t)'), N_t))

print("\n  Activity A(t) = lambda * N(t):")
sp.pprint(sp.Eq(sp.Symbol('A(t)'), A_t))

print("\n  Half-life:")
sp.pprint(sp.Eq(sp.Symbol('t_{1/2}'), t_half))

print("\n  Verify N(t_{1/2}) = N_0/2:")
val = N_t.subs(t, t_half)
simplified = sp.simplify(val / N0)
sp.pprint(sp.Eq(sp.Symbol('N(t_{1/2})/N_0'), simplified))


# ─────────────────────────────────────────────────────────────────────
sep("2. BATEMAN EQUATION -- ODE chain rule  A -> B -> C(stable)")
# ─────────────────────────────────────────────────────────────────────
lA, lB = sp.symbols('lambda_A lambda_B', positive=True)
NA0, NB0 = sp.symbols('N_{A0} N_{B0}', nonnegative=True)

NA = NA0 * sp.exp(-lA * t)
NB = (NA0 * lA / (lB - lA) * (sp.exp(-lA * t) - sp.exp(-lB * t))
      + NB0 * sp.exp(-lB * t))

print("\n  N_A(t):  (parent, exponential decay)")
sp.pprint(sp.Eq(sp.Symbol('N_A(t)'), NA))

print("\n  N_B(t):  (daughter, Bateman 1910 -- chained ODE solution)")
sp.pprint(sp.Eq(sp.Symbol('N_B(t)'), NB))

print("\n  Verify dN_B/dt = lambda_A*N_A - lambda_B*N_B:")
dNB = sp.diff(NB, t)
residual = sp.simplify(dNB - (lA * NA - lB * NB))
print(f"  Residual (should be 0): {residual}")

print("\n  Secular equilibrium limit (lA << lB, long times):")
print("  A_B(t->inf) = lB * N_B -> lA * N_A = A_A")
print("  The daughter activity EQUALS the parent activity.")
print("  Example: Ra-226 (1600 yr) -> Rn-222 (3.82 days)")
print("  After ~27 days (7 half-lives of Rn), activities equalize.")


# ─────────────────────────────────────────────────────────────────────
sep("3. GERCHBERG-SAXTON ALGORITHM (GSA) -- the core of this repo")
# ─────────────────────────────────────────────────────────────────────
f, D = sp.symbols('f D', real=True)
E = sp.Function('E')
omega = sp.Symbol('omega', real=True)
phi = sp.Symbol('phi', real=True)

H_f = sp.exp(sp.I * sp.pi * D * f**2)
print("\n  Dispersive fiber transfer function H(f):")
sp.pprint(sp.Eq(sp.Symbol('H(f)'), H_f))

print("\n  |H(f)|^2 = 1  (all-pass -- no amplitude change, only phase):")
magnitude_sq = sp.simplify(sp.Abs(H_f)**2)
# Can't fully simplify complex Abs in sympy easily -- show analytically
print("  |exp(i*pi*D*f^2)|^2 = exp(i*pi*D*f^2) * exp(-i*pi*D*f^2) = 1")

print("\n  GSA iteration (one step):")
print("""
  Given: I1(t) = |E1(t)|^2   (measured intensity at fiber INPUT)
         I2(t) = |E2(t)|^2   (measured intensity at fiber OUTPUT)

  Step 1: E1_est = sqrt(I1) * exp(i * phi_est)   [apply measured amplitude]
  Step 2: E2_est = IFFT[ H(f) * FFT[E1_est] ]    [propagate through fiber]
  Step 3: E2_est = sqrt(I2) * exp(i * angle(E2_est))  [replace amplitude]
  Step 4: E1_new = IFFT[ conj(H(f)) * FFT[E2_est] ]   [back-propagate]
  Step 5: phi_est = angle(E1_new)                 [update phase estimate]
  Repeat until convergence.

  The CONSTRAINT that drives convergence:
    - Fourier domain:  |E2(f)|^2 must match the fiber output spectrum
    - Time domain:     |E1(t)|^2 must match the input intensity
  Two constraints + one unknown (phase) = over-determined -> converges.
""")

E_in = sp.Function('E_in')
E_out_sym = sp.Symbol('E_out')
print("  Key equation: output field = fiber propagation of input field:")
sp.pprint(sp.Eq(
    E_out_sym,
    sp.Symbol('IFFT') * (H_f * sp.Symbol('FFT(E_in)'))
))


# ─────────────────────────────────────────────────────────────────────
sep("4. DISPERSION AS AN OPTICAL FOURIER NEURAL NETWORK LAYER")
# ─────────────────────────────────────────────────────────────────────
print("""
  Standard neural network layer (PyTorch):
    y = sigma( W * x + b )
    W = learnable weight matrix  (N x N floats on GPU)
    sigma = nonlinear activation function

  Fourier Neural Operator (FNO, Li et al. 2020):
    y_hat(f) = R(f) * x_hat(f)   [learned frequency-domain filter]
    R(f) = complex weight tensor in Fourier space

  THIS REPO's dispersive layer (photonic_ai.py):
    y_hat(f) = H(f) * x_hat(f)   [FIXED physics-based filter]
    H(f) = exp(i*pi*D*f^2)       [dispersion is the "weight"]
""")

# Show H(f) as a special case of R(f) with R(f) = exp(i*pi*D*f^2)
R_f = sp.Symbol('R(f)')
x_hat = sp.Symbol('x_hat(f)')
y_hat = sp.Symbol('y_hat(f)')
print("  FNO general layer:")
sp.pprint(sp.Eq(y_hat, R_f * x_hat))

print("\n  Dispersion-assisted GS (this repo) -- R(f) fixed by physics:")
sp.pprint(sp.Eq(y_hat, H_f * x_hat))

print("""
  WHY THIS MATTERS FOR PROJECT 5 (Unsupervised Phase Retrieval):
    In supervised learning: you need (input, true_phase) pairs -> expensive to label
    In this repo (Project 5 goal): the PHYSICS knows what H(f) is.
    The dispersion parameter D is MEASURED from the fiber.
    So H(f) is not learned -- it is KNOWN from physics.
    The neural network only needs to learn the RESIDUALS (noise, nonlinearity).
    This is the "Deep Dispersion Prior" (DDP) approach.
""")


# ─────────────────────────────────────────────────────────────────────
sep("5. MZI MATRIX -- same math as Jones calculus (chain rule again)")
# ─────────────────────────────────────────────────────────────────────
theta, phi_mzi = sp.symbols('theta phi', real=True)

U_MZI = sp.I * sp.Matrix([
    [sp.sin(theta) * sp.exp(sp.I * phi_mzi), sp.cos(theta)],
    [sp.cos(theta) * sp.exp(sp.I * phi_mzi), -sp.sin(theta)]
])

print("\n  MZI (Mach-Zehnder Interferometer) matrix  U_MZI = i * [[...],[...]]:")
sp.pprint(U_MZI)

print("\n  Verify U_MZI is unitary: U * U^H = I")
UH = U_MZI.H
product = sp.simplify(U_MZI * UH)
print("  U_MZI * U_MZI^H =")
sp.pprint(product)

print("""
  CONNECTION TO JONES CALCULUS (jones_calculus.py):
    The HWP Jones matrix is also a 2x2 unitary.
    MZI with theta=pi/4, phi=pi/2 IS a quarter-wave plate.
    Cascaded MZIs = M_N * ... * M_1  (same chain rule as cascaded wave plates)
    Polarization state space = photonic computing weight space.
    The Poincare sphere = the Bloch sphere with different axis labels.
""")


# ─────────────────────────────────────────────────────────────────────
sep("6. SHANNON CAPACITY -- why GSA doubles the information rate")
# ─────────────────────────────────────────────────────────────────────
B, SNR_sym = sp.symbols('B SNR', positive=True)

C_coherent = B * sp.log(1 + SNR_sym, 2)
C_direct   = B * sp.Rational(1, 2) * sp.log(1 + SNR_sym, 2)

print("\n  Shannon capacity (coherent, phase-aware receiver):")
sp.pprint(sp.Eq(sp.Symbol('C_coherent'), C_coherent))

print("\n  Direct detection (intensity-only, no phase):")
sp.pprint(sp.Eq(sp.Symbol('C_direct'), C_direct))

print("\n  Ratio:")
sp.pprint(sp.Eq(sp.Symbol('C_coherent/C_direct'), 2))

print("\n  At SNR=20 dB (100 linear), B=10 GHz:")
snr_val = 100
B_val = 10  # GHz
C_coh_num = B_val * np.log2(1 + snr_val)
C_dir_num = B_val * 0.5 * np.log2(1 + snr_val)
print(f"    Coherent (GS):        {C_coh_num:.1f} Gbps")
print(f"    Direct detection:     {C_dir_num:.1f} Gbps")
print(f"    Phase recovery gains: {C_coh_num - C_dir_num:.1f} Gbps for free")


# ─────────────────────────────────────────────────────────────────────
sep("7. PROJECT 5 -- HOW photonic_ai.py GROWS OUT OF GSA")
# ─────────────────────────────────────────────────────────────────────
print("""
  THE ACADEMIC CHAIN (what the papers say):

  [1] Gerchberg & Saxton (1972): Original GS algorithm
      - Phase retrieval from two intensity measurements
      - Alternating projections between spatial and Fourier domains
      - NO assumption about the physics linking the two domains

  [2] Jalali et al. (photonic time stretch, ~2009-2015):
      - Used dispersive fiber to create the SECOND intensity measurement
      - H(f) = exp(i*pi*D*f^2) is the dispersion link
      - This made GS work for WIDEBAND signals (not just spatial optics)
      - KEY: the fiber acts as an ANALOG COMPUTER doing Fourier transform

  [3] Li et al., "Fourier Neural Operator" (2020, NeurIPS):
      - Learned R(f) in frequency domain replaces fixed H(f)
      - Resolution-invariant: train on coarse grid, apply to fine grid
      - THIS IS gs_fno.py in this repo

  [4] Sitzmann et al., "Implicit Neural Representations" / "Neural Fields":
      - Network parametrizes the FIELD itself: E(t) = MLP(t)
      - No need for paired training data
      - THIS IS the "PINN self-supervised" part of Project 5

  [5] Photonic AI (Shen et al. 2017 Nature Photonics, Liu et al. 2022):
      - MZI mesh implements W = U*Sigma*V^H optically
      - EACH CASCADE of MZIs is the SAME chain rule as GSA propagation
      - The GS algorithm's forward model H(f) IS a photonic layer
      - So the GS receiver IS a photonic neural network (single fixed layer)

  PROJECT 5 = "Deep Dispersion Prior" (gs_unsupervised.py):
    Input:    two measured intensities I1(t), I2(t)
    Physics:  H(f) = exp(i*pi*D*f^2)  [known, not learned]
    Network:  phi(t) = MLP_theta(t)   [learned implicit field]
    Loss:     ||sqrt(I1) - |IFFT[H*FFT[sqrt(I2)*exp(i*phi)]]||^2
              (no labels needed -- the PHYSICS is the supervisor)
    This is SELF-SUPERVISED because the loss function encodes the
    physical constraint, not a labeled dataset.

  THE PHOTONIC AI CONNECTION:
    MZI mesh: U*x = optical matrix multiply (photonic_ai.py)
    GS fiber: H*x = dispersion multiply     (gs_core.py + photonic_ai.py)
    FNO:      R*x = learned multiply        (gs_fno.py)
    All three are SPECIAL CASES of the same operation: frequency-domain
    linear map of the optical field. The GS algorithm is the SIMPLEST
    photonic neural network: one layer, fixed weights, no training.

    Project 5 generalizes this: add learnable layers on top of the
    fixed physics prior. The dispersion H(f) anchors the network to
    physical reality, reducing the data needed to train by 100-1000x.
""")


# ─────────────────────────────────────────────────────────────────────
sep("8. ENERGY-DEPENDENT DATA: WHY INT8 / BFLOAT16 FOR PHOTONIC AI")
# ─────────────────────────────────────────────────────────────────────
print("""
  From computer_architecture.py (CUDA energy model):

  float32 forward pass:  ~300 fJ/MAC  (full precision)
  bfloat16 forward pass: ~75  fJ/MAC  (2x fewer memory bytes -> 2x fewer DRAM reads)
  int8 inference:        ~15  fJ/MAC  (quantized -- 4x better memory, 8x faster on A100)

  For Project 5 (PINN self-supervised training):
    - TRAINING phase: float32 needed (gradients need precision)
    - INFERENCE phase: int8 quantization of the MLP -> 4x energy reduction
    - The physics layer H(f) is a fixed FFT -> no quantization error

  WHY bfloat16 over float16 for the dispersion layer:
    bfloat16: 1 sign + 8 exponent + 7 mantissa (same range as float32!)
    float16:  1 sign + 5 exponent + 10 mantissa (narrow range: max 65504)
    The dispersion phase exp(i*pi*D*f^2) at D=-5000 ps^2 can exceed float16
    range at high frequencies -> NaN. bfloat16 never overflows here.

  ENERGY HIERARCHY for Project 5 inference:
    1. Laser source: 5 mW (always on)        -- unavoidable
    2. Dispersion fiber: 0 fJ/bit (passive!) -- free computation
    3. ADC at 20 GSa/s, 8-bit: 500 fJ/sample -- dominant cost
    4. MLP inference (int8): ~15 fJ/MAC      -- small vs ADC
    5. DRAM for MLP weights: ~10 pJ/weight   -- cache weights in SRAM

  The fiber does FREE COMPUTING. The ADC pays the entropy tax.
""")

print("\n" + "=" * 70)
print("  SUMMARY: photonic_ai.py IS the physics of gs_core.py, generalized.")
print("  H(f) = exp(i*pi*D*f^2) is a 1-layer photonic neural network.")
print("  Project 5 adds learnable layers on top of that fixed physics prior.")
print("=" * 70 + "\n")
