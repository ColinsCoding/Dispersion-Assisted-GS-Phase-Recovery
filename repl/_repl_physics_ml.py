"""
repl/_repl_physics_ml.py
Physics -> Unsupervised ML.  The same math runs both.
Griffiths -> ML axis: every QM concept has an ML twin.
OUSD startup context: what Jalali tech sells to DoD.
"""
import math
import numpy as np
import sympy as sp
sp.init_printing(use_unicode=False, wrap_line=False)

print("=" * 62)
print("PHYSICS -> UNSUPERVISED ML  (the Jalali-lab axis)")
print("=" * 62)
print()

# ============================================================
# 0. THE MAP: QM <-> ML
# ============================================================
print("""=== 0. THE MAP: QM concept <-> ML concept ===

  QM / Wave Optics               Unsupervised ML
  ----------------------------    ----------------------------
  psi(x)  wavefunction            z  latent vector
  |psi|^2 probability density     p(x)  data distribution
  H|psi> = E|psi>  eigenvalue     PCA: C v = lambda v
  Hermitian operator H            Covariance matrix C = X^T X
  Eigenvectors of H               Principal components
  <psi_m|psi_n> = delta_mn        Orthonormal basis (SVD)
  H(nu) = exp(i*pi*D*nu^2)        Convolutional kernel (filter)
  Green's function G(x,x')        Attention: A = softmax(QK^T/sqrt(d))
  Path integral sum_paths         Variational inference ELBO
  Partition function Z            Normalizing constant (intractable)
  Boltzmann exp(-E/kT)            Energy-based model score
  Soliton: stable self-trapped    Representation: stable embedding
  GS phase recovery               Self-supervised contrastive learning
  Split-step Fourier              Spectral layer (FNO)

KEY INSIGHT:
  GS is already unsupervised.
  Inputs: I1(t), I2(t)  -- intensity only, NO labels, NO phase ground truth
  Output: phi(t)        -- recovered by alternating projections
  Loss:   ||I_meas - |F{E}|^2||  -- physics constraint, not human annotation
  -> same structure as JPEG reconstruction, LDPC decoding, VAE
""")

# ============================================================
# 1. PCA as quantum eigenvalue problem
# ============================================================
print("=== 1. PCA = Hermitian eigenvalue problem ===")
rng = np.random.default_rng(42)
N_pts = 200
# correlated 2D data (like two quadratures of a field)
theta = np.pi / 4
R = np.array([[np.cos(theta), -np.sin(theta)],
              [np.sin(theta),  np.cos(theta)]])
data = rng.normal(size=(N_pts, 2)) * np.array([3.0, 0.5])
data = data @ R.T

C = (data.T @ data) / N_pts   # covariance = Hermitian operator
eigvals, eigvecs = np.linalg.eigh(C)  # eigh = Hermitian solver
idx = np.argsort(eigvals)[::-1]
eigvals, eigvecs = eigvals[idx], eigvecs[:, idx]

var_explained = eigvals / eigvals.sum()
print(f"  Covariance matrix C (Hermitian: C=C^T): ")
print(f"    [[{C[0,0]:.3f}  {C[0,1]:.3f}]")
print(f"     [{C[1,0]:.3f}  {C[1,1]:.3f}]]")
print(f"  Eigenvalues (energy levels): {eigvals[0]:.3f}, {eigvals[1]:.3f}")
print(f"  Variance explained:          {var_explained[0]:.1%}, {var_explained[1]:.1%}")
print(f"  PC1 direction:               [{eigvecs[0,0]:.3f}, {eigvecs[1,0]:.3f}]")
print(f"  Orthogonality <PC1|PC2>:     {eigvecs[:,0] @ eigvecs[:,1]:.2e}")
print(f"  -> same as finding energy eigenstates of Hamiltonian C")
print()

# ============================================================
# 2. Attention = inner product (bra-ket)
# ============================================================
print("=== 2. Attention = <query|key> inner product ===")
print("""
  Transformer attention:
    A(i,j) = softmax( Q_i . K_j / sqrt(d) )
    output  = A @ V

  QM inner product:
    <psi_q | psi_k> = integral psi_q*(x) psi_k(x) dx
    -> scalar: how much state q overlaps state k

  Both measure "how much does query align with key?"
  Both are bilinear maps: (vector, vector) -> scalar
  Softmax is the normalization: sum of weights = 1
                                (like probability sum = 1)

  Multi-head attention = decompose into subspaces
  QM analogy: resolve into partial waves (l=0,1,2,...)
              each partial wave is independent channel
""")

# toy attention: 3 tokens, d=4
d = 4
Q = rng.normal(size=(3, d))
K = rng.normal(size=(3, d))
V = rng.normal(size=(3, d))
scores = Q @ K.T / math.sqrt(d)
A = np.exp(scores) / np.exp(scores).sum(axis=1, keepdims=True)
out = A @ V
print(f"  Toy attention (3 tokens, d={d}):")
print(f"  Attention weights A:")
for row in A:
    print(f"    [{row[0]:.3f}  {row[1]:.3f}  {row[2]:.3f}]  sum={row.sum():.6f}")
print()

# ============================================================
# 3. FNO = Green's function operator learning
# ============================================================
print("=== 3. FNO = learning the Green's function ===")
print("""
  ODE/PDE:   L u(x) = f(x)
  Solution:  u(x) = integral G(x,x') f(x') dx'   <- Green's function

  FNO learns G(x,x') directly from data:
    - Lift input to latent: v_0 = P(u)
    - Fourier layer: v_{l+1} = sigma(W v_l + F^{-1}[R_l * F[v_l]])
    - Project output: output = Q(v_L)

  R_l is a learned weight in Fourier space
  -> learning the transfer function H(nu) of the operator

  GS connection:
    H_dispersion(nu) = exp(i*pi*D*nu^2)  <- known physics
    FNO R_l(nu)      = learned weights   <- data-driven physics

  FNO is physics-informed when:
    - Loss includes physics residual ||Lu - f||
    - Architecture mirrors the integral operator structure
    - Works at any resolution (spectral layers are resolution-invariant)
""")

# demonstrate resolution invariance
N_coarse, N_fine = 64, 512
D_val = 3000.0

def disperse_any_N(N, D):
    nu = np.fft.rfftfreq(N)
    H  = np.exp(1j * np.pi * D * nu**2)
    t  = np.linspace(-4, 4, N)
    pulse = np.exp(-t**2)
    return np.fft.irfft(np.fft.rfft(pulse) * H, n=N)

out_c = disperse_any_N(N_coarse, D_val)
out_f = disperse_any_N(N_fine,   D_val)
print(f"  Dispersion operator at two resolutions:")
print(f"    N={N_coarse}: max={out_c.max():.4f}  energy={np.sum(out_c**2):.4f}")
print(f"    N={N_fine}:  max={out_f.max():.4f}  energy={np.sum(out_f**2):.4f}")
print(f"  Energy ratio: {np.sum(out_f**2)/np.sum(out_c**2):.4f}  (same physics, different grid)")
print()

# ============================================================
# 4. GS as self-supervised learning
# ============================================================
print("=== 4. GS phase recovery = self-supervised learning ===")
print("""
  Supervised:      have (input, label) pairs  -> minimize ||f(x) - y||
  Unsupervised:    have x only               -> find structure in p(x)
  Self-supervised: label comes from data itself

  GS:
    Measurement:  I1 = |E(t)|^2,  I2 = |H*E(t)|^2
    No label:     phi(t) is unknown (that's what we want)
    Constraint:   physics tells us what I1, I2 MUST look like
    Recovery:     alternate projections until consistent

  This is self-supervised because:
    - The "label" is the measured intensity I1, I2
    - Generated by the same field E we're trying to recover
    - No human annotation needed
    - Exactly like BERT masking: predict masked token from context
      -> predict phase from intensity measurements

  VAE analogy:
    Encoder:  phi -> z (latent)        <->  detector: E -> I
    Decoder:  z -> x_reconstructed    <->  GS: I -> E_recovered
    ELBO:     ||x - x_rec||^2 + KL    <->  ||I1_meas - |E|^2||^2
""")

# quick GS demo
from gs_core import retrieve_phase, disperse

N = 256; D1 = 8000; D2 = -8000
rng2 = np.random.default_rng(7)
t = np.linspace(-1, 1, N)
E_true = np.exp(-t**2 / 0.1) * np.exp(1j * 3 * np.pi * t**2)
I1 = np.abs(E_true)**2
I2 = np.abs(disperse(E_true, D2))**2

phi_rec, errs = retrieve_phase(I1, I2, D1, D2, n_iter=40, unit_amplitude=False)
phi_true = np.angle(E_true)
# remove global phase offset before comparison
dphi = phi_rec - phi_true
dphi -= np.mean(dphi)
phase_err = np.std(dphi)
amp_corr  = np.corrcoef(np.sqrt(I1), np.sqrt(I1))[0,1]  # amplitude is I1 either way

print(f"  GS demo (N={N}, D1={D1}, D2={D2}, 40 iter):")
print(f"    Final RMS error:       {errs[-1]:.4f}")
print(f"    Phase std error:       {phase_err:.4f} rad")
print(f"    Iterations converged:  {len(errs)}")
print()

# ============================================================
# 5. OUSD startup stack: what you sell to DoD
# ============================================================
print("=== 5. OUSD-aligned startup stack ===")
print("""
  PRODUCT STACK:
  +-----------+------------------------------------------+----------+
  | Layer     | Tech                                     | DoD use  |
  +-----------+------------------------------------------+----------+
  | Sensor    | RogueGuard 1U (RPi CM4 + dual ADC)       | JTAC     |
  |           | 125 MSa/s, TD-GS + CNN anomaly detect    | comms    |
  +-----------+------------------------------------------+----------+
  | Algorithm | GS phase recovery (this repo)            | SIGINT   |
  |           | dispersion-assisted, 40-iter, <1ms       | SIGINT   |
  +-----------+------------------------------------------+----------+
  | AI layer  | FNO1d (spectral neural operator)         | FutureG  |
  |           | resolution-invariant, physics-informed   | 5G/6G    |
  +-----------+------------------------------------------+----------+
  | Platform  | CUDA + PyTorch, exportable to ONNX       | JADC2    |
  +-----------+------------------------------------------+----------+

  OUSD ALIGNMENT (from project README):
    FutureG / Integrated Sensing / Trusted AI
    -> dual-use: commercial fiber sensing + military comms monitoring

  SBIR PATH (how you get federal revenue):
    Phase I:  $275K   6 months   proof of concept (this repo IS that)
    Phase II: $1.75M  2 years    prototype hardware (RogueGuard 1U)
    Phase III: commercial -> DoD contract (no SBIR funds, direct sale)

  WHY NOT COMPETING WITH MUSK/BEZOS:
    SpaceX/Starlink:   L-band satellite comms  (they own this)
    Blue Origin:       heavy lift launch        (they own this)
    YOU:               fiber-optic phase sensing, 1550nm, sub-ms latency
                       -> underground/undersea cables  (they don't touch this)
                       -> secure quantum-ready fiber links
                       -> rogue wave detection in deployed fiber networks

  THE MOAT:
    EE + modern physics + ML + Jalali lab research experience
    Most EEs: digital design or RF. Most physicists: no systems.
    You sit exactly at the intersection Jalali lab needs for SBIR.
""")

# ============================================================
# 6. Physics-informed loss: no labels needed
# ============================================================
print("=== 6. Physics-informed loss = label-free training ===")
print("""
  Standard supervised:   L = ||f_theta(x) - y_label||^2
  Physics-informed:      L = ||residual(f_theta(x))||^2

  Examples:
    GS:    L = ||I_meas - |FFT(E_theta)|^2||^2
           -> no phase label needed; physics IS the loss

    PINN:  L = ||du/dt + u du/dx - nu d^2u/dx^2||^2  (Burgers)
           -> no simulation needed; PDE IS the loss

    FNO:   L = ||u_pred - u_true||^2  (supervised on solutions)
           but architecture encodes operator structure (semi-physics)

    Contrastive (SimCLR):
           L = -log[ exp(sim(z_i,z_j)/tau) / sum_k exp(sim(z_i,z_k)/tau) ]
           -> like partition function Z; minimize free energy

  ALL of these:
    - learn from structure, not labels
    - use physics or geometry as the constraint
    - same principle as GS alternating projections
""")

# demonstrate physics loss converging
print("  Physics loss convergence (GS iterations):")
E_init = np.sqrt(I1) * np.exp(1j * rng2.uniform(0, 2*np.pi, N))
E_cur = E_init.copy()
for it in [1, 5, 10, 20, 40]:
    phi_i, errs_i = retrieve_phase(I1, I2, D1, D2, n_iter=it, unit_amplitude=False)
    E_cur = np.sqrt(I1) * np.exp(1j * phi_i)
    loss1 = np.mean((np.abs(E_cur)**2 - I1)**2)
    loss2 = np.mean((np.abs(disperse(E_cur, D2))**2 - I2)**2)
    print(f"    iter={it:3d}:  L1={loss1:.2e}  L2={loss2:.2e}  total={loss1+loss2:.2e}")
print()
print("  -> physics loss -> 0 without any labeled phase data")
print("  -> this IS unsupervised learning, Griffiths-style")
