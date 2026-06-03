"""Append §8 (deep FT in Griffiths) and §9 (Griffiths→ML axis) to griffiths_prep.ipynb."""
import json

NB = 'notebooks/griffiths_prep.ipynb'
with open(NB, encoding='utf-8') as f:
    nb = json.load(f)

def code(src):
    return {"cell_type":"code","execution_count":None,"metadata":{},
            "outputs":[],"source":src}
def md(src):
    return {"cell_type":"markdown","metadata":{},"source":src}

S8_MD = """\
---
# §8 — Fourier Transform at Engineering Depth

Physics courses: define FT, do one Gaussian integral, done.
Engineering reality: four conventions, DFT bins, windowing, spectral
leakage, transfer functions, Z-transform. See `fourier_engineering_depth.ipynb`
for the full treatment. This section shows only the Griffiths-specific
FT identities you will use in chapters 2, 3, 9, and 11.
"""

S8_CODE = """\
sec('Fourier Transforms — What Griffiths Actually Needs')
from sympy import fourier_transform, inverse_fourier_transform
from sympy import exp, sqrt, pi, Abs, DiracDelta, Heaviside, sign
k_s, x_s, a_s, t_s, omega_s = symbols('k x a t omega', real=True, positive=False)
a_pos = symbols('a', positive=True)

sub('Convention: Griffiths (QM, eq 2.103)')
s('f(x) = (1/√2π) ∫ φ(k) eⁱᵏˣ dk')
s('φ(k) = (1/√2π) ∫ f(x) e⁻ⁱᵏˣ dx')
s('This is the PHYSICS convention — note the 1/√2π symmetric normalization')
s('Engineering convention uses 1/(2π) asymmetrically — watch for factor of 2π errors')

sub('Key FT pairs for QM')
pairs = [
    ('Gaussian (most important!)',
     exp(-a_pos*x_s**2),
     sqrt(pi/a_pos)*exp(-k_s**2/(4*a_pos))),
    ('Top-hat / rect function',
     Piecewise((1, Abs(x_s) < a_pos), (S.Half, Abs(x_s) <= a_pos), (0, True)),
     None),  # sin(ak)/k form
    ('Delta function',
     DiracDelta(x_s),
     1/(2*pi)),  # constant spectrum
    ('Derivative',
     None,
     None),
]
s('Gaussian → Gaussian (FT preserves Gaussian form):')
s('  f(x) = e^{-ax²}  →  φ(k) = √(π/a) e^{-k²/4a}')
s('  Width in x: σ_x = 1/√(2a)')
s('  Width in k: σ_k = √(2a)')
s('  Product: σ_x · σ_k = 1  → Heisenberg ΔxΔp = ħ/2 ✓')

s('')
s('Derivative rule (key for momentum operator):')
s('  FT{df/dx} = ik · φ(k)')
s('  Apply to Schrödinger: -ħ²/2m d²ψ/dx² + Vψ = Eψ')
s('  FT: ħ²k²/2m · φ(k) + FT{Vψ} = E · φ(k)')
s('  Pure kinetic (V=0): φ(k) = A·δ(k-k₀)  ← plane wave eigenstate')

s('')
s('Convolution theorem in QM — scattering theory:')
s('  ψ_scattered(r) = G₀(r) * V(r) · ψ_incident')
s('  where G₀ = free-particle Green function')
s('  FT: ψ̂_scattered(k) = Ĝ₀(k) · V̂*ψ̂_incident')
s('  Born approximation: scattering amplitude ∝ FT{V}(q) where q=k_f-k_i')
"""

S9_MD = """\
---
# §9 — Griffiths Forms → Machine Learning: The Axis

The math in Griffiths is not just physics — it is the foundation of
modern machine learning. This section makes the translation explicit.

| Griffiths | ML equivalent | Where it appears |
|-----------|--------------|-----------------|
| Eigenvectors of Hermitian op | Principal components (PCA) | Dimensionality reduction |
| Orthonormal basis expansion | Attention mechanism (Q,K,V) | Transformers |
| Green's function G(r,r') | Neural operator kernel K(x,y) | FNO, DeepONet |
| Fourier series on L² | Spectral convolution layer | FNO in gs_fno.py |
| Perturbation theory | Gradient descent + learning rate scheduling | Any NN training |
| Variational principle | Loss minimization = energy minimization | All of ML |
| Dirac delta sifting | Attention: softmax picks one key | Transformers |
| Schrödinger time evolution | Residual network layer | ResNet, FNO blocks |
"""

S9_CODE = """\
sec('Griffiths → Machine Learning: The Explicit Axis')
import numpy as np

sub('1. Hermitian operators → PCA / SVD')
s('Quantum: Â|n⟩ = aₙ|n⟩  (Hermitian, real eigenvalues, orthogonal eigenvectors)')
s('ML:      Cov(X)|v_k⟩ = λ_k|v_k⟩  ← PCA is diagonalizing the covariance operator')
s('Both find the natural "modes" of a system.')
s('In ML these modes are directions of maximum variance.')
s('In QM these modes are stationary states (eigenstates).')

A_cov = np.array([[3, 1, 0],
                  [1, 2, 1],
                  [0, 1, 1]], dtype=float)
vals, vecs = np.linalg.eigh(A_cov)
s('Example covariance matrix:', Matrix(A_cov.tolist()))
s('Eigenvalues (variance in each principal direction):', list(np.round(vals,3)))
s('Eigenvectors (principal axes):', Matrix(np.round(vecs,3).tolist()))

sub('2. Fourier series on L² → FNO spectral layer')
s('Griffiths: any normalizable wavefunction expands as ψ = Σ cₙ φₙ')
s('           where φₙ are eigenfunctions of Ĥ (complete orthonormal set)')
s('FNO:       any signal x(t) expands as x = Σ_k X[k] e^{i2πkt/N}')
s('           SpectralConv1d learns weights R[k] in this Fourier basis')
s('           Output: Σ_k R[k]·X[k]·e^{i2πkt/N}')
s('This is EXACTLY the quantum measurement expansion, but learned rather than derived.')

sub('3. Green\'s functions → Neural Operators')
s('Griffiths ch 9-11: solution to Lψ = f  is  ψ(r) = ∫ G(r,r\') f(r\') dr\'')
s('                   G is the propagator / resolvent of L')
s('Neural Operator:   output(x) = ∫ K(x,x\') · input(x\') dx\'   (learned kernel K)')
s('FNO implements this efficiently: K(x,x\') is a convolution kernel')
s('  → evaluated in Fourier domain as pointwise multiplication R[k]·X[k]')
s('  → O(N log N) instead of O(N²) for dense K')
s('')
s('In gs_fno.py:')
s('  SpectralConv1d = Fourier-domain neural operator')
s('  Input: I₁(t), I₂(t)  ←  f(r\') in Green\'s function language')
s('  Output: φ̂(t)          ←  ψ(r) = ∫G·f dr\'')
s('  Learned K(t,t\') captures the non-local dispersive phase structure')

sub('4. Variational principle → gradient descent')
s('Griffiths §7.1: ground state energy E₀ = min_{ψ} ⟨ψ|H|ψ⟩/⟨ψ|ψ⟩')
s('  → perturb the trial wavefunction to lower ⟨H⟩')
s('ML: optimal parameters θ* = argmin_θ L(θ)')
s('  → gradient descent: θ ← θ − η ∇_θ L')
s('Both are the same mathematical operation:')
s('  find the function (wavefunction / model parameters) that minimizes an energy functional')

sub('5. Uncertainty principle → model complexity tradeoff')
s('Griffiths: σ_x · σ_p ≥ ħ/2  (cannot localize in both x and p simultaneously)')
s('ML:  bias–variance tradeoff  (cannot minimize both simultaneously)')
s('     overfit in time → high bias in frequency  ←  sampling theorem')
s('     in GS: narrow window → good time resolution, poor frequency resolution (STFT §9)')

sub('Connection summary for this project')
s('')
s('TD-GS algorithm:')
s('  Core: two FTs per iteration  ←  §8 FT depth')
s('  Convergence: alternating projections = variational principle  ←  §9 item 4')
s('')
s('FNO (gs_fno.py):')
s('  Architecture: Green\\'s function operator  ←  §9 item 3')
s('  SpectralConv: Fourier expansion  ←  §9 item 2')
s('  Training: variational / gradient descent  ←  §9 item 4')
s('  Resolution invariance: comes from function-space formulation  ←  §3 Griffiths L²')
"""

new_cells = [md(S8_MD), code(S8_CODE), md(S9_MD), code(S9_CODE)]
nb['cells'].extend(new_cells)

with open(NB, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print(f'griffiths_prep.ipynb now has {len(nb["cells"])} cells (added §8 + §9).')
