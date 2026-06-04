"""
_repl_attention_projection.py
Attention = oblique projection in Hilbert space.
Paraxial wave modes = orthogonal basis in L^2.
GS alternating projection geometry.
SymPy + numpy + pandas.
"""
import numpy as np
import sympy as sp
import pandas as pd

# ============================================================
# 1. Projection in vector space: the core operation
# ============================================================
print("=== Projection: the one operation everything reduces to ===")
print("""
Orthogonal projection of v onto subspace spanned by u:

    P_u(v) = (v . u / u . u) * u     <- scalar projection * unit vector

In matrix form (onto column space of A):
    P_A = A (A^T A)^-1 A^T

Properties:
    P^2 = P          idempotent  (project twice = project once)
    P^T = P          symmetric   (orthogonal projection)
    eigenvalues 0,1  (you're either in the subspace or orthogonal to it)
""")

# numerical check
rng = np.random.default_rng(3)
u = rng.standard_normal(5)
v = rng.standard_normal(5)

proj = (np.dot(v, u) / np.dot(u, u)) * u
residual = v - proj

print(f"v.u = {np.dot(v,u):.4f}")
print(f"proj(v onto u).u = {np.dot(proj,u):.4f}  (same, as expected)")
print(f"residual . u = {np.dot(residual,u):.2e}  (orthogonal to u)")
print(f"P^2 = P check: {np.allclose(proj, (np.dot(proj,u)/np.dot(u,u))*u)}")
print()

# ============================================================
# 2. Attention IS a projection
# ============================================================
print("=== Attention = soft oblique projection ===")
print("""
Standard scaled dot-product attention:

    Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V

Geometry:
    Q (query)  = where you want to project TO
    K (keys)   = the subspace basis vectors
    V (values) = what gets mixed

    Q K^T / sqrt(d_k)  -> similarity scores (dot products)
    softmax(...)       -> soft weights summing to 1 (convex combination)
    * V                -> weighted sum of value vectors

Hard limit (argmax instead of softmax):
    -> exact projection onto nearest key
    -> softmax makes it differentiable + considers all keys

Connection to GS:
    GS constraint set C1 = project E onto {fields with |E_D1|^2 = I1}
    Attention query     = project input onto {value vectors matching key}
    Both: find the point in a subspace closest to current estimate.
""")

# small attention demo
d_k = 4
d_v = 4
seq_len = 6

Q = rng.standard_normal((seq_len, d_k))
K = rng.standard_normal((seq_len, d_k))
V = rng.standard_normal((seq_len, d_v))

scores = Q @ K.T / np.sqrt(d_k)           # (seq, seq)
scores -= scores.max(axis=-1, keepdims=True)
weights = np.exp(scores)
weights /= weights.sum(axis=-1, keepdims=True)   # softmax
output = weights @ V                             # (seq, d_v)

print(f"Q shape: {Q.shape}  K shape: {K.shape}  V shape: {V.shape}")
print(f"Attention weights (row sums = 1): {weights.sum(axis=1).round(4)}")
print(f"Output shape: {output.shape}")
print()

df_w = pd.DataFrame(weights.round(3),
    columns=[f'k{i}' for i in range(seq_len)],
    index=[f'q{i}' for i in range(seq_len)])
print("Attention weight matrix (query x key):")
print(df_w.to_string())
print()

# ============================================================
# 3. Paraxial wave modes: orthogonal basis in L^2
# ============================================================
print("=== Paraxial modes: Hermite-Gaussian orthogonal basis ===")
print("""
Paraxial wave equation (z-propagation, x-transverse):

    d/dz E = (i/2k) d^2/dx^2 E

Solutions: Hermite-Gaussian modes HG_n(x,z)
    HG_n(x) = H_n(x/w) * exp(-x^2/2w^2)   at waist

Orthogonality:
    integral HG_m(x) * HG_n(x) dx = delta_mn

+y, -y, 0 modes = HG_0 (Gaussian), HG_1 (first-order), TEM_00
Surface measurement: interferometric, phase recovered via GS or fringe analysis.
""")

x = np.linspace(-4, 4, 512)
w = 1.0

def hermite(n, x):
    if n == 0: return np.ones_like(x)
    if n == 1: return 2*x
    if n == 2: return 4*x**2 - 2
    if n == 3: return 8*x**3 - 12*x
    if n == 4: return 16*x**4 - 48*x**2 + 12
    return np.zeros_like(x)

modes = {}
for n in range(5):
    psi = hermite(n, x/w) * np.exp(-x**2/(2*w**2))
    norm = np.sqrt(np.trapezoid(psi**2, x))
    modes[n] = psi / norm

# orthogonality table
print("Orthogonality check <HG_m | HG_n>:")
rows = []
for m in range(5):
    row = {}
    for n in range(5):
        val = np.trapezoid(modes[m] * modes[n], x)
        row[f'HG{n}'] = round(val, 4)
    rows.append(row)
df_orth = pd.DataFrame(rows, index=[f'HG{m}' for m in range(5)])
print(df_orth.to_string())
print()
print("Diagonal = 1 (normalized), off-diagonal ~ 0 (orthogonal) = basis confirmed")
print()

# ============================================================
# 4. GS alternating projection geometry (SymPy)
# ============================================================
print("=== GS as alternating projection: SymPy fixed-point condition ===")
E_sym, phi_sym, I1_sym, I2_sym = sp.symbols('E phi I1 I2', positive=True)
D1_sym, D2_sym = sp.symbols('D1 D2', real=True)

print("""
Constraint sets in Hilbert space L^2(C^N):

    C1 = { E in C^N : |H(D1) E|^2 = I1 }    <- dispersed intensity matches
    C2 = { E in C^N : |H(D2) E|^2 = I2 }    <- second dispersed intensity

Projection onto C1:
    P_C1(E) = H(-D1) [ sqrt(I1) * exp(i * angle(H(D1) E)) ]
              ^^^^     ^^^^^^^^    ^^^^^^^^^^^^^^^^^^^^^^^^^
           undisperse  fix amp      keep phase from current estimate

GS iteration:
    E_{k+1} = P_C2( P_C1( E_k ) )

Converges to C1 intersect C2 when sets are non-empty and constraints compatible.
Diversity |D2 - D1| determines the angle between C1 and C2:
    large |delta_D| -> nearly orthogonal -> fast convergence, few traps
    small |delta_D| -> nearly parallel  -> slow, many metastable traps
""")

# angle between constraint sets numerically
print("Convergence rate vs dispersion diversity:")
rows2 = []
for delta_D in [100, 500, 1000, 2500, 5000, 10000]:
    D1 = -5000.0
    D2 = D1 - delta_D
    N = 64
    nu = np.fft.fftfreq(N)
    H1 = np.exp(1j * np.pi * D1 * nu**2)
    H2 = np.exp(1j * np.pi * D2 * nu**2)
    # angle between transfer functions as proxy for constraint set angle
    cos_angle = abs(np.dot(H1, np.conj(H2))) / (np.linalg.norm(H1) * np.linalg.norm(H2))
    rows2.append({'delta_D': delta_D, 'cos(angle)': round(float(cos_angle), 4),
                  'angle_deg': round(float(np.degrees(np.arccos(min(cos_angle,1.0)))), 2)})

df_angle = pd.DataFrame(rows2)
print(df_angle.to_string(index=False))
print()
print("Large angle -> orthogonal constraints -> GS converges fast with few traps")
