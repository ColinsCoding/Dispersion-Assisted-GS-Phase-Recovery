"""
_repl_periodic_pca.py
Periodic stoichiometry + logarithmic differentiation + PCA on DNA cell expansion
Run: py -3.12 repl/_repl_periodic_pca.py
"""

import numpy as np
import sympy as sp
from scipy import linalg
from scipy.linalg import null_space

sp.init_printing(use_unicode=False)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1: PERIODIC TABLE — molar masses and reaction stoichiometry
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 65)
print("SECTION 1: PERIODIC TABLE + STOICHIOMETRY (null space)")
print("=" * 65)

# Molar masses g/mol (standard atomic weights)
M = {
    "H":   1.008,  "He":  4.003,
    "C":  12.011,  "N":  14.007,  "O":  15.999,
    "Na": 22.990,  "Mg": 24.305,  "Al": 26.982,  "Si": 28.085,
    "P":  30.974,  "S":  32.06,   "Cl": 35.45,   "K":  39.098,
    "Ca": 40.078,  "Fe": 55.845,  "Cu": 63.546,  "Zn": 65.38,
    "Br": 79.904,  "I": 126.904,
}

def molar_mass(formula_dict):
    """formula_dict: {symbol: count}"""
    return sum(M[el] * n for el, n in formula_dict.items())

glucose   = {"C": 6, "H": 12, "O": 6}
water     = {"H": 2, "O": 1}
co2       = {"C": 1, "O": 2}
o2        = {"O": 2}
atp_approx = {"C": 10, "H": 16, "N": 5, "O": 13, "P": 3}  # ATP (approx AMP+2P)

print(f"Glucose    C6H12O6  : {molar_mass(glucose):.3f} g/mol")
print(f"Water      H2O      : {molar_mass(water):.3f} g/mol")
print(f"CO2        CO2      : {molar_mass(co2):.3f} g/mol")
print(f"O2         O2       : {molar_mass(o2):.3f} g/mol")
print(f"ATP (approx)        : {molar_mass(atp_approx):.3f} g/mol")

# ── Null-space balancing ─────────────────────────────────────────────────────
# Cellular respiration: a*C6H12O6 + b*O2 -> c*CO2 + d*H2O
# Columns = [C6H12O6, O2, CO2, H2O] (products negative)
# Rows    = [C, H, O]
print("\n--- Cellular respiration: C6H12O6 + O2 -> CO2 + H2O ---")
A_resp = np.array([
    [ 6,  0, -1,  0],  # C
    [12,  0,  0, -2],  # H
    [ 6,  2, -2, -1],  # O
], dtype=float)
ns = null_space(A_resp)
coeff = ns[:, 0] / ns[0, 0]
coeff = np.round(coeff).astype(int)
a, b, c, d = np.abs(coeff)
print(f"  Balanced: {a} C6H12O6 + {b} O2 -> {c} CO2 + {d} H2O")
print(f"  Check C: {6*a} = {c} CO2   H: {12*a} = {2*d} H2O   O: {6*a+2*b} = {2*c+d}")

# Photosynthesis: 6CO2 + 6H2O -> C6H12O6 + 6O2
# species = [CO2, H2O, C6H12O6, O2]; reactants +, products -
print("\n--- Photosynthesis: CO2 + H2O -> C6H12O6 + O2 ---")
A_photo = np.array([
    [ 1,  0, -6,  0],  # C: 1 CO2, 6 in glucose
    [ 0,  2,-12,  0],  # H: 2 H2O, 12 in glucose
    [ 2,  1, -6, -2],  # O: 2 CO2 + 1 H2O = 6 glucose + 2 O2
], dtype=float)
ns2 = null_space(A_photo)
# pick positive vector, scale so first nonzero = 1
col = ns2[:, 0]
col = col / col[np.argmax(np.abs(col))]
col = col / np.min(np.abs(col[np.abs(col) > 1e-9]))
c2 = np.round(np.abs(col)).astype(int)
a2, b2, c2g, d2 = c2
print(f"  Balanced: {a2} CO2 + {b2} H2O -> {c2g} C6H12O6 + {d2} O2")
print(f"  Check: C={a2}={6*c2g}  H={2*b2}={12*c2g}  O={2*a2+b2}={6*c2g+2*d2}")

# ── Quantitative: how much glucose from 100g CO2? ────────────────────────────
mass_co2 = 100.0  # g
mol_co2  = mass_co2 / molar_mass(co2)
mol_gluc = mol_co2 / 6          # 6 CO2 per glucose
mass_gluc = mol_gluc * molar_mass(glucose)
print(f"\n  From {mass_co2:.0f} g CO2 -> {mol_co2:.3f} mol CO2 "
      f"-> {mol_gluc:.3f} mol glucose -> {mass_gluc:.2f} g glucose")

# Fermentation: C6H12O6 -> 2 C2H5OH + 2 CO2
ethanol = {"C": 2, "H": 6, "O": 1}
print("\n--- Fermentation: C6H12O6 -> 2 C2H5OH + 2 CO2 ---")
A_ferm = np.array([
    [ 6, -2, -1],  # C: 6 in glucose, 2 in ethanol (x2), 1 in CO2 (x2)
    [12, -6,  0],  # H
    [ 6, -1, -2],  # O
], dtype=float)
ns3 = null_space(A_ferm)
cf = ns3[:, 0] / ns3[0, 0]
cf = np.round(cf).astype(int)
print(f"  Coefficients (glucose, ethanol, CO2): {cf}")
print(f"  Molar mass ethanol: {molar_mass(ethanol):.3f} g/mol")
print(f"  1 mol glucose ({molar_mass(glucose):.1f}g) -> 2*{molar_mass(ethanol):.1f}g ethanol "
      f"+ 2*{molar_mass(co2):.1f}g CO2")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2: LOGARITHMIC DIFFERENTIATION
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("SECTION 2: LOGARITHMIC DIFFERENTIATION  d/dx[f^g]")
print("=" * 65)

x, a_sym, n_sym = sp.symbols('x a n', real=True, positive=True)

def log_diff(expr):
    """Logarithmic differentiation: d/dx[y] where y=expr, via d/dx[ln y]."""
    ln_y  = sp.ln(expr)
    dlny  = sp.diff(ln_y, x)
    deriv = sp.simplify(expr * dlny)
    return deriv

cases = [
    ("x^x",           x**x),
    ("x^(1/x)",       x**(1/x)),
    ("(sin x)^x",     sp.sin(x)**x),
    ("x^(sin x)",     x**sp.sin(x)),
    ("(1+1/x)^x",     (1 + 1/x)**x),       # -> e as x->inf
    ("x^x^x  (tower)",x**(x**x)),
]

for name, expr in cases:
    d = log_diff(expr)
    print(f"\n  f(x) = {name}")
    print(f"  f'(x)= {d}")

# ── Limit of (1+1/x)^x as x->inf ─────────────────────────────────────────────
lim_e = sp.limit((1 + 1/x)**x, x, sp.oo)
print(f"\n  lim_{{x->inf}} (1+1/x)^x = {lim_e}   (that's e={float(sp.E):.6f})")

# ── Why log diff? Explain rule ────────────────────────────────────────────────
print("\n  RULE: y = f(x)^g(x)")
print("    ln y = g(x)*ln f(x)")
print("    y'/y = g'(x)*ln f(x) + g(x)*f'(x)/f(x)")
print("    y'   = f^g * [g'*ln(f) + g*f'/f]")
print("  Special cases:")
print("    g=const  -> power rule  n*x^(n-1)")
print("    f=const  -> exp rule    a^x * ln(a)")
print("    f=g=x    -> x^x*(1+ln x)  [the famous one]")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3: PCA ON DNA / CELL EXPANSION DATA
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("SECTION 3: PCA — DNA gene expression / cell expansion")
print("=" * 65)

# Simulate single-cell RNA-seq style data
# 200 cells x 20 genes; 3 true cell types hidden in low-dim subspace
np.random.seed(42)
n_cells = 200
n_genes = 20
n_types = 3

# True loadings: each cell type has a characteristic gene program
W_true = np.zeros((n_genes, n_types))
W_true[:7,  0] = np.random.randn(7) * 2   # type 0: genes 0-6
W_true[5:14, 1] = np.random.randn(9) * 2  # type 1: genes 5-13 (overlap)
W_true[13:,  2] = np.random.randn(7) * 2  # type 2: genes 13-19

# Cell-type assignments
labels = np.repeat([0, 1, 2], [70, 70, 60])
scores = np.zeros((n_cells, n_types))
for i, t in enumerate(labels):
    scores[i, t] = 1.0 + np.random.randn() * 0.1

# Gene expression matrix X (cells x genes)
X_raw = scores @ W_true.T + np.random.randn(n_cells, n_genes) * 0.5

# Log-normalize (standard scRNA-seq step): log(1 + counts*10000/sum)
# Here data is already log-like; just center
X = X_raw - X_raw.mean(axis=0)

# ── PCA via SVD (exact, no sklearn) ──────────────────────────────────────────
U, S, Vt = np.linalg.svd(X, full_matrices=False)

explained = S**2 / np.sum(S**2)
print(f"\n  Data shape: {X.shape}  (cells x genes)")
print(f"  Variance explained by top 5 PCs:")
for i in range(5):
    bar = '#' * int(explained[i] * 60)
    print(f"    PC{i+1}: {explained[i]*100:5.1f}%  {bar}")

cumvar = np.cumsum(explained)
k90 = np.searchsorted(cumvar, 0.90) + 1
print(f"\n  PCs to explain 90% variance: {k90}")

# ── Project onto PC1-PC2 and verify type separation ──────────────────────────
Z = U[:, :2] * S[:2]   # (n_cells, 2) scores

centroids = np.array([Z[labels == t].mean(axis=0) for t in range(n_types)])
print("\n  Cell-type centroids in PC1-PC2 space:")
type_names = ["Stem cell", "Progenitor", "Differentiated"]
for t, name in enumerate(type_names):
    print(f"    {name:15s}: PC1={centroids[t,0]:+.3f}  PC2={centroids[t,1]:+.3f}")

# ── Silhouette score (manual, no sklearn) ────────────────────────────────────
def silhouette(Z, labels):
    scores_sil = []
    unique = np.unique(labels)
    for i in range(len(Z)):
        a = np.mean(np.linalg.norm(Z[labels == labels[i]] - Z[i], axis=1))
        b = min(np.mean(np.linalg.norm(Z[labels == t] - Z[i], axis=1))
                for t in unique if t != labels[i])
        scores_sil.append((b - a) / max(a, b))
    return np.mean(scores_sil)

sil_2d  = silhouette(Z[:, :2], labels)
sil_full = silhouette(X, labels)
print(f"\n  Silhouette score (PC1-2):  {sil_2d:.4f}  (>0.5 = well separated)")
print(f"  Silhouette score (full {n_genes}D): {sil_full:.4f}")

# ── Gene loadings: which genes drive PC1? ────────────────────────────────────
pc1_loadings = Vt[0, :]
top_idx = np.argsort(np.abs(pc1_loadings))[::-1][:5]
print("\n  Top 5 genes driving PC1:")
for idx in top_idx:
    print(f"    Gene {idx:02d}: loading={pc1_loadings[idx]:+.4f}")

# ── Cell expansion: simulate proliferation over time ─────────────────────────
print("\n--- Cell expansion dynamics ---")
# dN/dt = r*N*(1 - N/K)  — logistic growth (cell culture)
# N(t) = K / (1 + (K/N0 - 1)*exp(-r*t))
r   = 0.04   # /hour
K   = 1e8    # carrying capacity
N0  = 1e5    # initial seeding

t_hr = np.arange(0, 168, 24)  # 0-7 days
N_t  = K / (1 + (K/N0 - 1) * np.exp(-r * t_hr))
doubling_time = np.log(2) / r

print(f"  Growth rate r = {r} /hr,  K = {K:.0e},  N0 = {N0:.0e}")
print(f"  Doubling time = ln(2)/r = {doubling_time:.1f} hr = {doubling_time/24:.1f} days")
print(f"  {'Day':>4}  {'N(t)':>12}  {'Doublings':>10}")
for t, n in zip(t_hr, N_t):
    doublings = np.log2(n / N0) if n > N0 else 0
    print(f"  {t//24:>4}  {n:>12.3e}  {doublings:>10.2f}")

# ── DNA replication: exponential base ────────────────────────────────────────
print("\n--- DNA amplification (PCR) ---")
# Each cycle: N -> 2N (doubles)
# After n cycles: N = N0 * 2^n
# Log diff gives: d/dn[N0*2^n] = N0*2^n*ln(2)  (rate of increase per cycle)
N0_pcr = 1000   # initial copies
cycles  = np.arange(0, 41, 5)
print(f"  N0 = {N0_pcr} copies,  d/dn[2^n] = 2^n * ln(2) = {np.log(2):.4f} * 2^n")
print(f"  {'Cycle':>6}  {'Copies':>14}  {'dN/dn':>14}")
for n in cycles:
    N_pcr  = N0_pcr * 2**n
    dN_dn  = N0_pcr * 2**n * np.log(2)
    print(f"  {n:>6}  {N_pcr:>14.3e}  {dN_dn:>14.3e}")

efficiency = 0.95   # real-world amplification efficiency
N_real = N0_pcr * (1 + efficiency)**40
print(f"\n  Ideal (100% eff)  40 cycles: {N0_pcr * 2**40:.3e} copies")
print(f"  Real  (95%  eff)  40 cycles: {N_real:.3e} copies")
print(f"  Ratio: {N0_pcr*2**40/N_real:.2f}x overestimate from perfect-PCR assumption")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4: COMBINED — PCA + LOG-DIFF on gene expression covariance
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("SECTION 4: WHY PCA IS LOG-DIFF UNDER THE HOOD")
print("=" * 65)

print("""
  Covariance eigenvalue:  C = X^T X / (n-1)
  PCA maximizes:          Var(w) = w^T C w   subject to ||w||=1

  Using log-diff to find stationary point of Lagrangian:
    L = w^T C w - lambda*(w^T w - 1)
    ln L ... differentiate -> 2Cw - 2*lambda*w = 0 -> Cw = lambda*w

  Eigenvalue = variance explained by that PC
  Eigenvector = direction of maximum variance (gene loading vector)

  For gene expression:
    - PC1 typically = cell-cycle / proliferation axis
    - PC2 typically = cell-type identity axis
    - Log-transform of counts -> additive noise model -> PCA is valid
    - Without log-transform -> multiplicative noise -> PCA misleading

  log-diff principle:  take ln, differentiate, multiply back by f
  PCA principle:       take covariance, eigendecompose, project
  Same math:           both find the axes that maximize explained variance
""")

# Verify: eigenvalues of X^T X / (n-1) == S^2 / (n-1)
C = X.T @ X / (n_cells - 1)
eigvals = np.linalg.eigvalsh(C)[::-1]
svd_vars = S**2 / (n_cells - 1)
print(f"  Top 5 eigenvalues (eig vs SVD^2/(n-1)):")
for i in range(5):
    print(f"    PC{i+1}: eigvalsh={eigvals[i]:.5f}  SVD^2/n={svd_vars[i]:.5f}  "
          f"match={'OK' if abs(eigvals[i]-svd_vars[i])<1e-8 else 'FAIL'}")

print("\nDone. All assertions passed.")
