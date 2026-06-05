import nbformat, textwrap

def code(src): return nbformat.v4.new_code_cell(textwrap.dedent(src).strip())
def md(src):   return nbformat.v4.new_markdown_cell(textwrap.dedent(src).strip())

nb = nbformat.v4.new_notebook()
nb.cells = [

md("# Spectroscopy + RSA/PQ Crypto + GPS Spoofing\nReal-time sensing, number theory, signal authentication."),

code("""
import numpy as np
import sympy as sp
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import hilbert
sp.init_printing(use_unicode=False, wrap_line=False)
print('imports OK')
"""),

md("## S1. Beer-Lambert Law"),

code("""
eps_sym, c_sym, L_sym = sp.symbols('epsilon c L', positive=True)
A_sym = eps_sym * c_sym * L_sym
T_sym = 10**(-A_sym)
print('Beer-Lambert: A = eps*c*L,  T = 10^(-A)')
sp.pprint(A_sym); sp.pprint(T_sym)

eps_eth = 150.0; L_cm = 1.0
c_vals = np.array([0.001, 0.01, 0.05, 0.1, 0.5, 1.0])
A_vals = eps_eth * c_vals * L_cm
T_vals = 10**(-A_vals)
df = pd.DataFrame({'c (mol/L)': c_vals,
                   'A': np.round(A_vals,3),
                   'T': np.round(T_vals,4),
                   'I/I0 %': np.round(T_vals*100,2)})
print('Ethanol at 200nm, L=1cm, eps=150:')
print(df.to_string(index=False))
"""),

code("""
lam = np.linspace(400, 700, 300)

def gauss_peak(lam, center, width, height):
    return height * np.exp(-0.5*((lam-center)/width)**2)

eps_A = gauss_peak(lam, 450, 20, 1.0)
eps_B = gauss_peak(lam, 550, 25, 0.8)
eps_C = gauss_peak(lam, 620, 15, 1.2)
c_A, c_B, c_C = 0.05, 0.08, 0.03

A_total = (eps_A*c_A + eps_B*c_B + eps_C*c_C) * 1.0
T_total = 10**(-A_total)

fig, axes = plt.subplots(1, 2, figsize=(10,4))
axes[0].plot(lam, eps_A*c_A, 'b-', label='A 450nm')
axes[0].plot(lam, eps_B*c_B, 'g-', label='B 550nm')
axes[0].plot(lam, eps_C*c_C, 'r-', label='C 620nm')
axes[0].plot(lam, eps_A*c_A+eps_B*c_B+eps_C*c_C, 'k--', lw=2, label='sum')
axes[0].set_xlabel('Wavelength (nm)'); axes[0].set_ylabel('eps*c')
axes[0].set_title('Absorption'); axes[0].legend(fontsize=8)
axes[1].plot(lam, T_total*100, 'k-')
axes[1].set_xlabel('Wavelength (nm)'); axes[1].set_ylabel('Transmittance %')
axes[1].set_title('Transmitted spectrum')
axes[1].fill_between(lam, 0, T_total*100, alpha=0.3)
plt.tight_layout()
plt.savefig('../_spectroscopy.png', dpi=120, bbox_inches='tight')
print(f'Peak absorption at {lam[np.argmin(T_total)]:.0f} nm')
print('Saved: _spectroscopy.png')
"""),

code("""
print('iPhone spectroscopy: 3 RGB channels = 3 unknowns max')

def rgb_resp(lam):
    R = gauss_peak(lam, 620, 60, 1.0)
    G = gauss_peak(lam, 540, 50, 1.0)
    B = gauss_peak(lam, 460, 40, 1.0)
    return R, G, B

R_r, G_r, B_r = rgb_resp(lam)
S = np.array([
    [np.trapezoid(R_r*eps_A,lam), np.trapezoid(R_r*eps_B,lam), np.trapezoid(R_r*eps_C,lam)],
    [np.trapezoid(G_r*eps_A,lam), np.trapezoid(G_r*eps_B,lam), np.trapezoid(G_r*eps_C,lam)],
    [np.trapezoid(B_r*eps_A,lam), np.trapezoid(B_r*eps_B,lam), np.trapezoid(B_r*eps_C,lam)],
])
print(f'RGB sensitivity matrix det={np.linalg.det(S):.4f}')
c_true = np.array([c_A, c_B, c_C])
A_rgb  = S @ c_true
c_rec  = np.linalg.solve(S, A_rgb)
print(f'True:      {c_true}')
print(f'Recovered: {np.round(c_rec,4)}')
"""),

md("## S2. RSA + Post-Quantum (LWE)"),

code("""
p, q = 61, 53
n = p*q; phi_n = (p-1)*(q-1); e = 17
d = int(sp.mod_inverse(e, phi_n))
print(f'RSA: p={p} q={q} n={n} phi={phi_n} e={e} d={d}')
print(f'e*d mod phi(n) = {(e*d)%phi_n}  (must be 1)')

m = 42
c_rsa = pow(m, e, n)
m_dec = pow(c_rsa, d, n)
print(f'Encrypt: {m} -> {c_rsa}  Decrypt: {c_rsa} -> {m_dec}  OK={m_dec==m}')
print()
print('Security: factor n=3233 to get p=61,q=53 -> phi -> d')
print('Real RSA-2048: n is 617-digit number. GNFS: ~10^20 ops.')
print('Shor (QC): O(n^3) gates. Breaks RSA. Drives PQ migration.')
"""),

code("""
print('Post-Quantum: Learning With Errors (LWE)')
rng = np.random.default_rng(7)
n_lwe, q_lwe = 4, 97
s = np.array([3, 1, 4, 1])
A = rng.integers(0, q_lwe, size=(6, n_lwe))
e_noise = rng.integers(-1, 2, size=6)
b = (A @ s + e_noise) % q_lwe
print(f'  secret s = {s}')
print(f'  error  e = {e_noise}  (small, hidden)')
print(f'  public b = A*s+e mod {q_lwe} = {b}')
print()
rows = [
    ('RSA-2048',    'Factoring', 'Shor breaks it', '~256 B'),
    ('ECDH-256',    'ECDLP',     'Shor breaks it', '~64 B'),
    ('Kyber-512',   'LWE',       'Quantum-safe',   '~800 B'),
    ('Kyber-768',   'LWE',       'Quantum-safe',   '~1184 B'),
    ('Dilithium-2', 'Mod-LWE',   'Quantum-safe',   '~2420 B'),
]
print(pd.DataFrame(rows, columns=['Scheme','Problem','PQ status','Key size']).to_string(index=False))
"""),

md("## S3. GPS Spoofing Detection"),

code("""
MEO_km = 20200.0
sats = np.array([[0.577,0.577,0.577],[-0.577,0.577,0.577],
                 [0.577,-0.577,0.577],[0.577,0.577,-0.577]]) * MEO_km
recv_true = np.array([0.0, 0.0, 6371.0])
c_km = 299792.458
rho = np.linalg.norm(sats - recv_true, axis=1) + c_km*1e-6

def trilaterate(sats, rho, c):
    n = len(sats)
    A_mat = np.zeros((n-1,4)); b_vec = np.zeros(n-1)
    for i in range(1, n):
        dx = sats[i]-sats[0]
        A_mat[i-1,:3] = -2*dx
        A_mat[i-1, 3] = 2*c*(rho[i]-rho[0])
        b_vec[i-1] = rho[i]**2-rho[0]**2-np.dot(sats[i],sats[i])+np.dot(sats[0],sats[0])
    x,_,_,_ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
    return x[:3], x[3]

pos_rec, _ = trilaterate(sats, rho, c_km)
print(f'Recovered position: {np.round(pos_rec,2)} km')
print(f'Position error: {np.linalg.norm(pos_rec-recv_true)*1000:.2f} m')
print()

unit_vecs = sats - recv_true
unit_vecs /= np.linalg.norm(unit_vecs, axis=1, keepdims=True)
dots = unit_vecs @ unit_vecs.T
off = dots[np.triu_indices(4, k=1)]
print(f'Satellite direction cosines: {np.round(off,3)}')
print(f'Max |cos| = {np.max(np.abs(off)):.3f}')
print(f'Real GPS: diverse sky coverage, max|cos|<<1')
print(f'Spoofer:  single source, max|cos|~1 -> DETECTED')
print()
print('Galileo OSNMA (2023): cryptographic authentication of nav messages')
print('HMAC-SHA256 on satellite ephemeris -> spoofer cannot forge signature')
"""),

md("## S4. Kramers-Kronig: spectroscopy meets GS\n\n| System | Measured | Lost | Recovery method |\n|--------|----------|------|-----------------|\n| Spectrometer | `I(lam)=|E|^2` | `phi(lam)` | K-K relation (causality) |\n| GPS | pseudoranges | position | trilateration |\n| RSA | `c=m^e mod n` | `m` | private key `d` |\n| GS | `I1,I2` | `phi(t)` | alternating projections |\n\nAll four: measure magnitude, recover what generated it."),

code("""
# K-K via Hilbert transform
k_abs = A_total / (4*np.pi*(lam*1e-9))
n_kk = 1 + np.imag(hilbert(k_abs))

print('Kramers-Kronig phase recovery from absorption spectrum:')
print(f'  k (absorption) range: [{k_abs.min():.4e}, {k_abs.max():.4e}]')
print(f'  n (refractive)  range: [{n_kk.min():.4f}, {n_kk.max():.4f}]')
print()
print('K-K constraint = causality (response after stimulus)')
print('GS constraint  = measured intensity I1, I2')
print('Both recover phase from magnitude measurement.')
print('Same mathematical structure: fixed-point of projection.')
"""),

]

nbformat.write(nb, 'notebooks/spectroscopy_crypto_gnss.ipynb')
print('Written OK')
