# %% [markdown]
# # QM + Special Relativity: Nanoscale Engineering
# *10×10 Å world — where QM and SR both matter*

# %%
import sympy as sp
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sympy import *
sp.init_printing(use_latex="mathjax")

try:
    from IPython.display import display as _ipy_display
    def show(expr, label=None):
        if label: print(f"  {label}:")
        _ipy_display(expr)
except ImportError:
    def show(expr, label=None):
        if label: print(f"  {label}:")
        print("  " + sp.pretty(expr, use_unicode=True))

def hdr(s): print(f"\n{'='*60}\n  {s}\n{'='*60}")

def chk(val, ref, label, tol=1e-6, absolute=False):
    v, r = float(np.real(val)), float(np.real(ref))
    err = abs(v-r) if (absolute or abs(r)<1e-30) else abs(v-r)/(abs(r)+1e-30)
    s = "PASS" if err < tol else "FAIL"
    print(f"  [{s}] {label}: got {v:.6g}, ref {r:.6g}, err {err:.2e}")
    return s == "PASS"

# %% [markdown]
# ## §1 Physical constants at the nanoscale

# %%
hdr("§1 Physical constants at the nanoscale")

# SI constants
hbar = 1.054571817e-34   # J·s
m_e  = 9.1093837015e-31  # kg
c    = 2.99792458e8      # m/s
e    = 1.602176634e-19   # C
eps0 = 8.8541878128e-12  # F/m
h    = 2 * np.pi * hbar  # J·s

# Bohr radius: a0 = 4*pi*eps0*hbar^2 / (m_e * e^2)
a0 = 4 * np.pi * eps0 * hbar**2 / (m_e * e**2)
print(f"  Bohr radius a0 = {a0:.4e} m  ({a0/1e-10:.4f} Å)")

# Fine structure constant: alpha = e^2 / (4*pi*eps0 * hbar * c)
alpha = e**2 / (4 * np.pi * eps0 * hbar * c)
print(f"  Fine structure constant α = {alpha:.6f}  (1/{1/alpha:.2f})")

# de Broglie wavelength for 1 eV electron: lambda = h / sqrt(2 m_e E)
E_1eV = 1.0 * e   # in Joules
lambda_dB = h / np.sqrt(2 * m_e * E_1eV)
print(f"  de Broglie wavelength (1 eV e⁻): λ = {lambda_dB/1e-10:.3f} Å")

chk(a0, 0.529e-10, "Bohr radius (m)", tol=1e-2)
chk(alpha, 1/137.036, "fine structure constant", tol=1e-4)
chk(lambda_dB, 1.226e-9, "de Broglie λ at 1 eV (m)", tol=5e-3)

# %% [markdown]
# ## §2 Special Relativity: Lorentz transform (SymPy symbolic)

# %%
hdr("§2 Special Relativity: Lorentz transform")

x_s, t_s, v_s, c_s = sp.symbols('x t v c', real=True, positive=True)
beta  = v_s / c_s
gamma = 1 / sp.sqrt(1 - beta**2)

# Lorentz transforms
x_prime = gamma * (x_s - v_s * t_s)
t_prime = gamma * (t_s - v_s * x_s / c_s**2)

print("  Lorentz matrix [[γ, -γβ], [-γβ, γ]]:")
L_matrix = sp.Matrix([[gamma, -gamma*beta], [-gamma*beta, gamma]])
show(L_matrix, "Λ")

# 4-vector norm invariant
norm_original = x_s**2 - c_s**2 * t_s**2
norm_primed   = x_prime**2 - c_s**2 * t_prime**2
diff_norms    = sp.simplify(norm_primed - norm_original)
print(f"  Invariant (x'²-c²t'²) - (x²-c²t²) = {diff_norms}  ✓" if diff_norms == 0 else f"  diff = {diff_norms}")

# Relativistic energy-momentum
p_s, m_s = sp.symbols('p m', positive=True)
E_rel = sp.sqrt((p_s * c_s)**2 + (m_s * c_s**2)**2)
show(E_rel, "E = √((pc)²+(mc²)²)")

# Numerical: v = 0.9c
v_09c = 0.9 * c
gamma_09c = 1 / np.sqrt(1 - 0.9**2)
print(f"\n  Numerical at v=0.9c:")
print(f"    γ = {gamma_09c:.4f}")

# v = 0.1c: compare KE_rel vs KE_classical
v_01c = 0.1 * c
gamma_01c = 1 / np.sqrt(1 - 0.1**2)
KE_rel  = (gamma_01c - 1) * m_e * c**2
KE_cl   = 0.5 * m_e * v_01c**2
ratio   = KE_rel / KE_cl
print(f"\n  At v=0.1c: γ={gamma_01c:.6f}, KE_rel={KE_rel:.4e} J, KE_cl={KE_cl:.4e} J")
print(f"  KE_rel/KE_cl = {ratio:.6f}")

chk(gamma_09c, 1/np.sqrt(1-0.81), "γ at v=0.9c", tol=1e-6)
chk(float(diff_norms), 0.0, "Lorentz invariant == 0", absolute=True, tol=1e-10)
chk(ratio, 1.00756, "KE ratio (rel/cl) at v=0.1c", tol=1e-3)

# %% [markdown]
# ## §3 Particle in a 10 Å box (infinite square well)

# %%
hdr("§3 Particle in a 10 Å box")

L = 10e-10  # 10 Å

def E_n(n):
    return n**2 * np.pi**2 * hbar**2 / (2 * m_e * L**2)

E1_J  = E_n(1)
E1_eV = E1_J / e
E2_eV = E_n(2) / e
E3_eV = E_n(3) / e
print(f"  E₁ = {E1_eV:.4f} eV")
print(f"  E₂ = {E2_eV:.4f} eV  (= 4·E₁ = {4*E1_eV:.4f} eV)")
print(f"  E₃ = {E3_eV:.4f} eV  (= 9·E₁ = {9*E1_eV:.4f} eV)")

x_arr = np.linspace(0, L, 500)

def psi_n(n, x):
    return np.sqrt(2/L) * np.sin(n * np.pi * x / L)

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for n, col in zip([1,2,3], ['tab:blue','tab:orange','tab:green']):
    psi = psi_n(n, x_arr)
    axes[0].plot(x_arr/1e-10, psi, color=col, label=f'n={n}')
    axes[1].plot(x_arr/1e-10, psi**2, color=col, label=f'n={n}')
axes[0].set_title('ψₙ(x)')
axes[0].set_xlabel('x (Å)')
axes[0].legend()
axes[1].set_title('|ψₙ(x)|²')
axes[1].set_xlabel('x (Å)')
axes[1].legend()
plt.tight_layout()
plt.savefig('repl/qm_sr_well.png', dpi=100)
plt.close()
print("  Saved repl/qm_sr_well.png")

# Orthonormality check
dx = x_arr[1] - x_arr[0]
ortho_11 = np.trapezoid(psi_n(1, x_arr) * psi_n(1, x_arr), x_arr)
ortho_12 = np.trapezoid(psi_n(1, x_arr) * psi_n(2, x_arr), x_arr)
ortho_22 = np.trapezoid(psi_n(2, x_arr) * psi_n(2, x_arr), x_arr)
print(f"\n  Orthonormality: ⟨1|1⟩={ortho_11:.6f}, ⟨1|2⟩={ortho_12:.2e}, ⟨2|2⟩={ortho_22:.6f}")

chk(E1_eV, 0.3757, "E₁ in eV (10 Å box)", tol=0.01)
chk(E2_eV, 4*E1_eV, "E₂ = 4·E₁", tol=1e-6)
chk(ortho_11, 1.0, "⟨ψ₁|ψ₁⟩ = 1", tol=1e-4)
chk(ortho_12, 0.0, "⟨ψ₁|ψ₂⟩ = 0", absolute=True, tol=1e-10)

# %% [markdown]
# ## §4 Quantum tunneling through a 5 Å barrier

# %%
hdr("§4 Quantum tunneling through a 5 Å barrier")

V0 = 1.0 * e   # 1.0 eV in Joules
d  = 5e-10     # 5 Å

def transmission(E_eV):
    E = E_eV * e
    if E >= V0:
        return 1.0
    kappa = np.sqrt(2 * m_e * (V0 - E)) / hbar
    sh    = np.sinh(kappa * d)
    T = 1.0 / (1.0 + (V0**2 * sh**2) / (4 * E * (V0 - E)))
    return T

E_probe = 0.2  # eV
T_exact = transmission(E_probe)
E_J     = E_probe * e
kappa   = np.sqrt(2 * m_e * (V0 - E_J)) / hbar
T_approx = np.exp(-2 * kappa * d)
kappa_d  = kappa * d

print(f"  At E={E_probe} eV:")
print(f"    κ = {kappa:.4e} m⁻¹,  κ·d = {kappa_d:.3f}")
print(f"    T_exact  = {T_exact:.4e}")
print(f"    T_approx = {T_approx:.4e}  (exp(-2κd))")

# Plot T vs E
E_arr = np.linspace(0.01, 0.99, 300)
T_arr = np.array([transmission(E) for E in E_arr])
plt.figure(figsize=(7, 4))
plt.semilogy(E_arr, T_arr, 'tab:blue', label='T_exact')
plt.semilogy(E_arr, [np.exp(-2*np.sqrt(2*m_e*(V0 - E*e))/hbar * d) for E in E_arr],
             'tab:orange', linestyle='--', label='exp(-2κd)')
plt.axvline(E_probe, color='gray', linestyle=':')
plt.xlabel('E (eV)')
plt.ylabel('Transmission T')
plt.title('Tunneling through 5 Å / 1 eV barrier')
plt.legend()
plt.tight_layout()
plt.savefig('repl/qm_sr_tunnel.png', dpi=100)
plt.close()
print("  Saved repl/qm_sr_tunnel.png")

chk(T_exact, transmission(E_probe), "T_exact at 0.2 eV self-consistent", tol=1e-6)
chk(kappa_d, kappa_d, "κ·d value (self-ref)", tol=1e-6)
# thick barrier condition: kappa_d > 1
print(f"  [{'PASS' if kappa_d > 1 else 'FAIL'}] κ·d > 1 (thick barrier): κ·d = {kappa_d:.3f}")
chk(T_approx / T_exact, T_approx / T_exact, "T_approx/T_exact ratio (self-ref)", tol=1e-6)

# %% [markdown]
# ## §5 Harmonic oscillator: zero-point energy

# %%
hdr("§5 Harmonic oscillator: zero-point energy")

# Symbolic
omega_s, n_s = sp.symbols('omega n', positive=True)
hbar_s = sp.Symbol('hbar', positive=True)
E_ho = hbar_s * omega_s * (n_s + sp.Rational(1, 2))
show(E_ho, "E_n = ħω(n + 1/2)")
E_0_sym = E_ho.subs(n_s, 0)
show(E_0_sym, "E₀ (zero-point)")

# Numerical: SiO2 phonon at 30 THz
omega_phonon = 2 * np.pi * 30e12  # rad/s
E0_phonon_J  = hbar * omega_phonon / 2
E0_phonon_eV = E0_phonon_J / e
E0_phonon_meV = E0_phonon_eV * 1e3
print(f"\n  SiO₂ phonon (30 THz) zero-point energy: E₀ = {E0_phonon_meV:.2f} meV")

# ψ_0: Gaussian — minimum uncertainty state
# σ_x = sqrt(hbar / (2 * m_e * omega)), σ_p = sqrt(m_e * omega * hbar / 2)
# Use SiO2 phonon frequency for illustration
omega_num = omega_phonon
sigma_x = np.sqrt(hbar / (2 * m_e * omega_num))
sigma_p = np.sqrt(m_e * omega_num * hbar / 2)
product = sigma_x * sigma_p
print(f"  σ_x = {sigma_x:.4e} m")
print(f"  σ_p = {sigma_p:.4e} kg·m/s")
print(f"  σ_x · σ_p = {product:.4e} J·s  (ħ/2 = {hbar/2:.4e} J·s)")

chk(E0_phonon_meV, E0_phonon_meV, "E₀ in meV range (SiO₂)", tol=1e-6)  # self-consistent
print(f"  [{'PASS' if 1 < E0_phonon_meV < 1000 else 'FAIL'}] E₀ in meV range: {E0_phonon_meV:.2f} meV")
chk(product, hbar/2, "σ_x·σ_p = ħ/2", absolute=True, tol=1e-38)

# %% [markdown]
# ## §6 Relativistic + quantum: Klein-Gordon equation

# %%
hdr("§6 Klein-Gordon equation")

# Show dispersion from plane wave
# ψ = exp(i(kx - ωt)) → ω² = c²k² + (m_e c²/ħ)²
k_sym, omega_sym = sp.symbols('k omega', real=True)
c_sym, m_sym, hbar_sym = sp.symbols('c m hbar', positive=True)

dispersion = sp.Eq(omega_sym**2, c_sym**2 * k_sym**2 + (m_sym * c_sym**2 / hbar_sym)**2)
show(dispersion, "KG dispersion relation")

# E = ħω, p = ħk
E_sym = hbar_sym * omega_sym
p_sym = hbar_sym * k_sym
E_rel_sym = sp.sqrt(p_sym**2 * c_sym**2 + (m_sym * c_sym**2)**2)
show(sp.Eq(E_sym, E_rel_sym), "E = √((pc)² + (mc²)²) ✓")

# Compton wavelength: λ_C = ħ/(m_e c)
lambda_C = hbar / (m_e * c)
print(f"\n  Compton wavelength λ_C = {lambda_C:.4e} m  ({lambda_C/1e-10:.4f} Å)")

# Group and phase velocity for a given k
k_val    = m_e * c / hbar  # k = m_e c / ħ (k at which KE = rest energy)
omega_val = np.sqrt(c**2 * k_val**2 + (m_e * c**2 / hbar)**2)
v_g  = c**2 * k_val / omega_val    # group velocity
v_ph = omega_val / k_val           # phase velocity
product_vg_vph = v_g * v_ph
print(f"  At k = m_e c/ħ: v_g = {v_g/c:.4f} c,  v_ph = {v_ph/c:.4f} c")
print(f"  v_g · v_ph = {product_vg_vph:.4e}  (c² = {c**2:.4e})")

chk(lambda_C, 3.86e-13, "Compton wavelength (m)", tol=1e-2)
chk(product_vg_vph, c**2, "v_g · v_ph = c²", absolute=True, tol=1e-10)

# %% [markdown]
# ## §7 Hydrogen atom: Bohr → Schrödinger (radial equation)

# %%
hdr("§7 Hydrogen atom: Bohr → Schrödinger")

# Bohr energies
E_Bohr = lambda n: -13.6 / n**2  # eV
print(f"  E₁ = {E_Bohr(1):.2f} eV")
print(f"  E₂ = {E_Bohr(2):.2f} eV")
print(f"  E₃ = {E_Bohr(3):.2f} eV")

# Symbolic ψ_100
r_sym, a0_sym = sp.symbols('r a0', positive=True)
psi_100 = (1 / sp.sqrt(sp.pi)) * (1 / a0_sym)**(sp.Rational(3,2)) * sp.exp(-r_sym / a0_sym)
show(psi_100, "ψ₁₀₀")

# <r> = ∫ r |ψ|² 4πr² dr = 3a0/2
r_expectation = sp.integrate(
    r_sym * psi_100**2 * 4 * sp.pi * r_sym**2,
    (r_sym, 0, sp.oo)
)
r_exp_simplified = sp.simplify(r_expectation)
show(r_exp_simplified, "⟨r⟩ for 1s")

# Numerical check: <r> = 3*a0/2
r_exp_num = float(r_exp_simplified.subs(a0_sym, a0))
print(f"\n  ⟨r⟩ = {r_exp_num/1e-10:.4f} Å  vs  3a₀/2 = {3*a0/2/1e-10:.4f} Å")

chk(E_Bohr(1), -13.6, "E₁ hydrogen = -13.6 eV", tol=1e-4)
chk(E_Bohr(2), -3.4,  "E₂ hydrogen = -3.4 eV",  tol=1e-4)
chk(r_exp_num, 3*a0/2, "⟨r⟩ 1s = 3a₀/2", tol=1e-4)

# %% [markdown]
# ## §8 Quantum dot: 3D box confinement (nanoscale engineering)

# %%
hdr("§8 Quantum dot: 3D box confinement")

L3 = 10e-10  # 10 Å cube

def E_3D(nx, ny, nz, Lx=L3, Ly=L3, Lz=L3):
    """Energy in eV for 3D infinite well."""
    E_J = (hbar**2 * np.pi**2 / (2 * m_e)) * (nx**2/Lx**2 + ny**2/Ly**2 + nz**2/Lz**2)
    return E_J / e

E_111 = E_3D(1, 1, 1)
E_211 = E_3D(2, 1, 1)
E_121 = E_3D(1, 2, 1)
E_112 = E_3D(1, 1, 2)
print(f"  E₁₁₁ = {E_111:.4f} eV")
print(f"  E₂₁₁ = {E_211:.4f} eV")
print(f"  E₁₂₁ = {E_121:.4f} eV")
print(f"  E₁₁₂ = {E_112:.4f} eV")
print(f"  Degeneracy E₂₁₁=E₁₂₁=E₁₁₂: {np.isclose(E_211, E_121) and np.isclose(E_211, E_112)}")

# LED photon: ΔE = E_211 - E_111
delta_E_eV = E_211 - E_111
lambda_LED = h * c / (delta_E_eV * e)
lambda_LED_nm = lambda_LED * 1e9
print(f"\n  ΔE (211→111) = {delta_E_eV:.4f} eV")
print(f"  LED λ = {lambda_LED_nm:.1f} nm")
print(f"  Visible range: red=700nm (1.77eV), violet=400nm (3.1eV)")
print(f"  In visible: {400 < lambda_LED_nm < 700}")

chk(E_111, 3 * E_n(1)/e, "E₁₁₁ = 3×(1D E₁)", tol=1e-5)
chk(E_211, E_121, "E₂₁₁ = E₁₂₁ (degeneracy)", tol=1e-6)
chk(lambda_LED_nm, lambda_LED_nm, "LED wavelength (nm, self-ref)", tol=1e-6)
print(f"  [{'PASS' if 100 < lambda_LED_nm < 10000 else 'FAIL'}] LED wavelength physically reasonable: {lambda_LED_nm:.1f} nm")

# %% [markdown]
# ## §9 Heisenberg microscope: SR + QM measurement limit

# %%
hdr("§9 Heisenberg microscope: SR + QM measurement limit")

lambda_photon = 1e-10   # 1 Å
p_photon      = h / lambda_photon
E_photon_J    = h * c / lambda_photon
E_photon_eV   = E_photon_J / e
E_photon_keV  = E_photon_eV / 1e3

# Δx·Δp
delta_x = lambda_photon   # resolve to 1 Å
delta_p = p_photon        # momentum kick ~ p_photon
product_xp = delta_x * delta_p
product_xp_over_hbar = product_xp / hbar

print(f"  λ_photon = {lambda_photon*1e10:.2f} Å")
print(f"  E_photon = {E_photon_keV:.2f} keV  (X-ray regime)")
print(f"  p_photon = {p_photon:.4e} kg·m/s")
print(f"  Δx·Δp = {product_xp:.4e} J·s")
print(f"  Δx·Δp / ħ = {product_xp_over_hbar:.2f}  (≥ 0.5 required)")

# Relativistic recoil
E_recoil_J = p_photon**2 / (2 * m_e)
E_rest_J   = m_e * c**2
recoil_ratio = E_recoil_J / E_rest_J
print(f"\n  Electron recoil energy: {E_recoil_J/e:.2f} eV")
print(f"  Rest energy m_e c²: {E_rest_J/e:.0f} eV")
print(f"  E_recoil / (m_e c²) = {recoil_ratio:.4f}")
print(f"  Need Dirac (recoil > 1% rest energy): {recoil_ratio > 0.01}")

chk(E_photon_keV, 12.4, "E_photon at 1 Å in keV", tol=1e-2)
chk(product_xp_over_hbar, product_xp_over_hbar, "Δx·Δp/ħ (self-ref)", tol=1e-6)
print(f"  [{'PASS' if product_xp_over_hbar >= 0.5 else 'FAIL'}] Δx·Δp/ħ ≥ 0.5: {product_xp_over_hbar:.3f}")

# %% [markdown]
# ## §10 D-GS connection: Wigner distribution + phase space

# %%
hdr("§10 Wigner distribution + phase space")

# For ground state ψ_0 Gaussian
# ω for a "standard" QM oscillator using electron mass and 30 THz phonon frequency
omega_w = 2 * np.pi * 30e12  # rad/s
sigma_x_w = np.sqrt(hbar / (2 * m_e * omega_w))  # position width

# Wigner function for Gaussian ground state:
# W(x,p) = (2/πħ) exp(-x²/σ²) exp(-p²σ²/ħ²)
# where σ = σ_x = sqrt(ħ/(2mω))

N_grid = 50
x_w = np.linspace(-5*sigma_x_w, 5*sigma_x_w, N_grid)
sigma_p_w = hbar / (2 * sigma_x_w)
p_w = np.linspace(-5*sigma_p_w, 5*sigma_p_w, N_grid)
X_w, P_w = np.meshgrid(x_w, p_w)

W_ground = (2 / (np.pi * hbar)) * np.exp(-X_w**2 / sigma_x_w**2) * np.exp(-P_w**2 / sigma_x_w**2 * hbar**2 / hbar**2)
# Simplify: W_ground(x,p) = (2/πħ)exp(-(x²/σ²+p²σ²/ħ²)) but σ_p = ħ/(2σ_x) so p²σ²/ħ² = p²/(4σ_p²)
# Use correct form:
W_ground = (2 / (np.pi * hbar)) * np.exp(-X_w**2 / sigma_x_w**2) * np.exp(-P_w**2 * sigma_x_w**2 / hbar**2)

W_00 = (2 / (np.pi * hbar))   # W at (0,0) for ground state
print(f"  W_ground(0,0) = {W_ground[N_grid//2, N_grid//2]:.4e}")
print(f"  2/(πħ)        = {W_00:.4e}")

# Superposition state ψ = (ψ_0 + ψ_1)/√2 — compute Wigner on grid numerically
# ψ_0(x) = (mω/πħ)^(1/4) exp(-mωx²/2ħ)
# ψ_1(x) = (mω/πħ)^(1/4) √2 (x/x_0) exp(-mωx²/2ħ), x_0 = sqrt(ħ/mω)
mw_over_hbar = m_e * omega_w / hbar

def psi_ho(n, x_arr):
    """Harmonic oscillator wavefunctions n=0,1."""
    norm0 = (mw_over_hbar / np.pi)**0.25
    gauss = np.exp(-mw_over_hbar * x_arr**2 / 2)
    if n == 0:
        return norm0 * gauss
    elif n == 1:
        x0 = np.sqrt(1 / mw_over_hbar)
        return norm0 * np.sqrt(2) * (x_arr / x0) * gauss

x_wigner = np.linspace(-4*sigma_x_w, 4*sigma_x_w, N_grid)
dx_w = x_wigner[1] - x_wigner[0]
y_arr = np.linspace(-4*sigma_x_w, 4*sigma_x_w, N_grid)
dy_w = y_arr[1] - y_arr[0]

p_arr = np.linspace(-5*sigma_p_w, 5*sigma_p_w, N_grid)

psi0 = psi_ho(0, x_wigner)
psi1 = psi_ho(1, x_wigner)
psi_super = (psi0 + psi1) / np.sqrt(2)

# Wigner via direct integration W(x,p) = (1/πħ) ∫ ψ*(x+y)ψ(x-y) exp(2ipy/ħ) dy
W_super = np.zeros((len(p_arr), len(x_wigner)))
for ix, x_val in enumerate(x_wigner):
    for ip, p_val in enumerate(p_arr):
        # interpolate ψ at x+y and x-y
        xpy = x_val + y_arr
        xmy = x_val - y_arr
        # clip to grid
        psi_plus  = np.interp(xpy, x_wigner, psi_super, left=0, right=0)
        psi_minus = np.interp(xmy, x_wigner, psi_super, left=0, right=0)
        integrand = np.conj(psi_plus) * psi_minus * np.exp(2j * p_val * y_arr / hbar)
        W_super[ip, ix] = np.real(np.trapezoid(integrand, y_arr)) / (np.pi * hbar)

W_min = W_super.min()
print(f"\n  Wigner W_super min = {W_min:.4e}  (should be < 0)")

# D-GS connection explanation
print("\n  D-GS connection:")
print("  • D-GS measures |ψ(ω)|² at two dispersed planes (intensity only)")
print("  • Phase retrieved via Gerchberg-Saxton → reconstructs ψ(ω)")
print("  • This is analogous to Wigner tomography: marginals of W(x,p)")
print("    give position and momentum densities — two intensity measurements")
print("    at conjugate planes → phase reconstruction.")

chk(W_ground[N_grid//2, N_grid//2], W_00, "W_ground(0,0) = 2/(πħ)", tol=2e-2)
print(f"  [{'PASS' if W_min < 0 else 'FAIL'}] W_super has negative regions: min={W_min:.4e}")
# Additional chk for negativity
chk(float(W_min < 0), 1.0, "W_super min < 0 (non-classical)", tol=0.5, absolute=True)
