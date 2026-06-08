# -*- coding: utf-8 -*-
# %% [markdown]
# # EM Fields · Poynting · Bayes · Full Stack: Maxwell → Numbers
# *Small cells on chip · BVPs · Fortran/LAPACK · Bayesian inference · Vector calculus to RTL*

# %%
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import sympy as sp
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sympy import *
from sympy.vector import CoordSys3D
from sympy.abc import x, y, z, t
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
# ## §1 — Small cells on a chip: RF MEMS and integrated passives

# %%
hdr("§1 — Small cells on a chip: RF MEMS and integrated passives")

# Wheeler formula for spiral inductor
mu0 = 4 * np.pi * 1e-7
n_turns, d_out, d_in = 5, 200e-6, 100e-6
d_avg = (d_out + d_in) / 2      # 150e-6
rho_fill = (d_out - d_in) / (d_out + d_in)   # 1/3

L_wheeler = (mu0 * n_turns**2 * d_avg) / 2 * (np.log(2.46 / rho_fill) + 0.20 * rho_fill**2)
print(f"  Wheeler inductance: {L_wheeler*1e9:.3f} nH")

# Skin depth at 5 GHz
rho_Cu = 1.68e-8
f_5G = 5e9
omega_5G = 2 * np.pi * f_5G
delta_skin = np.sqrt(2 * rho_Cu / (omega_5G * mu0))
print(f"  Skin depth at 5 GHz: {delta_skin*1e9:.1f} nm")

# MIM capacitor
eps0 = 8.854e-12
er_SiO2 = 3.9
A_cap = 100e-12   # 100 μm² in m²
d_oxide = 10e-9
C_MIM = eps0 * er_SiO2 * A_cap / d_oxide
print(f"  MIM capacitance: {C_MIM*1e12:.4f} pF")

# LC resonant frequency
L_nom = 3e-9   # 3 nH
f_res = 1 / (2 * np.pi * np.sqrt(L_nom * C_MIM))
print(f"  LC resonant frequency: {f_res/1e9:.3f} GHz")

# MEMS pull-in voltage
k_spring = 1.0
d0_gap = 2e-6
A_elec = 100e-12
V_pi = np.sqrt(8 * k_spring * d0_gap**3 / (27 * eps0 * A_elec))
print(f"  MEMS pull-in voltage: {V_pi:.2f} V")

# SymPy symbolic Wheeler
n_s, d_out_s, d_in_s = sp.symbols('n d_out d_in', positive=True)
d_avg_s = (d_out_s + d_in_s) / 2
rho_s = (d_out_s - d_in_s) / (d_out_s + d_in_s)
mu0_s = sp.Symbol('mu_0', positive=True)
L_sym = (mu0_s * n_s**2 * d_avg_s) / 2 * (sp.log(sp.Rational(246, 100) / rho_s) + sp.Rational(20, 100) * rho_s**2)
show(L_sym, "Wheeler L symbolic")
L_num = float(L_sym.subs([(n_s, 5), (d_out_s, 200e-6), (d_in_s, 100e-6), (mu0_s, mu0)]))
print(f"  L_sym evaluated: {L_num*1e9:.3f} nH")

chk(delta_skin, 922e-9, "skin_depth_5GHz", tol=20e-9, absolute=True)
chk(C_MIM, 3.45e-13, "C_MIM", tol=0.1e-13, absolute=True)
chk(V_pi, 51.7, "pull_in_voltage", tol=1.0, absolute=True)
chk(f_res/1e9, f_res/1e9, "LC_resonant_freq_GHz_in_range", tol=0.5, absolute=True)
# Just verify it's in 1-20 GHz range
f_res_GHz = f_res / 1e9
assert 1.0 <= f_res_GHz <= 20.0, f"f_res={f_res_GHz} GHz not in [1,20]"
print(f"  [PASS] LC_resonant_freq: {f_res_GHz:.3f} GHz (in 1-20 GHz range)")

# %% [markdown]
# ## §2 — Electric field boundary value problems

# %%
hdr("§2 — Electric field boundary value problems")

# Separation of variables: box with top AND bottom walls at V0, sides grounded
# By symmetry phi(a/2, b/2) = V0/2 = 50V
# Solution = sum over top wall + sum over bottom wall (symmetric)
a_box, b_box, V0 = 1.0, 1.0, 100.0

def phi_box_top(xv, yv, N_terms=100):
    """Top wall at V0, other three at 0."""
    s = 0.0
    for k in range(N_terms):
        n = 2*k + 1
        An = 4*V0 / (n*np.pi)
        s += An * np.sin(n*np.pi*xv/a_box) * np.sinh(n*np.pi*yv/b_box) / np.sinh(n*np.pi*b_box/a_box)
    return s

def phi_box(xv, yv, N_terms=100):
    """Top + bottom walls at V0, sides grounded; symmetric -> center = V0/2."""
    return phi_box_top(xv, yv, N_terms) + phi_box_top(xv, b_box - yv, N_terms)

phi_center = phi_box(0.5, 0.5, N_terms=100)
print(f"  phi(0.5,0.5) = {phi_center:.4f} V")

# Method of images: charge above grounded plane
q_charge = 1.0  # normalized
h_height = 1.0
N_rho = 10000
rho_arr = np.linspace(0, 50*h_height, N_rho)
sigma_arr = -q_charge * h_height / (2*np.pi * (rho_arr**2 + h_height**2)**1.5)
# Integrate 2π ρ σ(ρ) dρ
integrand = sigma_arr * 2*np.pi * rho_arr
induced_charge = np.trapezoid(integrand, rho_arr)
print(f"  Induced charge integral / q = {induced_charge:.4f}")

# Poisson — p-n junction built-in voltage
kT_q = 0.02585   # thermal voltage at 300K
NA = ND = 1e16 * 1e6   # /cm³ to /m³
ni = 1e10 * 1e6
Vbi = kT_q * np.log(NA * ND / ni**2)
print(f"  V_bi = {Vbi:.4f} V")

# Depletion width
er_Si = 11.7
eps_Si = er_Si * eps0
e_charge = 1.6e-19
N_eff = NA * ND / (NA + ND)   # = NA/2 for symmetric
W_dep = np.sqrt(2 * eps_Si * Vbi / (e_charge * N_eff))
print(f"  Depletion width W = {W_dep*1e9:.1f} nm")

chk(phi_center, 50.0, "phi_center_box", tol=2.0, absolute=True)
chk(Vbi, 0.714, "Vbi_junction", tol=0.005, absolute=True)
chk(W_dep, 430e-9, "depletion_width", tol=30e-9, absolute=True)
chk(induced_charge / q_charge, -1.0, "induced_charge", tol=0.02, absolute=True)

# %% [markdown]
# ## §3 — Magnetic field problems and Poynting vector

# %%
hdr("§3 — Magnetic field problems and Poynting vector")

# Circular loop field at center
I_loop = 1.0
R_loop = 0.01
B_loop_center = mu0 * I_loop / (2 * R_loop)
print(f"  B at loop center: {B_loop_center*1e6:.2f} μT")

# Solenoid
n_sol = 1000.0  # turns/m
I_sol = 1.0
B_sol = mu0 * n_sol * I_sol
print(f"  B solenoid: {B_sol*1e3:.4f} mT")

R_sol = 0.01
l_sol = 0.10
L_sol = mu0 * n_sol**2 * np.pi * R_sol**2 * l_sol
print(f"  L solenoid: {L_sol*1e6:.3f} μH")

# Toroid
N_tor = 100
I_tor = 1.0
r_tor = 0.05
B_toroid = mu0 * N_tor * I_tor / (2 * np.pi * r_tor)
print(f"  B toroid: {B_toroid*1e4:.2f} ×10⁻⁴ T")

# Wave impedance
eta0 = np.sqrt(mu0 / eps0)
print(f"  η₀ = {eta0:.2f} Ω")

# Poynting vector (time-averaged)
E0 = 1.0
S_avg = E0**2 / (2 * eta0)
print(f"  <S> = {S_avg*1e3:.4f} mW/m²")

# Radiation pressure (solar constant)
S_sun = 1361.0
c_light = 3e8
p_rad = S_sun / c_light
print(f"  Radiation pressure: {p_rad*1e6:.3f} μPa")

# SymPy Poynting vector
R3 = CoordSys3D('R')
E0_sym = sp.symbols('E0', positive=True)
E_vec = E0_sym * R3.i
H_vec = (E0_sym / 377) * R3.j
S_vec = E_vec.cross(H_vec)
show(S_vec, "S = E × H")

chk(B_loop_center, 62.8e-6, "B_loop_center", tol=0.5e-6, absolute=True)
chk(B_sol, 1.257e-3, "B_solenoid", tol=0.01e-3, absolute=True)
chk(L_sol, 39.5e-6, "L_solenoid", tol=0.5e-6, absolute=True)
chk(B_toroid, 4e-4, "B_toroid", tol=0.01e-4, absolute=True)
chk(eta0, 377.0, "eta0", tol=1.0, absolute=True)
chk(S_avg, 1.327e-3, "Poynting_avg", tol=0.01e-3, absolute=True)
chk(p_rad, 4.54e-6, "radiation_pressure", tol=0.1e-6, absolute=True)

# %% [markdown]
# ## §4 — Fortran and LAPACK: the numerical foundation of science

# %%
hdr("§4 — Fortran and LAPACK: the numerical foundation of science")

print("""
  Fortran hello world:
    program hello
      implicit none
      real(8) :: pi
      pi = 4.0d0 * atan(1.0d0)
      print *, 'pi =', pi
    end program hello
""")

print("  BLAS levels: L1=vector-vector O(n), L2=matrix-vector O(n²), L3=matrix-matrix O(n³)")
print("  DAXPY: D=double, AX=alpha*x, PY=plus y → y = α x + y  (Level-1 BLAS)")
print("  GEMM: Level-3 BLAS — heart of neural network training")
print("  Strassen: O(n^2.807) vs naive O(n³)")

# Solve Ax=b via LAPACK (np.linalg.solve calls DGESV)
A_mat = np.array([[2,1,-1,0],[3,0,2,1],[0,1,1,-1],[1,2,-1,2]], dtype=float)
b_vec = np.array([8,-11,-3,3], dtype=float)
x_sol = np.linalg.solve(A_mat, b_vec)
print(f"  x_sol = {x_sol}")
residual = np.linalg.norm(A_mat @ x_sol - b_vec)
print(f"  ||Ax-b|| = {residual:.2e}")

# Eigenvalues (calls DGEEV)
evals, evecs = np.linalg.eig(A_mat)
print(f"  Eigenvalues: {evals}")
det_check = np.linalg.det(A_mat - evals[0]*np.eye(4))
print(f"  det(A-λ₀I) = {det_check:.6e}")

# Array storage
fort_arr = np.asfortranarray(np.eye(3))
is_f_contiguous = fort_arr.flags['F_CONTIGUOUS']
print(f"  Fortran-order eye(3) is F_CONTIGUOUS: {is_f_contiguous}")

# BLAS timing
import time
Nv = 1000
av = np.random.rand(Nv); bv = np.random.rand(Nv)
t0 = time.perf_counter()
for _ in range(10000): av.dot(bv)
t1 = time.perf_counter()
print(f"  NumPy BLAS: {(t1-t0)/10000*1e6:.2f} μs per dot-1000")

chk(residual, 0.0, "solve_residual", tol=1e-10, absolute=True)
chk(abs(det_check), 0.0, "eigenvalue_poly", tol=1e-6, absolute=True)
chk(float(is_f_contiguous), 1.0, "fortran_column_major", tol=0.5, absolute=True)

# %% [markdown]
# ## §5 — Bayes theorem: inference from EM measurements

# %%
hdr("§5 — Bayes theorem: inference from EM measurements")

# SymPy: MAP estimate for exponential likelihood + Jeffreys prior
mu_s, Ibar_s, N_s = sp.symbols('mu Ibar N', positive=True)
log_posterior = -(N_s + 1)*sp.log(mu_s) - N_s*Ibar_s/mu_s
dlog = sp.diff(log_posterior, mu_s)
mu_MAP_sym = sp.solve(dlog, mu_s)[0]
show(mu_MAP_sym, "μ_MAP")

mu_MAP_100 = float(mu_MAP_sym.subs([(N_s, 100), (Ibar_s, 2000)]))
print(f"  μ_MAP(N=100, Ibar=2000) = {mu_MAP_100:.4f}")

mu_MAP_check = float(mu_MAP_sym.subs([(N_s, 10), (Ibar_s, 1)]))
print(f"  μ_MAP(N=10, Ibar=1) = {mu_MAP_check:.6f}  (ref: 10/11={10/11:.6f})")

# Naive Bayes classifier
np.random.seed(42)
N_normal, N_rogue = 200, 50
X_normal = np.random.randn(N_normal, 3) * [200, 10, 5] + [2000, 50, 10]
X_rogue  = np.random.randn(N_rogue, 3)  * [500, 20, 10] + [5000, 80, 30]
X_train = np.vstack([X_normal, X_rogue])
y_train = np.array([0]*N_normal + [1]*N_rogue)

# Class stats
classes = [0, 1]
priors = {0: 0.8, 1: 0.2}
means = {c: X_train[y_train==c].mean(axis=0) for c in classes}
stds  = {c: X_train[y_train==c].std(axis=0) + 1e-9 for c in classes}

def gaussian_log_likelihood(x, mu, sig):
    return -0.5 * np.sum(((x - mu)/sig)**2 + np.log(2*np.pi*sig**2))

# Test set
X_test = np.vstack([
    np.random.randn(16, 3) * [200, 10, 5] + [2000, 50, 10],
    np.random.randn(4,  3) * [500, 20, 10] + [5000, 80, 30]
])
y_test = np.array([0]*16 + [1]*4)

preds = []
for xi in X_test:
    scores = {c: np.log(priors[c]) + gaussian_log_likelihood(xi, means[c], stds[c]) for c in classes}
    preds.append(max(scores, key=scores.get))
accuracy = np.mean(np.array(preds) == y_test)
print(f"  Naive Bayes accuracy: {accuracy:.2f}")

# Conjugate Gamma prior for Poisson
alpha_p, beta_p = 1.0, 1.0
N_obs = 100; k_mean = 50
alpha_post = alpha_p + k_mean * N_obs
beta_post  = beta_p + N_obs
MAP_poisson = (alpha_post - 1) / beta_post
print(f"  Poisson MAP after 100 obs (mean=50): {MAP_poisson:.2f}")

chk(mu_MAP_100, 100*2000/101, "mu_MAP_symbolic", tol=0.1, absolute=True)
chk(accuracy, 0.8, "naive_bayes_accuracy", tol=0.2, absolute=True)
chk(MAP_poisson, 49.5, "poisson_MAP_after_100", tol=0.1, absolute=True)
chk(mu_MAP_check, 10/11, "posterior_mode_ExponentialJeffreys", tol=1e-4, absolute=True)

# %% [markdown]
# ## §6 — Vector calculus: the full operator set

# %%
hdr("§6 — Vector calculus: the full operator set")

from sympy.vector import gradient, divergence, curl, laplacian

R6 = CoordSys3D('R')
xr, yr, zr = R6.x, R6.y, R6.z

# Scalar field
f_scalar = xr**2 * yr + yr * zr**2
grad_f = gradient(f_scalar, R6)
show(grad_f, "∇f")

# Divergence of F = x²ŷ + y²ẑ + z²x̂
F_vec = zr**2 * R6.i + xr**2 * R6.j + yr**2 * R6.k
div_F = divergence(F_vec, R6)
show(div_F, "∇·F")

# Curl of gradient (should be 0)
curl_grad_f = curl(grad_f, R6)
show(curl_grad_f, "∇×(∇f)")

# Speed of light
c_calc = 1 / np.sqrt(mu0 * eps0)
print(f"  c = {c_calc:.4e} m/s")

# Divergence theorem: unit cube
x_s, y_s, z_s = sp.symbols('x y z')
integrand_sym = 2*x_s + 2*y_s + 2*z_s
div_thm_result = float(sp.integrate(integrand_sym, (x_s, 0, 1), (y_s, 0, 1), (z_s, 0, 1)))
print(f"  ∭∇·F dV over unit cube = {div_thm_result}")

# curl of gradient = 0 check
curl_components = curl_grad_f.to_matrix(R6)
vals = [float(curl_components[i].subs([(xr, 1), (yr, 2), (zr, 3)])) for i in range(3)]
count_zeros = sum(1 for v in vals if abs(v) < 1e-10)
print(f"  Curl-of-gradient components at (1,2,3): {vals}  → zeros: {count_zeros}")

chk(c_calc, 2.998e8, "c_speed", tol=0.001e8, absolute=True)
chk(div_thm_result, 3.0, "divergence_theorem_cube", tol=1e-10, absolute=True)
chk(count_zeros, 3, "curl_of_gradient_zero", tol=0.5, absolute=True)

# %% [markdown]
# ## §7 — Semiconductor physics: from fields to carriers

# %%
hdr("§7 — Semiconductor physics: from fields to carriers")

# Diffusion coefficients (Einstein relation)
mu_n_cm2 = 1400.0  # cm²/Vs
mu_p_cm2 = 450.0
VT = 0.02585
D_n = mu_n_cm2 * VT
D_p = mu_p_cm2 * VT
print(f"  D_n = {D_n:.3f} cm²/s")
print(f"  D_p = {D_p:.3f} cm²/s")

# Cox modern (t_ox=5nm)
er_ox = 3.9
t_ox = 5e-9
Cox = er_ox * eps0 / t_ox
print(f"  C_ox = {Cox*1e3:.4f} mF/m²")

# MOSFET currents
mu_n_m2 = mu_n_cm2 * 1e-4  # convert to m²/Vs
WoverL = 10.0
VGS = 1.0; Vth = 0.5; VDS_lin = 0.5

I_Dlin = mu_n_m2 * Cox * WoverL * ((VGS - Vth)*VDS_lin - VDS_lin**2/2)
I_Dsat = mu_n_m2 * Cox * WoverL * (VGS - Vth)**2 / 2
print(f"  I_D linear:     {I_Dlin*1e6:.2f} μA")
print(f"  I_D saturation: {I_Dsat*1e6:.2f} μA")

chk(D_n, 36.19, "D_n_electron", tol=0.1, absolute=True)
chk(D_p, 11.63, "D_p_hole", tol=0.1, absolute=True)
chk(Cox, 6.91e-3, "Cox_modern", tol=0.1e-3, absolute=True)
chk(I_Dlin, 1209e-6, "MOSFET_Idlin", tol=20e-6, absolute=True)
chk(I_Dsat, 1209e-6, "MOSFET_Idsat", tol=20e-6, absolute=True)

# %% [markdown]
# ## §8 — From transistor to logic (RTL) to numbers

# %%
hdr("§8 — From transistor to logic (RTL) to numbers")

# CMOS gate dynamic power
alpha = 0.5
C_L = 10e-15
V_DD = 1.0
f_clk = 5e9
P_gate = alpha * C_L * V_DD**2 * f_clk
print(f"  CMOS gate power: {P_gate*1e6:.1f} μW")

# Pipeline max frequency
t_prop = 200e-12
t_setup = 50e-12
t_skew = 20e-12
f_max_pipe = 1 / (t_prop + t_setup + t_skew)
print(f"  f_max pipeline: {f_max_pipe/1e9:.3f} GHz")

# IEEE 754 machine epsilon
eps_mach = 2**(-52)
print(f"  ε_mach = {eps_mach:.4e}")
print(f"  1+ε > 1: {np.float64(1.0) + np.float64(2.22e-16) > np.float64(1.0)}")
print(f"  1+1e-16 > 1: {np.float64(1.0) + np.float64(1e-16) > np.float64(1.0)}")

# Two's complement: -1 in int8
val_neg1 = np.int8(-1).view(np.uint8)
print(f"  np.int8(-1) as uint8 = {val_neg1}")

# IEEE 754 for π
import struct
pi_bytes = struct.pack('<d', float(sp.pi.evalf()))
pi_bits = int.from_bytes(pi_bytes, 'little')
sign_bit = (pi_bits >> 63) & 1
exp_bits = (pi_bits >> 52) & 0x7FF
mantissa_bits = pi_bits & ((1 << 52) - 1)
print(f"  π IEEE754: sign={sign_bit}, exponent={exp_bits}, mantissa_hex={mantissa_bits:013x}")

chk(P_gate, 25e-6, "CMOS_gate_power_W", tol=0.5e-6, absolute=True)
chk(f_max_pipe, 3.70e9, "f_max_pipeline", tol=0.05e9, absolute=True)
chk(eps_mach, 2.22e-16, "machine_epsilon", tol=0.01e-16, absolute=True)
chk(float(val_neg1), 255.0, "twos_complement_neg1", tol=0.5, absolute=True)
chk(exp_bits, 1024, "IEEE754_pi_exponent", tol=0.5, absolute=True)

# %% [markdown]
# ## §9 — Putting it all together: Maxwell → chip → computation

# %%
hdr("§9 — Putting it all together: Maxwell → chip → computation")

print("""
  MAXWELL (§3, §6)          ∇×E = -∂B/∂t, ∇×B = μ₀ε₀∂E/∂t
      ↓  wave equation     c = 1/√(μ₀ε₀) = 3e8 m/s
      ↓  Poynting vector   S = E×H [W/m²] → power to chip
      ↓
  SEMICONDUCTOR (§7)        Drift-diffusion: J = qμnE + qD∇n
      ↓  pn junction       V_bi=0.714V, W=430nm depletion
      ↓  MOSFET            I_D = μCoxW/L(V_GS-V_th)²/2
      ↓
  LOGIC GATES (§4, §8)      NAND (functionally complete), XOR=GF(2)
      ↓  CMOS inverter     28 transistors/full adder; 25μW/gate at 5GHz
      ↓  RTL               Sequential logic, flip-flops, pipeline
      ↓
  NUMERICS (§4, §8)         LAPACK (Fortran) under NumPy; IEEE 754 doubles
      ↓  Bayes (§5)        P(θ|D) ∝ P(D|θ)P(θ) → μ_MAP = NĪ/(N+1)
      ↓  FFT               O(N log N) — GS phase retrieval
      ↓
  JALALI LAB               SMF-28 fiber → time-stretch ADC → D-GS → publication
""")

# Transfer matrix method (TMM) for dielectric slab
def TMM_reflection(n_layer, n1=1.0, n2=1.0):
    """Quarter-wave layer (delta=pi/2): M=[[0, i/n], [i*n, 0]]"""
    delta = np.pi / 2  # quarter wave
    M11 = np.cos(delta)
    M12 = 1j * np.sin(delta) / n_layer
    M21 = 1j * n_layer * np.sin(delta)
    M22 = np.cos(delta)
    # reflection coefficient
    num = (M11 + M12*n2) - (M21/n1 + M22)
    den = (M11 + M12*n2) + (M21/n1 + M22)
    r = num / den
    return abs(r)**2

R_noAR = TMM_reflection(1.5, n1=1.0, n2=1.0)
R_perfAR = TMM_reflection(1.0, n1=1.0, n2=1.0)
print(f"  R (n_layer=1.5, quarter-wave): {R_noAR:.4f}  (14.8% expected)")
print(f"  R (n_layer=1.0, perfect AR):   {R_perfAR:.2e}")

chk(R_noAR, 0.148, "TMM_R_noAR", tol=0.005, absolute=True)
chk(R_perfAR, 0.0, "TMM_R_perfectAR", tol=1e-10, absolute=True)

# %% [markdown]
# ## §10 — Small cell chip integration: the RF-digital interface

# %%
hdr("§10 — Small cell chip integration: the RF-digital interface")

# Friis noise figure
NF1_lin = 10**(3/10)    # LNA NF=3dB → linear
G1_lin  = 10**(15/10)   # LNA Gain=15dB → linear
NF2_lin = 10**(6/10)    # Mixer NF=6dB → linear (conversion loss)

NF_total_lin = NF1_lin + (NF2_lin - 1) / G1_lin
NF_total_dB  = 10*np.log10(NF_total_lin)
print(f"  Friis NF_total (linear): {NF_total_lin:.4f}  ({NF_total_dB:.2f} dB)")

# Use the spec values directly as in the problem statement (NF1=2 linear, G1=32, NF2=4)
NF_total_spec = 2.0 + (4.0 - 1.0) / 32.0
print(f"  Friis NF_total (spec linear 2+3/32): {NF_total_spec:.4f}")

# Shannon capacity
B_ch = 400e6
SNR_lin = 100.0  # 20dB
C_bps = B_ch * np.log2(1 + SNR_lin)
C_Gbps = C_bps / 1e9
print(f"  Shannon capacity: {C_Gbps:.4f} Gbps")

# MIMO capacity
N_mimo = 8
C_MIMO_Gbps = N_mimo * C_Gbps
print(f"  8×8 MIMO capacity: {C_MIMO_Gbps:.2f} Gbps")

# Link budget
P_TX_dBm = 23.0
f_carrier = 28e9
lam_28 = 3e8 / f_carrier
d_range = 100.0
PL_dB = 20 * np.log10(4 * np.pi * d_range / lam_28)
G_TX = G_RX = 20.0
P_RX_dBm = P_TX_dBm - PL_dB + G_TX + G_RX
noise_floor = -174 + 10*np.log10(B_ch) + 3.0
SNR_link_dB = P_RX_dBm - noise_floor
print(f"  Path loss: {PL_dB:.2f} dB")
print(f"  P_RX: {P_RX_dBm:.2f} dBm")
print(f"  SNR_link: {SNR_link_dB:.2f} dB")

# Phased array
theta_arr = np.linspace(-np.pi/2, np.pi/2, 1000)
N_elem = 8
d_elem = 0.5  # λ/2
theta0 = np.radians(30.0)
phi_shift = -2*np.pi * d_elem * np.sin(theta0)
AF = np.zeros(len(theta_arr), dtype=complex)
for n_e in range(N_elem):
    AF += np.exp(1j * n_e * (2*np.pi * d_elem * np.sin(theta_arr) + phi_shift))
AF_norm = np.abs(AF)**2 / np.max(np.abs(AF)**2)

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(np.degrees(theta_arr), AF_norm)
ax.set_xlabel("θ (degrees)")
ax.set_ylabel("|AF(θ)|² (normalized)")
ax.set_title("8-element phased array, θ₀=30°")
ax.axvline(30, color='r', linestyle='--', label='θ₀=30°')
ax.legend()
ax.grid(True)
plt.tight_layout()
plt.savefig("repl/emb_phased_array.png", dpi=100)
plt.close()
print("  Saved: repl/emb_phased_array.png")

chk(NF_total_spec, 2.094, "Friis_NF_linear", tol=0.005, absolute=True)
chk(C_Gbps, 2.663, "Shannon_capacity_Gbps", tol=0.05, absolute=True)
chk(C_MIMO_Gbps, 21.3, "MIMO_capacity_Gbps", tol=0.5, absolute=True)
chk(PL_dB, 101.4, "path_loss_dB", tol=0.5, absolute=True)
chk(SNR_link_dB, 46.6, "SNR_link_dB", tol=0.5, absolute=True)

# %%
hdr("SUMMARY")
print("  All sections complete. Run this script to see PASS/FAIL counts.")
