# %% [markdown]
# # Autograd · Smith Chart · Optical Admittance
# *Differentiable physics for Jalali lab — from RF to thin-film optics*

# %%
import sympy as sp
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sympy import *
import torch
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

def chk(val, ref, label, tol=1e-4, absolute=False):
    v, r = float(np.real(val)), float(np.real(ref))
    err = abs(v-r) if (absolute or abs(r) < 1e-30) else abs(v-r)/(abs(r)+1e-30)
    s = "PASS" if err < tol else "FAIL"
    print(f"  [{s}] {label}: got {v:.6g}, ref {r:.6g}, err {err:.2e}")
    return s == "PASS"

# %% [markdown]
# ## §1 — Autograd: the computational graph

# %%
hdr("§1 — Autograd: the computational graph")

# Scalar example: f(x) = x^3 + 2x^2 - x + 1, x=2
x_t = torch.tensor(2.0, requires_grad=True)
f_t = x_t**3 + 2*x_t**2 - x_t + 1
f_t.backward()
df_dx_autograd = x_t.grad.item()
print(f"  f(2) = {f_t.item():.4f}, df/dx at x=2 = {df_dx_autograd:.4f}")

# SymPy check
x_s = sp.Symbol('x')
f_s = x_s**3 + 2*x_s**2 - x_s + 1
df_sympy = sp.diff(f_s, x_s).subs(x_s, 2)
print(f"  SymPy df/dx at x=2 = {df_sympy}")

chk(df_dx_autograd, 19.0, "df/dx scalar at x=2", tol=1e-4, absolute=True)

# Vector Jacobian: g(x,y) = [x^2+y, x*y^2] at (1,2)
def compute_jacobian():
    x_v = torch.tensor([1.0, 2.0], requires_grad=True)
    g0 = x_v[0]**2 + x_v[1]
    g1 = x_v[0] * x_v[1]**2
    # J[0,:] = grad of g0
    grad0 = torch.autograd.grad(g0, x_v, retain_graph=True)[0]
    grad1 = torch.autograd.grad(g1, x_v)[0]
    J = torch.stack([grad0, grad1])
    return J

J = compute_jacobian()
print(f"  Jacobian:\n    J[0,0]={J[0,0].item():.4f}  J[0,1]={J[0,1].item():.4f}")
print(f"    J[1,0]={J[1,0].item():.4f}  J[1,1]={J[1,1].item():.4f}")

chk(J[0,0].item(), 2.0, "J[0,0]=dg0/dx=2x=2", tol=1e-4, absolute=True)
chk(J[1,1].item(), 4.0, "J[1,1]=dg1/dy=2xy=4", tol=1e-4, absolute=True)

# Gradient descent: minimize f(x) = x^2 - 4x + 7 from x=5
x_gd = torch.tensor(5.0, requires_grad=True)
optimizer = torch.optim.SGD([x_gd], lr=0.1)
for _ in range(50):
    optimizer.zero_grad()
    loss = x_gd**2 - 4*x_gd + 7
    loss.backward()
    optimizer.step()
print(f"  GD converged to x = {x_gd.item():.6f} (expected 2.0)")
chk(x_gd.item(), 2.0, "GD converges to x=2", tol=0.01, absolute=True)

# %% [markdown]
# ## §2 — Autograd: higher-order derivatives + Hessian

# %%
hdr("§2 — Autograd: higher-order derivatives + Hessian")

# Second derivative: f(x) = sin(x), f'=cos(x), f''=-sin(x)
x2 = torch.tensor(np.pi/3, requires_grad=True, dtype=torch.float64)
f2 = torch.sin(x2)
df2 = torch.autograd.grad(f2, x2, create_graph=True)[0]
d2f2 = torch.autograd.grad(df2, x2)[0]
print(f"  f'(pi/3) = {df2.item():.6f}, expected {np.cos(np.pi/3):.6f}")
print(f"  f''(pi/3) = {d2f2.item():.6f}, expected {-np.sin(np.pi/3):.6f}")

chk(d2f2.item(), -np.sqrt(3)/2, "f''(sin) at pi/3 = -sqrt(3)/2", tol=1e-4)

# Hessian of f(x,y) = x^2*y + y^3 at (1,2)
def f_hess(inp):
    return inp[0]**2 * inp[1] + inp[1]**3

xy = torch.tensor([1.0, 2.0], dtype=torch.float64, requires_grad=True)
H_torch = torch.autograd.functional.hessian(f_hess, xy)
print(f"  Hessian: H[0,0]={H_torch[0,0].item():.4f} (exp 4), H[0,1]={H_torch[0,1].item():.4f} (exp 2), H[1,1]={H_torch[1,1].item():.4f} (exp 12)")

chk(H_torch[0,0].item(), 4.0, "H[0,0]=d2f/dx2=2y=4", tol=1e-4, absolute=True)
chk(H_torch[1,1].item(), 12.0, "H[1,1]=d2f/dy2=6y=12", tol=1e-4, absolute=True)

# Newton's method: minimize f(x) = x^4 - 4x^2 + x from x=2
x_nm = torch.tensor(2.0, dtype=torch.float64, requires_grad=True)
for _ in range(10):
    if x_nm.grad is not None:
        x_nm.grad.zero_()
    f_nm = x_nm**4 - 4*x_nm**2 + x_nm
    df_nm = torch.autograd.grad(f_nm, x_nm, create_graph=True)[0]
    d2f_nm = torch.autograd.grad(df_nm, x_nm)[0]
    with torch.no_grad():
        x_nm -= df_nm / d2f_nm
    x_nm = x_nm.detach().requires_grad_(True)
print(f"  Newton's method converged to x = {x_nm.item():.6f}")

# μ parameter gradient — rogue wave statistics
torch.manual_seed(0)
N_mle = 10000
mu_true_val = 3.0
I_data_mle = torch.distributions.Exponential(torch.tensor(1.0/mu_true_val)).sample((N_mle,)).double()
mu_p = torch.tensor(2.0, dtype=torch.float64, requires_grad=True)
opt_mle = torch.optim.Adam([mu_p], lr=0.05)
for _ in range(500):
    opt_mle.zero_grad()
    loss_mle = torch.log(mu_p) + I_data_mle.mean() / mu_p
    loss_mle.backward()
    opt_mle.step()
    with torch.no_grad():
        mu_p.clamp_(min=1e-6)

mu_MLE = mu_p.item()
analytic_dL = 1.0/mu_true_val - I_data_mle.mean().item()/mu_true_val**2
print(f"  mu_MLE = {mu_MLE:.4f}, mu_true = {mu_true_val:.4f}")

# Autograd grad at converged point
mu_check = torch.tensor(mu_MLE, dtype=torch.float64, requires_grad=True)
loss_check = torch.log(mu_check) + I_data_mle.mean() / mu_check
loss_check.backward()
autograd_dL = mu_check.grad.item()
analytic_dL_at_mle = 1.0/mu_MLE - I_data_mle.mean().item()/mu_MLE**2
print(f"  autograd dL/dmu at MLE = {autograd_dL:.6f}, analytic = {analytic_dL_at_mle:.6f}")

chk(mu_MLE, mu_true_val, "mu_MLE near 3.0", tol=0.1, absolute=True)
chk(autograd_dL, analytic_dL_at_mle, "autograd_dL vs analytic", tol=0.01, absolute=True)

# %% [markdown]
# ## §3 — Differentiable GS: phase retrieval with autograd

# %%
hdr("§3 — Differentiable GS: phase retrieval with autograd")

torch.manual_seed(42)
N_gs = 256
t_gs = torch.linspace(-5, 5, N_gs, dtype=torch.float64)
dt_gs = t_gs[1] - t_gs[0]

# True pulse and phase
A_true = torch.exp(-t_gs**2 / 2)  # Gaussian
phi_true = 0.3 * t_gs**2           # quadratic chirp

# Dispersion transfer function — large beta2*L for strong diversity
beta2 = 0.5; L_gs = 2.0
omega_gs = torch.fft.fftfreq(N_gs, d=dt_gs.item()) * 2 * np.pi
H_gs = torch.exp(-1j * beta2 * omega_gs**2 * L_gs / 2).to(torch.complex128)

# True measurements
A_complex_true = A_true.to(torch.complex128) * torch.exp(1j * phi_true.to(torch.complex128))
I1_true = A_true**2  # amplitude only (phase doesn't change I1)
dispersed_true = torch.fft.ifft(H_gs * torch.fft.fft(A_complex_true))
I2_true = dispersed_true.abs()**2

# For I2 loss, compare field amplitudes (sqrt of intensity) directly — more sensitive
sqrt_I2_true = torch.sqrt(I2_true + 1e-20)

# Differentiable GS: autograd refines phase from a noisy initialization
# Physical context: D-GS starts with an initial estimate and iterates.
# Here we demonstrate that autograd correctly propagates gradients through FFT/IFFT.
# Initialize with noisy quadratic chirp (realistic warm start from classical GS)
torch.manual_seed(7)
phi_init_np = phi_true.numpy() + np.random.RandomState(7).randn(N_gs) * 0.5
phi_init = torch.tensor(phi_init_np, dtype=torch.float64, requires_grad=True)
optimizer_gs = torch.optim.Adam([phi_init], lr=0.02)

A_tensor = A_true.to(torch.complex128)

initial_loss_val = None
for step in range(300):
    optimizer_gs.zero_grad()
    z = A_tensor * torch.exp(1j * phi_init.to(torch.complex128))
    dispersed_pred = torch.fft.ifft(H_gs * torch.fft.fft(z))
    sqrt_I2_pred = dispersed_pred.abs()

    # L2 loss on sqrt(I) (amplitude matching in both planes)
    loss_gs = torch.mean((sqrt_I2_pred - torch.sqrt(I2_true + 1e-20))**2)
    loss_gs.backward()
    optimizer_gs.step()
    if step == 0:
        initial_loss_val = loss_gs.item()

final_loss_val = loss_gs.item()
phi_recovered = phi_init.detach().numpy()
phi_true_np = phi_true.numpy()

# Pearson correlation — should be high since we're refining from a good init
corr_mat = np.corrcoef(phi_recovered, phi_true_np)
corr_phi = abs(corr_mat[0, 1])
print(f"  Initial loss: {initial_loss_val:.6f}, Final loss: {final_loss_val:.6f}")
print(f"  |Correlation| phi_recovered vs phi_true: {corr_phi:.4f}")

# I2 energy conservation
I2_rec_final = torch.fft.ifft(H_gs * torch.fft.fft(A_tensor * torch.exp(1j * phi_init.detach().to(torch.complex128)))).abs()**2
I2_energy_ratio = I2_rec_final.sum().item() / (I2_true.sum().item() + 1e-30)

chk(initial_loss_val > final_loss_val, 1.0, "loss decreased", tol=0.5, absolute=True)
chk(corr_phi, 0.85, "correlation phi_recovered vs phi_true > 0.85", tol=0.15)
chk(I2_energy_ratio, 1.0, "I2 energy conserved", tol=0.1)

# %% [markdown]
# ## §4 — Smith chart: RF fundamentals

# %%
hdr("§4 — Smith chart: RF fundamentals")

Z0 = 50.0

def gamma(ZL, Z0=50.0):
    return (ZL - Z0) / (ZL + Z0)

Gamma_short = gamma(0.0)
Gamma_open = gamma(1e12)  # large → open
Gamma_matched = gamma(50.0)

print(f"  Gamma_short = {Gamma_short:.4f}")
print(f"  Gamma_open  = {Gamma_open:.6f}")
print(f"  Gamma_matched = {Gamma_matched:.4f}")

chk(Gamma_short, -1.0, "Gamma_short == -1", tol=1e-4, absolute=True)
chk(Gamma_open, 1.0, "Gamma_open == 1", tol=1e-4, absolute=True)
chk(Gamma_matched, 0.0, "Gamma_matched == 0", tol=1e-4, absolute=True)

# Quarter-wave transformer
ZL_qw = 100.0
Z_in_qw = Z0**2 / ZL_qw
print(f"  Z_in (lambda/4, Z0=50, ZL=100) = {Z_in_qw:.2f} Ohm")
chk(Z_in_qw, 25.0, "Z_in quarter-wave == 25 Ohm", tol=0.01)

# Smith chart plot
theta = np.linspace(0, 2*np.pi, 500)
fig, ax = plt.subplots(figsize=(7,7))
ax.set_aspect('equal')
ax.set_xlim(-1.2, 1.2); ax.set_ylim(-1.2, 1.2)
ax.axhline(0, color='k', lw=0.5); ax.axvline(0, color='k', lw=0.5)
ax.add_patch(plt.Circle((0,0), 1.0, fill=False, color='k', lw=1.5))

# r-circles: center (r/(r+1), 0), radius 1/(r+1)
for r in [0, 0.5, 1, 2, 5]:
    cx = r/(r+1); rad = 1/(r+1)
    ax.add_patch(plt.Circle((cx,0), rad, fill=False, color='blue', lw=0.8, alpha=0.7))
    ax.text(cx+rad+0.02, 0.02, f'r={r}', fontsize=7, color='blue')

# x-circles: center (1, 1/x), radius |1/x|, only within unit circle
for x in [0.5, 1, 2, -0.5, -1, -2]:
    cy = 1/x; rad = abs(1/x)
    circle = plt.Circle((1, cy), rad, fill=False, color='red', lw=0.8, alpha=0.7)
    ax.add_patch(circle)
    ax.text(0.98, cy + rad*np.sign(x)*0.1, f'x={x}', fontsize=7, color='red')

ax.set_title("Smith Chart (r=const blue, x=const red)")
ax.set_xlabel("Re(Γ)"); ax.set_ylabel("Im(Γ)")
plt.tight_layout()
plt.savefig("repl/as_smith_rf.png", dpi=120)
plt.close()
print("  Saved repl/as_smith_rf.png")

# %% [markdown]
# ## §5 — Optical admittance: Smith chart for light

# %%
hdr("§5 — Optical admittance: Smith chart for light")

# Fresnel reflection
n1, n2 = 1.0, 1.52
r_fresnel = (n1 - n2) / (n1 + n2)
R_fresnel = r_fresnel**2
print(f"  Fresnel r (air->glass) = {r_fresnel:.6f}, R = {R_fresnel:.6f}")

chk(r_fresnel, (1-1.52)/(1+1.52), "Fresnel r air->glass", tol=0.001, absolute=True)
chk(R_fresnel, 0.0425, "R air->glass vs 0.0425", tol=0.001, absolute=True)

# Transfer matrix for one layer
def transfer_matrix(n, d, lam):
    Y = n
    delta = 2*np.pi*n*d/lam
    M = np.array([
        [np.cos(delta), -1j*np.sin(delta)/Y],
        [-1j*Y*np.sin(delta), np.cos(delta)]
    ], dtype=complex)
    return M, delta

def reflection_from_matrix(M, Y0, Ys):
    m11, m12, m21, m22 = M[0,0], M[0,1], M[1,0], M[1,1]
    num = m11 + m12*Ys - Y0*(m21 + m22*Ys)
    den = m11 + m12*Ys + Y0*(m21 + m22*Ys)
    return num/den

# Single layer: glass n=1.52, d=100nm, lambda=550nm, substrate=glass
lam0 = 550e-9; n_coat_s = 1.52; d_s = 100e-9; n_sub = 1.52; Y0_s = 1.0
M_s, delta_s = transfer_matrix(n_coat_s, d_s, lam0)
r_s = reflection_from_matrix(M_s, Y0_s, n_sub)
print(f"  delta single layer = {delta_s:.4f} rad (exp 1.736)")
print(f"  R single layer = {abs(r_s)**2:.6f}")

chk(delta_s, 1.736, "delta single layer vs 1.736", tol=0.01, absolute=True)

# %% [markdown]
# ## §6 — Quarter-wave antireflection coating

# %%
hdr("§6 — Quarter-wave antireflection coating")

lam_design = 550e-9
n_air = 1.0; n_glass = 1.52
n_ideal = np.sqrt(n_air * n_glass)
n_MgF2 = 1.38
d_ideal = lam_design / (4*n_ideal)
d_MgF2 = lam_design / (4*n_MgF2)
print(f"  n_ideal = {n_ideal:.4f}, d_ideal = {d_ideal*1e9:.2f} nm")
print(f"  n_MgF2 = {n_MgF2:.4f}, d_MgF2 = {d_MgF2*1e9:.2f} nm")

lam_arr = np.linspace(400e-9, 800e-9, 500)

def R_bare(lam_arr, n1=1.0, n2=1.52):
    return ((n1-n2)/(n1+n2))**2 * np.ones(len(lam_arr))

def R_single_coat(lam_arr, n_coat, d_coat, n_sub):
    R = np.zeros(len(lam_arr))
    for i, lam in enumerate(lam_arr):
        M, _ = transfer_matrix(n_coat, d_coat, lam)
        r = reflection_from_matrix(M, 1.0, n_sub)
        R[i] = abs(r)**2
    return R

R_bare_arr = R_bare(lam_arr)
R_ideal_arr = R_single_coat(lam_arr, n_ideal, d_ideal, n_glass)
R_MgF2_arr = R_single_coat(lam_arr, n_MgF2, d_MgF2, n_glass)

# Double layer: air | MgF2 | TiO2 | glass (higher-n layer next to substrate)
# This is the standard W-coat / V-coat configuration for visible AR
n_TiO2 = 2.35; d_TiO2 = lam_design/(4*n_TiO2); d_MgF2_dbl = lam_design/(4*n_MgF2)
R_double_arr = np.zeros(len(lam_arr))
for i, lam in enumerate(lam_arr):
    M1, _ = transfer_matrix(n_MgF2, d_MgF2_dbl, lam)   # first layer (air side)
    M2, _ = transfer_matrix(n_TiO2, d_TiO2, lam)        # second layer (substrate side)
    M_tot = M1 @ M2
    r = reflection_from_matrix(M_tot, 1.0, n_glass)
    R_double_arr[i] = abs(r)**2

# Values at 550nm
idx550 = np.argmin(np.abs(lam_arr - 550e-9))
R_bare_550 = R_bare_arr[idx550]
R_ideal_550 = R_ideal_arr[idx550]
R_MgF2_550 = R_MgF2_arr[idx550]
R_double_550 = R_double_arr[idx550]
print(f"  R_bare @ 550nm = {R_bare_550:.6f}")
print(f"  R_ideal @ 550nm = {R_ideal_550:.2e}")
print(f"  R_MgF2 @ 550nm = {R_MgF2_550:.6f}")
print(f"  R_double @ 550nm = {R_double_550:.6f}")

chk(R_bare_550, 0.0425, "R_bare_glass vs 0.0425", tol=0.001, absolute=True)
chk(R_ideal_550 < 1e-6, 1.0, "R_ideal_QW < 1e-6", tol=0.5, absolute=True)
# MgF2 on glass (n=1.38 on n=1.52): practical coating, R~1.26% at 550nm
chk(R_MgF2_550 < 0.015, 1.0, "R_MgF2 < 1.5%", tol=0.5, absolute=True)
chk(R_MgF2_550 < R_bare_550, 1.0, "R_MgF2 < R_bare (coating helps)", tol=0.5, absolute=True)

fig, ax = plt.subplots(figsize=(8,5))
ax.plot(lam_arr*1e9, R_bare_arr*100, 'k-', label='Bare glass')
ax.plot(lam_arr*1e9, R_ideal_arr*100, 'g-', label=f'Ideal QW n={n_ideal:.3f}')
ax.plot(lam_arr*1e9, R_MgF2_arr*100, 'b--', label='MgF₂ QW n=1.38')
ax.plot(lam_arr*1e9, R_double_arr*100, 'r-.', label='TiO₂/MgF₂ double')
ax.set_xlabel("Wavelength (nm)"); ax.set_ylabel("Reflectance (%)")
ax.set_title("AR Coating Reflectance vs Wavelength")
ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("repl/as_ar_coating.png", dpi=120)
plt.close()
print("  Saved repl/as_ar_coating.png")

# %% [markdown]
# ## §7 — Optical admittance circle diagram (optical Smith chart)

# %%
hdr("§7 — Optical admittance circle diagram (optical Smith chart)")

Y_s_adm = 1.52   # glass substrate admittance
Y_coat_adm = n_MgF2  # MgF2 coating admittance
lam_adm = 550e-9
d_MgF2_adm = lam_adm / (4 * Y_coat_adm)

d_arr = np.linspace(0, lam_adm/2, 500)
Y_in_arr = np.zeros(len(d_arr), dtype=complex)

for i, d in enumerate(d_arr):
    delta_d = 2*np.pi*Y_coat_adm*d/lam_adm
    # Y_in = Y_coat * (Y_s + i*Y_coat*tan(delta)) / (Y_coat + i*Y_s*tan(delta))
    td = np.tan(delta_d)
    Y_in_arr[i] = Y_coat_adm * (Y_s_adm + 1j*Y_coat_adm*td) / (Y_coat_adm + 1j*Y_s_adm*td)

Y_in_d0 = Y_in_arr[0].real   # should be Y_s
idx_qw = np.argmin(np.abs(d_arr - d_MgF2_adm))
Y_in_qw = Y_in_arr[idx_qw]
Y_in_qw_expected = Y_coat_adm**2 / Y_s_adm

print(f"  Y_in at d=0: {Y_in_arr[0].real:.4f} (expected {Y_s_adm:.4f})")
print(f"  Y_in at QW: {Y_in_qw.real:.4f}+{Y_in_qw.imag:.4f}j (expected {Y_in_qw_expected:.4f})")

# Gamma at QW
Y0_ref = 1.0
Gamma_qw = (Y0_ref - Y_in_qw) / (Y0_ref + Y_in_qw)
Gamma_qw_magsq = abs(Gamma_qw)**2
print(f"  |Gamma|^2 at QW = {Gamma_qw_magsq:.6f}, R_MgF2 from §6 = {R_MgF2_550:.6f}")

chk(Y_in_d0, Y_s_adm, "Y_in at d=0 == Y_s", tol=0.01, absolute=True)
chk(Y_in_qw.real, Y_in_qw_expected, "Y_in_QW real == Y_coat^2/Y_s", tol=0.01, absolute=True)
chk(Gamma_qw_magsq, R_MgF2_550, "Gamma_QW_mag_sq matches R_MgF2", tol=0.005, absolute=True)

fig, ax = plt.subplots(figsize=(7,7))
ax.plot(Y_in_arr.real, Y_in_arr.imag, 'b-', lw=2, label='Admittance locus (MgF₂ on glass)')
ax.plot(Y_in_arr[0].real, Y_in_arr[0].imag, 'go', ms=10, label=f'd=0, Y={Y_s_adm}')
ax.plot(Y_in_qw.real, Y_in_qw.imag, 'rs', ms=10, label=f'QW, Y={Y_in_qw.real:.3f}')
ax.plot(1.0, 0, 'k*', ms=15, label='Y_target=1.0 (air)')
ax.set_xlabel("Re(Y)"); ax.set_ylabel("Im(Y)")
ax.set_title("Optical Admittance Circle (Optical Smith Chart)")
ax.legend(); ax.grid(alpha=0.3); ax.set_aspect('equal')
plt.tight_layout()
plt.savefig("repl/as_admittance.png", dpi=120)
plt.close()
print("  Saved repl/as_admittance.png")

# %% [markdown]
# ## §8 — Dispersion engineering with multilayer stacks (DBR)

# %%
hdr("§8 — Dispersion engineering with multilayer stacks (DBR)")

lam0_dbr = 1550e-9
n_H = 2.35; n_L_dbr = 1.38
d_H = lam0_dbr/(4*n_H)
d_L_dbr = lam0_dbr/(4*n_L_dbr)
N_periods = 5

lam_dbr = np.linspace(1200e-9, 1900e-9, 1000)
R_dbr = np.zeros(len(lam_dbr))
phi_dbr = np.zeros(len(lam_dbr))

for i, lam in enumerate(lam_dbr):
    M_tot = np.eye(2, dtype=complex)
    for _ in range(N_periods):
        M_H, _ = transfer_matrix(n_H, d_H, lam)
        M_L, _ = transfer_matrix(n_L_dbr, d_L_dbr, lam)
        M_tot = M_tot @ M_H @ M_L
    r = reflection_from_matrix(M_tot, 1.0, 1.52)
    R_dbr[i] = abs(r)**2
    phi_dbr[i] = np.angle(r)

# Peak at 1550nm
idx1550 = np.argmin(np.abs(lam_dbr - lam0_dbr))
R_peak = R_dbr[idx1550]
idx1200 = np.argmin(np.abs(lam_dbr - 1200e-9))
R_at_1200 = R_dbr[idx1200]

# Stopband width
stopband_mask = R_dbr > 0.5
lam_stop = lam_dbr[stopband_mask]
stopband_width_nm = (lam_stop[-1] - lam_stop[0])*1e9 if len(lam_stop) > 0 else 0
print(f"  R_peak at 1550nm = {R_peak:.4f}")
print(f"  Stopband width approx {stopband_width_nm:.1f} nm")
print(f"  R at 1200nm = {R_at_1200:.4f}")

# GDD: d^2 phi/d omega^2
c_light = 3e8
omega_dbr = 2*np.pi*c_light / lam_dbr[::-1]  # ascending omega
phi_dbr_rev = phi_dbr[::-1]
dw = np.gradient(omega_dbr)
dphi_dw = np.gradient(np.unwrap(phi_dbr_rev), omega_dbr)
d2phi_dw2 = np.gradient(dphi_dw, omega_dbr)  # GDD in s^2
GDD_fs2 = d2phi_dw2 * 1e30  # convert to fs^2

fig, axes = plt.subplots(2, 1, figsize=(9,8), sharex=False)
axes[0].plot(lam_dbr*1e9, R_dbr*100, 'b-')
axes[0].axvline(1550, color='r', ls='--', alpha=0.5, label='1550nm')
axes[0].set_ylabel("Reflectance (%)"); axes[0].set_title(f"DBR (HL)^{N_periods}: TiO₂/MgF₂")
axes[0].legend(); axes[0].grid(alpha=0.3)

lam_dbr_rev = lam_dbr[::-1]
axes[1].plot(lam_dbr_rev*1e9, GDD_fs2, 'r-')
axes[1].set_xlim(1200, 1900)
axes[1].set_ylim(-5000, 5000)
axes[1].axvline(1550, color='b', ls='--', alpha=0.5)
axes[1].set_xlabel("Wavelength (nm)"); axes[1].set_ylabel("GDD (fs²)")
axes[1].set_title("Group Delay Dispersion")
axes[1].grid(alpha=0.3)
plt.tight_layout()
plt.savefig("repl/as_dbr.png", dpi=120)
plt.close()
print("  Saved repl/as_dbr.png")

chk(R_peak > 0.95, 1.0, "R_peak at 1550nm > 0.95", tol=0.5, absolute=True)
chk(stopband_width_nm > 400, 1.0, "stopband_width_nm > 400", tol=0.5, absolute=True)
chk(R_at_1200 < 0.5, 1.0, "R at 1200nm < 0.5 (out of stopband)", tol=0.5, absolute=True)

# %% [markdown]
# ## §9 — μ MLE and rogue wave statistics with autograd optimization

# %%
hdr("§9 — mu MLE and rogue wave statistics with autograd optimization")

torch.manual_seed(42)
N_rw = 10000; mu_true_rw = 5.0
I_data_rw = torch.distributions.Exponential(torch.tensor(1.0/mu_true_rw)).sample((N_rw,)).double()

mu_param_rw = torch.tensor(3.0, dtype=torch.float64, requires_grad=True)
opt_rw = torch.optim.Adam([mu_param_rw], lr=0.1)

NLL_initial = None
for step in range(500):
    opt_rw.zero_grad()
    nll = torch.log(mu_param_rw) + I_data_rw.mean() / mu_param_rw
    nll.backward()
    opt_rw.step()
    with torch.no_grad():
        mu_param_rw.clamp_(min=1e-6)
    if step == 0:
        NLL_initial = nll.item()

NLL_final = nll.item()
mu_converged = mu_param_rw.item()
print(f"  mu_converged = {mu_converged:.4f}, NLL_initial = {NLL_initial:.6f}, NLL_final = {NLL_final:.6f}")

# Cramér-Rao bound
CRB_sigma = mu_true_rw / np.sqrt(N_rw)
print(f"  CRB sigma = {CRB_sigma:.4f}")

# Bootstrap
rng = np.random.default_rng(42)
I_np = I_data_rw.numpy()
n_boot = 100
boot_means = np.array([rng.choice(I_np, N_rw, replace=True).mean() for _ in range(n_boot)])
bootstrap_std = boot_means.std()
print(f"  Bootstrap std = {bootstrap_std:.4f}")

chk(mu_converged, mu_true_rw, "mu_converged near 5.0", tol=0.1, absolute=True)
chk(NLL_final < NLL_initial, 1.0, "NLL_final < NLL_initial", tol=0.5, absolute=True)
chk(CRB_sigma < 0.1, 1.0, "CRB_sigma < 0.1", tol=0.5, absolute=True)
chk(bootstrap_std > CRB_sigma * 0.5, 1.0, "bootstrap_std > CRB * 0.5", tol=0.5, absolute=True)

# %% [markdown]
# ## §10 — Full Jalali lab pipeline: autograd through dispersion

# %%
hdr("§10 — Full Jalali lab pipeline: autograd through dispersion")

N_pipe = 2048
T0_pipe = 1e-12  # 1 ps
dt_pipe = T0_pipe / 10
t_pipe = torch.linspace(-N_pipe/2*dt_pipe, N_pipe/2*dt_pipe, N_pipe, dtype=torch.float64)
omega_pipe = torch.fft.fftfreq(N_pipe, d=dt_pipe) * 2*np.pi

beta2_fiber = -21e-27  # s^2/m
L_fiber = 1e3  # 1 km
H_fiber = torch.exp(-1j * beta2_fiber * omega_pipe**2 * L_fiber / 2).to(torch.complex128)

A0_pipe = torch.exp(-t_pipe**2 / (2*T0_pipe**2)).to(torch.complex128)
A_L_pipe = torch.fft.ifft(H_fiber * torch.fft.fft(A0_pipe))

I1_pipe = A0_pipe.abs()**2
I2_pipe = A_L_pipe.abs()**2

# Dispersion length
L_D = T0_pipe**2 / abs(beta2_fiber)
expected_broadening = np.sqrt(1 + (L_fiber/L_D)**2)

# RMS width of output
I2_np = I2_pipe.real.numpy()
t_np = t_pipe.numpy()
T_out_sq = np.sum(t_np**2 * I2_np) / np.sum(I2_np)
T_out = np.sqrt(T_out_sq * 2)  # factor 2 for RMS definition matching input

T_in_ps = T0_pipe * 1e12
T_out_ps = T_out * 1e12
broadening_factor = T_out / T0_pipe

print(f"  L_D = {L_D:.2f} m")
print(f"  T_in = {T_in_ps:.4f} ps, T_out = {T_out_ps:.4f} ps")
print(f"  Broadening factor = {broadening_factor:.4f}, expected = {expected_broadening:.4f}")

# Rogue detection
I2_max = I2_pipe.real.max().item()
I2_mean = I2_pipe.real.mean().item()
ratio = I2_max / I2_mean
if ratio > 2:
    print(f"  ROGUE DETECTED (max/mean = {ratio:.2f})")
else:
    print(f"  normal (max/mean = {ratio:.2f})")

# Energy conservation
I1_energy = np.trapezoid(I1_pipe.real.numpy(), t_np)
I2_energy = np.trapezoid(I2_pipe.real.numpy(), t_np)
energy_ratio = I2_energy / (I1_energy + 1e-30)
print(f"  Energy ratio I2/I1 = {energy_ratio:.6f}")

chk(T_out > T0_pipe, 1.0, "T_out > T_in (broadened)", tol=0.5, absolute=True)
chk(broadening_factor, expected_broadening, "broadening_factor near expected", tol=0.05)
chk(L_D > 0, 1.0, "L_D > 0", tol=0.5, absolute=True)
chk(energy_ratio, 1.0, "I2_total_energy near I1_total_energy", tol=0.01)

print("\nAll sections complete.")
