"""
_repl_optical_materials.py
Material library (real+imaginary n), normal vectors + vectorization,
human visual perception, Griffiths EM complex waves
Run: py -3.12 repl/_repl_optical_materials.py
"""

import numpy as np
import sympy as sp
import time

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1: MATERIAL LIBRARY — COMPLEX REFRACTIVE INDEX n + ik
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 65)
print("SECTION 1: COMPLEX REFRACTIVE INDEX  n_tilde = n + i*k")
print("=" * 65)

print("""
  Griffiths §9.4:  plane wave in lossy medium
    E(z,t) = E0 * exp(i*(k_tilde*z - omega*t))
    k_tilde = (omega/c) * n_tilde = (omega/c)*(n + i*k)

    Real part n  -> phase velocity  v_phi = c/n
    Imag part k  -> absorption      alpha = 2*omega*k/c = 4*pi*k/lambda0

  Dielectric function:
    epsilon_tilde = epsilon_r + i*epsilon_i = n_tilde^2 = (n+ik)^2
    epsilon_r = n^2 - k^2
    epsilon_i = 2*n*k
    (invert: n = sqrt((|eps| + eps_r)/2),  k = sqrt((|eps| - eps_r)/2))
""")

# Material optical constants at lambda = 550 nm (green, peak human vision)
# n = real refractive index, k = extinction coefficient
materials = {
    "Air":            (1.0000,   0.0),
    "Water (550nm)":  (1.3330,   1.96e-9),
    "Glass (BK7)":    (1.5168,   0.0),
    "Silicon":        (4.150,    0.044),
    "Gold":           (0.468,    2.415),
    "Silver":         (0.129,    3.199),
    "Copper":         (1.156,    2.424),
    "Al2O3 (sapph)":  (1.768,    0.0),
    "ITO (TCO)":      (1.870,    0.043),
    "GaAs":           (3.881,    0.194),
    "TiO2 (anatase)": (2.561,    0.0),
}

lambda0_nm = 550.0
lambda0_m  = lambda0_nm * 1e-9
c_light    = 2.998e8
omega      = 2*np.pi*c_light / lambda0_m

print(f"  lambda = {lambda0_nm:.0f} nm  (peak photopic vision)\n")
print(f"  {'Material':20s}  {'n':>7}  {'k':>9}  {'alpha(1/m)':>12}  {'skin(nm)':>10}  {'eps_r':>8}  {'eps_i':>8}")
print("-" * 80)

for name, (n, k) in materials.items():
    alpha   = 4*np.pi*k / lambda0_m      # absorption coefficient m^-1
    skin_nm = (1/alpha * 1e9) if k > 0 else np.inf
    eps_r   = n**2 - k**2
    eps_i   = 2*n*k
    skin_str = f"{skin_nm:>10.1f}" if np.isfinite(skin_nm) else "     inf  "
    print(f"  {name:20s}  {n:>7.4f}  {k:>9.4f}  {alpha:>12.3e}  {skin_str}  {eps_r:>8.4f}  {eps_i:>8.4f}")

# ── Sellmeier equation (dispersion of glass) ─────────────────────────────────
print("\n--- Sellmeier dispersion: BK7 glass ---")
print("  n^2(lambda) = 1 + B1*lambda^2/(lambda^2-C1) + B2*... + B3*...")
# BK7 Sellmeier coefficients (lambda in micrometers)
B = [1.03961212, 0.231792344, 1.01046945]
C = [0.00600069867, 0.0200179144, 103.560653]   # um^2

lam_um = np.array([0.365, 0.405, 0.486, 0.546, 0.587, 0.656, 0.852, 1.060])
n_sell = np.sqrt(1 + sum(b*lam_um**2/(lam_um**2 - c) for b, c in zip(B, C)))
GVD    = np.gradient(np.gradient(n_sell, lam_um*1e-6), lam_um*1e-6)  # crude

print(f"\n  {'lambda(nm)':>10}  {'n(Sellmeier)':>14}  {'GVD sign':>10}")
for lam, n_s, gvd in zip(lam_um*1000, n_sell, GVD):
    print(f"  {lam:>10.0f}  {n_s:>14.6f}  {'normal (D>0)' if gvd > 0 else 'anomalous':>10}")

# ── Drude model for metals ────────────────────────────────────────────────────
print("\n--- Drude model for gold ---")
print("  eps_tilde(omega) = 1 - omega_p^2 / (omega^2 + i*gamma_D*omega)")
# Gold Drude parameters
omega_p_eV  = 9.03     # eV plasma frequency
gamma_D_eV  = 0.0270   # eV damping
eV_to_rad   = 1.5193e15  # rad/s per eV

omega_p = omega_p_eV * eV_to_rad
gamma_D = gamma_D_eV * eV_to_rad

lam_test_nm = np.array([400, 550, 700, 1000, 1550])
omega_test   = 2*np.pi*c_light / (lam_test_nm * 1e-9)
eps_drude    = 1 - omega_p**2 / (omega_test**2 + 1j*gamma_D*omega_test)
n_drude      = np.sqrt(eps_drude)
# take branch with positive imaginary part (absorbing)
n_drude = np.where(n_drude.imag < 0, -n_drude, n_drude)

print(f"\n  {'lambda(nm)':>10}  {'n_Drude':>10}  {'k_Drude':>10}  {'skin_nm':>10}")
for lam, nd in zip(lam_test_nm, n_drude):
    k_d    = nd.imag
    alpha_d = 4*np.pi*k_d / (lam*1e-9)
    skin_d  = 1/alpha_d*1e9 if k_d > 0 else np.inf
    print(f"  {lam:>10}  {nd.real:>10.4f}  {k_d:>10.4f}  {skin_d:>10.1f}")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2: NORMAL VECTORS + VECTORIZATION
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("SECTION 2: SURFACE NORMAL VECTORS + NUMPY VECTORIZATION")
print("=" * 65)

print("""
  Surface f(x,y,z) = 0: normal vector n_hat = grad(f) / |grad(f)|
  Parametric surface r(u,v): normal = dr/du x dr/dv  (cross product)
  For n_tilde optical surface: normal determines Fresnel reflection.
""")

# ── Gradient as normal: sphere x^2+y^2+z^2=R^2 ───────────────────────────────
x_s, y_s, z_s, R_s = sp.symbols('x y z R', real=True, positive=True)
f_sphere = x_s**2 + y_s**2 + z_s**2 - R_s**2
grad_f = sp.Matrix([sp.diff(f_sphere, v) for v in (x_s, y_s, z_s)])
grad_mag = sp.sqrt(grad_f.dot(grad_f))
n_hat_sym = grad_f / grad_mag
print(f"  Sphere f=x^2+y^2+z^2-R^2:")
print(f"  grad(f) = {grad_f.T}")
print(f"  |grad|  = {sp.simplify(grad_mag)}")
print(f"  n_hat   = {sp.simplify(n_hat_sym).T}  (= r_hat as expected)")

# ── Vectorized normal computation on a mesh ───────────────────────────────────
print("\n--- Vectorized surface normals on 256x256 mesh ---")
N_mesh = 256
u = np.linspace(0, 2*np.pi, N_mesh)
v = np.linspace(0, np.pi,   N_mesh)
U, V = np.meshgrid(u, v)

# Parametric torus: r(u,v) where R_big=3, r_small=1
R_big, r_small = 3.0, 1.0
X = (R_big + r_small*np.cos(V)) * np.cos(U)
Y = (R_big + r_small*np.cos(V)) * np.sin(U)
Z = r_small * np.sin(V)

# Loop version
def normals_loop(X, Y, Z):
    Nv, Nu = X.shape
    normals = np.zeros((Nv, Nu, 3))
    for i in range(Nv-1):
        for j in range(Nu-1):
            du = np.array([X[i,j+1]-X[i,j], Y[i,j+1]-Y[i,j], Z[i,j+1]-Z[i,j]])
            dv = np.array([X[i+1,j]-X[i,j], Y[i+1,j]-Y[i,j], Z[i+1,j]-Z[i,j]])
            n  = np.cross(du, dv)
            mag = np.linalg.norm(n)
            normals[i,j] = n / mag if mag > 0 else n
    return normals

# Vectorized version
def normals_vec(X, Y, Z):
    # finite difference tangent vectors along u and v
    dXdu = np.diff(X, axis=1, append=X[:, :1])   # shape (Nv, Nu)
    dYdu = np.diff(Y, axis=1, append=Y[:, :1])
    dZdu = np.diff(Z, axis=1, append=Z[:, :1])
    dXdv = np.diff(X, axis=0, append=X[:1, :])
    dYdv = np.diff(Y, axis=0, append=Y[:1, :])
    dZdv = np.diff(Z, axis=0, append=Z[:1, :])
    # Cross product: du x dv
    Nx = dYdu*dZdv - dZdu*dYdv
    Ny = dZdu*dXdv - dXdu*dZdv
    Nz = dXdu*dYdv - dYdu*dXdv
    mag = np.sqrt(Nx**2 + Ny**2 + Nz**2) + 1e-30
    return np.stack([Nx/mag, Ny/mag, Nz/mag], axis=-1)

# Benchmark
t0 = time.perf_counter()
N_loop = normals_loop(X[:32, :32], Y[:32, :32], Z[:32, :32])   # 32x32 only (slow)
t_loop = time.perf_counter() - t0

t0 = time.perf_counter()
N_vec  = normals_vec(X, Y, Z)    # full 256x256
t_vec  = time.perf_counter() - t0

t0 = time.perf_counter()
N_loop_eq = normals_loop(X[:32,:32], Y[:32,:32], Z[:32,:32])
t_loop_eq = time.perf_counter() - t0

# Estimate loop time for full 256x256
scale = (256*256) / (32*32)
t_loop_est = t_loop * scale

print(f"  Mesh: {N_mesh}x{N_mesh} = {N_mesh**2:,} surface points")
print(f"  Loop  32x32  time: {t_loop*1000:.2f} ms")
print(f"  Loop  256x256 est: {t_loop_est*1000:.1f} ms  (extrapolated)")
print(f"  Vec   256x256 time: {t_vec*1000:.2f} ms")
print(f"  Speedup: ~{t_loop_est/t_vec:.0f}x")

# Verify agreement on 32x32
N_vec_small = normals_vec(X[:32,:32], Y[:32,:32], Z[:32,:32])
err = np.max(np.abs(N_loop[:31,:31] - N_vec_small[:31,:31]))
print(f"  Max error loop vs vec (32x32): {err:.2e}")

# ── Why vectorization works: memory layout and SIMD ──────────────────────────
print("""
  WHY VECTORIZATION IS FAST:
    Loop:  Python overhead per iteration ~100 ns; 256x256=65536 iters -> ~6.5 ms overhead alone
    Vec:   NumPy sends entire array to C/Fortran BLAS; CPU SIMD (AVX-512) does 8 doubles/clock
    Rule:  any loop over array elements -> replace with array op
           any nested loop -> replace with broadcasting / einsum / outer product

  Broadcasting rule: shapes (A,B,1) and (1,B,C) -> broadcast to (A,B,C)
  Normal vector formula as einsum:
    n = du x dv = eps_ijk * du_j * dv_k   (Levi-Civita)
    np.einsum('ijk,j...,k...->i...', levi_civita, du, dv)
""")

# ── Fresnel reflection using vectorized complex n ─────────────────────────────
print("--- Fresnel equations with complex n_tilde ---")
print("  R_s = |cos(theta_i) - n_tilde*cos(theta_t)|^2 / |...|^2")
print("  Uses Snell: n1*sin(theta_i) = n_tilde*sin(theta_t)")

n1    = 1.0                             # air
theta_i = np.linspace(0, np.pi/2, 1000)  # incident angles

def fresnel_rs(n1, n2_tilde, theta_i):
    cos_i = np.cos(theta_i)
    sin_i = np.sin(theta_i)
    cos_t = np.sqrt(1 - (n1/n2_tilde * sin_i)**2 + 0j)
    rs = (n1*cos_i - n2_tilde*cos_t) / (n1*cos_i + n2_tilde*cos_t)
    return np.abs(rs)**2

def fresnel_rp(n1, n2_tilde, theta_i):
    cos_i = np.cos(theta_i)
    sin_i = np.sin(theta_i)
    cos_t = np.sqrt(1 - (n1/n2_tilde * sin_i)**2 + 0j)
    rp = (n2_tilde*cos_i - n1*cos_t) / (n2_tilde*cos_i + n1*cos_t)
    return np.abs(rp)**2

print(f"\n  Normal incidence (theta=0) reflectance R = |(n-1)/(n+1)|^2:")
for name, (n, k) in list(materials.items())[:6]:
    n_tilde = n + 1j*k
    R_0 = np.abs((n_tilde - 1)/(n_tilde + 1))**2
    print(f"    {name:20s}: R={R_0:.4f} = {R_0*100:.1f}%")

# Brewster angle for dielectrics (k=0): tan(theta_B) = n2/n1
print(f"\n  Brewster angle (Rp=0, p-polarization vanishes):")
for name, (n, k) in list(materials.items())[:4]:
    if k == 0:
        theta_B = np.degrees(np.arctan(n/n1))
        print(f"    {name:20s}: theta_B = {theta_B:.2f} deg")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3: HUMAN VISUAL PERCEPTION — CIE + WEBER-FECHNER
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("SECTION 3: HUMAN VISUAL PERCEPTION")
print("=" * 65)

print("""
  Photoreceptors:
    Rods:   rhodopsin peak ~498 nm,  scotopic (dim light), no color
    L-cone: peak ~564 nm  (red)
    M-cone: peak ~534 nm  (green)
    S-cone: peak ~420 nm  (blue)

  CIE 1931 color matching functions x_bar, y_bar, z_bar:
    Tristimulus: X = integral S(lambda)*x_bar(lambda) d_lambda
    Luminance Y = integral S(lambda)*y_bar(lambda) d_lambda
    y_bar peaks at 555 nm (photopic peak sensitivity)

  Weber-Fechner law:  dS = k * dI/I
    Perception S = k * ln(I/I0)  (logarithmic!)
    Just-noticeable difference:  delta_I/I = constant ~ 0.02 (2%)
    Consequence: decibels, musical octaves, brightness all log-scaled
""")

# Approximate CIE 1931 using Gaussian fits (Stockman/Sharpe)
lam = np.linspace(380, 780, 401)

def cie_gauss(lam, peaks, widths, weights):
    return sum(w * np.exp(-0.5*((lam-p)/s)**2)
               for p, s, w in zip(peaks, widths, weights))

# x_bar has two lobes
x_bar = (cie_gauss(lam, [598.8], [33.0], [1.065]) +
         cie_gauss(lam, [445.0], [19.0], [0.366]))
y_bar =  cie_gauss(lam, [556.3], [46.0], [1.014])
z_bar =  cie_gauss(lam, [449.8], [22.0], [1.839])

# Normalize (y_bar integrates to 1 in standard CIE)
y_bar /= np.trapezoid(y_bar, lam)
x_bar /= np.trapezoid(y_bar, lam)  # keep ratio
z_bar /= np.trapezoid(y_bar, lam)

# Monochromatic spectrum: find peak visual sensitivity
peak_y_idx = np.argmax(y_bar)
peak_x_idx = np.argmax(x_bar[:200])   # short-wave lobe
print(f"  y_bar (luminosity) peak: {lam[peak_y_idx]:.1f} nm")
print(f"  x_bar short lobe  peak:  {lam[peak_x_idx]:.1f} nm")

# Spectral sensitivity at key wavelengths
print(f"\n  {'lambda(nm)':>10}  {'x_bar':>8}  {'y_bar':>8}  {'z_bar':>8}  {'color'}")
key_lam = [420, 450, 490, 520, 555, 590, 620, 660, 700]
for lam_k in key_lam:
    idx = np.argmin(np.abs(lam - lam_k))
    color_names = {420:'violet', 450:'blue', 490:'cyan', 520:'green',
                   555:'yellow-grn', 590:'yellow', 620:'orange', 660:'red', 700:'deep-red'}
    print(f"  {lam_k:>10}  {x_bar[idx]:>8.4f}  {y_bar[idx]:>8.4f}  "
          f"{z_bar[idx]:>8.4f}  {color_names[lam_k]}")

# Weber-Fechner: just-noticeable differences
print("\n--- Weber-Fechner: logarithmic perception ---")
k_WF = 0.02    # Weber fraction for luminance
I0   = 1.0
I_range = np.array([1, 10, 100, 1000, 10000])
print(f"  Weber fraction k = {k_WF} (2% JND for luminance)")
print(f"  {'I (cd/m^2)':>12}  {'S=k*ln(I/I0)':>14}  {'JND delta_I':>12}")
for I in I_range:
    S   = np.log(I / I0)      # Fechner: perception ~ log intensity
    dI  = k_WF * I
    print(f"  {I:>12.0f}  {S:>14.4f}  {dI:>12.2f}")

print("\n  Stevens' power law (refined): S = C * I^n")
print("  Brightness: n=0.33  (cube root -- heavily compressed)")
print("  Loudness:   n=0.60")
print("  Pain:       n=3.5   (amplified!)")
print("  Electric shock: n=3.5  <- that's why tasers work")

# ── Color in terms of complex n: thin film interference ──────────────────────
print("\n--- Thin film: color from destructive interference (soap bubble) ---")
print("  2*n*t*cos(theta) = m*lambda  -> constructive")
print("  2*n*t*cos(theta) = (m+1/2)*lambda -> destructive")
n_soap  = 1.34   # soap film
theta_0 = 0.0    # normal incidence
t_nm    = np.array([100, 150, 200, 300, 500])  # film thickness nm

print(f"\n  {'t (nm)':>8}  Constructive wavelengths (m=1):")
for t in t_nm:
    lam_c = 2 * n_soap * t   # m=1, lambda = 2nt
    print(f"  {t:>8}  lambda = {lam_c:.1f} nm", end="")
    if 380 < lam_c < 700:
        if lam_c < 450:   color = " (violet)"
        elif lam_c < 490: color = " (blue)"
        elif lam_c < 560: color = " (green)"
        elif lam_c < 620: color = " (yellow)"
        else:             color = " (red)"
        print(color)
    else:
        print(" (UV/IR -- appears dark/black)")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4: GRIFFITHS EM — WAVES IN ABSORBING MEDIA (§9.4)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("SECTION 4: GRIFFITHS SEC 9.4 — WAVES IN ABSORBING MEDIA")
print("=" * 65)

print("""
  Wave equation in linear medium with conductivity sigma:
    nabla^2 E = mu*eps * d^2E/dt^2 + mu*sigma * dE/dt

  Plane wave ansatz E = E0*exp(i*(k_tilde*z - omega*t)):
    k_tilde^2 = mu*eps*omega^2 + i*mu*sigma*omega
               = (omega/c)^2 * (eps_r + i*sigma/(eps_0*omega))
               = (omega/c)^2 * eps_tilde

  Write k_tilde = K + i*kappa:
    K     = real part = omega*n/c   (phase constant)
    kappa = imag part = omega*k/c   (attenuation constant)
    alpha = 2*kappa = skin depth coefficient

  E(z,t) = E0 * exp(-kappa*z) * exp(i*(K*z - omega*t))
           |___decay envelope__| |___propagating wave___|
""")

# ── Quantitative: skin depth table at 550nm, radio, microwave ────────────────
print("  Skin depth delta = 1/kappa = c/(omega*k) = lambda0/(2*pi*k)")
print()

# (material, n, k, description)
conductors = [
    ("Copper (550nm)",    1.156, 2.424, 550e-9),
    ("Gold (550nm)",      0.468, 2.415, 550e-9),
    ("Silver (550nm)",    0.129, 3.199, 550e-9),
    ("Copper (1GHz RF)",  1.0,   4.7e4,  0.30),    # skin depth ~2 um
    ("Copper (100MHz)",   1.0,   1.5e5,  3.0),
    ("Seawater (1MHz)",   1.34,  7.5e3,  300.0),   # radio penetration in ocean
]

print(f"  {'Material':22s}  {'lambda0':>10}  {'n':>8}  {'k':>10}  {'skin depth':>12}")
for name, n_m, k_m, lam_m in conductors:
    alpha_m  = 4*np.pi*k_m / lam_m
    skin_m   = 1/alpha_m
    if skin_m < 1e-6:
        skin_str = f"{skin_m*1e9:.2f} nm"
    elif skin_m < 1e-3:
        skin_str = f"{skin_m*1e6:.2f} um"
    elif skin_m < 1:
        skin_str = f"{skin_m*1e3:.2f} mm"
    else:
        skin_str = f"{skin_m:.2f} m"
    print(f"  {name:22s}  {lam_m*1e9:>7.1f} nm  {n_m:>8.4f}  {k_m:>10.3e}  {skin_str:>12}")

# ── Phase delta between E and H in conductor ─────────────────────────────────
print("\n--- Phase lag between E and H in conductor (Griffiths eq 9.138) ---")
print("""
  k_tilde = K + i*kappa = |k_tilde| * exp(i*phi_k)
  H0 = (k_tilde / (mu*omega)) * E0
  phi_k = arctan(kappa/K) = arctan(k/n)

  Perfect conductor: k >> n -> phi_k -> 45 deg
  Perfect dielectric: k=0 -> phi_k = 0 (E and H in phase)
""")
for name, (n_m, k_m) in list(materials.items()):
    phi_deg = np.degrees(np.arctan2(k_m, n_m))
    print(f"  {name:20s}: phi = {phi_deg:7.2f} deg  "
          f"({'conductor' if phi_deg>20 else 'dielectric':>12})")

# ── Energy: time-averaged Poynting vector in absorbing medium ─────────────────
print("\n--- Time-averaged Poynting vector <S> = (1/2)*Re[E x H*] ---")
print("  In absorbing medium:")
print("  <S_z> = (1/2) * (|k_tilde|/(mu*omega)) * |E0|^2 * exp(-2*kappa*z)")
print("  Energy decays as exp(-alpha*z) where alpha = 2*kappa = 2*omega*k/c")
print()

E0_sq = 1.0  # normalized
z_test = np.array([0, 1, 2, 5]) * 1e-9   # nm to m
for name, (n_m, k_m) in [("Silver (550nm)", (0.129, 3.199)),
                           ("Gold (550nm)",   (0.468, 2.415))]:
    kappa = 2*np.pi*k_m / (550e-9)
    print(f"  {name}:")
    for z_nm in [0, 5, 10, 25]:
        I_frac = np.exp(-2*kappa * z_nm*1e-9)
        print(f"    z={z_nm:3d} nm: I/I0 = {I_frac:.4f} = {I_frac*100:.1f}%")

print("\nDone.")
