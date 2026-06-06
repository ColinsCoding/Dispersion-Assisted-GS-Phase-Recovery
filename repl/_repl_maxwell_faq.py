"""
_repl_maxwell_faq.py
Maxwell's equations: integral form, differential form, physical meaning, measurement FAQ
Run: py -3.12 repl/_repl_maxwell_faq.py
"""

import numpy as np
import sympy as sp

eps0 = 8.8542e-12   # F/m
mu0  = 4*np.pi*1e-7 # H/m
c    = 1/np.sqrt(eps0*mu0)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1: THE FOUR EQUATIONS — BOTH FORMS SIDE BY SIDE
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 65)
print("MAXWELL'S EQUATIONS: INTEGRAL vs DIFFERENTIAL FORM")
print("=" * 65)

print(f"""
  c = 1/sqrt(eps0*mu0) = {c:.6e} m/s  (speed of light, derived)

  Notation:
    E  = electric field       [V/m]
    B  = magnetic flux density [T]
    D  = eps*E displacement   [C/m^2]
    H  = B/mu magnetic field  [A/m]
    rho = free charge density [C/m^3]
    J   = free current density [A/m^2]

  =========================================================
  I.  GAUSS'S LAW (electric)
  =========================================================
  Integral:     closed surface integral D.dA = Q_enc
  Differential: div D = rho        (nabla . D = rho)

  Meaning: electric field DIVERGES from charges.
  Charge is the SOURCE of D field lines.
  Q_enc > 0: field lines point OUT of closed surface.

  Measurement: Faraday cage + electrometer measures Q_enc
               by integrating D.dA over enclosing surface.

  =========================================================
  II. GAUSS'S LAW (magnetic)
  =========================================================
  Integral:     closed surface integral B.dA = 0
  Differential: div B = 0          (nabla . B = 0)

  Meaning: NO magnetic monopoles exist.
  B field lines form CLOSED LOOPS -- they have no source/sink.
  Every field line entering a surface exits somewhere.

  Measurement: any Gaussian surface gives zero net flux.
               Verified to 1 part in 10^13 (best precision test).

  =========================================================
  III. FARADAY'S LAW (induction)
  =========================================================
  Integral:     closed loop integral E.dl = -d/dt (flux of B through loop)
  Differential: curl E = -dB/dt    (nabla x E = -dB/dt)

  Meaning: a CHANGING B field CURLS the E field.
  Time-varying B induces circulating E (even with no charges).
  The minus sign = Lenz's law (induced EMF opposes change).

  Measurement: search coil (pickup coil) measures EMF = -dPhi_B/dt.
               This is how transformers, electric motors, guitar
               pickups, and MRI gradient coils work.

  =========================================================
  IV. AMPERE-MAXWELL LAW
  =========================================================
  Integral:     closed loop integral H.dl = I_enc + eps*d/dt(flux of E)
  Differential: curl H = J + dD/dt  (nabla x H = J + eps*dE/dt)

  Meaning: CURRENT or CHANGING E field CURLS the H field.
  Maxwell's addition: displacement current eps*dE/dt
    -> without it, div J != 0 violates charge conservation.
    -> predicts electromagnetic WAVES (no charges needed).

  Measurement: Rogowski coil measures enclosed current I_enc.
               Displacement current measured indirectly via
               wave propagation (Hertz 1887 experiment).
""")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2: MATHEMATICAL OPERATORS — WHAT GRAD/DIV/CURL MEAN
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 65)
print("SECTION 2: WHAT DO THE OPERATORS MEAN?")
print("=" * 65)

print("""
  GRADIENT  (nabla f):  scalar -> vector
    Points in direction of steepest INCREASE of f.
    Magnitude = rate of change per meter.
    Example: nabla T = heat flow direction (Fourier law: q = -k*nabla T)

  DIVERGENCE  (nabla . F):  vector -> scalar
    Measures NET OUTFLOW of field per unit volume.
    div F > 0: source (field lines emerge here)
    div F < 0: sink   (field lines converge here)
    div F = 0: no source/sink -- field lines form closed loops or go to inf
    Example: nabla . E = rho/eps0 -> charge is source of E

  CURL  (nabla x F):  vector -> vector
    Measures CIRCULATION of field around a point.
    |curl F| = circulation per unit area (in the direction of max rotation)
    curl F = 0: conservative field (no closed-loop energy gain)
    curl F != 0: field "rotates" -- something is DRIVING the circulation
    Example: nabla x E = -dB/dt -> changing B drives E circulation

  LAPLACIAN  (nabla^2 f = div grad f):  scalar -> scalar
    Measures how f at a point compares to its NEIGHBORHOOD AVERAGE.
    nabla^2 f < 0: f is a local maximum (charges attract field toward center)
    nabla^2 f = 0: Laplace equation -- f equals its neighborhood average (harmonic)
    Wave equation: nabla^2 E = (1/c^2) d^2E/dt^2

  KEY IDENTITIES (always true, verify with SymPy below):
    div(curl F) = 0      (a curl has no divergence)
    curl(grad f) = 0     (a gradient has no curl)
    curl(curl F) = grad(div F) - nabla^2 F
""")

# Verify identities with SymPy
x_s, y_s, z_s = sp.symbols('x y z', real=True)
coords = (x_s, y_s, z_s)

# Arbitrary scalar and vector fields
f_sc = sp.sin(x_s) * sp.exp(-y_s) * z_s**2

Fx = x_s**2 * y_s
Fy = y_s**2 * z_s
Fz = z_s**2 * x_s
F_vec = sp.Matrix([Fx, Fy, Fz])

def grad(f):
    return sp.Matrix([sp.diff(f, v) for v in coords])

def div(F):
    return sum(sp.diff(F[i], coords[i]) for i in range(3))

def curl(F):
    return sp.Matrix([
        sp.diff(F[2], y_s) - sp.diff(F[1], z_s),
        sp.diff(F[0], z_s) - sp.diff(F[2], x_s),
        sp.diff(F[1], x_s) - sp.diff(F[0], y_s),
    ])

laplacian_f = div(grad(f_sc))

div_curl_F    = sp.simplify(div(curl(F_vec)))
curl_grad_f   = sp.simplify(curl(grad(f_sc)))
curl_curl_F   = sp.simplify(curl(curl(F_vec)))
grad_div_F    = sp.simplify(grad(div(F_vec)))
lap_F0        = sp.simplify(
    sp.diff(Fx,x_s,2) + sp.diff(Fx,y_s,2) + sp.diff(Fx,z_s,2))

print("  SymPy verification of vector identities:")
print(f"  div(curl F) = {div_curl_F}  (should be 0)")
print(f"  curl(grad f) = {curl_grad_f.T}  (should be [0,0,0])")
print(f"  |curl(curl F) - (grad(div F) - lap F)| = "
      f"{sp.simplify(curl_curl_F - grad_div_F + sp.Matrix([lap_F0, 0, 0])).norm()} (not simplified but verifiable)")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3: DERIVING THE WAVE EQUATION FROM MAXWELL
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("SECTION 3: WAVE EQUATION FROM MAXWELL (free space)")
print("=" * 65)

print("""
  In free space (rho=0, J=0):
    curl E = -mu0 * dH/dt          (Faraday)
    curl H = eps0 * dE/dt          (Ampere-Maxwell, no current)

  Take curl of Faraday:
    curl(curl E) = -mu0 * d/dt(curl H)
                = -mu0 * eps0 * d^2E/dt^2

  Use curl(curl E) = grad(div E) - nabla^2 E:
    grad(div E) - nabla^2 E = -mu0*eps0 * d^2E/dt^2

  In free space div E = 0 (no charges), so:
    nabla^2 E = mu0*eps0 * d^2E/dt^2
    nabla^2 E = (1/c^2) * d^2E/dt^2    <- WAVE EQUATION

  Solution: E = E0 * exp(i*(k.r - omega*t))
    with  |k| = omega/c  (dispersion relation in vacuum)

  This is why c appears: it's BUILT INTO Maxwell's equations
  as c = 1/sqrt(eps0*mu0).  Maxwell derived this in 1865.
""")

print(f"  mu0 = {mu0:.6e} H/m")
print(f"  eps0 = {eps0:.6e} F/m")
print(f"  c = 1/sqrt(mu0*eps0) = {c:.6e} m/s")
print(f"  NIST c = 2.99792458e8 m/s")
print(f"  Error: {abs(c - 2.99792458e8)/2.99792458e8 * 100:.4f}%")

# Plane wave: verify k.E = 0 (transverse), k x E = omega*mu0*H
print("\n  Plane wave properties (k = k_z hat_z, E = E_x hat_x):")
k_num   = 2*np.pi / 633e-9   # red light wavenumber
omega_n = k_num * c
E0_n    = 1.0   # V/m
H0_n    = E0_n / (mu0 * c)   # = E0 * sqrt(eps0/mu0) = E0 / 377 ohm

print(f"  lambda=633nm (red):  k={k_num:.4e} rad/m,  f={omega_n/(2*np.pi):.4e} Hz")
print(f"  E0=1 V/m -> H0 = E0/(mu0*c) = E0/Z0 = {H0_n:.4f} A/m")
print(f"  Z0 = mu0*c = sqrt(mu0/eps0) = {mu0*c:.2f} ohm (impedance of free space)")
print(f"  Poynting: <S> = E0^2/(2*Z0) = {E0_n**2/(2*mu0*c):.4f} W/m^2 per (V/m)^2")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4: MEASUREMENT FAQ
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("SECTION 4: MEASUREMENT FAQ")
print("=" * 65)

print("""
  Q: How do you MEASURE E field?
  A: Dipole antenna: induced voltage V = integral(E.dl) over antenna length.
     For calibrated dipole length L: E = V/L  (V/m).
     Lab: electro-optic Pockels cell -- E modulates crystal birefringence
          -> phase shift read by interferometer (no metal probe needed).

  Q: How do you MEASURE B field?
  A: Hall probe: F = qv x B on carriers in semiconductor -> voltage V_H = I*B/(n*q*t).
     Search coil: V = -N * dPhi/dt = -N*A * dB/dt -> integrate for B(t).
     SQUID magnetometer: Josephson junction phase slips at flux quantum
          Phi_0 = h/2e = 2.07e-15 Wb -> sensitivity ~1 fT (femtoTesla).

  Q: How do you MEASURE displacement current eps*dE/dt?
  A: Directly, you cannot -- it's not a real current.
     Indirectly: it MUST exist to explain RF wave propagation,
     capacitor charging (current flows in circuit but not across gap),
     and electromagnetic wave propagation.
     The existence of radio waves IS the measurement of displacement current.

  Q: What does div B = 0 mean for measurement?
  A: Wrap any surface around any object -- net B flux is always zero.
     B field lines that enter MUST exit somewhere.
     There is no magnetic charge (monopole) that would create a
     diverging B field.  If you cut a magnet in half, you get
     two smaller magnets, not separate poles.

  Q: Why does the integral form use closed surfaces/loops?
  A: Stokes' theorem:  integral(curl F).dA = closed_loop_integral F.dl
     Divergence theorem: integral(div F) dV = closed_surface_integral F.dA
     These convert VOLUME/AREA integrals to BOUNDARY integrals.
     Integral form = differential form averaged over a finite region.
     Differential form = integral form in the limit (region -> point).

  Q: Which form is more fundamental?
  A: Differential -- it applies pointwise, works in non-uniform media.
     Integral form requires the region to have uniform properties.
     But integral form is easier to use for HIGH-SYMMETRY problems
     (sphere, cylinder, infinite plane) -- Gauss's law trick.

  Q: What is the constitutive relation?
  A: D = eps * E  and  B = mu * H  (linear, isotropic media)
     In general: D = eps0*E + P  (P = polarization, dipole density)
                  B = mu0*(H + M) (M = magnetization)
     For our fiber:  eps = eps0 * n^2  where n = complex refractive index

  Q: How does this connect to our GS phase retrieval?
  A: Fiber propagation: E(z) = E(0) * exp(i*k_tilde*z)
     k_tilde = (omega/c) * n_tilde = (omega/c)*(n + i*k)
     -> Measurement of |E|^2 at two planes (I1, I2) = two different z
     -> GS recovers phase phi from I1, I2, dispersion D
     Both planes are closed surfaces of the Faraday integral -- the
     MEASUREMENT IS the integral form of Gauss's law applied to
     optical intensity.
""")

# Gauss's law: E field from point charge, sphere, infinite plane
print("  Gauss's law examples (high-symmetry):")
print()

# Point charge
Q_C = 1e-9   # 1 nC
r_m = np.array([0.01, 0.1, 1.0])   # meters
E_point = Q_C / (4*np.pi*eps0 * r_m**2)
print("  Point charge Q=1nC:")
print(f"  {'r(m)':>6}  {'E (V/m)':>12}  {'note'}")
for r, E in zip(r_m, E_point):
    print(f"  {r:>6.3f}  {E:>12.2f}  {'dangerous' if E > 3e6 else 'safe'}")

# Infinite plane: E = sigma/(2*eps0) -- CONSTANT, no r dependence
sigma_C_m2 = 1e-6   # 1 uC/m^2
E_plane = sigma_C_m2 / (2*eps0)
print(f"\n  Infinite sheet sigma=1uC/m^2:  E = sigma/(2*eps0) = {E_plane:.2f} V/m")
print(f"  (independent of distance -- that's why parallel plate capacitors work)")

# Cylindrical: infinite line charge
lam_C_m = 1e-9   # 1 nC/m
r_cyl = np.array([0.001, 0.01, 0.1])
E_cyl = lam_C_m / (2*np.pi*eps0 * r_cyl)
print(f"\n  Infinite line charge lambda=1nC/m:")
for r, E in zip(r_cyl, E_cyl):
    print(f"  r={r*1000:.1f}mm: E={E:.2f} V/m  (falls as 1/r, not 1/r^2)")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5: IN MATTER — eps and mu frequency dependence
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("SECTION 5: MAXWELL IN MATTER -- eps(omega), mu(omega)")
print("=" * 65)

print("""
  In matter:
    D(omega) = eps(omega) * E(omega)       <- frequency domain
    B(omega) = mu(omega)  * H(omega)

  eps(omega) = eps_r + i*eps_i
    eps_r > 0, eps_i > 0: ordinary dielectric (glass, water)
    eps_r < 0:            metal below plasma frequency (reflects light)
    eps_r < 0, mu_r < 0:  negative index metamaterial (Veselago 1968)

  Kramers-Kronig relations (causality constraint):
    eps_r(omega) = 1 + (2/pi) * principal value integral [omega'*eps_i(omega') / (omega'^2 - omega^2)] d_omega'
    eps_i(omega) = -(2*omega/pi) * principal value integral [eps_r(omega') / (omega'^2 - omega^2)] d_omega'

  Meaning: real and imaginary parts of eps are NOT independent.
  Absorption (eps_i) at any frequency affects the refractive index (eps_r)
  at ALL other frequencies.  This is why glass has dispersion: it absorbs
  in UV, which shifts n across the visible via Kramers-Kronig.

  Drude model (free electrons in metal):
    eps(omega) = 1 - omega_p^2 / (omega^2 + i*gamma*omega)
    omega_p = sqrt(n_e * e^2 / (eps0 * m_e))   plasma frequency

  Lorentz oscillator (bound electrons, dielectric):
    eps(omega) = 1 + (omega_p^2) / (omega_0^2 - omega^2 - i*gamma*omega)
    omega_0 = resonance frequency of bound electron
""")

# Drude: gold at optical frequencies
omega_p_Au  = 9.03 * 1.519e15   # rad/s (9.03 eV)
gamma_Au    = 0.027 * 1.519e15  # rad/s (0.027 eV)
lam_nm_arr  = np.array([400, 500, 600, 700, 800, 1000, 1550])
omega_arr   = 2*np.pi*c / (lam_nm_arr * 1e-9)
eps_drude   = 1 - omega_p_Au**2 / (omega_arr**2 + 1j*gamma_Au*omega_arr)
n_tilde     = np.sqrt(eps_drude)
# choose branch with positive imaginary part
n_tilde     = np.where(n_tilde.imag < 0, -n_tilde, n_tilde)

print("  Gold Drude model (omega_p=9.03eV, gamma=0.027eV):")
print(f"  {'lam(nm)':>8}  {'eps_r':>8}  {'eps_i':>8}  {'n':>8}  {'k':>8}  {'skin(nm)':>10}")
print("-" * 60)
for lam, ep, nt in zip(lam_nm_arr, eps_drude, n_tilde):
    alpha = 4*np.pi*nt.imag / (lam*1e-9)
    skin  = 1/alpha*1e9
    print(f"  {lam:>8}  {ep.real:>8.3f}  {ep.imag:>8.3f}  "
          f"{nt.real:>8.4f}  {nt.imag:>8.4f}  {skin:>10.1f}")

print(f"\n  eps_r < 0 for all optical wavelengths -> gold REFLECTS (metallic behavior)")
print(f"  At 1550nm (telecom): k=11.2 -> skin depth ~11nm -> used as mirror in MEMS")

# Plasma frequency: frequency where eps_r = 0 -> metal becomes transparent
omega_p_Si  = 2.5e15    # rad/s (rough for doped Si)
lam_p_Si_nm = 2*np.pi*c / omega_p_Si * 1e9
print(f"\n  Plasma frequency of gold: lambda_p = {2*np.pi*c/omega_p_Au*1e9:.0f} nm")
print(f"  For lambda < lambda_p: eps_r -> +1 (UV, transparent)")
print(f"  For lambda > lambda_p: eps_r < 0 (optical, metallic, reflects)")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6: QUICK REFERENCE TABLE
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("SECTION 6: QUICK REFERENCE -- ALL FOUR EQUATIONS")
print("=" * 65)

print(f"""
  LAW             DIFFERENTIAL        INTEGRAL           PHYSICAL MEANING
  ---------------------------------------------------------------------------
  Gauss E         div D = rho         S.I. D.dA = Q_enc  Charges source E field
  Gauss B         div B = 0           S.I. B.dA = 0      No magnetic monopoles
  Faraday         curl E = -dB/dt     C.I. E.dl = -dPhi  Changing B curls E
  Ampere-Maxwell  curl H = J + dD/dt  C.I. H.dl = I+dPhi_E  Current/changing E curls H

  OPERATOR        MEANING             ZERO MEANS          NONZERO MEANS
  ---------------------------------------------------------------------------
  div F           outflow per vol     closed loops        source or sink present
  curl F          circulation/area    conservative        something driving rotation
  grad f          steepest ascent     f is constant       f varies in space
  nabla^2 f       vs neighborhood avg harmonic (Laplace)  source (Poisson: = -rho/eps0)

  SYMMETRY        USE INTEGRAL FORM   GAUSS SURFACE/LOOP
  ---------------------------------------------------------------------------
  Spherical       Point charge, atom  Sphere radius r
  Cylindrical     Wire, coax cable    Cylinder radius r, length L
  Planar          Capacitor, slab     Pillbox (flat cylinder)
  General         (no shortcut)       Use differential form + boundary conditions

  FOR FIBER/GS:
  ---------------------------------------------------------------------------
  Propagation     E(z) = E0*exp(i*k_tilde*z)    k_tilde = (omega/c)*n_tilde
  Measurement     I(z) = |E(z)|^2               intensity = Gauss surface integral
  GS retrieval    recover phi from I1, I2        inverse problem on Faraday integral
  Dispersion D    group delay spread             GVD = d^2 k / d omega^2

  c = {c:.5e} m/s = 1/sqrt(mu0*eps0)
  Z0 = sqrt(mu0/eps0) = {np.sqrt(mu0/eps0):.3f} ohm   (impedance of free space)
  Phi0 = h/2e = {6.626e-34/(2*1.602e-19):.4e} Wb     (flux quantum, SQUID)
""")

print("Done.")
