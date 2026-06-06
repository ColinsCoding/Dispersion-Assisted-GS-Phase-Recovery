"""
_repl_imaging_dew.py

S1: 2^n -- binomial expansion, Pascal's triangle, SymPy series
S2: Expansion microscopy (ExM) + cryo-EM / frozen insects
S3: Direct imaging -- coronagraph contrast, planet SNR, JWST
S4: Directed energy -- irradiance, M^2, atmospheric propagation, ANSI MPE
S5: Ethical hacking -- CIA triad, complexity, CTF patterns (educational)
S6: SymPy architecture errors -- common pitfalls and correct patterns
S7: COMSOL analog design -- FEM weak form, op-amp virtual ground
"""

import numpy as np
import sympy as sp
import math

SEP = "=" * 65

# ------------------------------------------------------------------ #
# S1: 2^n EXPANSIONS
# ------------------------------------------------------------------ #
print(SEP)
print("SECTION 1: 2^n -- BINOMIAL EXPANSION")
print(SEP)

x, y, n_sym, k_sym = sp.symbols("x y n k", positive=True)

# binomial theorem: (x+y)^n = sum C(n,k) x^(n-k) y^k
print("\n  Binomial theorem: (x+y)^n = sum_{k=0}^{n} C(n,k) x^(n-k) y^k")
for n_val in [2, 3, 4, 5]:
    expr = sp.expand((x + y)**n_val)
    print(f"  (x+y)^{n_val} = {expr}")

# Pascal's triangle -- C(n,k) = C(n-1,k-1) + C(n-1,k)
print("\n  Pascal's triangle  C(n,k) = n! / (k!(n-k)!):")
for row in range(7):
    coeffs = [math.comb(row, k) for k in range(row + 1)]
    print(f"  n={row}: {coeffs}   sum={sum(coeffs)} = 2^{row}")

# generating function: sum_{n>=0} x^n = 1/(1-x), |x|<1
print("\n  Generating function: sum_{n=0}^{inf} x^n = 1/(1-x)")
x_s = sp.Symbol("x")
gf = sp.series(1 / (1 - x_s), x_s, 0, 8)
print(f"  Taylor: {gf}")

# 2^n counts subsets of n-element set
print("\n  Combinatorial: 2^n = number of subsets of n-element set")
for n_val in [4, 8, 16, 32, 64]:
    print(f"  n={n_val:2d}: 2^n = {2**n_val:>12,}")

# binomial series for non-integer n: (1+x)^alpha = sum C(alpha,k) x^k
alpha = sp.Rational(1, 2)
binom_half = sp.series((1 + x_s)**alpha, x_s, 0, 6)
print(f"\n  Fractional binomial (1+x)^(1/2):")
print(f"  {binom_half}")
print("  -> used in sqrt(1 + beta^2) approximations in SR / fiber optics")

# ------------------------------------------------------------------ #
# S2: EXPANSION MICROSCOPY + CRYO-EM / FROZEN INSECTS
# ------------------------------------------------------------------ #
print()
print(SEP)
print("SECTION 2: EXPANSION MICROSCOPY + CRYO-EM")
print(SEP)

print("""
  EXPANSION MICROSCOPY (ExM):
  Chen et al. Science 2015 -- physically expand biological sample
  with a swellable polyelectrolyte hydrogel BEFORE imaging.

  Protocol:
    1. Label with fluorescent antibodies (NHS-ester anchored)
    2. Infuse acrylamide + sodium acrylate monomer into tissue
    3. Gelate (polymerize) -- proteins anchored to gel mesh
    4. Protease digest (proteinase K) -- breaks protein backbone,
       leaves fluorophore anchors in gel
    5. Swell in deionized water: gel expands ~4x linear (64x volume)
    6. Image expanded gel on standard confocal

  Key equation:
    resolution_effective = diffraction_limit / expansion_factor
    diffraction_limit (confocal) = lambda / (2*NA) ~ 250 nm
    expansion_factor F = 4 (standard ExM), 10 (ultraExM, 10x gel)
    -> effective resolution = 250 / 4 = 62.5 nm (standard)
    -> effective resolution = 250 / 10 = 25 nm  (ultraExM)

  Why it works:
    Hydrogel swelling is isotropic and uniform to ~1% across sample.
    Euclidean distance between anchors scales as F * original_distance.
    Signal-to-noise IMPROVES: background autofluorescence does NOT expand,
    so SNR per pixel increases by ~F^2 = 16x.
""")

# ExM resolution numbers
lam_ex = 488e-9  # excitation nm
NA_obj = 1.2     # water immersion
d_diffraction = lam_ex / (2 * NA_obj) * 1e9  # nm
print(f"  Diffraction limit (lambda=488nm, NA=1.2): {d_diffraction:.1f} nm")
for F in [4, 10, 20]:
    print(f"  ExM expansion {F:2d}x -> effective resolution = {d_diffraction/F:.1f} nm")

print("""
  CRYO-EM AND FROZEN INSECTS:
  Vitrification: rapid cooling (>10^6 K/s) prevents ice crystal formation.
  Sample goes directly from liquid to AMORPHOUS ice (vitreous ice).
  If cooling too slow -> hexagonal ice crystals -> destroy ultrastructure.

  Tools:
    Vitrobot / Leica GP2: blot specimen grid, plunge into liquid ethane
    (-183 C).  Liquid N2 is too slow (Leidenfrost vapor barrier).
    Liquid propane also used.

  For INSECTS and large organisms:
    High-pressure freezing (HPF): 2100 bar applied simultaneously with
    cryo-plunge.  Pressure suppresses ice nucleation for samples up to
    ~200 um thick.  Used for:
      - Drosophila neuromuscular junction
      - C. elegans whole worm
      - Bee wing joints
    After HPF: freeze substitution (FS) in acetone + OsO4 at -90 C -> -20 C
    -> room temp.  Dehydrates and stains simultaneously.
    Embed in Epon resin.  Section at 70-100 nm.

  CTF (Contrast Transfer Function):
    CTF(q) = -2*A*sin(pi*lam*Cs*q^4/2 - pi*lam*dz*q^2)
    q = spatial frequency (1/nm)
    lam = electron wavelength (300 kV: lam = 1.97 pm)
    Cs = spherical aberration (~2 mm for FEI Titan)
    dz = defocus (typical: -1 to -3 um, negative = underfocus)

  CTF zeros (Thon rings) appear at:
    q_k where sin(...)=0 -> information lost at those frequencies
    CTF correction phase-flips and weights Fourier components.

  Fourier Shell Correlation (FSC) resolution criterion:
    Split dataset into two halves -> 3D reconstruct independently
    Correlate in Fourier shells -> FSC(q)
    Resolution = 1/q where FSC(q) = 0.143 (gold standard, Rosenthal & Henderson 2003)
    Or 0.5 (older criterion)
""")

# electron wavelength at 300 kV
m_e = 9.109e-31
e_c = 1.602e-19
h_p = 6.626e-34
c_light = 2.998e8
V_kV = 300e3  # volts
# relativistic de Broglie
lam_e = h_p / np.sqrt(2 * m_e * e_c * V_kV * (1 + e_c * V_kV / (2 * m_e * c_light**2)))
print(f"\n  Electron wavelength at 300 kV: {lam_e*1e12:.3f} pm")
print(f"  Compare: visible light 400-700 nm -> 200,000x shorter")
print(f"  Theoretical resolution: ~1 Angstrom (limited by Cs, not diffraction)")

# ------------------------------------------------------------------ #
# S3: DIRECT IMAGING
# ------------------------------------------------------------------ #
print()
print(SEP)
print("SECTION 3: DIRECT IMAGING -- CORONAGRAPHS, PLANET DETECTION")
print(SEP)

print("""
  DIRECT IMAGING: photograph the planet directly, not via stellar wobble.
  Challenge: star-planet contrast ratio.
    Sun/Jupiter at 10 pc: contrast ~ 1e-9 (infrared)
                           contrast ~ 1e-10 (visible, reflected light)
    Sun/Earth:             contrast ~ 1e-10 (infrared)
                           contrast ~ 1e-10 (visible)

  CORONAGRAPH:
    Lyot coronagraph: opaque focal-plane mask blocks star PSF core.
    IWA (inner working angle): minimum angular separation to detect planet.
      IWA = N_lambda * lambda / D   (N_lambda ~ 2-4 for modern designs)

  Angular separation:
    theta (radians) = a_planet / d_star
    a_planet = semi-major axis (AU), d_star = distance (parsec)
    1 AU / 1 pc = 1 arcsecond (definition of parsec)
    theta (arcsec) = a_AU / d_pc
""")

# planet angular separations
planets = [
    ("Jupiter analog", 5.2, 10.0),
    ("Saturn analog",  9.5, 10.0),
    ("HR 8799 b",      68,  39.4),
    ("Beta Pic b",     9.2,  19.3),
    ("Earth analog",   1.0,  10.0),
]
print(f"  {'Planet':<20} {'a (AU)':>8} {'d (pc)':>8} {'sep (arcsec)':>14} {'sep (mas)':>10}")
print("  " + "-" * 62)
for name, a_AU, d_pc in planets:
    sep_arcsec = a_AU / d_pc
    print(f"  {name:<20} {a_AU:>8.1f} {d_pc:>8.1f} {sep_arcsec:>14.4f} {sep_arcsec*1000:>10.2f}")

print("""
  JWST NIRCam coronagraph:
    IWA ~ 0.40 arcsec at 4.44 um (MASK430R)
    Leakage contrast: ~1e-5 raw, ~1e-6 with reference PSF subtraction
    -> Can detect Jupiter-like planets at 5-10 AU around nearby stars

  ELT / GMT (30-m class):
    D=39m, lambda=1um -> lambda/D = 5.3 mas
    IWA ~ 30 mas -> Earth-like planets at 1 AU within 30 pc
    Planned: Roman Space Telescope hybrid Lyot + SPC -> 1e-9 contrast

  SNR for direct imaging:
    SNR = F_planet * t / sqrt(F_planet*t + F_speckle*t + (sigma_read)^2)
    Speckle noise (residual starlight) often dominates over shot noise.
    ADI (Angular Differential Imaging): rotate field, subtract star PSF
    KLIP (Karhunen-Loeve Image Processing): PCA-based speckle subtraction
    -> same math as PCA/SVD we used for cell expansion data
""")

# IWA for different telescopes
print("  IWA comparison (N_lambda=3, lambda=1um):")
for name, D_m in [("HST", 2.4), ("JWST", 6.5), ("ELT", 39.0), ("VLT", 8.2)]:
    lam_m = 1e-6
    IWA_rad = 3 * lam_m / D_m
    IWA_mas = np.degrees(IWA_rad) * 3600 * 1000
    print(f"  {name:<6} D={D_m:5.1f}m  IWA={IWA_mas:.1f} mas")

# ------------------------------------------------------------------ #
# S4: DIRECTED ENERGY
# ------------------------------------------------------------------ #
print()
print(SEP)
print("SECTION 4: DIRECTED ENERGY -- LASERS, BEAM QUALITY, ANSI MPE")
print(SEP)

print("""
  IRRADIANCE on target:
    I = P / A_spot      [W/m^2]
    A_spot = pi * w^2   for Gaussian beam radius w

  GAUSSIAN BEAM PROPAGATION:
    w(z) = w0 * sqrt(1 + (z/z_R)^2)
    z_R = pi * w0^2 / lambda   (Rayleigh range)
    theta_div = lambda / (pi * w0)   (half-angle divergence, far field)

  BEAM QUALITY M^2:
    Real beam: w(z) = w0 * sqrt(1 + (M^2 * z/z_R)^2)
    M^2 = 1: perfect TEM00 Gaussian (diffraction limited)
    M^2 > 1: multimode, aberrated, fibers
    Typical values: single-mode fiber -> M^2 ~ 1.0-1.1
                    multimode diode bar -> M^2 ~ 10-100
                    VCSEL array        -> M^2 ~ 5-20
""")

lam_dew = 1064e-9   # Nd:YAG
w0_dew  = 0.05      # 5 cm aperture beam waist (exit aperture)
P_dew   = 10e3      # 10 kW
z_R_dew = np.pi * w0_dew**2 / lam_dew
theta_dew = lam_dew / (np.pi * w0_dew)

print(f"  Example: Nd:YAG 1064 nm, w0=5cm, P=10 kW")
print(f"  Rayleigh range z_R = {z_R_dew/1e3:.1f} km")
print(f"  Far-field divergence theta = {np.degrees(theta_dew)*1e6:.2f} urad")

print(f"\n  Irradiance vs range (M^2=1.5):")
M2 = 1.5
print(f"  {'Range (km)':>12} {'w(z) (cm)':>12} {'I (kW/cm^2)':>14}")
for z_km in [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]:
    z_m = z_km * 1e3
    w_z = w0_dew * np.sqrt(1 + (M2 * z_m / z_R_dew)**2)
    A_z = np.pi * w_z**2  # m^2
    I_z = P_dew / A_z / 1e4  # W/cm^2 -> kW/cm^2
    print(f"  {z_km:>12.1f} {w_z*100:>12.2f} {I_z:>14.4f}")

print("""
  ATMOSPHERIC EFFECTS:
    1. Absorption: Beer-Lambert I(z) = I0 * exp(-alpha_atm * z)
       alpha_atm ~ 0.1-1 dB/km (clear day at 1064nm)
    2. Turbulence: Kolmogorov spectrum -> r0 (Fried parameter)
       r0 = 0.185 * (lambda^(6/5)) / (Cn^2 * L)^(3/5)
       Cn^2 ~ 1e-14 m^(-2/3) typical daytime
       Short-term beam wander + long-term beam spread
    3. Thermal blooming: absorbed power heats air -> lensing defocus
       Focus shifts with wind speed v_wind
    4. Adaptive optics: deformable mirror corrects wavefront up to kHz,
       r0 sets number of actuators needed: N_act ~ (D/r0)^2

  ANSI Z136.1 MPE (Maximum Permissible Exposure):
    Eye MPE (1064nm, 0.1s exposure):
      Cornea:  50 J/m^2  (~ 5 mJ/cm^2)
      For P=1 mW HeNe, beam_area=1mm^2: I=1e4 W/m^2, t=0.1s -> E=1000 J/m^2
      -> ABOVE MPE -> laser safety eyewear required

    Skin MPE (1064nm): 2000 J/m^2 for 0.1s (much higher than eye)

    Nominal Ocular Hazard Distance (NOHD):
      NOHD = (1/theta_div) * sqrt(P_peak / (pi/4 * MPE_E)) - z_R
""")

# NOHD calculation
MPE_E = 50      # J/m^2 eye MPE at 1064nm
P_peak_nohd = 10e-3  # 10 mW laser pointer class
theta_nohd = 1.5e-3  # rad divergence (full angle)
NOHD = (2 / theta_nohd) * np.sqrt(P_peak_nohd / (np.pi * MPE_E)) - z_R_dew
print(f"  NOHD example: P=10mW, theta=1.5mrad, MPE=50J/m^2")
print(f"  NOHD ~ {max(NOHD,0):.1f} m (stay further away than this)")

# ------------------------------------------------------------------ #
# S5: ETHICAL HACKING -- CIA TRIAD, COMPLEXITY (EDUCATIONAL)
# ------------------------------------------------------------------ #
print()
print(SEP)
print("SECTION 5: ETHICAL HACKING -- CIA TRIAD, COMPLEXITY (EDUCATIONAL)")
print(SEP)

print("""
  SCOPE: authorized penetration testing, CTF competitions, security
  research, and building defensive systems.  Never use against systems
  you do not own or have written permission to test.

  THE CIA TRIAD:
    Confidentiality -- only authorized parties read data
    Integrity       -- data is not modified without authorization
    Availability    -- systems work when legitimate users need them

  ATTACKER MODELS (for designing defenses):
    Passive: observe/eavesdrop (Eve)
    Active:  modify/inject/replay (Mallory)
    Insider: authorized user abuses access
    Supply chain: compromise a dependency before deployment

  CRYPTOGRAPHIC COMPLEXITY:
    AES-128: brute force = 2^128 operations ~ 3.4e38
    AES-256: brute force = 2^256 ~ 1.16e77
    RSA-2048: factoring hardness ~ exp(cbrt(n)) -- sub-exponential
    ECDSA-256: discrete log ~ 2^128 (same security as AES-128)

    At 1e12 guesses/sec (modern GPU cluster):
""")

gpu_rate = 1e12  # guesses/sec
age_universe_s = 4.3e17
for bits in [40, 64, 80, 128, 256]:
    ops = 2**bits
    years = ops / gpu_rate / 3.15e7
    if years < 1e6:
        note = f"{years:.2e} years"
    elif years < age_universe_s / 3.15e7:
        note = f"{years:.2e} years  (universe age: {age_universe_s/3.15e7:.1e} yrs)"
    else:
        note = f"{years:.2e} years  (>> age of universe)"
    print(f"  2^{bits:3d}: {ops:.2e} ops -> {note}")

print("""
  HASH FUNCTIONS (integrity / password storage):
    MD5:    128-bit output, BROKEN (collision found 2004)
    SHA-1:  160-bit, BROKEN (SHAttered collision 2017)
    SHA-256: 256-bit, SECURE (no known collision)
    bcrypt/Argon2: KEY STRETCHING -- deliberately slow
      work factor=12: ~250ms/hash -> 4 hashes/sec per GPU
      vs MD5: ~1e9 hashes/sec -> 250M slower

  PASSWORD ATTACK COMPLEXITY:
    Dictionary: N_words ~ 1e6 candidates
    Rule-based: append 1-3 digits, capitalize, substitute -> ~1e9
    Brute 8-char alphanumeric: 62^8 ~ 2.18e14
    Brute 12-char alphanumeric: 62^12 ~ 3.22e21

  CTF COMMON PATTERNS:
    1. ROT13 / Caesar cipher: frequency analysis or brute 25 shifts
    2. Base64 decode -> then further encoding layer
    3. XOR with key: known-plaintext -> XOR with ciphertext to get key
    4. Padding oracle: bit-flip CBC IV byte -> deduce plaintext byte
    5. SQL injection: ' OR 1=1 -- (test in authorized lab only)
    6. Buffer overflow (ret2libc): overwrite return address with system()
    7. JWT none alg: change alg to "none", remove signature -> bypass

  DEFENSE CHECKLIST:
    - Input validation at every boundary (never trust user input)
    - Parameterized queries (not f-string SQL)
    - TLS 1.3 everywhere, HSTS headers
    - Bcrypt/Argon2 for passwords, never MD5/SHA-1 for passwords
    - Principle of least privilege (PoLP)
    - Secrets in environment variables, NOT in source code
    - Dependency scanning (Dependabot, Snyk)
    - WAF + rate limiting for web APIs
""")

# XOR demo -- educational
print("  XOR cipher example (educational):")
plaintext = b"ATTACK AT DAWN"
key_byte  = 0x42  # single-byte key
ciphertext = bytes([b ^ key_byte for b in plaintext])
recovered  = bytes([b ^ key_byte for b in ciphertext])
print(f"  Plaintext : {plaintext}")
print(f"  Key       : 0x{key_byte:02X}")
print(f"  Ciphertext: {ciphertext.hex()}")
print(f"  Recovered : {recovered}")
print(f"  XOR key recovery (known plaintext): key = pt[0] ^ ct[0] = 0x{plaintext[0]^ciphertext[0]:02X}")

# ------------------------------------------------------------------ #
# S6: SYMPY ARCHITECTURE ERRORS -- COMMON PITFALLS
# ------------------------------------------------------------------ #
print()
print(SEP)
print("SECTION 6: SYMPY COMMON ERRORS AND CORRECT PATTERNS")
print(SEP)

print("""
  ERROR 1: Using Python == for symbolic equality -> always True/False
    WRONG:  if expr == 0:      (tests Python object identity)
    RIGHT:  if sp.simplify(expr) == 0:
            if sp.Eq(expr, 0):         (unevaluated equation)
            if expr.equals(0):         (numerical comparison fallback)

  ERROR 2: Mixing float and symbolic -> loses exact form
    WRONG:  x = sp.Symbol('x'); expr = x + 3.14159
            -> 3.14159 becomes Float, not Rational
    RIGHT:  expr = x + sp.pi        (exact)
            expr = x + sp.Rational('3.14159')  (exact rational)
            expr = x + sp.Float('3.14159', 30) (30 sig figs)

  ERROR 3: sqrt(-1) returns nan instead of I
    WRONG:  sp.sqrt(-1)   -> I  (this is fine, SymPy handles it)
    BUT:    np.sqrt(-1)   -> RuntimeWarning + nan
    RIGHT:  use sp.sqrt or sp.I explicitly for symbolic work

  ERROR 4: diff returns 0 when variable not detected
    WRONG:  sp.diff(x**2, y)  -> 0  (silent wrong answer)
    RIGHT:  verify your symbol names match; use sp.Symbol('x')
            not x = 'x' (string)

  ERROR 5: series() on Laurent series (negative powers)
    WRONG:  sp.series(sp.exp(t - 1/t), t, 0, 4)
            -> raises ValueError or returns O(t^4)=0 terms only
    RIGHT:  use numerical integral for generating functions
            or sp.residue() for coefficients
            or sp.Laurent() [SymPy >= 1.12]

  ERROR 6: solve() vs solveset() -- solve() can miss solutions
    WRONG:  sp.solve(sp.sin(x), x)  -> [0]  (misses n*pi)
    RIGHT:  sp.solveset(sp.sin(x), x, domain=sp.S.Reals)
            -> ImageSet(Lambda(n, n*pi), Integers)

  ERROR 7: lambdify with numpy -- some SymPy functions not in numpy
    WRONG:  f = sp.lambdify(x, sp.Heaviside(x))
            f(np.array([-1,0,1]))  -> AttributeError Heaviside
    RIGHT:  f = sp.lambdify(x, sp.Heaviside(x), modules=['scipy'])
            or provide custom dict: {"Heaviside": np.heaviside}

  ERROR 8: doit() needed for unevaluated integrals
    WRONG:  sp.Integral(x**2, (x,0,1))  -> Integral object, not 1/3
    RIGHT:  sp.Integral(x**2, (x,0,1)).doit()  -> 1/3

  ERROR 9: Matrix inverse on singular matrix -> raises ShapeError
    WRONG:  M = sp.Matrix([[1,2],[2,4]]); M.inv()
            -> NonInvertibleMatrixError
    RIGHT:  check sp.det(M) first; use M.pinv() for pseudoinverse

  ERROR 10: assumptions on symbols matter for simplification
    WRONG:  x = sp.Symbol('x')
            sp.sqrt(x**2)  -> sqrt(x^2)  (can't simplify)
    RIGHT:  x = sp.Symbol('x', real=True, positive=True)
            sp.sqrt(x**2)  -> x
""")

# live demo of correct patterns
print("  Live demos:")
x_d = sp.Symbol("x", real=True, positive=True)
print(f"  sqrt(x^2) with positive=True: {sp.sqrt(x_d**2)}")

x_d2 = sp.Symbol("x")  # no assumption
print(f"  sqrt(x^2) without assumption: {sp.sqrt(x_d2**2)}")

# solveset vs solve
print(f"  solveset(sin(x), x, Reals) -> {sp.solveset(sp.sin(x_d2), x_d2, domain=sp.S.Reals)}")

# lambdify with numpy
import numpy as np
f_lam = sp.lambdify(x_d2, sp.cos(x_d2)**2 + sp.sin(x_d2)**2, modules="numpy")
print(f"  lambdify cos^2 + sin^2 at x=1.23: {f_lam(1.23):.6f}  (should be 1.0)")

# Rational vs float
expr_rat = sp.Rational(1, 3) * x_d2**2
expr_flt = 0.333333 * x_d2**2
print(f"  Rational form: diff(x^2/3, x) = {sp.diff(expr_rat, x_d2)}")
print(f"  Float form:    diff(0.333*x^2,x) = {sp.diff(expr_flt, x_d2)}")

# ------------------------------------------------------------------ #
# S7: COMSOL ANALOG DESIGN -- FEM WEAK FORM, OP-AMP VIRTUAL GROUND
# ------------------------------------------------------------------ #
print()
print(SEP)
print("SECTION 7: COMSOL ANALOG DESIGN -- FEM WEAK FORM + OP-AMP")
print(SEP)

print("""
  FEM (Finite Element Method) -- COMSOL solves PDE weak forms.

  STRONG FORM (PDE):
    -div(D * grad(u)) = f   in domain Omega
    u = g               on Dirichlet boundary
    D * grad(u) . n = h on Neumann boundary

  WEAK FORM (multiply by test function v, integrate):
    integral[ D * grad(u) . grad(v) ] dOmega
    = integral[ f * v ] dOmega + integral[ h * v ] dGamma_N

  Why weak form?
    1. Only needs u in H^1 (square-integrable first derivative), not C^2
    2. Neumann BC appears NATURALLY in the boundary term
    3. Symmetric bilinear form -> symmetric stiffness matrix K
    4. K * u = F  is the linear system COMSOL assembles

  MESH CONVERGENCE:
    FEM error ~ h^p where h = mesh element size, p = polynomial order
    P1 (linear): error ~ h^2 in energy norm
    P2 (quadratic): error ~ h^3
    Rule of thumb: halve mesh size -> 4x more DOF, 4x better error (P1)
    COMSOL physics-controlled mesh: fine/finer/extra-fine levels

  COMSOL WORKFLOW:
    1. Geometry: build CAD in COMSOL or import STEP/IGES
    2. Materials: assign eps_r, sigma, mu_r (or custom expressions)
    3. Physics: Electrostatics (es), RF (emw), ACDC (mef)
       Weak form contribution: Physics -> Equation View -> Weak Expression
    4. Mesh: auto (physics-controlled) or manual (boundary layers for skin depth)
    5. Study: Stationary / Frequency Domain / Time Dependent / Eigenfrequency
    6. Parametric sweep: Study -> Parametric Sweep over frequency, geometry
    7. Results: derived quantities, line/surface integrals, far-field pattern

  COMSOL FOR FIBER / WAVEGUIDE:
    Mode analysis (eigenfrequency study):
      Port boundary -> solves for k_z (propagation constant)
      n_eff = k_z * c / omega
      GVD = d^2 k_z / d omega^2
    This directly connects to our GS phase retrieval:
      D = -lambda^2 / (2*pi*c) * GVD [ps^2/km -> phase steps]
""")

# FEM 1D Poisson demo with numpy (tridiagonal K)
print("  1D Poisson FEM demo (Laplacian u = -f, P1 elements):")
print("  -u''(x) = 1 on [0,1],  u(0)=u(1)=0  -> exact: u=x(1-x)/2")
N_fem = 8  # internal nodes
h_fem = 1.0 / (N_fem + 1)
# assemble stiffness K and load F
K_fem = (1/h_fem) * (2*np.eye(N_fem) - np.eye(N_fem, k=1) - np.eye(N_fem, k=-1))
F_fem = h_fem * np.ones(N_fem)
u_fem = np.linalg.solve(K_fem, F_fem)
x_fem = np.linspace(h_fem, 1 - h_fem, N_fem)
u_exact = x_fem * (1 - x_fem) / 2
err_L2 = np.sqrt(h_fem * np.sum((u_fem - u_exact)**2))
print(f"  N={N_fem} elements, h={h_fem:.4f}, L2 error = {err_L2:.2e}")
print(f"  Max u (should be 0.125): {u_fem.max():.5f}")

print("""
  OP-AMP VIRTUAL GROUND AND ANALOG DESIGN:

  IDEAL OP-AMP RULES:
    1. V+ = V-       (virtual short at inputs, infinite open-loop gain)
    2. I_in = 0      (infinite input impedance, no current into inputs)

  INVERTING AMPLIFIER:
    Gain = -R_f / R_in
    Virtual ground: V- = V+ = 0 (if V+ tied to GND)
    I through R_in = (V_in - 0) / R_in
    Same I through R_f: V_out = -I * R_f = -V_in * R_f / R_in

  INTEGRATOR (active low-pass):
    Replace R_f with C: V_out = -(1/RC) * integral(V_in) dt
    Gain at frequency f: H(f) = -1 / (j*2*pi*f*RC)
    Used in: signal integration, ADC DAC, active filters

  TRANSIMPEDANCE AMPLIFIER (TIA) -- photodetector readout:
    I_pd (photocurrent) -> R_f -> V_out = -I_pd * R_f
    Bandwidth: f_-3dB = 1 / (2*pi*R_f*C_f)  [C_f feedback cap for stability]
    Noise: Johnson = sqrt(4*k_B*T*R_f) [V/sqrt(Hz)] -> dominates at large R_f
    DESIGN TRADEOFF: large R_f = high transimpedance (sensitivity) but low BW
    For RogueGuard ADC front end: R_f ~ 1 kohm -> f_3dB ~ 160 MHz for C_f=1pF
""")

# TIA design
k_B = 1.381e-23
T_room = 300  # K
for R_f_ohm in [100, 1e3, 10e3, 100e3]:
    C_f = 1e-12  # 1 pF
    f_3dB = 1 / (2 * np.pi * R_f_ohm * C_f)
    V_noise = np.sqrt(4 * k_B * T_room * R_f_ohm)  # V/sqrt(Hz)
    print(f"  TIA R_f={R_f_ohm/1e3:6.2f}kohm: BW={f_3dB/1e6:.1f}MHz, "
          f"noise={V_noise*1e9:.2f} nV/rtHz")

print()
print(SEP)
print("Done.")
print(SEP)
