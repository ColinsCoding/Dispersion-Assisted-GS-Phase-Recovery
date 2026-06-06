"""
_repl_hamiltonian_fiber_nuc.py

S1: Mass matrix -- structural dynamics, M*x'' + K*x = F, modal analysis
S2: Hamiltonian mechanics -- dp/dt dq/dt, phase space, Poisson brackets
S3: Double fiber -- PM fiber, dual-fiber coherent sensing, Sagnac
S4: Photolithography -- public light patterning, resolution, EUV
S5: Nuclear -- binding energy, decay, fission/fusion, Q-value
"""

import numpy as np
import sympy as sp
from scipy import linalg

SEP = "=" * 65

# ------------------------------------------------------------------ #
# S1: MASS MATRIX -- STRUCTURAL DYNAMICS
# ------------------------------------------------------------------ #
print(SEP)
print("SECTION 1: MASS MATRIX -- STRUCTURAL DYNAMICS")
print(SEP)

print("""
  EQUATION OF MOTION (FEM assembled):
    M * x'' + C * x' + K * x = F(t)

    M = mass matrix       [kg]      symmetric positive definite
    C = damping matrix    [kg/s]    Rayleigh: C = alpha*M + beta*K
    K = stiffness matrix  [N/m]     symmetric positive semi-definite
    x = displacement DOF vector
    F = external force vector

  UNDAMPED EIGENVALUE PROBLEM:
    (K - omega^2 * M) * phi = 0
    omega_i = natural frequencies (rad/s)
    phi_i   = mode shapes (eigenvectors)

    Orthogonality: phi_i^T * M * phi_j = 0  (i != j)
                   phi_i^T * K * phi_j = 0  (i != j)
    Mass-normalize: phi_i^T * M * phi_i = 1  -> modal mass = 1

  MODAL DECOUPLING:
    x = Phi * q   (transform to modal coordinates q)
    Phi^T M Phi = I  (identity -- mass normalized)
    Phi^T K Phi = diag(omega_i^2)
    -> N uncoupled SDOFs:  q_i'' + omega_i^2 * q_i = f_i(t)

  2-DOF SPRING-MASS EXAMPLE:
    m1--k1--[m1]--k2--[m2]--k3--wall
    M = [[m1, 0 ], [0,  m2]]
    K = [[k1+k2, -k2], [-k2, k2+k3]]
""")

# 2-DOF system
m1, m2 = 1.0, 2.0    # kg
k1, k2, k3 = 100.0, 200.0, 100.0  # N/m

M = np.array([[m1, 0],
              [0,  m2]])
K = np.array([[k1+k2, -k2],
              [-k2,   k2+k3]])

# solve generalized eigenvalue problem K*phi = omega^2 * M * phi
eigvals, eigvecs = linalg.eigh(K, M)   # returns SORTED eigenvalues
omega_sq = eigvals
omega_n  = np.sqrt(omega_sq)           # rad/s
freq_Hz  = omega_n / (2 * np.pi)

print(f"  m1={m1}kg  m2={m2}kg  k1={k1} k2={k2} k3={k3} N/m")
for i in range(2):
    phi = eigvecs[:, i]
    # verify mass normalization
    mass_norm = phi @ M @ phi
    print(f"  Mode {i+1}: omega={omega_n[i]:.3f} rad/s  f={freq_Hz[i]:.3f} Hz")
    print(f"           phi = [{phi[0]:+.4f}, {phi[1]:+.4f}]  "
          f"phi^T M phi = {mass_norm:.4f}")

# verify orthogonality
cross = eigvecs[:,0] @ M @ eigvecs[:,1]
print(f"  Cross-orthogonality phi1^T M phi2 = {cross:.2e}  (should be ~0)")

print("""
  RAYLEIGH DAMPING:  C = alpha*M + beta*K
    alpha = 2*xi*omega_1*omega_2 / (omega_1 + omega_2)
    beta  = 2*xi / (omega_1 + omega_2)
    Gives critical damping ratio xi at both omega_1 and omega_2.
    Damping ratio at other frequencies: xi(omega) = alpha/(2*omega) + beta*omega/2

  FREQUENCY RESPONSE FUNCTION (FRF):
    H(omega) = (-omega^2*M + i*omega*C + K)^(-1)
    |H(omega)| peaks at resonance, phase shifts 180 deg through it.
    Used in modal testing: tap hammer + accelerometer -> FFT -> FRF.

  COMSOL CONNECTION:
    COMSOL Structural Mechanics module assembles M, K automatically.
    Study -> Eigenfrequency -> solves K*phi = lambda*M*phi.
    Study -> Frequency Domain -> solves H(omega) for each omega.
""")

# FRF magnitude at mode frequencies
print("  FRF magnitude |H11(omega)| sweep (no damping, xi=0.05 added):")
xi = 0.05
C_ray = np.zeros((2,2))
for i in range(2):
    C_ray += (2*xi*omega_n[i]) * (eigvecs[:,i:i+1] @ eigvecs[:,i:i+1].T @ M)

omega_sweep = np.linspace(1, 30, 500)  # rad/s
H_max = []
for w in omega_sweep:
    Z = -w**2 * M + 1j*w * C_ray + K
    H = np.linalg.inv(Z)
    H_max.append(abs(H[0,0]))

peaks_idx = [np.argmin(abs(omega_sweep - omega_n[i])) for i in range(2)]
for i, idx in enumerate(peaks_idx):
    print(f"  omega={omega_sweep[idx]:.2f} rad/s: |H11|={H_max[idx]:.4f} m/N  "
          f"(resonance {i+1})")

# ------------------------------------------------------------------ #
# S2: HAMILTONIAN MECHANICS
# ------------------------------------------------------------------ #
print()
print(SEP)
print("SECTION 2: HAMILTONIAN MECHANICS -- dp/dt, dq/dt, pq")
print(SEP)

print("""
  HAMILTONIAN  H(q, p, t):
    q = generalized coordinates  (position-like)
    p = generalized momenta      (momentum-like, p = dL/dq_dot)
    H = total energy = T + V  (for conservative systems)

  CANONICAL EQUATIONS (Hamilton's equations):
    dq/dt = +dH/dp      "position evolves with momentum gradient"
    dp/dt = -dH/dq      "momentum evolves against position gradient"

  COMPARE to Lagrangian:
    Lagrangian: L = T - V, second-order ODE in q
    Hamiltonian: H = T + V, two first-order ODEs in (q,p)
    Hamiltonian preferred for: phase space analysis, perturbation theory,
    quantum mechanics (p -> -i*hbar*d/dx), statistical mechanics.

  SIMPLE HARMONIC OSCILLATOR:
    H = p^2/(2m) + m*omega^2*q^2/2
    dq/dt = p/m
    dp/dt = -m*omega^2*q
    -> q(t) = A*cos(omega*t + phi)  (circles in phase space)

  PHASE SPACE:
    State = point (q, p).  Trajectory = curve through phase space.
    SHO: ellipses (q^2/A^2 + p^2/B^2 = 1, constant H = E)
    Pendulum: ellipses near origin, separatrix at E=2mgl (unstable fixed pt)
    Liouville's theorem: phase space volume is CONSERVED (incompressible flow).

  POISSON BRACKET:
    {f, g} = sum_i (df/dq_i * dg/dp_i - df/dp_i * dg/dq_i)
    {q_i, p_j} = delta_ij    (fundamental)
    {q_i, q_j} = 0
    {p_i, p_j} = 0
    df/dt = {f, H} + df/dt_explicit
    Conserved quantity: df/dt = 0 -> {f, H} = 0

  QUANTUM CORRESPONDENCE:
    Poisson bracket -> commutator:  {f,g} -> (1/ih) [F, G]
    [x, p] = ih  (canonical commutation relation)
    This is why Hamiltonian mechanics is the natural bridge to QM.
    Schrodinger: ih * dpsi/dt = H_hat * psi  (H_hat = quantum Hamiltonian)

  CONNECTION TO MASS MATRIX (above):
    For mechanical system with M, K:
    T = (1/2) x'^T M x'   -> p = M * x'  -> x' = M^(-1) p
    H = (1/2) p^T M^(-1) p + (1/2) x^T K x
    dp/dt = -K*x   (Newton F=ma in disguise)
    dq/dt = M^(-1)*p
""")

# SymPy Hamiltonian demo
q_s, p_s, m_s, w_s, t_s = sp.symbols("q p m omega t", real=True, positive=True)
H_sho = p_s**2 / (2*m_s) + m_s * w_s**2 * q_s**2 / 2

dq_dt = sp.diff(H_sho, p_s)
dp_dt = -sp.diff(H_sho, q_s)

print(f"  SymPy SHO:  H = {H_sho}")
print(f"  dq/dt = +dH/dp = {dq_dt}")
print(f"  dp/dt = -dH/dq = {dp_dt}")

# phase space trajectory (numerical)
print()
print("  Phase space: SHO numerical trajectory (m=1, omega=2, E=1):")
m_val, w_val = 1.0, 2.0
E_val = 1.0  # total energy
A_q = np.sqrt(2*E_val / (m_val * w_val**2))  # amplitude
A_p = np.sqrt(2*E_val * m_val)

theta = np.linspace(0, 2*np.pi, 5, endpoint=False)
print(f"  {'theta':>10} {'q':>10} {'p':>10} {'H=E':>10}")
for th in theta:
    q_val = A_q * np.cos(th)
    p_val = -A_p * np.sin(th)
    H_val = p_val**2/(2*m_val) + m_val*w_val**2*q_val**2/2
    print(f"  {np.degrees(th):>10.1f} {q_val:>10.4f} {p_val:>10.4f} {H_val:>10.4f}")

# ------------------------------------------------------------------ #
# S3: DOUBLE FIBER -- PM FIBER, DUAL-FIBER SENSING, SAGNAC
# ------------------------------------------------------------------ #
print()
print(SEP)
print("SECTION 3: DOUBLE FIBER -- PM FIBER, DUAL-FIBER, SAGNAC")
print(SEP)

print("""
  POLARIZATION-MAINTAINING (PM) FIBER:
    Standard SMF: circular core, two degenerate polarization modes.
    Tiny perturbation (bend, stress) mixes them -> polarization wanders.
    PM fiber: intentional asymmetry (PANDA, bow-tie, elliptical core)
    -> lifts degeneracy: n_fast != n_slow
    Beat length: L_B = lambda / (n_slow - n_fast)
    Birefringence B = n_slow - n_fast ~ 3e-4 to 5e-4 (Panda fiber, 1550nm)
    L_B ~ 1550e-9 / 4e-4 = 3.9 mm  (polarization rotates every 3.9mm)

  DUAL-FIBER COHERENT SENSING:
    Two fibers: reference arm + sensing arm.
    Beat signal: I = I_ref + I_sig + 2*sqrt(I_ref*I_sig)*cos(phi_sig - phi_ref)
    Measures phase difference -> strain, temperature, refractive index change.
    OTDR (optical time-domain reflectometer): single fiber, pulse + backscatter.
    phi-OTDR: coherent detection, phase-sensitive -> acoustic sensing.

  SAGNAC INTERFEROMETER (fiber gyroscope):
    Light split into two counter-propagating beams around a loop.
    Rotation rate Omega -> phase difference:
      delta_phi = (8*pi*N*A / (lambda*c)) * Omega
      N = number of fiber turns, A = loop area
    Fiber optic gyroscope (FOG): N=1000 turns, A=pi*(0.05)^2 m^2
    -> delta_phi / Omega = 8*pi*1000*pi*0.0025 / (1550e-9 * 3e8)

  DUAL-FIBER for DISPERSION MEASUREMENT (our GS setup):
    I1(x) = |E(z1, x)|^2   -> measured at z1 (after D1 dispersion)
    I2(x) = |E(z2, x)|^2   -> measured at z2 (after D2 dispersion)
    GS retrieves phi(nu) from {I1, I2, D1, D2}
    Two fibers with DIFFERENT D values -> maximum diversity
    diversity metric: corr(I1, I2) should be < 0.5 for good retrieval
    |D1 - D2| >= 5000 (from GS convergence fix in memory)
""")

# FOG scale factor
N_fib   = 1000
r_fib   = 0.05      # m, coil radius
A_fib   = np.pi * r_fib**2
lam_fib = 1550e-9
c_light = 3e8

SF = 8 * np.pi * N_fib * A_fib / (lam_fib * c_light)
print(f"  FOG scale factor: {SF:.4f} rad/(rad/s)")
print(f"  Earth rotation = 7.27e-5 rad/s -> phase = {SF*7.27e-5*1000:.3f} mrad")
print(f"  Sensitivity: 1 mrad/sqrt(Hz) -> {1e-3/SF*1e6:.3f} urad/s/sqrt(Hz)")

print("""
  FIBER TYPES COMPARISON:
  Type            Core/Clad  n_core  n_clad  NA     Typical use
  ------------------------------------------------------------------
  SMF-28          9/125 um   1.4677  1.4625  0.13   Telecom, sensor
  PM (PANDA)      8/125 um   same    same    0.12   Gyro, coherent
  DCF (dispersion compensating) small core  hi n  high NL  GVD=-100
  HNLF (nonlinear)  3/125   1.48    1.46    0.3    FWM, rogue waves
  LMA (large mode area)  25/250  1.45   low  0.06  High power laser
  Photonic crystal  varies  air holes        low   Endlessly SM, THz

  ROGUE WAVES in HNLF:
    HNLF has gamma_NL ~ 10 W^{-1}km^{-1} vs SMF 1.3 W^{-1}km^{-1}
    Modulation instability (MI) gain: g(omega) = gamma*P * sqrt(1-(omega/omega_MI)^2)
    MI breaks CW into soliton train -> rare events -> rogue waves
    RogueGuard monitors I(t) for events > 2*sigma_Hs (significant wave height)
""")

# GS diversity check
print("  GS dispersion diversity check:")
for D1, D2 in [(0, 5000), (0, -600), (1000, 6000), (-500, 4500), (0, 300)]:
    delta = abs(D1 - D2)
    ok = "OK" if delta >= 5000 else "FAIL -- too similar"
    print(f"  D1={D1:6d}  D2={D2:6d}  |D1-D2|={delta:5d}  {ok}")

# ------------------------------------------------------------------ #
# S4: PHOTOLITHOGRAPHY -- PUBLIC LIGHT
# ------------------------------------------------------------------ #
print()
print(SEP)
print("SECTION 4: PHOTOLITHOGRAPHY -- PUBLIC LIGHT IN SEMICONDUCTOR FAB")
print(SEP)

print("""
  PHOTOLITHOGRAPHY: use light to pattern a photoresist,
  then etch to transfer the pattern to silicon.

  STEPS:
    1. Spin-coat photoresist on wafer (positive or negative tone)
    2. Pre-bake (soft bake) -- remove solvent, harden resist
    3. Expose: project mask pattern onto wafer with stepper/scanner
    4. Post-exposure bake (PEB) -- acid diffusion in chemically amplified resist
    5. Develop: dissolve exposed (positive) or unexposed (negative) resist
    6. Inspect, etch or implant, then strip resist

  RESOLUTION (Rayleigh criterion for lithography):
    CD_min = k1 * lambda / NA
    CD = critical dimension (minimum feature size)
    k1 = process factor (0.25 < k1 < 0.8; k1=0.25 is diffraction limit)
    lambda = exposure wavelength
    NA = numerical aperture of projection lens

  DEPTH OF FOCUS:
    DOF = k2 * lambda / NA^2
    Increasing NA improves resolution but SHRINKS DOF.
    At NA=1.35 (immersion): DOF ~ 100nm -> wafer flatness critical.

  WAVELENGTH EVOLUTION:
  Generation  lambda    NA      CD_min   Year (approx)
  ----------------------------------------------------------
  i-line      365 nm   0.70    250 nm   1990s
  KrF         248 nm   0.80    180 nm   late 1990s
  ArF (dry)   193 nm   0.93    130 nm   early 2000s
  ArF immersion 193nm  1.35     38 nm   2010s (water, n=1.44)
  EUV         13.5 nm  0.33      7 nm   2020s (ASML NXE)
  High-NA EUV 13.5 nm  0.55      2 nm   2025+ (ASML EXE)

  IMMERSION LITHOGRAPHY:
    Fill gap between lens and wafer with water (n=1.44).
    Effective wavelength: lambda_eff = lambda / n_water = 193/1.44 = 134 nm.
    NA_immersion = n_water * sin(theta) -> max NA = 1.35.

  EUV (Extreme Ultraviolet):
    lambda = 13.5 nm (Sn plasma source, 13.5 nm from 13.5 eV photons).
    No refractive optics -- all reflective (Mo/Si multilayer mirrors, 70% reflectivity).
    Requires vacuum (EUV absorbed by air in mm).
    Light source: Sn droplets + CO2 laser -> 200W EUV power.
    Throughput: ~100-125 wafers/hr (ASML NXE:3400).

  DOUBLE/MULTIPLE PATTERNING:
    To go below k1=0.25 with 193nm ArF: expose TWICE with offset mask.
    LELE (Litho-Etch-Litho-Etch): pattern odd + even lines separately.
    SADP (Self-Aligned Double Patterning): spacer = half original pitch.
    SAQP (quad patterning): pitch/4 features.
    These are "double fiber" equivalents -- two passes of light.
""")

# lithography resolution table
print("  Resolution vs NA (lambda=193nm ArF, k1=0.28):")
print(f"  {'NA':>6} {'CD_min (nm)':>13} {'DOF (nm)':>10} {'lambda_eff (nm)':>16}")
print("  " + "-" * 46)
lam_arf = 193.0  # nm
k1_val  = 0.28
k2_val  = 0.5
for NA_val in [0.60, 0.75, 0.93, 1.10, 1.20, 1.35]:
    CD = k1_val * lam_arf / NA_val
    DOF = k2_val * lam_arf / NA_val**2
    lam_eff = lam_arf / NA_val if NA_val <= 1 else lam_arf / NA_val  # in medium
    print(f"  {NA_val:>6.2f} {CD:>13.1f} {DOF:>10.1f} {lam_eff:>16.1f}")

# ------------------------------------------------------------------ #
# S5: NUCLEAR PHYSICS
# ------------------------------------------------------------------ #
print()
print(SEP)
print("SECTION 5: NUCLEAR PHYSICS -- BINDING ENERGY, DECAY, FISSION/FUSION")
print(SEP)

# constants
u_kg    = 1.66054e-27   # kg (atomic mass unit)
c_sq    = (2.998e8)**2  # m^2/s^2
MeV_J   = 1.602e-13     # J per MeV
m_p_u   = 1.007276      # proton mass in u
m_n_u   = 1.008665      # neutron mass in u
m_e_u   = 0.000549      # electron mass in u

print("""
  NUCLEAR NOTATION:
    A = mass number (protons + neutrons)
    Z = atomic number (protons)
    N = A - Z (neutrons)
    Nuclide: ^A_Z X  e.g. ^4_2 He (alpha particle)

  BINDING ENERGY:
    B = (Z*m_p + N*m_n - M_nucleus) * c^2
    Mass defect: delta_m = Z*m_p + N*m_n - M_nucleus
    B/A = binding energy per nucleon [MeV]
    Peak: Fe-56, Ni-62 at ~8.8 MeV/nucleon (most stable)
    Light nuclei: lower B/A -> fusion releases energy
    Heavy nuclei: lower B/A -> fission releases energy

  SEMI-EMPIRICAL MASS FORMULA (Bethe-Weizsacker):
    B = a_V*A - a_S*A^(2/3) - a_C*Z^2/A^(1/3) - a_A*(A-2Z)^2/A + delta
    a_V = 15.75 MeV   (volume: each nucleon has ~15MeV binding)
    a_S = 17.80 MeV   (surface: nucleons on surface less bound)
    a_C =  0.711 MeV  (Coulomb repulsion of protons)
    a_A = 23.70 MeV   (asymmetry: equal N,Z preferred)
    delta: pairing (+12/sqrt(A) even-even, 0 odd-A, -12/sqrt(A) odd-odd)
""")

a_V = 15.75; a_S = 17.80; a_C = 0.711; a_A = 23.70

def SEMF_binding(A, Z):
    N = A - Z
    if A % 2 == 1:
        delta = 0
    elif Z % 2 == 0:   # even-even
        delta = 12.0 / np.sqrt(A)
    else:              # odd-odd
        delta = -12.0 / np.sqrt(A)
    B = (a_V*A - a_S*A**(2/3) - a_C*Z**2/A**(1/3)
         - a_A*(A-2*Z)**2/A + delta)
    return B

nuclides = [
    ("H-1",   1,  1), ("He-4",  4,  2), ("Li-6",  6,  3),
    ("C-12", 12,  6), ("O-16", 16,  8), ("Fe-56", 56, 26),
    ("Ni-62",62, 28), ("Sr-90", 90, 38), ("U-235",235, 92),
    ("U-238",238, 92), ("Pu-239",239,94),
]

print(f"  {'Nuclide':<10} {'A':>4} {'Z':>4} {'B (MeV)':>10} {'B/A (MeV)':>11}")
print("  " + "-" * 40)
for name, A, Z in nuclides:
    B = SEMF_binding(A, Z)
    print(f"  {name:<10} {A:>4} {Z:>4} {B:>10.2f} {B/A:>11.4f}")

print("""
  RADIOACTIVE DECAY:
    N(t) = N0 * exp(-lambda_d * t)
    lambda_d = ln(2) / t_half    (decay constant)
    Activity: A(t) = lambda_d * N(t)   [Becquerel = decays/s]
    1 Curie = 3.7e10 Bq (activity of 1g Ra-226)

  DECAY MODES:
    Alpha:  ^A_Z X -> ^(A-4)_(Z-2) Y + ^4_2 He   (heavy nuclei)
    Beta-:  n -> p + e^- + anti-neutrino           (neutron-rich)
    Beta+:  p -> n + e^+ + neutrino                (proton-rich)
    Gamma:  excited nucleus -> ground + photon     (follows alpha/beta)
    EC:     electron capture: p + e^- -> n + neutrino

  Q-VALUE (energy released):
    Q = (M_parent - M_daughters) * c^2   [MeV]
    Q > 0: exothermic (spontaneous)
    Q < 0: endothermic (requires energy input)
""")

# decay calculation
print("  Radioactive decay examples:")
examples = [
    ("C-14",   5730,    "yr",  3.156e7),   # t_half in years, conv to s
    ("I-131",  8.02,    "day", 86400),
    ("Cs-137", 30.17,   "yr",  3.156e7),
    ("U-238",  4.47e9,  "yr",  3.156e7),
    ("Po-214", 164e-6,  "s",   1),
]
print(f"  {'Nuclide':<10} {'t_half':>14} {'lambda (1/s)':>14} {'N/N0 at 1yr':>14}")
print("  " + "-" * 55)
for name, t_half, unit, conv in examples:
    lam_d = np.log(2) / (t_half * conv)
    t_1yr = 3.156e7  # 1 year in seconds
    ratio = np.exp(-lam_d * t_1yr)
    print(f"  {name:<10} {t_half:>10.3g} {unit:<3} {lam_d:>14.3e} {ratio:>14.6f}")

print("""
  FISSION (U-235 + n):
    n + U-235 -> Ba-141 + Kr-92 + 3n + Q
    Q ~ 200 MeV per fission (compare: chemical bond ~ 1 eV)
    1 kg U-235 fully fissioned ~ 20 kilotons TNT equivalent
    Critical mass (bare sphere, weapons grade): ~52 kg U-235
    Reflected (Be/U reflector): ~15 kg

  FUSION (D-T):
    D + T -> He-4 + n   Q = 17.6 MeV
    D-D -> He-3 + n  OR  T + p  Q ~ 3-4 MeV
    Ignition condition (Lawson): n * tau_E > 1e20 m^-3 s (for T=10 keV)
    ITER: 500 MW fusion, 50 MW input (Q=10)
    Inertial confinement (NIF): 3.15 MJ fusion from 2.05 MJ laser (Q=1.54, 2023)

  CONNECTION TO OPTICS / ROGUEGUARD:
    Gamma rays from decay are EM waves: E = hf, f ~ 10^20 Hz
    Same Maxwell equations, same photon concept as our 1550nm laser
    Nuclear timing: gamma coincidence ~ 1 ns resolution
    Scintillator + photodetector: same TIA front end as RogueGuard ADC
    Phoswich detector: NaI(Tl) crystal + PMT -> same Poisson statistics
    as photon counting in our fiber sensor (shot noise ~ sqrt(N))
""")

# energy comparison
print("  Energy scale comparison:")
energies = [
    ("Thermal kT (room)",     0.025,         "eV"),
    ("Visible photon 550nm",  2.25,          "eV"),
    ("Chemical bond (C-C)",   3.6,           "eV"),
    ("Hydrogen ionization",   13.6,          "eV"),
    ("X-ray photon 10keV",    10e3,          "eV"),
    ("Beta decay (avg)",      0.5e6,         "eV"),
    ("Alpha particle U-238",  4.27e6,        "eV"),
    ("D-T fusion Q",          17.6e6,        "eV"),
    ("Fission U-235 Q",       200e6,         "eV"),
    ("1 gram U-235 fissioned", 200e6*6.02e23/235*1e6, "eV"),
]
for name, E_eV, unit in energies:
    if E_eV < 1e6:
        print(f"  {name:<30}  {E_eV:.3g} eV")
    else:
        print(f"  {name:<30}  {E_eV:.3g} eV  =  {E_eV*1.602e-19:.3e} J")

print()
print(SEP)
print("Done.")
print(SEP)
