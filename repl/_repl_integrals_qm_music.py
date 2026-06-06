"""
_repl_integrals_qm_music.py

S1: SymPy integrals -- definite, indefinite, improper, with matplotlib
S2: QM integrals -- normalization, <x> <p> <H>, harmonic oscillator
S3: Music and Fourier -- harmonics up to ~2kHz, spectrum, equal temperament
S4: Physics -> ML career path -- the exact curriculum mapping
S5: Fiber Sagnac interferometer -- phase, sensitivity, gyroscope
"""

import numpy as np
import sympy as sp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

SEP = "=" * 65
OUT = os.path.join(os.path.dirname(__file__), "_out_integrals_qm_music.png")

# ------------------------------------------------------------------ #
# S1: SYMPY INTEGRALS + MATPLOTLIB
# ------------------------------------------------------------------ #
print(SEP)
print("SECTION 1: SYMPY INTEGRALS")
print(SEP)

x, t, a, b, n_sym = sp.symbols("x t a b n", real=True, positive=True)

# --- indefinite integrals ---
print("\n  INDEFINITE INTEGRALS (antiderivatives):")
indef = [
    (x**n_sym,          "x^n"),
    (sp.sin(x),         "sin(x)"),
    (sp.exp(-a*x),      "exp(-a*x)"),
    (1/(1+x**2),        "1/(1+x^2)"),
    (sp.log(x),         "ln(x)"),
    (x*sp.exp(-x),      "x*exp(-x)"),
    (1/sp.sqrt(1-x**2), "1/sqrt(1-x^2)"),
]
for expr, label in indef:
    result = sp.integrate(expr, x)
    print(f"  int {label} dx = {result}")

# --- definite integrals ---
print("\n  DEFINITE INTEGRALS:")
defin = [
    (sp.exp(-x**2),    (x, -sp.oo, sp.oo),    "Gaussian int"),
    (sp.sin(x),        (x, 0, sp.pi),          "sin 0 to pi"),
    (x**2,             (x, 0, 1),              "x^2 0 to 1"),
    (1/x,              (x, 1, sp.exp(1)),      "1/x 1 to e"),
    (sp.exp(-x),       (x, 0, sp.oo),          "exp(-x) 0 to inf"),
    (x*sp.exp(-x**2),  (x, 0, sp.oo),          "x*exp(-x^2)"),
]
for expr, bounds, label in defin:
    result = sp.integrate(expr, bounds)
    print(f"  int({label}) = {result} = {float(result.evalf()):.6f}")

# --- integration by parts ---
print("\n  INTEGRATION BY PARTS: int u dv = uv - int v du")
ibp_cases = [
    (x*sp.exp(-x),   "x*exp(-x)"),
    (x*sp.cos(x),    "x*cos(x)"),
    (x**2*sp.sin(x), "x^2*sin(x)"),
]
for expr, label in ibp_cases:
    result = sp.integrate(expr, x)
    print(f"  int {label} dx = {result}")

# --- matplotlib: visualize integrals ---
fig, axes = plt.subplots(2, 3, figsize=(14, 8))
fig.suptitle("Integrals: Area Under Curve", fontsize=14)

x_num = np.linspace(-3, 3, 500)
plots = [
    (np.exp(-x_num**2),   r"$e^{-x^2}$, int=$\sqrt{\pi}$", (-3,3)),
    (np.sin(x_num),        r"$\sin(x)$, int[0,pi]=2",       (-3,3)),
    (x_num**2,             r"$x^2$, int[0,1]=1/3",           (0,1.5)),
    (1/(1+x_num**2),       r"$1/(1+x^2)$, int=arctan",      (-3,3)),
    (x_num*np.exp(-x_num), r"$x e^{-x}$, int[0,inf]=1",      (0,3)),
    (np.abs(np.sin(x_num)), r"$|\sin(x)|$",                  (-3,3)),
]
for ax, (y, title, (xa, xb)) in zip(axes.flat, plots):
    mask = (x_num >= xa) & (x_num <= xb)
    ax.plot(x_num, y, "b-", lw=1.5)
    ax.fill_between(x_num[mask], y[mask], alpha=0.3, color="blue")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_title(title, fontsize=9)
    ax.set_xlim(xa - 0.5, xb + 0.5)
    ax.grid(True, alpha=0.3)

plt.tight_layout()

# ------------------------------------------------------------------ #
# S2: QM INTEGRALS -- HARMONIC OSCILLATOR
# ------------------------------------------------------------------ #
print()
print(SEP)
print("SECTION 2: QUANTUM MECHANICS INTEGRALS")
print(SEP)

print("""
  QUANTUM STATE: psi(x) = amplitude at position x
    Probability density: |psi(x)|^2
    Normalization: integral |psi|^2 dx = 1  (particle must be somewhere)
    Expectation value <A> = integral psi* A_hat psi dx

  QUANTUM HARMONIC OSCILLATOR:
    H = p^2/(2m) + (1/2)*m*omega^2*x^2
    Energy eigenvalues: E_n = hbar*omega*(n + 1/2),  n = 0, 1, 2, ...
    Ground state (n=0):
      psi_0(x) = (m*omega/(pi*hbar))^(1/4) * exp(-m*omega*x^2 / (2*hbar))
    General:
      psi_n(x) = (1/sqrt(2^n * n!)) * (m*omega/(pi*hbar))^(1/4)
                 * H_n(sqrt(m*omega/hbar)*x) * exp(-m*omega*x^2/(2*hbar))
    H_n = Hermite polynomial
      H_0 = 1,  H_1 = 2xi,  H_2 = 4xi^2-2,  H_3 = 8xi^3-12xi

  KEY INTEGRALS:
    Gaussian: int exp(-alpha*x^2) dx = sqrt(pi/alpha)
    Gaussian moments: int x^2 * exp(-alpha*x^2) dx = (1/2)*sqrt(pi/alpha^3)
    Orthogonality: int psi_m* psi_n dx = delta_mn
""")

xi = sp.Symbol("xi", real=True)
hbar_s, m_s, omega_s = sp.symbols("hbar m omega", positive=True)
alpha_s = m_s * omega_s / hbar_s

# SymPy: normalize ground state
psi0_s = (alpha_s / sp.pi)**sp.Rational(1,4) * sp.exp(-alpha_s * xi**2 / 2)
norm_sq = sp.integrate(psi0_s**2, (xi, -sp.oo, sp.oo))
print(f"  Normalization check: int |psi_0|^2 = {sp.simplify(norm_sq)}")

# expectation values
x_exp = sp.integrate(xi * psi0_s**2, (xi, -sp.oo, sp.oo))
x2_exp = sp.integrate(xi**2 * psi0_s**2, (xi, -sp.oo, sp.oo))
print(f"  <x>   = {sp.simplify(x_exp)}  (zero by symmetry)")
print(f"  <x^2> = {sp.simplify(x2_exp)}  (ground state position spread)")
print(f"  <x^2> in terms of hbar,m,omega = hbar/(2*m*omega)")

# momentum via -i*hbar*d/dx
dpsi0 = sp.diff(psi0_s, xi)
p_exp = -sp.I * sp.integrate(psi0_s * dpsi0, (xi, -sp.oo, sp.oo))
print(f"  <p>   = {sp.simplify(p_exp)}  (zero by symmetry)")

# <p^2> = -hbar^2 int psi* d^2psi/dx^2
d2psi0 = sp.diff(psi0_s, xi, 2)
p2_exp = -hbar_s**2 * sp.integrate(psi0_s * d2psi0, (xi, -sp.oo, sp.oo))
print(f"  <p^2> = {sp.simplify(p2_exp)}")
print(f"  <p^2> should be m*hbar*omega/2 = (m*omega*hbar)/2")

print("""
  HEISENBERG UNCERTAINTY:
    sigma_x = sqrt(<x^2> - <x>^2) = sqrt(hbar/(2*m*omega))
    sigma_p = sqrt(<p^2> - <p>^2) = sqrt(m*omega*hbar/2)
    sigma_x * sigma_p = hbar/2  (minimum uncertainty for ground state!)
    Ground state saturates the uncertainty principle.

  ENERGY:
    <H> = <p^2>/(2m) + (1/2)*m*omega^2*<x^2>
        = (hbar*omega/2)/2 + (hbar*omega/2)/2... wait:
    <T> = <p^2>/(2m) = (m*omega*hbar/2)/(2m) = hbar*omega/4
    <V> = m*omega^2/2 * <x^2> = m*omega^2/2 * hbar/(2*m*omega) = hbar*omega/4
    <H> = <T> + <V> = hbar*omega/2 = E_0  (virial theorem: <T>=<V> for SHO)

  INFINITE SQUARE WELL (particle in box):
    psi_n(x) = sqrt(2/L) * sin(n*pi*x/L),  0 < x < L
    E_n = n^2 * pi^2 * hbar^2 / (2*m*L^2)
    <x> = L/2  (by symmetry)
    <x^2> = L^2*(1/3 - 1/(2*n^2*pi^2))
""")

# numerics for harmonic oscillator wavefunctions
print("  HO wavefunctions at x_scaled = xi = x*sqrt(m*omega/hbar):")
print("  (using dimensionless xi; set hbar=m=omega=1)")
from scipy.special import hermite, factorial
import numpy as np

xi_num = np.linspace(-4, 4, 300)

fig2, ax2 = plt.subplots(1, 2, figsize=(12, 5))
fig2.suptitle("Quantum Harmonic Oscillator Wavefunctions", fontsize=13)

colors = ["blue", "red", "green", "purple", "orange"]
for n_val in range(5):
    Hn = hermite(n_val)
    norm = 1.0 / np.sqrt(2**n_val * float(factorial(n_val)) * np.sqrt(np.pi))
    psi_n = norm * Hn(xi_num) * np.exp(-xi_num**2 / 2)
    E_n = n_val + 0.5
    ax2[0].plot(xi_num, psi_n + E_n, color=colors[n_val],
                label=f"n={n_val}, E={E_n}")
    ax2[1].plot(xi_num, psi_n**2, color=colors[n_val],
                label=f"n={n_val}")
    print(f"  n={n_val}: E={(n_val+0.5):.1f} hbar*omega, "
          f"nodes={n_val}, max|psi|={abs(psi_n).max():.4f}")

V = 0.5 * xi_num**2
ax2[0].plot(xi_num, V, "k--", lw=1, label="V(xi)")
ax2[0].set_xlabel("xi = x sqrt(m*omega/hbar)")
ax2[0].set_ylabel("psi_n(xi) + E_n")
ax2[0].set_title("Wavefunctions (offset by energy)")
ax2[0].legend(fontsize=8)
ax2[0].set_ylim(-0.5, 6)
ax2[0].grid(True, alpha=0.3)

ax2[1].set_xlabel("xi")
ax2[1].set_ylabel("|psi_n|^2")
ax2[1].set_title("Probability Densities")
ax2[1].legend(fontsize=8)
ax2[1].grid(True, alpha=0.3)

plt.tight_layout()

# ------------------------------------------------------------------ #
# S3: MUSIC AND FOURIER -- HARMONICS UP TO ~2000 Hz
# ------------------------------------------------------------------ #
print()
print(SEP)
print("SECTION 3: MUSIC, FOURIER, HARMONICS UP TO ~2000 Hz")
print(SEP)

print("""
  EQUAL TEMPERAMENT (12-TET):
    12 semitones per octave.
    Frequency ratio per semitone: 2^(1/12) = 1.0595
    A4 = 440 Hz (concert pitch, ISO 16).
    Note frequency: f(n) = 440 * 2^((n-69)/12)  where n=MIDI note number.

  HARMONIC SERIES:
    A vibrating string produces: f, 2f, 3f, 4f, 5f, ... (overtones).
    Timbre = relative amplitudes of harmonics.
    Sine wave: only fundamental (pure, boring).
    Violin: rich harmonic series -> warm.
    Clarinet: odd harmonics only (cylindrical bore, closed end).
    Trumpet: full harmonic series.

  NYQUIST AND MUSIC:
    CD: f_s = 44100 Hz -> Nyquist = 22050 Hz (covers 20 Hz - 20 kHz).
    Human pitch perception: 20 Hz - 20000 Hz.
    Fundamental of piano: A0 = 27.5 Hz ... C8 = 4186 Hz.
    "Up to 1999 Hz": covers all piano fundamentals + first few harmonics.
    Above 2 kHz: mostly harmonics/timbre, not melodic pitch.

  EQUAL TEMPERAMENT FREQUENCIES (A4=440, selected notes up to ~2000 Hz):
""")

# equal temperament
def note_freq(midi):
    return 440.0 * 2**((midi - 69) / 12.0)

note_names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
print(f"  {'Note':<6} {'MIDI':>5} {'Freq (Hz)':>12}")
print("  " + "-" * 24)
for octave in range(3, 8):
    for semitone in range(12):
        midi = 12 * (octave + 1) + semitone
        freq = note_freq(midi)
        if freq > 2100:
            break
        name = f"{note_names[semitone]}{octave}"
        print(f"  {name:<6} {midi:>5} {freq:>12.2f}")
    if note_freq(12*(octave+1)) > 2100:
        break

# synthesize a chord and compute spectrum
f_s = 44100
duration = 1.0
t_audio = np.linspace(0, duration, int(f_s * duration), endpoint=False)

# A major chord: A4=440, C#5=554.37, E5=659.26
A4 = 440.0; Cs5 = note_freq(73); E5 = note_freq(76)
chord = (np.sin(2*np.pi*A4*t_audio) +
         np.sin(2*np.pi*Cs5*t_audio) +
         np.sin(2*np.pi*E5*t_audio))

# FFT
N_fft = len(t_audio)
freqs = np.fft.rfftfreq(N_fft, 1/f_s)
spectrum = np.abs(np.fft.rfft(chord)) / N_fft

# peaks near expected frequencies
print(f"\n  A-major chord spectrum (A4={A4:.1f}, C#5={Cs5:.2f}, E5={E5:.2f} Hz):")
for target in [A4, Cs5, E5]:
    idx = np.argmin(np.abs(freqs - target))
    print(f"  Peak at {freqs[idx]:.2f} Hz, amplitude = {spectrum[idx]:.4f}")

# plot spectrum
fig3, axes3 = plt.subplots(2, 2, figsize=(12, 8))
fig3.suptitle("Music, Fourier, and Harmonics", fontsize=13)

# chord spectrum up to 2000 Hz
mask = freqs <= 2000
axes3[0,0].plot(freqs[mask], spectrum[mask], "b-", lw=0.8)
axes3[0,0].set_xlabel("Frequency (Hz)")
axes3[0,0].set_ylabel("Amplitude")
axes3[0,0].set_title("A-major chord spectrum (0-2000 Hz)")
axes3[0,0].grid(True, alpha=0.3)
for f0, name in [(A4,"A4"), (Cs5,"C#5"), (E5,"E5")]:
    axes3[0,0].axvline(f0, color="red", alpha=0.6, ls="--")
    axes3[0,0].text(f0+5, spectrum[np.argmin(np.abs(freqs-f0))]*0.8,
                    name, fontsize=8, color="red")

# equal temperament frequency lattice
midi_range = np.arange(48, 85)
freqs_et = np.array([note_freq(m) for m in midi_range])
axes3[0,1].semilogy(midi_range, freqs_et, "bo-", ms=3)
axes3[0,1].axhline(2000, color="red", ls="--", label="2000 Hz")
axes3[0,1].set_xlabel("MIDI note number")
axes3[0,1].set_ylabel("Frequency (Hz, log scale)")
axes3[0,1].set_title("Equal temperament: exponential frequency")
axes3[0,1].legend()
axes3[0,1].grid(True, alpha=0.3)

# harmonic series for A2=110 Hz
f_fund = 110.0
n_harm = np.arange(1, 19)
f_harm = f_fund * n_harm
amp_harm = 1.0 / n_harm  # 1/n falloff (sawtooth-like)
axes3[1,0].stem(f_harm[f_harm <= 2000],
                amp_harm[f_harm <= 2000],
                linefmt="b-", markerfmt="bo", basefmt="k-")
axes3[1,0].set_xlabel("Frequency (Hz)")
axes3[1,0].set_ylabel("Amplitude (1/n)")
axes3[1,0].set_title(f"Harmonic series: A2={f_fund}Hz, 1/n rolloff")
axes3[1,0].grid(True, alpha=0.3)

# piano keyboard frequencies
all_notes = [(f"{note_names[s]}{o}", note_freq(12*(o+1)+s))
             for o in range(0,9) for s in range(12)]
piano_freqs = [f for _,f in all_notes if 27 <= f <= 4200]
piano_log = np.log2(np.array(piano_freqs) / 27.5)
axes3[1,1].hist(piano_log, bins=50, color="gray", edgecolor="black", lw=0.3)
axes3[1,1].axvline(np.log2(2000/27.5), color="red", ls="--", label="2000 Hz")
axes3[1,1].set_xlabel("log2(f / A0)")
axes3[1,1].set_ylabel("Count")
axes3[1,1].set_title("Piano note distribution (log scale)")
axes3[1,1].legend()
axes3[1,1].grid(True, alpha=0.3)

plt.tight_layout()

# ------------------------------------------------------------------ #
# S4: PHYSICS -> ML CAREER PATH
# ------------------------------------------------------------------ #
print()
print(SEP)
print("SECTION 4: PHYSICS -> ML/AI CAREER PATH")
print(SEP)

print("""
  THE MAPPING (physics curriculum -> ML foundations):

  PHYSICS COURSE          ML EQUIVALENT           Why it transfers
  -----------------------------------------------------------------------
  Linear algebra          Matrices everywhere     weight matrices, PCA, SVD
  Classical mechanics     Optimization            Lagrangian -> loss minimization
  E&M / Maxwell           CNNs                   conv = cross-correlation = FT trick
  Quantum mechanics       Attention / Transformers Hermitian ops -> self-attention
  Statistical mechanics   Probabilistic ML        partition function = normalization
  Griffiths QM            Tensor operations       Dirac notation = einsum notation
  Fourier analysis        Signal processing in ML  frequency domain training tricks
  PDEs / Green's functions Neural operators (FNO)  G(x,y) -> kernel attention
  Thermodynamics          Diffusion models        score function, Langevin dynamics
  Special relativity      Invariant representations equivariant networks
  Numerical methods       Autodiff / backprop      chain rule = backprop

  CAREER PATH (physics undergrad -> Bay Area ML):

  STEP 1 (undergrad, years 1-3):
    Physics major: classical, E&M, QM, stat mech.
    Add: linear algebra, real analysis, probability.
    Code: Python, NumPy, SymPy. Start GitHub. Build projects.
    This session IS that curriculum -- compressed, with code.

  STEP 2 (year 3-4 or gap year):
    ML foundations: Goodfellow "Deep Learning" textbook.
    PyTorch: build a CNN, RNN, Transformer from scratch.
    Kaggle: 1 competition to learn data pipelines.
    Research: find a physics professor using ML (Radulaski, our advisor).
    Key insight: your physics intuition = UNFAIR ADVANTAGE in ML research.
    ML people don't understand Green's functions. You will.

  STEP 3 (grad school or industry):
    PhD route: apply to ML/CS programs with physics background.
      Strong programs: Stanford, Berkeley, MIT, CMU, UCSD, UW.
      Pitch: "I use physics to design better neural architectures."
    Industry route: ML engineer at FAANG.
      Physics PhD -> $200-400K at Google DeepMind, OpenAI, Anthropic.
      Entry: SWE + ML knowledge sufficient with GitHub portfolio.

  BAY AREA ML PROFESSOR EXAMPLES (physics background):
    Stefano Ermon (Stanford): physics undergrad -> statistical ML.
    Surya Ganguli (Stanford): physics PhD -> theoretical neuroscience/ML.
    Anima Anandkumar (Caltech->Nvidia->Caltech): EE/physics -> tensors.
    Pieter Abbeel (KU Leuven physics -> Berkeley CS) -> robotics/RL.
    Ilya Sutskever (open U -> Toronto): physics influence on deep learning.
    Yann LeCun (ESIEE Paris, physics-adjacent -> Bell Labs -> NYU/Meta).

  THE PHYSICS ADVANTAGE:
    You know: dimensional analysis (catch ML bugs by checking units).
    You know: perturbation theory (approximations in ML proofs).
    You know: symmetry -> conservation (equivariant nets, gauge theories).
    You know: phase transitions (loss landscape topology, grokking).
    You know: the FT / Green's function / convolution toolkit.
    ML researchers learn these AFTER they need them. You have them first.

  TIMELINE FROM YOUR POSITION (23-24, CSUS CS+physics):
    Now (2026):        This project. GitHub portfolio. SBIR publication.
    2026-2027:         Apply to MS/PhD at UCB/Stanford/UCSD.
    2027-2029:         MS or first 2 years PhD. 1-2 ML+photonics papers.
    2029-2032:         PhD completion. Thesis: photonic ML or phase retrieval ML.
    2032-2034:         Postdoc (optional) or ML research at Nvidia/Google/OpenAI.
    2034+:             Faculty track if desired. Or: startup (this project is one).
""")

# physics concepts as ML building blocks
print("  Physics concept -> ML tool (concrete examples):")
mappings = [
    ("Schrodinger eqn",    "Attention: Q,K,V = eigenvectors of H"),
    ("Fourier transform",  "FFT layers: frequency domain conv, FNO"),
    ("Gaussian integral",  "Variational inference: ELBO, KL divergence"),
    ("Green's function",   "Neural operator: K(x,y) = learned G(x,y)"),
    ("Langevin dynamics",  "Score diffusion models: dx = -grad E dt + noise"),
    ("Partition function", "Softmax: Z = sum exp(-E_i), p_i = exp(-E_i)/Z"),
    ("Phase transition",   "Grokking: sudden generalization in training"),
    ("Renorm group",       "Multiscale architectures, U-Net, wavelets"),
    ("Hamiltonian",        "Energy-based models, contrastive learning"),
    ("Gauge invariance",   "Equivariant networks: E(3)-equivariant GNNs"),
]
print(f"  {'Physics':<28} {'ML equivalent'}")
print("  " + "-" * 60)
for phys, ml in mappings:
    print(f"  {phys:<28} {ml}")

# ------------------------------------------------------------------ #
# S5: FIBER SAGNAC INTERFEROMETER
# ------------------------------------------------------------------ #
print()
print(SEP)
print("SECTION 5: FIBER SAGNAC INTERFEROMETER")
print(SEP)

print("""
  SAGNAC EFFECT:
    Counter-propagating beams in a rotating loop acquire a phase difference.
    delta_phi = (4*pi*A*N / (lambda*c)) * Omega
    A     = loop area  [m^2]
    N     = number of fiber turns
    lambda= wavelength [m]
    Omega = rotation rate [rad/s]

  PHYSICAL ORIGIN:
    In the rotating frame, CW beam travels longer path (frame moves into it).
    CCW beam travels shorter path.
    Path length difference: delta_L = 4*A*Omega/c (first order in Omega/c).
    Phase difference: delta_phi = (2*pi/lambda) * delta_L.
    This is a GENERAL RELATIVISTIC effect (Sagnac is in GR, not SR).

  FIBER OPTIC GYROSCOPE (FOG):
    delta_phi = S * Omega    where S = 4*pi*N*A / (lambda*c)
    Minimum detectable Omega:  Omega_min = delta_phi_min / S
    Shot noise limit: delta_phi_min = 1/sqrt(N_photons) = 1/sqrt(P*T/(hf))
    -> Omega_min = (lambda*c) / (4*pi*N*A) * sqrt(hf/(P*T))

  BIAS AND ERRORS:
    Shupe effect: thermally-induced non-reciprocal phase (slow temperature gradient).
    Faraday effect: magnetic field -> circular birefringence -> phase shift.
    Backscattering: Rayleigh backscattering in fiber creates coherent noise.
    Fix: use broadband source (low coherence) -> coherence length << fiber length.

  APPLICATIONS:
    Inertial navigation (INS): aircraft, missiles, submarines.
    Geodesy: measure Earth's rotation precisely.
    General relativity tests: Lense-Thirring frame dragging.
    SEISMIC sensing: detect ground rotation during earthquakes.

  ROGUEGUARD EXTENSION:
    Sagnac can also detect: vibration, acoustic waves, pressure changes.
    A fiber loop around a pipeline: acoustic emission from flow anomalies.
    Combines with our GS phase retrieval: intensity I_out = I_0*cos^2(delta_phi/2)
    -> phase delta_phi encoded in output intensity -> GS decodes phi.
    Same kernel H(nu) = exp(i*pi*D*nu^2) applies to dispersed Sagnac output.
""")

# Sagnac calculations
lam_s  = 1550e-9   # m
c_s    = 3e8       # m/s
N_s    = 1000      # turns
r_s    = 0.05      # m coil radius
A_s    = np.pi * r_s**2
h_p    = 6.626e-34
f_opt  = c_s / lam_s
P_mW   = 1.0e-3    # 1 mW source power
T_int  = 1.0       # 1 second integration

S_factor = 4 * np.pi * N_s * A_s / (lam_s * c_s)
N_phot   = P_mW * T_int / (h_p * f_opt)
dphi_min = 1.0 / np.sqrt(N_phot)
Omega_min = dphi_min / S_factor

print(f"  FOG parameters: N={N_s}, r={r_s}m, lambda={lam_s*1e9:.0f}nm, P={P_mW*1e3:.0f}mW")
print(f"  Scale factor S = {S_factor:.4f} rad/(rad/s)")
print(f"  Photons in 1 sec: {N_phot:.3e}")
print(f"  Shot-noise phase: {dphi_min*1e6:.3f} urad")
print(f"  Min detectable rotation: {Omega_min*1e6:.4f} urad/s")
print(f"  Earth rotation = 72.9 urad/s -> SNR = {72.9e-6/Omega_min:.0f}")

print()
print("  Sagnac phase vs rotation rate:")
omegas = np.array([1e-6, 1e-5, 1e-4, 1e-3, 0.01, 0.1, 72.9e-6, 1.0])
print(f"  {'Omega (rad/s)':>16} {'delta_phi (urad)':>18} {'Detectable?':>13}")
print("  " + "-" * 48)
for om in sorted(omegas):
    dphi = S_factor * om * 1e6  # urad
    det = "YES" if om > Omega_min else "NO"
    print(f"  {om:>16.2e} {dphi:>18.4f} {det:>13}")

# save figure
plt.figure(1)
plt.savefig(OUT, dpi=100, bbox_inches="tight")
print(f"\n  Plots saved to {OUT}")

print()
print(SEP)
print("Done.")
print(SEP)
