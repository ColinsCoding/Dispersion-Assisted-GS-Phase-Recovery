"""
_build_resolution_nb.py
Builds notebooks/angular_resolution_genomics.ipynb
Textbook-style: radians/revolution, resolution philosophy, cMOS Nyquist, genomic sequencing
Run: py -3.12 scripts/_build_resolution_nb.py
"""

import nbformat
c = nbformat.v4

cells = []
def md(src):  cells.append(c.new_markdown_cell(src))
def code(src): cells.append(c.new_code_cell(src))

# ── Title ────────────────────────────────────────────────────────────────────
md("""# Angular Units, Resolution, cMOS Nyquist, and Genomic Sequencing
### A unified engineering textbook chapter

**Topics**
1. Radians, revolutions, and the complex exponential
2. Resolution — from Abbe to Shannon to super-resolution
3. cMOS sensor Nyquist: pixels, full-well capacity, dynamic range
4. Genomic sequencing as a sampling problem (Phred, coverage, Shannon DNA)

*All derivations shown. Run cells top-to-bottom.*
""")

# ── Setup ────────────────────────────────────────────────────────────────────
code("""\
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from scipy.special import j1
sp.init_printing(use_unicode=True, use_latex='mathjax')

kB   = 1.380649e-23
h    = 6.626e-34
c    = 2.998e8
e_c  = 1.602e-19
print("Setup complete.")
""")

# ════════════════════════════════════════════════════════════════════════════
# SECTION 1: RADIANS AND REVOLUTIONS
# ════════════════════════════════════════════════════════════════════════════
md("""---
## Section 1 — Radians, Revolutions, and the Complex Exponential

### 1.1 Why radians are natural

A **radian** is defined so that arc length = radius × angle.
For a circle of radius $r$: $s = r\\theta$.

One full revolution = circumference / radius = $2\\pi r / r = 2\\pi$ radians.

| Unit | Symbol | Per revolution |
|---|---|---|
| Radian | rad | $2\\pi$ |
| Degree | ° | 360 |
| Revolution | rev | 1 |
| Grad | gon | 400 |

**The key insight:** $2\\pi$ appears because a circle has circumference $2\\pi r$, which is a *geometric* fact, not a convention.
""")

code("""\
theta = sp.Symbol('theta', real=True)
x_s, y_s = sp.cos(theta), sp.sin(theta)

print("Unit circle: (cos theta, sin theta)")
print()
for deg in [0, 30, 45, 60, 90, 180, 270, 360]:
    rad = deg * np.pi / 180
    print(f"  {deg:>4} deg = {rad/np.pi:.4f}*pi rad  "
          f"  cos={np.cos(rad):+.4f}  sin={np.sin(rad):+.4f}")
""")

md("""### 1.2 Angular frequency $\\omega$ vs ordinary frequency $f$

$$\\omega = 2\\pi f \\quad [\\text{rad/s}]$$

**Why $\\omega$?** The complex exponential $e^{i\\omega t}$ is the eigenfunction of all LTI systems.
Derivatives and integrals become multiplication:

$$\\frac{d}{dt} e^{i\\omega t} = i\\omega \\cdot e^{i\\omega t}$$

This is why Fourier transforms use $\\omega$ (or $\\nu$ in optics): differentiation → multiplication by $i\\omega$.

| Quantity | Symbol | Formula | Unit |
|---|---|---|---|
| Ordinary freq | $f$ | cycles/sec | Hz |
| Angular freq | $\\omega$ | $2\\pi f$ | rad/s |
| Wavenumber | $k$ | $2\\pi / \\lambda$ | rad/m |
| Spatial freq | $\\nu$ | $1/\\lambda$ | cycles/m |
""")

code("""\
t_s, omega_s, f_s = sp.symbols('t omega f', real=True, positive=True)

psi = sp.exp(sp.I * omega_s * t_s)

d_psi  = sp.diff(psi, t_s)
d2_psi = sp.diff(psi, t_s, 2)

print("Eigenfunction property:")
print(f"  d/dt  [e^(i*omega*t)] = {d_psi}")
print(f"  d2/dt2[e^(i*omega*t)] = {d2_psi}")
print()
print("Euler's formula at key angles:")
for angle_name, angle in [("0", 0), ("pi/2", sp.pi/2), ("pi", sp.pi), ("2*pi", 2*sp.pi)]:
    val = sp.exp(sp.I * angle)
    print(f"  e^(i*{angle_name}) = {sp.simplify(val)}")
""")

code("""\
# Visualize: revolution in complex plane
theta_arr = np.linspace(0, 4*np.pi, 500)
z = np.exp(1j * theta_arr)

fig, axes = plt.subplots(1, 3, figsize=(14, 4))

# Complex plane
ax = axes[0]
ax.plot(z.real, z.imag, 'b-', lw=2)
ax.plot([1], [0], 'ro', ms=8, label='theta=0')
ax.plot([0], [1], 'gs', ms=8, label='theta=pi/2')
ax.axhline(0, color='k', lw=0.5); ax.axvline(0, color='k', lw=0.5)
ax.set_aspect('equal'); ax.set_title('Unit circle in complex plane')
ax.set_xlabel('Re'); ax.set_ylabel('Im'); ax.legend(fontsize=8)

# cos and sin vs revolutions
revs = theta_arr / (2*np.pi)
ax = axes[1]
ax.plot(revs, np.cos(theta_arr), 'b-', label='cos(2*pi*rev)')
ax.plot(revs, np.sin(theta_arr), 'r--', label='sin(2*pi*rev)')
ax.set_xlabel('Revolutions'); ax.set_ylabel('Amplitude')
ax.set_title('cos and sin vs revolutions'); ax.legend(); ax.grid(True, alpha=0.3)

# Frequency domain: 1 revolution/s -> delta at f=1
dt = 0.001; t_arr = np.arange(0, 4, dt)
signal = np.cos(2*np.pi*1.5*t_arr) + 0.5*np.cos(2*np.pi*3.0*t_arr)
freqs  = np.fft.rfftfreq(len(t_arr), dt)
spec   = np.abs(np.fft.rfft(signal)) * dt

ax = axes[2]
ax.plot(freqs[:80], spec[:80], 'b-', lw=1.5)
ax.set_xlabel('f (Hz)'); ax.set_ylabel('|FFT|')
ax.set_title('Spectrum: 1.5Hz + 3Hz tones'); ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/tmp/angular_units.png', dpi=100, bbox_inches='tight')
plt.show()
print("Fig 1.1: Angular units and Fourier connection")
""")

# ════════════════════════════════════════════════════════════════════════════
# SECTION 2: RESOLUTION PHILOSOPHY
# ════════════════════════════════════════════════════════════════════════════
md("""---
## Section 2 — Resolution: From Abbe to Shannon

### 2.1 Diffraction limit (Abbe, 1873)

A lens of numerical aperture $\\text{NA} = n\\sin\\theta$ cannot resolve two points separated by less than:

$$d_\\text{Abbe} = \\frac{\\lambda}{2\\,\\text{NA}} \\qquad \\text{(coherent illumination)}$$

$$d_\\text{Rayleigh} = \\frac{0.61\\,\\lambda}{\\text{NA}} \\qquad \\text{(incoherent, first Airy-null criterion)}$$

**Why?** High spatial frequencies $k_\\perp > k_0 \\text{NA}$ become evanescent and don't propagate to the far field.
The lens is a **low-pass filter** in spatial frequency space.

### 2.2 Shannon resolution (information theory)

The **Shannon limit** asks: given SNR in an optical system, what is the minimum resolvable feature?

$$d_\\text{Shannon} = \\frac{\\lambda}{2\\,\\text{NA} \\cdot \\sqrt{1 + \\text{SNR}}}$$

At SNR=0: $d_\\text{Shannon} \\to \\infty$. At SNR$\\to\\infty$: $d_\\text{Shannon} \\to d_\\text{Abbe} / \\sqrt{2}$.

**Resolution is not just physics — it's information.**

### 2.3 Super-resolution techniques

| Technique | Principle | Resolution |
|---|---|---|
| Widefield | Direct imaging | $\\lambda / (2\\,\\text{NA})$ |
| Confocal | Point illumination + pinhole | $\\sim 0.37\\lambda/\\text{NA}$ |
| STED | Stimulated depletion of outer ring | $\\sim 30$ nm (live cells) |
| SIM | Structured illumination, Moiré | $2\\times$ Abbe limit |
| PALM/STORM | Single-molecule localization | $\\sim 10$ nm |
| Expansion microscopy | Physical expansion of sample | $\\sim 10$ nm (cheap!) |
""")

code("""\
lam_nm  = 488.0   # excitation wavelength
NA_vals = np.linspace(0.1, 1.49, 200)
SNR_vals = [1, 10, 100, 1000]

d_abbe    = lam_nm / (2 * NA_vals)
d_rayleigh = 0.61 * lam_nm / NA_vals

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

ax = axes[0]
ax.semilogy(NA_vals, d_abbe,    'b-',  lw=2, label='Abbe (coherent)')
ax.semilogy(NA_vals, d_rayleigh,'r--', lw=2, label='Rayleigh (incoherent)')
ax.axhline(10, color='gray', lw=1, ls=':', label='PALM/STORM ~10nm')
ax.axhline(30, color='green',lw=1, ls=':', label='STED ~30nm')
ax.set_xlabel('NA'); ax.set_ylabel('Resolution (nm)')
ax.set_title(f'Diffraction limits  lambda={lam_nm:.0f}nm')
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
ax.set_ylim(5, 3000)

ax = axes[1]
colors = plt.cm.plasma(np.linspace(0.2, 0.9, len(SNR_vals)))
for SNR, col in zip(SNR_vals, colors):
    d_shannon = lam_nm / (2 * NA_vals * np.sqrt(1 + SNR))
    ax.semilogy(NA_vals, d_shannon, color=col, lw=2, label=f'SNR={SNR}')
ax.semilogy(NA_vals, d_abbe, 'k--', lw=1.5, alpha=0.6, label='Abbe limit')
ax.set_xlabel('NA'); ax.set_ylabel('Shannon resolution (nm)')
ax.set_title('Shannon resolution vs SNR'); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/tmp/resolution.png', dpi=100, bbox_inches='tight')
plt.show()
print("Fig 2.1: Resolution limits")
""")

code("""\
# Point Spread Function: Airy pattern I(r) = [2*J1(x)/x]^2
from scipy.special import j1

NA_psf  = 1.40
r_nm    = np.linspace(0.01, 600, 1000)
x_psf   = np.pi * r_nm * 1e-9 * NA_psf / (lam_nm * 1e-9)
I_wide  = (2 * j1(x_psf) / x_psf)**2          # widefield
I_conf  = (2 * j1(x_psf) / x_psf)**4          # confocal ~ squared

FWHM_wide = 2 * 0.514 * lam_nm / NA_psf       # nm
FWHM_conf = FWHM_wide / np.sqrt(2)

plt.figure(figsize=(8, 4))
plt.plot(r_nm, I_wide, 'b-',  lw=2, label=f'Widefield  FWHM={FWHM_wide:.0f}nm')
plt.plot(r_nm, I_conf, 'r--', lw=2, label=f'Confocal   FWHM={FWHM_conf:.0f}nm')
plt.axvline(0.61*lam_nm/NA_psf, color='gray', lw=1, ls=':', label=f'Rayleigh={0.61*lam_nm/NA_psf:.0f}nm')
plt.xlabel('r (nm)'); plt.ylabel('I / I_0')
plt.title(f'PSF: Airy pattern  NA={NA_psf}, lambda={lam_nm:.0f}nm')
plt.legend(); plt.grid(True, alpha=0.3)
plt.savefig('/tmp/psf.png', dpi=100, bbox_inches='tight')
plt.show()
print(f"Widefield FWHM: {FWHM_wide:.1f} nm")
print(f"Confocal  FWHM: {FWHM_conf:.1f} nm  (sqrt(2) improvement)")
print(f"Airy first zero (Rayleigh): {0.61*lam_nm/NA_psf:.1f} nm")
""")

# ════════════════════════════════════════════════════════════════════════════
# SECTION 3: cMOS NYQUIST
# ════════════════════════════════════════════════════════════════════════════
md("""---
## Section 3 — cMOS Sensor: Nyquist, Full-Well, and Dynamic Range

### 3.1 Nyquist–Shannon sampling theorem (images)

A band-limited image with maximum spatial frequency $f_\\text{max}$ requires pixel spacing:

$$\\Delta x \\leq \\frac{1}{2 f_\\text{max}} = \\frac{d_\\text{Abbe}}{2}$$

For a microscope: $f_\\text{max} = \\text{NA}/\\lambda$, so:

$$\\boxed{\\text{pixel size at sample} \\leq \\frac{\\lambda}{4 \\cdot \\text{NA}}}$$

(Factor of 4, not 2, because $d_\\text{Abbe} = \\lambda/2\\text{NA}$ and we need pixel $\\leq d/2$.)

### 3.2 cMOS pixel architecture

A cMOS pixel is a **photodiode + source-follower amplifier** on silicon.

| Parameter | Symbol | Typical value |
|---|---|---|
| Pixel pitch | $p$ | 1.6–10 µm |
| Full-well capacity | FWC | 5K–100K e⁻ |
| Read noise | $\\sigma_r$ | 1–5 e⁻ rms |
| Dark current | $I_d$ | 1–100 e⁻/s/pixel |
| QE (peak) | — | 60–95% |
| ADC depth | — | 10–16 bit |

### 3.3 Dynamic range

$$\\text{DR} = \\frac{\\text{FWC}}{\\sigma_r} \\qquad \\text{(electrons)} \\qquad = 20\\log_{10}\\left(\\frac{\\text{FWC}}{\\sigma_r}\\right) \\text{ dB}$$

**Shot noise** from $N$ photons: $\\sigma_\\text{shot} = \\sqrt{N}$.
At low light ($N < \\sigma_r^2$): read-noise limited.
At high light ($N > \\sigma_r^2$): shot-noise limited.
""")

code("""\
# cMOS Nyquist: pixel size vs objective
print("cMOS Nyquist: required pixel size at sample")
print(f"{'Objective':18s}  {'NA':>5}  {'Abbe(nm)':>9}  {'Nyquist pix(nm)':>16}  {'6.5um cam (nm)':>15}  {'OK?':>5}")
print("-" * 75)

camera_pixel_um = 6.5
objectives = {
    "10x air":   (0.30, 10),
    "20x air":   (0.45, 20),
    "40x air":   (0.65, 40),
    "40x water": (0.80, 40),
    "60x oil":   (1.25, 60),
    "100x oil":  (1.40, 100),
}

for name, (NA, M) in objectives.items():
    d_abbe_nm  = lam_nm / (2*NA)
    pix_nyq_nm = d_abbe_nm / 2
    pix_cam_nm = camera_pixel_um / M * 1000
    ok = "OK" if pix_cam_nm <= pix_nyq_nm else "UNDER"
    print(f"  {name:18s}  {NA:>5.2f}  {d_abbe_nm:>9.1f}  {pix_nyq_nm:>16.1f}  {pix_cam_nm:>15.1f}  {ok:>5}")

print()
print("UNDER = pixel is too big -> aliased (undersampled)")
print("Nyquist pixel = Abbe_limit / 2 = lambda / (4*NA)")
""")

code("""\
# Dynamic range vs FWC and read noise
FWC_vals   = np.array([5000, 10000, 25000, 50000, 100000])  # electrons
sigma_r    = np.array([1, 2, 3, 5])   # read noise electrons

print("Dynamic range (dB) = 20*log10(FWC / sigma_read)")
print(f"{'FWC (e-)':>10}", end="")
for sr in sigma_r:
    print(f"  sigma_r={sr:1.0f}e-", end="")
print()
print("-" * 60)
for fwc in FWC_vals:
    print(f"{fwc:>10,}", end="")
    for sr in sigma_r:
        DR_dB = 20 * np.log10(fwc / sr)
        print(f"  {DR_dB:>9.1f} dB", end="")
    print()

print()
print("Human eye: ~120 dB.  Best sCMOS: FWC=100K, sigma=1 -> 100 dB.")
print("Typical sCMOS: FWC=30K, sigma=2 -> 83.5 dB.")
""")

code("""\
# Photon transfer curve: noise model
N_photons = np.logspace(0, 5, 500)  # 1 to 100K photons
QE        = 0.82    # quantum efficiency
FWC       = 30000   # electrons
sigma_r   = 2.0     # read noise
I_dark_s  = 5.0     # dark current e-/s
t_exp     = 0.1     # 100ms exposure

N_signal  = QE * N_photons
N_dark    = I_dark_s * t_exp
noise_shot   = np.sqrt(N_signal)
noise_dark   = np.sqrt(N_dark)
noise_read   = sigma_r
noise_total  = np.sqrt(noise_shot**2 + noise_dark**2 + noise_read**2)
SNR_dB       = 20 * np.log10(N_signal / noise_total)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

ax = axes[0]
ax.loglog(N_photons, noise_total, 'k-',  lw=2, label='Total noise')
ax.loglog(N_photons, noise_shot,  'b--', lw=1.5, label='Shot noise sqrt(N)')
ax.loglog(N_photons, [sigma_r]*len(N_photons), 'r:', lw=1.5, label=f'Read noise {sigma_r} e-')
ax.axvline(sigma_r**2/QE, color='purple', lw=1, ls='--', label='Shot=Read crossover')
ax.set_xlabel('Incident photons'); ax.set_ylabel('Noise (e-)')
ax.set_title('Photon transfer curve'); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

ax = axes[1]
ax.semilogx(N_photons, SNR_dB, 'b-', lw=2)
ax.axhline(20, color='r', ls='--', lw=1, label='SNR=20dB (10:1)')
ax.axhline(40, color='g', ls='--', lw=1, label='SNR=40dB (100:1)')
crossover = sigma_r**2 / QE
ax.axvline(crossover, color='purple', lw=1, ls='--', label=f'Read-limited below {crossover:.0f} ph')
ax.set_xlabel('Incident photons'); ax.set_ylabel('SNR (dB)')
ax.set_title(f'SNR vs photons  (QE={QE}, FWC={FWC//1000}K, sigma_r={sigma_r})');
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/tmp/cmos_ptc.png', dpi=100, bbox_inches='tight')
plt.show()
print(f"Read-noise limited below {crossover:.0f} photons ({crossover*QE:.0f} e-)")
print(f"Shot-noise limited above {crossover:.0f} photons")
print(f"Full well: {FWC} e- -> max SNR = {20*np.log10(FWC/noise_read):.1f} dB at saturation")
""")

md("""### 3.4 The real cMOS Nyquist numbers

For a **6.5 µm pixel sCMOS** (standard research camera):

| Objective | M | Pixel at sample | Abbe limit | Nyquist OK? |
|---|---|---|---|---|
| 10× air NA=0.3 | 10 | **650 nm** | 813 nm | OK (0.8×) |
| 20× air NA=0.45 | 20 | **325 nm** | 542 nm | OK (0.6×) |
| 40× air NA=0.65 | 40 | **163 nm** | 375 nm | OK (0.43×) |
| 60× oil NA=1.25 | 60 | **108 nm** | 195 nm | OK (0.56×) |
| 100× oil NA=1.4 | 100 | **65 nm** | 174 nm | **2.7× oversampled** |

At 100×: pixel is 65 nm but Nyquist only requires <87 nm — you're **2.7× oversampled** and wasting pixels on noise.
Downbinning 2×2 → 130 nm pixels recovers SNR without aliasing.
""")

# ════════════════════════════════════════════════════════════════════════════
# SECTION 4: GENOMIC SEQUENCING AS A SAMPLING PROBLEM
# ════════════════════════════════════════════════════════════════════════════
md("""---
## Section 4 — Genomic Sequencing: Nyquist Applied to DNA

### 4.1 DNA as a discrete signal

The human genome is a **3.2 billion base** sequence over alphabet $\\{A, T, G, C\\}$.
Each base carries $\\log_2(4) = 2$ bits of information → genome = **6.4 Gbits raw**.

Compression: real genomes have correlations (repeats, codons, gene structure) → ~800 MB gzipped.

### 4.2 Sequencing as sampling

A sequencer reads short **reads** of length $L$ bases.
**Coverage** (depth) $C$ is the number of times each base is independently read:

$$C = \\frac{N_\\text{reads} \\times L}{G} \\qquad G = \\text{genome size}$$

**Nyquist analogy:** coverage is sampling rate; genome is signal; read length is aperture.

**Lander-Waterman statistics** (Poisson model):
$$P(\\text{base covered at least once}) = 1 - e^{-C}$$

For $P > 99\\%$: need $C > -\\ln(0.01) \\approx 4.6\\times$.
Clinical standard: $30\\times$ for germline, $100\\times$ for somatic variants.

### 4.3 Phred quality scores

Phred score $Q$ encodes base-call error probability $p$:

$$Q = -10 \\log_{10}(p) \\qquad \\Leftrightarrow \\qquad p = 10^{-Q/10}$$

| Q score | Error prob | Accuracy | SNR equivalent |
|---|---|---|---|
| Q10 | 1/10 | 90% | 10 dB |
| Q20 | 1/100 | 99% | 20 dB |
| Q30 | 1/1000 | 99.9% | 30 dB |
| Q40 | 1/10000 | 99.99% | 40 dB |
| Q60 | 1/10⁶ | 99.9999% | 60 dB |

**Phred is just dB** — the same logarithmic scale used in optics and RF.
""")

code("""\
# Phred scores: same math as dB
Q_vals = np.arange(0, 65, 5)
p_err  = 10**(-Q_vals / 10)
acc    = (1 - p_err) * 100

print("Phred score = -10*log10(p_error)  [identical to optical SNR in dB]")
print()
print(f"  {'Q':>4}  {'p_error':>12}  {'Accuracy':>10}  {'dB equiv':>10}  {'Illumina grade'}")
print("-" * 60)
grades = {0:'fail', 10:'low', 20:'acceptable', 30:'good', 40:'excellent', 60:'near-perfect'}
for Q, p, a in zip(Q_vals, p_err, acc):
    grade = grades.get(Q, '')
    print(f"  {Q:>4}  {p:>12.2e}  {a:>9.4f}%  {Q:>9} dB  {grade}")
""")

code("""\
# Lander-Waterman coverage model
C_range    = np.linspace(0.1, 40, 500)
P_covered  = 1 - np.exp(-C_range)

# Coverage needed for different P targets
P_targets = [0.90, 0.95, 0.99, 0.999, 0.9999]
C_needed  = [-np.log(1-P) for P in P_targets]

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

ax = axes[0]
ax.plot(C_range, P_covered * 100, 'b-', lw=2)
for P, C in zip(P_targets, C_needed):
    ax.axhline(P*100, color='r', lw=0.8, ls='--', alpha=0.7)
    ax.axvline(C,     color='g', lw=0.8, ls='--', alpha=0.7)
    ax.annotate(f'{C:.1f}x', xy=(C, P*100), xytext=(C+0.5, P*100-3), fontsize=7)
ax.set_xlabel('Coverage depth (x)'); ax.set_ylabel('% bases covered')
ax.set_title('Lander-Waterman: P(covered) = 1 - e^(-C)')
ax.grid(True, alpha=0.3)

# Shannon information in a read
print("Coverage required for target completeness (Lander-Waterman):")
for P, C in zip(P_targets, C_needed):
    reads_needed = int(C * 3.2e9 / 150)  # 150bp reads, 3.2Gb genome
    data_GB = reads_needed * 150 * 1 / 1e9  # 1 byte/base (uncompressed)
    print(f"  P={P:.4f}: C={C:.2f}x -> {reads_needed/1e6:.1f}M reads -> {data_GB:.1f} GB raw")

# Shannon entropy of DNA
ax = axes[1]
# Base composition: uniform = max entropy; CpG islands = low entropy
p_bases = np.array([0.25, 0.25, 0.25, 0.25])  # A,T,G,C
H_uniform = -np.sum(p_bases * np.log2(p_bases))

# Vary GC content
GC_frac = np.linspace(0.01, 0.99, 200)
AT_frac = 1 - GC_frac
p_G = GC_frac / 2; p_C = GC_frac / 2
p_A = AT_frac / 2; p_T = AT_frac / 2
H_GC = -(p_G*np.log2(p_G+1e-30) + p_C*np.log2(p_C+1e-30) +
         p_A*np.log2(p_A+1e-30) + p_T*np.log2(p_T+1e-30))

ax.plot(GC_frac * 100, H_GC, 'b-', lw=2)
ax.axvline(50, color='r', ls='--', lw=1, label='50% GC = max entropy (2 bits/base)')
ax.axvline(41, color='g', ls='--', lw=1, label='Human genome ~41% GC')
ax.set_xlabel('GC content (%)'); ax.set_ylabel('Shannon entropy H (bits/base)')
ax.set_title('DNA information content vs GC bias'); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/tmp/genomics.png', dpi=100, bbox_inches='tight')
plt.show()
print(f"\\nUniform base freq: H = {H_uniform:.4f} bits/base (maximum = log2(4) = 2)")
print(f"Human genome GC=41%: H = {H_GC[np.argmin(np.abs(GC_frac-0.41))]:.4f} bits/base")
print(f"Human genome raw information: {H_GC[np.argmin(np.abs(GC_frac-0.41))]*3.2e9/8e9:.3f} GB (uncompressed)")
""")

md("""### 4.4 SEALS: Sequence Exact Alignment with Light Scattering

**SEALS** (Sorting and Extraction of Aligned Long-read Sequences) applies
fiber-optic light scattering to sort cells by expressed gene signature:

1. **Cells labeled** with fluorescent probes targeting mRNA sequences
2. **Optical tweezer** traps individual cells in microfluidic channel
3. **Ring resonator** reads binding signal → digital base call
4. **GS phase recovery** (this repo) reconstructs wavefront → cell type
5. **Output**: sorted populations ready for downstream sequencing

The connection to this repo:

$$\\text{DNA base call} \\leftrightarrow \\text{GS phase retrieval}$$

Both solve an **inverse problem**: recover a discrete state (base / phase)
from noisy intensity measurements, using prior knowledge (ATGC / support constraint).
""")

code("""\
# Unified SNR: Phred (DNA), dB (optics), same formula
print("=" * 55)
print("UNIFIED SNR SCALE  Q = -10*log10(p_error) = dB")
print("=" * 55)
print()
print("Domain          Signal     Noise      Q/dB    Meaning")
print("-" * 55)

rows = [
    ("DNA base call", "correct base", "error prob",  "Q30 = 30dB", "1 error/1000 bases"),
    ("GS phase recov","phi_true",     "phi_error",   "30 dB",      "phase err < 0.063 rad"),
    ("Optical link",  "P_signal",     "P_noise",     "OSNR 30dB",  "BER < 1e-9"),
    ("cMOS pixel",    "N_signal e-",  "sigma_r e-",  "DR 80dB",    "100:1 dynamic range"),
    ("Audiophile",    "music",        "hiss",        "SNR 100dB",  "24-bit audio"),
]
for domain, sig, noise, dB, meaning in rows:
    print(f"  {domain:16s}  {dB:>12}  {meaning}")

print()
print("All use:  SNR = 10*log10(signal_power / noise_power)")
print("Phred is optical SNR applied to molecular biology.")
""")

# ── Build notebook ────────────────────────────────────────────────────────────
nb = c.new_notebook(cells=cells)
nb.metadata['kernelspec'] = {
    'display_name': 'Python 3 (ipykernel)',
    'language': 'python',
    'name': 'python3'
}
nb.metadata['language_info'] = {'name': 'python', 'version': '3.12.0'}

out_path = 'notebooks/angular_resolution_genomics.ipynb'
with open(out_path, 'w', encoding='utf-8') as f:
    nbformat.write(nb, f)
print(f"Written: {out_path}  ({len(cells)} cells)")
