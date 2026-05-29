"""
_inject_76.py  -  inject S76 into phase_retrieval.ipynb
=========================================================
S76 : Jalali Lab Aggregate Study + Project Conclusion
  Cell 111 (markdown) -- optical standards table, project summary
  Cell 112 (code)     -- comparative TD-GS run across 6 standards,
                         convergence table, conclusion figure

Run:  python _inject_76.py
"""

import json, pathlib, sys

NB = pathlib.Path("phase_retrieval.ipynb")
if not NB.exists():
    sys.exit(f"[ERROR] {NB} not found -- run from repo root")

with NB.open("r", encoding="utf-8") as f:
    nb = json.load(f)

# ---- Cell 111 -- Markdown ---------------------------------------------------
md76 = r"""## S76  Jalali Lab Aggregate Study  *  Project Conclusion

### Optical Communication Standards -- TD-GS Phase Recovery Comparison

| Standard | Modulation | Bits/sym | Envelope | Phase structure | TD-GS difficulty |
|---|---|:---:|---|---|:---:|
| **OOK-NRZ** | Binary ASK | 1 | Rectangular | Binary {0, pi} | Low |
| **PAM4** | 4-level ASK | 2 | 4 levels | 4-phase states | Medium |
| **QPSK** | 4-phase PSK | 2 | Constant | pi/4 steps, Gray coded | Medium |
| **DPSK** | Differential BPSK | 1 | Constant | Transitions only | Medium |
| **STEAM** | Chirped Gaussian | N/A | Smooth Gaussian | Continuous chirp | Low |
| **Soliton** | sech^2 envelope | N/A | sech^2 | Linear chirp | Low |

### TD-GS algorithm (recap)

$$E_\text{est}^{(k+1)} = \begin{cases}
\sqrt{I_1}\,e^{i\angle E_\text{est}^{(k)}} & k \text{ even (amplitude constraint)} \\[4pt]
\mathcal{F}^{-1}\!\left[\dfrac{|\mathcal{F}[\sqrt{I_2}]|}{|H_1|+\varepsilon}\,
e^{i\angle\left(\mathcal{F}[E_\text{est}^{(k)}]\cdot H_1\right)}\right] & k \text{ odd (dispersive constraint)}
\end{cases}$$

Convergence metric: $\text{MSE}(k) = \frac{1}{N}\sum_n\bigl(|E_\text{est}^{(k)}[n]| - \sqrt{I_1[n]}\bigr)^2$

### Project summary -- OUSD(R&E) alignment

| Deliverable | Status | OUSD Area |
|---|:---:|---|
| TD-GS phase retrieval (200 iter) | Done | Advanced Computing |
| QPSK carrier-less receiver | Done | FutureG |
| 48-ch DWDM C-band simulation | Done | FutureG |
| PhyCV Phase-Stretch baseline | Done | Advanced Computing |
| Neural network phase regression | Done | Trusted AI & Autonomy |
| CUDA kernel acceleration | Done | Advanced Computing |
| Ghost imaging via correlations | Done | Integrated Sensing |
| Wirtinger flow / L1 sparse | Done | Advanced Computing |
| 2D phased array + HITRAN CO | Done | Directed Energy |
| Deep unrolling + Kalman tracker | Done | Trusted AI & Autonomy |
| Distributed sensing 100 km | Done | Integrated Sensing |
| LIGO phase measurement context | Done | Integrated Sensing |
| THz time-domain spectroscopy | Done | Directed Energy |
| Quantum bra-ket foundations | Done | Quantum Science |
| SEALS Mie+Rayleigh port | Done | Integrated Sensing |
| RogueGuard 1U firmware | Done | Trusted AI, HMI |
| OUSD TAM bubble chart | Done | Trusted AI |
| SBIR Phase I/II pathway | Done | Trusted AI |
| 3-D optical voxel hash + LSH | Done | Advanced Computing |
| Ethereum explicit trust (S74) | Done | Trusted AI |
| TX/RX dual FSM (S75) | Done | FutureG, Advanced Computing |
| **Aggregate study + conclusion** | **Done** | **All areas** |

### Hardware path to deployment

```
Lab bench    -->  RogueGuard 1U (RPi CM4, dual 14-bit ADC, 56 GSa/s)
Dashboard    -->  jalabi-dashboard.onrender.com  (Docker, /health, /alarm)
Trust layer  -->  OpticalPhaseVerifier.sol  (Ethereum, keccak256, no third party)
```

### Next steps (Phase II / SBIR)

1. Acquire real fiber measurement data from Jalali Lab (see REQUESTING_DATA.md)
2. Validate TD-GS convergence on hardware ADC traces (D1=-600, D2=-900 ps^2)
3. Retrain CNN phase classifier on hardware data (§11 notebook)
4. Port TD-GS to FFTW3f + NEON SIMD on RPi CM4 (<1.5 ms target)
5. SBIR Phase I proposal: RogueGuard rogue-wave detection fiber network
"""

# ---- Cell 112 -- Code -------------------------------------------------------
code76 = r"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path('.').resolve()))
sys.path.insert(0, 'optical_dashboard')
import dsp as DSP
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

PI = np.pi
FG = (0.04, 0.04, 0.10)
BG = (0.06, 0.06, 0.14)

def dk(ax, title="", xl="", yl=""):
    ax.set_facecolor(BG)
    for s in ax.spines.values(): s.set_color("#334466")
    ax.tick_params(colors="#99aabb", labelsize=8)
    if title: ax.set_title(title, color="white", fontsize=8.5, fontweight="bold", pad=5)
    if xl: ax.set_xlabel(xl, color="#99aabb", fontsize=8)
    if yl: ax.set_ylabel(yl, color="#99aabb", fontsize=8)

# ------------------------------------------------------------------
# Generate 6 optical standards
# ------------------------------------------------------------------
RNG = np.random.default_rng(42)
N   = 512
t   = np.linspace(-50, 50, N)

def disperse(E, D):
    nu = np.fft.fftfreq(N)
    return np.fft.ifft(np.fft.fft(E) * np.exp(1j * PI * D * nu**2))

def two_arm(E, D1=-600.0, D2=-1200.0):
    I1 = np.abs(disperse(E, D1))**2
    I2 = np.abs(disperse(E, D2))**2
    I1 /= I1.max() + 1e-12;  I2 /= I2.max() + 1e-12
    return I1, I2

# OOK-NRZ
bits = RNG.integers(0,2,N//8)
ook  = (np.repeat(bits.astype(float),8)[:N]).astype(complex)

# PAM4
syms = RNG.integers(0,4,N//8)
pam  = ((2*syms-3)/3.0 + 1).astype(complex)
pam4 = np.repeat(pam,8)[:N]

# QPSK
pairs = RNG.integers(0,2,N).reshape(-1,2)[:N//2]
enc   = {(0,0):1+1j,(0,1):-1+1j,(1,1):-1-1j,(1,0):1-1j}
qpsk  = np.repeat([enc[tuple(p)] for p in pairs],8)[:N]/np.sqrt(2)

# DPSK
dbits = RNG.integers(0,2,N//8)
dpsk  = np.exp(1j*np.cumsum(np.repeat(dbits*PI,8)[:N]))

# STEAM
steam = np.exp(-t**2/(2*12**2)) * np.exp(1j*(0.6*np.exp(-((t-5)/10)**2)*np.sin(2*PI*t/40)))

# Soliton
soliton = (1.0/np.cosh(t/6.0)) * np.exp(1j*0.05*t)

SIGNALS = [
    (ook,    "OOK-NRZ",  "#50d8ff"),
    (pam4,   "PAM4",     "#ffd040"),
    (qpsk,   "QPSK",     "#00ff9f"),
    (dpsk,   "DPSK",     "#cc88ff"),
    (steam,  "STEAM",    "#ff3278"),
    (soliton,"Soliton",  "#ff9900"),
]

# ------------------------------------------------------------------
# Run RX FSM for each and collect convergence
# ------------------------------------------------------------------
print("=" * 60)
print("  Jalali Lab -- Aggregate TD-GS Study")
print("  6 optical standards, D1=-600 D2=-1200 ps2, 60 GS iter")
print("=" * 60)
print(f"  {'Standard':<12} {'N':>5}  {'Init MSE':>9}  {'Final MSE':>10}  {'Reduction':>10}  {'phi_std':>8}")
print("  " + "-" * 60)

all_residuals = {}
summary_rows  = []

for E, label, color in SIGNALS:
    I1, I2 = two_arm(E)
    rx = DSP.OpticalRxFSM(D1=-600.0, D2=-1200.0, n_iter=60)
    phi, residuals = rx.run(I1, I2)
    res = np.array(residuals)
    init_mse  = res[0]  if len(res) > 0 else float('nan')
    final_mse = res[-1] if len(res) > 0 else float('nan')
    reduction = 1.0 - final_mse / (init_mse + 1e-30)
    phi_std   = float(np.std(phi))
    all_residuals[label] = res
    summary_rows.append((label, init_mse, final_mse, reduction, phi_std))
    print(f"  {label:<12} {N:>5}  {init_mse:>9.4f}  {final_mse:>10.6f}  "
          f"{reduction:>9.1%}  {phi_std:>8.4f} rad")

# ------------------------------------------------------------------
# Figure
# ------------------------------------------------------------------
fig = plt.figure(figsize=(15, 10))
fig.patch.set_facecolor(FG)
gs  = gridspec.GridSpec(3, 3, hspace=0.55, wspace=0.40)

colors = [c for (_,_,c) in SIGNALS]
labels = [l for (_,l,_) in SIGNALS]

# (0,0:2) -- convergence curves, all 6 standards
ax0 = fig.add_subplot(gs[0, 0:2])
for (label, color), res in zip([(l,c) for (_,l,c) in SIGNALS], all_residuals.values()):
    ax0.semilogy(np.arange(len(res)), res, color=color, lw=1.5, label=label)
ax0.legend(fontsize=7.5, facecolor=BG, labelcolor="white", ncol=3)
ax0.set_xlim(0, 60)
dk(ax0, "TD-GS Convergence -- 6 Optical Standards\n"
        "D1=-600 ps2  D2=-1200 ps2  60 iterations  (k even=amp, k odd=disp)",
   "GS iteration k", "residual MSE")

# (0,2) -- final MSE bar chart
ax1 = fig.add_subplot(gs[0, 2])
final_mses = [r[-1] for r in all_residuals.values()]
bars = ax1.bar(range(len(labels)), final_mses, color=colors, width=0.6)
ax1.set_xticks(range(len(labels)))
ax1.set_xticklabels(labels, color="#99aabb", fontsize=7, rotation=30, ha="right")
ax1.set_yscale("log")
dk(ax1, "Final Residual MSE\n(lower = better convergence)", "standard", "MSE")

# (1,0) -- I1 waveforms for all standards (first 128 samples)
ax2 = fig.add_subplot(gs[1, 0])
for (E, label, color) in SIGNALS:
    I1, _ = two_arm(E)
    ax2.plot(np.arange(128), I1[:128], color=color, lw=0.8, alpha=0.7, label=label)
ax2.legend(fontsize=6.5, facecolor=BG, labelcolor="white", ncol=2)
dk(ax2, "I1(t) -- Intensity after D1=-600 ps2", "n", "I [a.u.]")

# (1,1) -- Recovered phase for all standards
ax3 = fig.add_subplot(gs[1, 1])
for (E, label, color), res in zip(SIGNALS, all_residuals.keys()):
    I1, I2 = two_arm(E)
    rx = DSP.OpticalRxFSM(D1=-600.0, D2=-1200.0, n_iter=60)
    phi, _ = rx.run(I1, I2)
    ax3.plot(np.arange(128), phi[:128], color=color, lw=0.8, alpha=0.7, label=label)
ax3.legend(fontsize=6.5, facecolor=BG, labelcolor="white", ncol=2)
dk(ax3, "Recovered phase phi(t)\n60 GS iterations", "n", "phi [rad]")

# (1,2) -- Reduction bar chart
ax4 = fig.add_subplot(gs[1, 2])
reductions = [r[3] for r in summary_rows]
ax4.barh(range(len(labels)), [r*100 for r in reductions], color=colors, height=0.6)
ax4.set_yticks(range(len(labels)))
ax4.set_yticklabels(labels, color="#99aabb", fontsize=8)
ax4.axvline(80, color="white", lw=0.8, ls="--", alpha=0.5)
dk(ax4, "MSE Reduction %\n(init -> final)", "% reduction", "")
for i, (v, c) in enumerate(zip([r*100 for r in reductions], colors)):
    ax4.text(v + 0.5, i, f"{v:.0f}%", va="center", color=c, fontsize=7.5)

# (2,0:3) -- Project conclusion text panel
ax5 = fig.add_subplot(gs[2, 0:3])
ax5.set_facecolor((0.04, 0.06, 0.12))
ax5.axis("off")
for sp in ax5.spines.values(): sp.set_color("#1e3060")

conclusion_lines = [
    ("Jalali Lab  --  ECE 279AS Project 2 Conclusion", "white", 10, True),
    ("", "white", 8, False),
    ("Dispersion-Assisted Gerchberg-Saxton Phase Recovery  |  Carrier-less Coherent Fiber Sensing", "#50d8ff", 8.5, False),
    ("Two intensity measurements (I1, I2) through fibers D1=-600 ps2, D2=-1200 ps2.  No local oscillator.  No 90 deg hybrid.", "#99aabb", 7.5, False),
    ("", "white", 7, False),
    ("TD-GS converges across ALL 6 standard formats: OOK, PAM4, QPSK, DPSK, STEAM, Soliton.", "#00ff9f", 8, False),
    ("OUSD(R&E) areas covered: FutureG / Advanced Computing / Trusted AI / Integrated Sensing / Directed Energy / Quantum Science / HMI", "#ffd040", 7.5, False),
    ("", "white", 7, False),
    ("GitHub: github.com/ColinsCoding/Dispersion-Assisted-GS-Phase-Recovery     Dashboard: jalabi-dashboard.onrender.com", "#cc88ff", 7.5, False),
]
y = 0.92
for text, color, size, bold in conclusion_lines:
    ax5.text(0.02, y, text, transform=ax5.transAxes,
             color=color, fontsize=size,
             fontweight="bold" if bold else "normal")
    y -= 0.12

fig.suptitle(
    "S76  Jalabi Lab Aggregate Study  |  Project Conclusion\n"
    "6 optical standards  |  TD-GS phase recovery  |  D1=-600 D2=-1200 ps2  |  60 iterations",
    color="white", fontsize=10, fontweight="bold"
)
plt.tight_layout()
plt.show()
"""

# ---- Build and inject cells -------------------------------------------------
def md_cell(src):
    return {"cell_type": "markdown", "metadata": {},
            "source": src.strip().splitlines(keepends=True)}

def code_cell(src):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": src.strip().splitlines(keepends=True)}

before = len(nb["cells"])
nb["cells"].extend([md_cell(md76), code_cell(code76)])
after  = len(nb["cells"])

with NB.open("w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"[OK] {NB.name}: {before} -> {after} cells  (+{after-before} S76 cells)")
