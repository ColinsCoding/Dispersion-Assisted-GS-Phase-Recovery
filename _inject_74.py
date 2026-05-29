"""
_inject_74.py  —  inject §74 into phase_retrieval.ipynb
========================================================
§74 : Odd/Even · Arithmetic Circuits · Ethereum Explicit Trust
  Cell 107 (markdown) — 6-step theory
  Cell 108 (code)     — demo: odd/even FFT, field butterfly, commitments, on-chain sim

Run:  python _inject_74.py
"""

import json, pathlib, sys

NB = pathlib.Path("phase_retrieval.ipynb")
if not NB.exists():
    sys.exit(f"[ERROR] {NB} not found — run from repo root")

with NB.open("r", encoding="utf-8") as f:
    nb = json.load(f)

# ── Cell 107 — Markdown ──────────────────────────────────────────────────────
md74 = r"""## §74  Odd / Even  ·  Arithmetic Circuits  ·  Ethereum Explicit Trust

### Why these three ideas connect

The GS phase retrieval algorithm calls `np.fft.fft` twice per iteration.
The FFT is a **Cooley-Tukey radix-2 circuit**: at every level it splits
the input into **even-indexed** and **odd-indexed** sub-arrays, then
recombines with a **butterfly gate**:

$$X[k] = \underbrace{X_\text{even}[k]}_{\text{even out}} + \omega^k \cdot X_\text{odd}[k]$$
$$X[k+N/2] = \underbrace{X_\text{even}[k]}_{\text{even out}} - \omega^k \cdot X_\text{odd}[k]$$

That butterfly gate — one multiply + two adds — **is** an arithmetic circuit gate.
The same gate works over any field, including the prime fields Ethereum uses.

---

### 6 Steps: from float to on-chain commitment

| Step | Operation | Math |
|------|-----------|------|
| **1** | Odd/even split | $x_\text{even} = x[0::2]$, $x_\text{odd} = x[1::2]$ |
| **2** | Butterfly gate over $\mathbb{C}$ | $(a,b,\omega) \to (a+\omega b,\; a-\omega b)$ |
| **3** | Map float to $\mathbb{F}_p$ | $I[n] \mapsto \lfloor I[n] \cdot 2^{32} \rfloor \bmod p$ |
| **4** | Butterfly gate over $\mathbb{F}_p$ | same gate, all ops mod $p$ — one R1CS row |
| **5** | keccak256 commitment | $C_{I_1} = \text{keccak256}(I_1\text{.tobytes()})$ |
| **6** | On-chain explicit trust | `OpticalPhaseVerifier.verify(id, I1_raw, I2_raw)` → `bool` |

---

### Explicit trust (no trusted party)

```
Lab runs GS:    I1, I2 → φ
Commits:        C_I1 = keccak256(I1_bytes)
                C_I2 = keccak256(I2_bytes)
                C_φ  = keccak256(phi_bytes)
On-chain:       submit(C_I1, C_I2)  →  measurement_id
                attachSolution(id, C_φ)
Anyone verifies: verify(id, I1_raw, I2_raw)  →  true / false
                 verifySolution(id, phi_raw) →  true / false
```

A 1-bit tamper in $I_1$ → different keccak256 → `verify()` returns `false`.
The Ethereum EVM enforces this. No authority needed.

---

### Field parameters

| Parameter | Value |
|-----------|-------|
| BN128 scalar prime $p$ | `0x30644e72e131a029b85045b68181585d2833e84879b9709142e1f0121b70900d` |
| Primitive root $g$ | $5$ |
| $N$-th root of unity | $\omega_N = g^{(p-1)/N} \bmod p$ |
| $N$-point FFT butterfly count | $\frac{N}{2} \log_2 N$ gates |
| GS iteration circuit gates | $N \log_2 N$ (forward + inverse FFT) |
"""

# ── Cell 108 — Code ──────────────────────────────────────────────────────────
code74 = r"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path('.').resolve()))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Import from dsp module (already in optical_dashboard/)
sys.path.insert(0, 'optical_dashboard')
import dsp as DSP

PI = np.pi

# ─────────────────────────────────────────────────────────────────────────────
# Run the full 6-step demo
# ─────────────────────────────────────────────────────────────────────────────
result = DSP.optical_zk_demo(N=64, snr_db=20.0)

print("=" * 60)
print("  STEP 1: Odd / even split")
print("=" * 60)
I1       = np.array(result["I1"])
I1_even  = np.array(result["I1_even"])
I1_odd   = np.array(result["I1_odd"])
print(f"  I1 length        : {len(I1)}")
print(f"  I1_even[0:4]     : {I1_even[:4].round(4)}")
print(f"  I1_odd [0:4]     : {I1_odd[:4].round(4)}")
print(f"  Even indices: 0,2,4,...  |  Odd indices: 1,3,5,...")

print()
print("=" * 60)
print("  STEP 2: Butterfly gate over C")
print("=" * 60)
a, b = complex(I1[0]), complex(I1[1])
w    = np.exp(-2j * PI * 0 / 4)
top, bot = DSP.fft_butterfly(a, b, w)
print(f"  a = {a:.4f},  b = {b:.4f},  omega = {w:.4f}")
print(f"  top    = a + omega*b = {top:.4f}   (even combination)")
print(f"  bottom = a - omega*b = {bot:.4f}   (odd  combination)")
stats = result["circuit_stats"]
print(f"\n  N={stats['N']}  circuit_depth={stats['circuit_depth']}  "
      f"butterflies={stats['n_butterflies']}  "
      f"(GS iter={stats['gates_per_GS_iter']} gates)")
print(f"  FFT radix-2 matches numpy.fft.fft: {result['fft_match']}")

print()
print("=" * 60)
print("  STEP 3: Float -> F_p (BN128 prime field)")
print("=" * 60)
for i, (f_val, fe_hex, rec) in enumerate(zip(
        result["I1_float_sample"],
        result["I1_field_sample"],
        result["I1_recovered"])):
    print(f"  I1[{i}] = {f_val:.6f}  ->  F_p: {fe_hex}  ->  recover: {rec:.6f}")

print()
print("=" * 60)
print("  STEP 4: Field butterfly over F_p (R1CS gate)")
print("=" * 60)
for b_res in result["butterfly_results"]:
    print(f"  i={b_res['i']}  a={b_res['a'][:10]}...  w={b_res['w'][:10]}...")
    print(f"        top={b_res['top'][:10]}...  bot={b_res['bot'][:10]}...")

print()
print("=" * 60)
print("  STEP 5: keccak256 commitments")
print("=" * 60)
print(f"  C_I1 = {result['C_I1'][:18]}...")
print(f"  C_I2 = {result['C_I2'][:18]}...")
print(f"  C_phi= {result['C_phi'][:18]}...")

print()
print("=" * 60)
print("  STEP 6: On-chain explicit trust simulation")
print("=" * 60)
print(f"  verify(id, I1_raw,    I2_raw)   = {result['verify_I1'] and result['verify_I2']}")
print(f"  verify(id, tampered,  I2_raw)   = {not result['tamper_rejected']}  (1-bit change rejected)")
print(f"  tamper_rejected                 = {result['tamper_rejected']}")

# ─────────────────────────────────────────────────────────────────────────────
# Plots
# ─────────────────────────────────────────────────────────────────────────────
FG=(0.04,0.04,0.10); BG=(0.06,0.06,0.14)
def dk(ax, title="", xl="", yl=""):
    ax.set_facecolor(BG)
    for s in ax.spines.values(): s.set_color("#334466")
    ax.tick_params(colors="#99aabb", labelsize=8)
    if title: ax.set_title(title, color="white", fontsize=8.5, fontweight="bold", pad=5)
    if xl: ax.set_xlabel(xl, color="#99aabb", fontsize=8)
    if yl: ax.set_ylabel(yl, color="#99aabb", fontsize=8)

t    = np.array(result["t"])
I2   = np.array(result["I2"])
phi_true = np.array(result["phi_true"])
phi_est  = np.array(result["phi_est"])

fig = plt.figure(figsize=(14, 9)); fig.patch.set_facecolor(FG)
gs  = gridspec.GridSpec(2, 3, hspace=0.52, wspace=0.40)

# (0,0) Intensity measurements
ax0 = fig.add_subplot(gs[0, 0])
ax0.plot(t, I1, color="#50d8ff", lw=1.2, label="I1  D1=-600ps²")
ax0.plot(t, I2, color="#ffd040", lw=1.2, label="I2  D2=-1200ps²", alpha=0.8)
ax0.legend(fontsize=7, facecolor=BG, labelcolor="white")
dk(ax0, "Step 1: Intensity Measurements I1, I2", "t", "I [a.u.]")

# (0,1) Odd/even split of I1
ax1 = fig.add_subplot(gs[0, 1])
ax1.stem(range(len(I1_even)), I1_even, linefmt="#50d8ff", markerfmt="o",
         basefmt="gray", label="even  I1[0::2]")
ax1.stem(range(len(I1_odd)),  I1_odd,
         linefmt="#ff3278", markerfmt="s",
         basefmt="gray", label="odd   I1[1::2]")
ax1.legend(fontsize=7, facecolor=BG, labelcolor="white")
dk(ax1, "Step 1: Odd / Even Index Split\nCooley-Tukey radix-2 level 1", "n/2", "I1")

# (0,2) FFT spectrum — numpy vs radix2
X_np = np.fft.fft(I1)
X_r2 = DSP.fft_radix2_dit(I1)
freqs = np.fft.fftfreq(len(I1))
sort_idx = np.argsort(freqs)
ax2 = fig.add_subplot(gs[0, 2])
ax2.plot(freqs[sort_idx], np.abs(X_np)[sort_idx], color="#ffd040", lw=1.5,
         label="numpy.fft.fft")
ax2.plot(freqs[sort_idx], np.abs(X_r2)[sort_idx], color="#ff3278", lw=1.0,
         ls="--", label="fft_radix2_dit (explicit)")
ax2.legend(fontsize=7, facecolor=BG, labelcolor="white")
dk(ax2, f"Step 2: FFT Match = {result['fft_match']}\n"
        f"{stats['n_butterflies']} butterfly gates, depth={stats['circuit_depth']}",
   "freq", "|X[k]|")

# (1,0:2) Phase retrieval result
ax3 = fig.add_subplot(gs[1, 0:2])
ax3.plot(t, phi_true, color="#50d8ff", lw=2.0, label="phi_true")
ax3.plot(t, phi_est,  color="#ffd040", lw=1.4, ls="--", label="phi_est (50 GS iter)")
residual = np.std(phi_true - phi_est)
ax3.fill_between(t, phi_true - residual, phi_true + residual,
                 color="#50d8ff", alpha=0.12)
ax3.legend(fontsize=7, facecolor=BG, labelcolor="white")
dk(ax3, f"Step 2-4: GS Phase Retrieval  residual std={residual:.3f} rad",
   "t", "phi [rad]")

# (1,2) Commitment hex comparison
ax4 = fig.add_subplot(gs[1, 2])
labels = ["C_I1", "C_I2", "C_phi", "C_tamper"]
C_tamper = DSP.commit_intensity(np.array(result["I1"])); I1t = np.array(result["I1"]); I1t[0]+=1; C_tamper=DSP.commit_intensity(I1t)
vals = [result["C_I1"], result["C_I2"], result["C_phi"], C_tamper]
colors_c = ["#50d8ff","#ffd040","#00ff9f","#ff3278"]
for i,(lab,val,col) in enumerate(zip(labels,vals,colors_c)):
    ax4.text(0.02, 0.88 - i*0.22, f"{lab}:", transform=ax4.transAxes,
             color=col, fontsize=7.5, fontweight="bold")
    ax4.text(0.22, 0.88 - i*0.22, val[:20]+"...", transform=ax4.transAxes,
             color=col, fontsize=6.5, fontfamily="monospace")
ax4.text(0.02, 0.02, "verify(tampered) = False", transform=ax4.transAxes,
         color="#ff3278", fontsize=7.5, fontweight="bold")
ax4.set_xlim(0,1); ax4.set_ylim(0,1); ax4.axis("off")
dk(ax4, "Step 5-6: keccak256 Commitments\nExplicit Trust — no third party")

fig.suptitle(
    "§74  Odd/Even FFT Circuits  |  F_p Field Butterfly  |  Ethereum Explicit Trust\n"
    "butterfly gate (a,b,w) -> (a+wb, a-wb)  same over C and F_p (BN128)",
    color="white", fontsize=10, fontweight="bold"
)
plt.tight_layout()
plt.show()
"""

# ── Build and inject cells ────────────────────────────────────────────────────
def md_cell(src):
    return {"cell_type": "markdown", "metadata": {},
            "source": src.strip().splitlines(keepends=True)}

def code_cell(src):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": src.strip().splitlines(keepends=True)}

before = len(nb["cells"])
nb["cells"].extend([md_cell(md74), code_cell(code74)])
after  = len(nb["cells"])

with NB.open("w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"[OK] {NB.name}: {before} -> {after} cells  (+{after-before} S74 cells)")
