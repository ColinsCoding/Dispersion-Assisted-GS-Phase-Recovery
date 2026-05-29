"""
_inject_75.py  -  inject S75 into phase_retrieval.ipynb
=========================================================
S75 : TX/RX Dual FSM  *  Modular Arithmetic State Machines
  Cell 109 (markdown) -- 5-step theory  (male/female complementary pair)
  Cell 110 (code)     -- demo: state traces, mod tables, GS residuals, plots

Run:  python _inject_75.py
"""

import json, pathlib, sys

NB = pathlib.Path("phase_retrieval.ipynb")
if not NB.exists():
    sys.exit(f"[ERROR] {NB} not found -- run from repo root")

with NB.open("r", encoding="utf-8") as f:
    nb = json.load(f)

# ---- Cell 109 -- Markdown ---------------------------------------------------
md75 = r"""## S75  TX / RX Dual FSM  *  Modular Arithmetic State Machines

### Male (TX) FSM -- 5 states, cycling mod 5

$$s_{TX}(t) = t \bmod 5 \qquad s \in \{0,1,2,3,4\}$$

| step mod 5 | State | Action |
|:---:|---|---|
| 0 | **IDLE** | load bit stream $\{b_n\}$ |
| 1 | **LOAD_BITS** | verify $N$ bits ready |
| 2 | **MODULATE** | QPSK encode: $b_{2k},b_{2k+1} \to s_k \in \{\pm 1 \pm j\}/\sqrt{2}$ |
| 3 | **DISPERSE** | apply transfer function: $\tilde{E}_m(\nu) = E(\nu)\,e^{i\pi D_m \nu^2}$ |
| 4 | **SAMPLE** | square-law detect: $I_m[n] = |\tilde{E}_m[n]|^2$ |

### Female (RX) FSM -- 6 states, cycling mod 6

$$s_{RX}(t) = t \bmod 6 \qquad s \in \{0,1,2,3,4,5\}$$

| step mod 6 | State | Action |
|:---:|---|---|
| 0 | **IDLE** | await upload |
| 1 | **UPLOAD** | accept $I_1[n], I_2[n]$ |
| 2 | **VALIDATE** | clip NaN, normalise $\max=1$ |
| 3 | **PREPROCESS** | $\hat{E} = \sqrt{I_1}\,e^{i\cdot 0}$ |
| 4 | **GS_ITERATE** | run $n_{iter}$ GS steps (see below) |
| 5 | **CONVERGED** | $\hat{\phi}(t) = \angle\hat{E}$ |

### GS iteration -- modular alternation on $k \bmod 2$

$$k \bmod 2 = 0 \;\Rightarrow\; \text{amplitude constraint:}\quad \hat{E} = \sqrt{I_1}\,e^{i\angle\hat{E}}$$

$$k \bmod 2 = 1 \;\Rightarrow\; \text{dispersive constraint:}\quad \hat{E} = \mathcal{F}^{-1}\!\left[\frac{|\hat{E}_2(\nu)|}{|H_1(\nu)|}\,e^{i\angle(\mathcal{F}[\hat{E}]\cdot H_1)}\right]$$

### Paired complementary design

```
TX  (male)         RX  (female)
IDLE     <---> IDLE
SAMPLE   <---> UPLOAD      I1, I2 flows across link
         AWGN channel      noise added at snr_db
DISPERSE <---> GS_ITERATE  H(nu) forward / inverse
```

The two FSMs are **complementary** -- TX dispersion and RX phase constraint are
inverses of each other under the TD-GS fixed-point iteration.
"""

# ---- Cell 110 -- Code -------------------------------------------------------
code75 = r"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path('.').resolve()))
sys.path.insert(0, 'optical_dashboard')
import dsp as DSP
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrowPatch

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
# Run full TX->AWGN->RX demo
# ------------------------------------------------------------------
result = DSP.optical_link_fsm_demo(n_bits=256, snr_db=15.0)

tx_hist  = result["tx_history"]   # list of (step, state, note)
rx_hist  = result["rx_history"]
tx_mod   = result["tx_mod_table"]  # [(step, step%5, state_name), ...]
rx_mod   = result["rx_mod_table"]
gs_mod   = result["gs_mod_table"]  # [(k, k%2, constraint_name), ...]
residuals = result["residuals"]
I1       = np.array(result["I1"])
I2       = np.array(result["I2"])
I1_noisy = np.array(result["I1_noisy"])
phi_est  = np.array(result["phi_est"])

# ------------------------------------------------------------------
# Print modular arithmetic tables
# ------------------------------------------------------------------
print("=" * 56)
print("  TX FSM -- modular arithmetic  (step mod 5)")
print("=" * 56)
print(f"  {'step':>4}  {'step mod 5':>10}  state")
print(f"  {'-'*4}  {'-'*10}  {'-'*15}")
for step, mod_val, state in tx_mod:
    marker = " <-- SAMPLE" if state == "SAMPLE" else ""
    print(f"  {step:>4}  {mod_val:>10}  {state}{marker}")

print()
print("=" * 56)
print("  RX FSM -- modular arithmetic  (step mod 6)")
print("=" * 56)
print(f"  {'step':>4}  {'step mod 6':>10}  state")
print(f"  {'-'*4}  {'-'*10}  {'-'*15}")
for step, mod_val, state in rx_mod:
    marker = " <-- CONVERGED" if state == "CONVERGED" else ""
    print(f"  {step:>4}  {mod_val:>10}  {state}{marker}")

print()
print("=" * 56)
print("  GS iteration -- k mod 2 constraint alternation")
print("=" * 56)
print(f"  {'k':>4}  {'k mod 2':>7}  constraint")
print(f"  {'-'*4}  {'-'*7}  {'-'*20}")
for k, mod_val, constraint in gs_mod:
    print(f"  {k:>4}  {mod_val:>7}  {constraint}")

print()
print("=" * 56)
print("  TX history")
print("=" * 56)
for step, state, note in tx_hist:
    print(f"  step {step:>2}  [{state:<12}]  {note}")

print()
print("=" * 56)
print("  RX history")
print("=" * 56)
for step, state, note in rx_hist:
    print(f"  step {step:>2}  [{state:<12}]  {note}")

# ------------------------------------------------------------------
# Figure
# ------------------------------------------------------------------
fig = plt.figure(figsize=(14, 10))
fig.patch.set_facecolor(FG)
gs  = gridspec.GridSpec(3, 3, hspace=0.58, wspace=0.42)

# (0,0) -- TX state trace
ax0 = fig.add_subplot(gs[0, 0])
tx_steps  = [h[0] for h in tx_hist]
tx_states = [DSP.OpticalTxFSM.STATES.index(h[1]) for h in tx_hist]
ax0.step(tx_steps, tx_states, color="#50d8ff", lw=2, where="post")
ax0.scatter(tx_steps, tx_states, color="#ffd040", s=60, zorder=3)
ax0.set_yticks(range(len(DSP.OpticalTxFSM.STATES)))
ax0.set_yticklabels(DSP.OpticalTxFSM.STATES, color="#99aabb", fontsize=7)
dk(ax0, "TX FSM  (male)  state trace\nstep = t mod 5", "step", "state")

# (0,1) -- RX state trace
ax1 = fig.add_subplot(gs[0, 1])
rx_steps  = [h[0] for h in rx_hist]
rx_states = [DSP.OpticalRxFSM.STATES.index(h[1]) for h in rx_hist]
ax1.step(rx_steps, rx_states, color="#ffd040", lw=2, where="post")
ax1.scatter(rx_steps, rx_states, color="#50d8ff", s=60, zorder=3)
ax1.set_yticks(range(len(DSP.OpticalRxFSM.STATES)))
ax1.set_yticklabels(DSP.OpticalRxFSM.STATES, color="#99aabb", fontsize=7)
dk(ax1, "RX FSM  (female)  state trace\nstep = t mod 6", "step", "state")

# (0,2) -- GS k mod 2 alternation
ax2 = fig.add_subplot(gs[0, 2])
ks       = [g[0] for g in gs_mod]
mods     = [g[1] for g in gs_mod]
colors_k = ["#50d8ff" if m == 0 else "#ff3278" for m in mods]
ax2.bar(ks, mods, color=colors_k, width=0.6)
ax2.set_yticks([0, 1])
ax2.set_yticklabels(["amplitude\n(even k)", "dispersion\n(odd k)"],
                    color="#99aabb", fontsize=7)
ax2.set_xticks(ks)
dk(ax2, "GS iteration: k mod 2\nBlue=amplitude  Red=dispersion", "GS step k", "constraint")

# (1,0) -- I1 clean vs noisy
ax3 = fig.add_subplot(gs[1, 0])
t_plot = np.arange(len(I1[:128]))
ax3.plot(t_plot, I1[:128],       color="#50d8ff", lw=1.2, label="I1 clean")
ax3.plot(t_plot, I1_noisy[:128], color="#ffd040", lw=1.0, alpha=0.75,
         ls="--", label="I1 noisy")
ax3.legend(fontsize=7, facecolor=BG, labelcolor="white")
dk(ax3, f"I1 clean vs noisy  SNR={result['snr_db']:.0f} dB", "n", "I [a.u.]")

# (1,1) -- I2 measurement
ax4 = fig.add_subplot(gs[1, 1])
I2       = np.array(result["I2"])
I2_noisy = np.array(result["I2_noisy"])
ax4.plot(t_plot, I2[:128],       color="#00ff9f", lw=1.2, label="I2 clean")
ax4.plot(t_plot, I2_noisy[:128], color="#ff3278", lw=1.0, alpha=0.75,
         ls="--", label="I2 noisy")
ax4.legend(fontsize=7, facecolor=BG, labelcolor="white")
dk(ax4, f"I2  D2={result['D2']:.0f} ps2", "n", "I [a.u.]")

# (1,2) -- Recovered phase
ax5 = fig.add_subplot(gs[1, 2])
ax5.plot(np.arange(len(phi_est)), phi_est, color="#cc88ff", lw=1.0)
ax5.axhline(0, color="gray", lw=0.5, ls=":")
dk(ax5, f"Recovered phase  phi(t)\n{result['D1']:.0f}/{result['D2']:.0f} ps2  "
        f"60 GS iter", "n", "phi [rad]")

# (2,0:2) -- GS convergence residuals
ax6 = fig.add_subplot(gs[2, 0:2])
n_res = len(residuals)
k_axis = np.arange(n_res)
even_idx = k_axis[k_axis % 2 == 0]
odd_idx  = k_axis[k_axis % 2 == 1]
res_arr  = np.array(residuals)
ax6.semilogy(k_axis, res_arr, color="#50d8ff", lw=1.5, alpha=0.6, label="all")
ax6.semilogy(even_idx, res_arr[even_idx], "o", color="#50d8ff",
             ms=4, label="k even (amplitude)")
ax6.semilogy(odd_idx,  res_arr[odd_idx],  "s", color="#ff3278",
             ms=4, label="k odd  (dispersion)")
ax6.legend(fontsize=7.5, facecolor=BG, labelcolor="white")
ax6.set_xlim(0, n_res)
dk(ax6, "GS Convergence -- residual vs iteration\n"
        "blue=amplitude constraint (k even)   red=dispersion constraint (k odd)",
   "GS iteration k", "residual MSE")

# (2,2) -- mod tables text panel
ax7 = fig.add_subplot(gs[2, 2])
ax7.set_facecolor(BG); ax7.axis("off")
lines = ["TX  step mod 5:", ""] + \
        [f"  {r[0]:>2} mod 5 = {r[1]}  {r[2]}" for r in tx_mod[:5]] + \
        ["", "RX  step mod 6:", ""] + \
        [f"  {r[0]:>2} mod 6 = {r[1]}  {r[2]}" for r in rx_mod[:6]]
for i, line in enumerate(lines[:14]):
    ax7.text(0.05, 0.97 - i * 0.066, line, transform=ax7.transAxes,
             color="#99aabb", fontsize=6.8, fontfamily="monospace")
for sp in ax7.spines.values(): sp.set_color("#334466")
ax7.set_title("Modular Arithmetic Tables", color="white",
              fontsize=8.5, fontweight="bold", pad=5)

fig.suptitle(
    "S75  TX/RX Dual FSM  |  Modular Arithmetic State Machines\n"
    "TX (male): 5 states  step%5 | "
    "RX (female): 6 states  step%6 | "
    "GS alternation k%2",
    color="white", fontsize=9.5, fontweight="bold"
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
nb["cells"].extend([md_cell(md75), code_cell(code75)])
after  = len(nb["cells"])

with NB.open("w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"[OK] {NB.name}: {before} -> {after} cells  (+{after-before} S75 cells)")
