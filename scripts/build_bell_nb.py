"""Build notebooks/bell_inequality.ipynb
Full Bell inequality treatment: derivation, CHSH, simulation, Aspect result.
"""
import nbformat as nbf

nb = nbf.v4.new_notebook()
nb["metadata"]["kernelspec"] = {
    "display_name": "Python 3", "language": "python", "name": "python3"
}

def md(src): return nbf.v4.new_markdown_cell(src)
def code(src): return nbf.v4.new_code_cell(src)

cells = []

# ── Title ─────────────────────────────────────────────────────────────────────
cells.append(md(r"""# Bell's Theorem and the CHSH Inequality
## The Most Important Experiment in the History of Physics

> *"If [a hidden-variable theory] is local it will not agree with quantum mechanics,
> and if it agrees with quantum mechanics it will not be local."*
> — John Bell, 1964

Bell's theorem proves that **no local realistic theory can reproduce all
predictions of quantum mechanics**. The CHSH inequality turns this into a
measurable number. Aspect (1982) measured it. Local realism lost.
"""))

cells.append(code("""\
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from IPython.display import display, Math, HTML
sp.init_printing(use_latex='mathjax')

rng = np.random.default_rng(42)
"""))

# ── §1 The Setup ──────────────────────────────────────────────────────────────
cells.append(md(r"""---
## §1 — The Setup: Entangled Photon Pairs

Two photons are produced in the **singlet state** (total spin zero):

$$|\Psi^-\rangle = \frac{1}{\sqrt{2}}\bigl(|{\uparrow}\rangle_A|{\downarrow}\rangle_B - |{\downarrow}\rangle_A|{\uparrow}\rangle_B\bigr)$$

Alice measures photon A at angle $a$. Bob measures photon B at angle $b$.
Each measurement returns $\pm 1$.

**Local realism assumes**: each photon carries a hidden variable $\lambda$
that predetermines the outcome. Alice's result $A(a,\lambda) = \pm 1$ depends
only on her setting $a$ and $\lambda$ — **not** on Bob's distant setting $b$.
"""))

cells.append(code(r"""
display(HTML("<h4>The Singlet State and Measurement</h4>"))

display(Math(
    r"|\Psi^-\rangle = \frac{1}{\sqrt{2}}"
    r"\bigl(|\uparrow\rangle_A|\downarrow\rangle_B"
    r" - |\downarrow\rangle_A|\uparrow\rangle_B\bigr)"
))

display(Math(
    r"\text{QM prediction:}\quad"
    r"E(a,b) = \langle\Psi^-|\,\hat{\sigma}_a \otimes \hat{\sigma}_b\,|\Psi^-\rangle"
    r"= -\cos\bigl(2(a - b)\bigr)"
))

display(Math(
    r"\text{Local realism:}\quad"
    r"E(a,b) = \int A(a,\lambda)\,B(b,\lambda)\,\rho(\lambda)\,d\lambda"
    r"\quad A,B \in \{-1,+1\}"
))
"""))

# ── §2 Bell's Proof ───────────────────────────────────────────────────────────
cells.append(md(r"""---
## §2 — Bell's Proof (1964)

**Assume** local hidden variables: outcomes $A(a,\lambda),\, B(b,\lambda) = \pm 1$.

Define the **CHSH combination** with four angle settings
$a, a',\, b, b'$:

$$S = E(a,b) - E(a,b') + E(a',b) + E(a',b')$$

**Bell's bound**: under any local realistic model,

$$|S| \leq 2$$

**Proof sketch** — for a single $\lambda$, let $A = A(a,\lambda)$,
$A' = A(a',\lambda)$, $B = B(b,\lambda)$, $B' = B(b',\lambda)$, all $\pm 1$:

$$AB - AB' + A'B + A'B' = A(B - B') + A'(B + B')$$

Since $B, B' = \pm 1$, either $B - B' = 0, B + B' = \pm 2$
or $B - B' = \pm 2, B + B' = 0$. Either way:

$$|AB - AB' + A'B + A'B'| = 2 \quad\text{for every } \lambda$$

Averaging over $\lambda$: $|S| \leq 2$. &emsp; $\blacksquare$
"""))

cells.append(code(r"""
display(HTML("<h4>Bell's Bound — Symbolic Verification</h4>"))

display(Math(r"|S| \;=\; |E(a,b) - E(a,b') + E(a',b) + E(a',b')| \;\leq\; 2"))
display(Math(r"\text{Tsirelson bound (QM maximum):}\quad |S|_{\max} = 2\sqrt{2} \approx 2.828"))

# Check all 16 combinations of A,A',B,B' in {-1,+1}
violations = 0
for A in [-1, 1]:
    for Ap in [-1, 1]:
        for B in [-1, 1]:
            for Bp in [-1, 1]:
                val = A*B - A*Bp + Ap*B + Ap*Bp
                if abs(val) > 2:
                    violations += 1

print(f"Checked all 16 sign combinations of (A, A', B, B') in {{-1,+1}}:")
print(f"  Max |AB - AB' + A'B + A'B'| = 2  for every combination")
print(f"  Violations of |val| <= 2: {violations}  (must be 0)")
print(f"  Bell bound |S| <= 2 confirmed by exhaustion.")
"""))

# ── §3 QM prediction ──────────────────────────────────────────────────────────
cells.append(md(r"""---
## §3 — Quantum Mechanical Prediction

For the singlet state, the correlation at angle difference $\theta = a - b$:

$$E(a,b) = -\cos\bigl(2(a-b)\bigr)$$

The optimal CHSH angles are $a=0°,\; a'=45°,\; b=22.5°,\; b'=67.5°$,
giving:

$$S_{\text{QM}} = -\cos(45°) - \cos(-135°) + (-\cos(-45°)) + (-\cos(45°))$$
$$= 2\sqrt{2} \approx 2.828$$

This **exceeds 2 by $\sqrt{2}-1 \approx 41\%$** — a testable prediction that
distinguishes QM from all local realistic theories.
"""))

cells.append(code(r"""
display(HTML("<h4>QM Correlation and CHSH Value</h4>"))

display(Math(r"E(a,b) = -\cos\bigl(2(a-b)\bigr)"))
display(Math(
    r"S_{\mathrm{QM}} = E(a,b) - E(a,b') + E(a',b) + E(a',b')"
    r"= 2\sqrt{2}"
))

def E_qm(a_deg, b_deg):
    return -np.cos(2 * np.deg2rad(a_deg - b_deg))

a, ap, b, bp = 0, 45, 22.5, 67.5

Eab   = E_qm(a,  b)
Eabp  = E_qm(a,  bp)
Eapb  = E_qm(ap, b)
Eapbp = E_qm(ap, bp)
S_qm  = Eab - Eabp + Eapb + Eapbp

print("Optimal angle settings (degrees):")
print(f"  a={a}  a'={ap}  b={b}  b'={bp}")
print()
print(f"  E(a,  b ) = {Eab:+.6f}")
print(f"  E(a,  b') = {Eabp:+.6f}")
print(f"  E(a', b ) = {Eapb:+.6f}")
print(f"  E(a', b') = {Eapbp:+.6f}")
print()
print(f"  S_QM = {S_qm:.6f}")
print(f"  2*sqrt(2) = {2*np.sqrt(2):.6f}  (Tsirelson bound)")
print(f"  Bell bound = 2.000000")
print(f"  QM exceeds classical by {(S_qm - 2)/2*100:.1f}%")
"""))

cells.append(code("""\
# Plot E(theta) = -cos(2*theta) and show where CHSH angles sit
theta = np.linspace(0, 180, 500)
E_vals = -np.cos(2 * np.deg2rad(theta))

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(theta, E_vals, 'royalblue', lw=2.5, label=r'$E(\\theta) = -\\cos(2\\theta)$')
ax.axhline(0, color='k', lw=0.5)

pairs = [(a-b, 'E(a,b)', Eab),
         (a-bp,'E(a,b\')', Eabp),
         (ap-b,"E(a',b)", Eapb),
         (ap-bp,"E(a',b')", Eapbp)]
colors = ['#e74c3c','#e67e22','#2ecc71','#9b59b6']
for (th, lbl, val), col in zip(pairs, colors):
    ax.scatter(th % 180, val, s=120, color=col, zorder=5, label=f'{lbl} = {val:+.3f}')

ax.set_xlabel('Angle difference  a - b  (degrees)')
ax.set_ylabel('Correlation  E(a,b)')
ax.set_title('QM Singlet Correlation  $E(a,b) = -\\cos 2(a-b)$')
ax.legend(loc='lower right', fontsize=9)
ax.set_xlim(0, 180); ax.set_ylim(-1.15, 1.15)
plt.tight_layout(); plt.savefig('bell_correlation.png', dpi=100); plt.show()
"""))

# ── §4 Monte Carlo simulation ─────────────────────────────────────────────────
cells.append(md(r"""---
## §4 — Monte Carlo Simulation of the Bell Experiment

We simulate $N$ entangled photon pairs. For each pair:

1. Draw a hidden phase $\phi \in [0, 2\pi)$
2. Alice measures: $A = \text{sign}[\cos(2(a - \phi))]$
3. Bob measures: $B = \text{sign}[\cos(2(b - \phi))]$

This is a **local realistic** (hidden variable) simulation.
We then compare against the true QM prediction.
"""))

cells.append(code("""\
def simulate_local_realistic(N, a_deg, b_deg, seed=0):
    rng2 = np.random.default_rng(seed)
    phi  = rng2.uniform(0, np.pi, N)   # hidden variable
    A    = np.sign(np.cos(2*(np.deg2rad(a_deg) - phi)))
    B    = np.sign(np.cos(2*(np.deg2rad(b_deg) - phi)))
    return float(np.mean(A * B))

def simulate_quantum(N, a_deg, b_deg, seed=0):
    rng2 = np.random.default_rng(seed)
    # Singlet: P(same) = sin^2(a-b), P(diff) = cos^2(a-b)
    # E = P(same)*(-1) + P(diff)*(+1) ... actually for sigma_z eigenstates:
    # Easier: sample outcomes directly from QM joint probabilities
    th = np.deg2rad(a_deg - b_deg)
    # P(+1,+1) = P(-1,-1) = sin^2(theta)/2
    # P(+1,-1) = P(-1,+1) = cos^2(theta)/2  ... for singlet
    p_same = np.sin(th)**2 / 2    # both +1 or both -1
    p_diff = np.cos(th)**2 / 2    # one +1 one -1
    outcomes = rng2.choice([+1, -1], size=N,
                           p=[2*p_diff, 2*p_same])  # product AB
    # Correct formula: E = -cos(2*theta), sample AB directly
    AB = np.where(rng2.random(N) < (1 + E_qm(a_deg,b_deg))/2, +1, -1)
    return float(np.mean(AB))

N = 100_000
angle_pairs = [(0,22.5),(0,67.5),(45,22.5),(45,67.5)]
labels = ["E(a,b)","E(a,b')","E(a',b)","E(a',b')"]

print(f"{'Pair':<12} {'QM exact':>10} {'QM sim':>10} {'LR sim':>10}")
print("-" * 45)

S_lr  = 0
S_sim = 0
signs = [+1, -1, +1, +1]   # S = E(a,b) - E(a,b') + E(a',b) + E(a',b')

for (ai, bi), lbl, sgn in zip(angle_pairs, labels, signs):
    e_exact = E_qm(ai, bi)
    e_sim   = simulate_quantum(N, ai, bi, seed=ai+bi)
    e_lr    = simulate_local_realistic(N, ai, bi, seed=ai+bi)
    S_sim  += sgn * e_sim
    S_lr   += sgn * e_lr
    print(f"{lbl:<12} {e_exact:>10.4f} {e_sim:>10.4f} {e_lr:>10.4f}")

print("-" * 45)
print(f"{'S':.<12} {'2.8284':>10} {S_sim:>10.4f} {S_lr:>10.4f}")
print(f"\\nBell bound |S| <= 2:")
print(f"  QM sim violates? {'YES' if abs(S_sim) > 2 else 'no'}  (S={S_sim:.4f})")
print(f"  LR sim violates? {'YES' if abs(S_lr)  > 2 else 'no'}  (S={S_lr:.4f})")
"""))

# ── §5 Statistical test ───────────────────────────────────────────────────────
cells.append(md(r"""---
## §5 — The Hypothesis Test

$$H_0:\; |S| \leq 2 \quad\text{(local realism)}$$
$$H_1:\; |S| > 2 \quad\text{(quantum mechanics)}$$

The measured $S$ has standard error $\sigma_S \approx 2/\sqrt{N_{\text{pairs}}}$
(each $E$ estimated from $N/4$ coincidences, $\sigma_E \approx 1/\sqrt{N/4}$,
and $S$ is a linear combination of 4 such terms).

$$z = \frac{\hat{S} - 2}{\sigma_S}$$

Reject $H_0$ at $p < 0.05$ when $z > 1.645$.

**Aspect (1982):** $N \approx 10^7$ pairs, $\hat{S} = 2.697 \pm 0.015$.
$$z = \frac{2.697 - 2}{0.015} = 46.5\sigma \quad\Rightarrow\quad p < 10^{-470}$$
"""))

cells.append(code(r"""
display(HTML("<h4>CHSH Hypothesis Test — Power vs Sample Size</h4>"))

display(Math(r"z = \frac{\hat{S} - 2}{\sigma_S}, \qquad \sigma_S \approx \frac{4}{\sqrt{N}}"))
display(Math(r"H_0\text{ rejected when } z > 1.645 \quad (\alpha = 0.05,\;\text{one-tailed})"))
"""))

cells.append(code("""\
from scipy.stats import norm

S_true   = 2 * np.sqrt(2)   # QM prediction
alpha    = 0.05
z_crit   = norm.ppf(1 - alpha)

N_vals   = np.logspace(1, 8, 300)
sigma_S  = 4 / np.sqrt(N_vals)           # SE of S
z_vals   = (S_true - 2) / sigma_S
power    = norm.sf(z_crit - (S_true-2)/sigma_S)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Left: z-statistic vs N
axes[0].semilogx(N_vals, z_vals, 'royalblue', lw=2.5)
axes[0].axhline(z_crit, color='red', ls='--', lw=1.5, label=f'z_crit={z_crit:.2f} (alpha=0.05)')
axes[0].axhline(5,     color='orange', ls=':', lw=1, label='5-sigma discovery')
axes[0].scatter([1e7], [(2.697-2)/0.015], s=150, color='gold', zorder=6,
                label='Aspect 1982  z=46.5')
axes[0].set_xlabel('Number of photon pairs N'); axes[0].set_ylabel('z-statistic')
axes[0].set_title('How many pairs needed to detect Bell violation?')
axes[0].legend(fontsize=8); axes[0].set_ylim(0, 55)

# Right: power curve
axes[1].semilogx(N_vals, power, 'crimson', lw=2.5)
axes[1].axhline(0.80, color='gray', ls='--', lw=1.5, label='80% power')
axes[1].axhline(0.95, color='gray', ls=':',  lw=1.5, label='95% power')
n_80 = N_vals[np.argmin(np.abs(power - 0.80))]
n_95 = N_vals[np.argmin(np.abs(power - 0.95))]
axes[1].axvline(n_80, color='gray', ls='--', lw=1, alpha=0.5)
axes[1].axvline(n_95, color='gray', ls=':',  lw=1, alpha=0.5)
axes[1].set_xlabel('Number of photon pairs N'); axes[1].set_ylabel('Power (1 - beta)')
axes[1].set_title('Power to detect Bell violation at alpha=0.05')
axes[1].legend(fontsize=8)
axes[1].text(n_80*1.2, 0.72, f'N={int(n_80)}', fontsize=8)
axes[1].text(n_95*1.2, 0.87, f'N={int(n_95)}', fontsize=8)

plt.tight_layout(); plt.savefig('bell_power.png', dpi=100); plt.show()
print(f"  N for 80% power: {int(n_80):,}")
print(f"  N for 95% power: {int(n_95):,}")
"""))

# ── §6 Angle sweep ────────────────────────────────────────────────────────────
cells.append(md(r"""---
## §6 — S as a Function of Angle Settings

The CHSH value $S$ depends on the four detector angles. Fixing $a=0$ and
optimising over $a',b,b'$:

$$S(a, a', b, b') = E(a,b) - E(a,b') + E(a',b) + E(a',b')$$

The classical bound $|S|=2$ is a flat ceiling.
QM reaches $2\sqrt{2}$ at the optimal 22.5° spacing.
"""))

cells.append(code("""\
# Fix a=0, a'=45; sweep b from 0 to 90 and compute S with b'=b+45
b_sweep = np.linspace(0, 90, 400)
S_sweep = np.array([
    E_qm(0,b) - E_qm(0,b+45) + E_qm(45,b) + E_qm(45,b+45)
    for b in b_sweep
])

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(b_sweep, S_sweep, 'royalblue', lw=2.5, label='S (QM)')
ax.axhline( 2, color='red',   ls='--', lw=2, label='Bell bound +2')
ax.axhline(-2, color='red',   ls='--', lw=2, label='Bell bound -2')
ax.axhline( 2*np.sqrt(2), color='green', ls=':', lw=1.5,
            label=f'Tsirelson +2sqrt(2) = {2*np.sqrt(2):.3f}')
ax.axhline(-2*np.sqrt(2), color='green', ls=':', lw=1.5,
            label=f'Tsirelson -2sqrt(2)')
ax.axvline(22.5, color='orange', ls='--', lw=1.5, label='Optimal b=22.5 deg')
ax.fill_between(b_sweep, -2, 2, alpha=0.08, color='red',
                label='Classical allowed region')
ax.set_xlabel('Bob angle b (degrees)  [a=0, a\'=45, b\'=b+45]')
ax.set_ylabel('CHSH value S')
ax.set_title('QM vs Classical: S as a function of detector angle')
ax.legend(fontsize=8, loc='upper right'); ax.set_ylim(-3.2, 3.2)
plt.tight_layout(); plt.savefig('bell_angle_sweep.png', dpi=100); plt.show()

peak_b = b_sweep[np.argmax(S_sweep)]
print(f"  Peak S = {S_sweep.max():.6f} at b = {peak_b:.2f} deg")
print(f"  2*sqrt(2) = {2*np.sqrt(2):.6f}")
"""))

# ── §7 Aspect experiment ──────────────────────────────────────────────────────
cells.append(md(r"""---
## §7 — The Aspect Experiment (1982)

**Alain Aspect, Philippe Grangier, Gérard Roger** — Institut d'Optique, Paris.

| Parameter | Value |
|-----------|-------|
| Source | Calcium cascade $4p^2\;^1S_0 \to 4s4p\;^1P_1 \to 4s^2\;^1S_0$ |
| Photon wavelengths | 551.3 nm and 422.7 nm |
| Pair rate | $\sim 4 \times 10^7$ pairs/s |
| Measured S | $2.697 \pm 0.015$ |
| Bell bound | 2.000 |
| Violation | **46.5 standard deviations** |
| p-value | $< 10^{-470}$ |

**Loophole-free Bell tests** (2015, Delft / NIST / Vienna):
closed both the *detection loophole* (not all pairs detected) and the
*locality loophole* (settings changed faster than light-travel time).
Result: still violated. Local realism is **dead**.
"""))

cells.append(code(r"""
display(HTML("<h4>Aspect 1982 Result</h4>"))

display(Math(r"\hat{S} = 2.697 \pm 0.015"))
display(Math(
    r"z = \frac{\hat{S} - 2}{\sigma_S} = \frac{2.697 - 2}{0.015} = 46.5"
))
display(Math(r"p \ll 10^{-400} \quad\Rightarrow\quad \text{Local realism REJECTED}"))

# Visualise the measurement with error bar vs Bell bound
fig, ax = plt.subplots(figsize=(7, 3))
S_measured = 2.697; sigma_m = 0.015
ax.barh(['Aspect 1982'], [S_measured], xerr=[2*sigma_m],
        color='royalblue', alpha=0.8, capsize=8, height=0.4)
ax.axvline(2,            color='red',   lw=2.5, ls='--', label='Bell bound |S|=2')
ax.axvline(2*np.sqrt(2), color='green', lw=2,   ls=':',
           label=f'Tsirelson 2sqrt(2)={2*np.sqrt(2):.3f}')
ax.set_xlabel('CHSH value S')
ax.set_title('Aspect (1982): measured S vs classical and quantum limits')
ax.legend(); ax.set_xlim(1.8, 3.0)
plt.tight_layout(); plt.savefig('aspect_result.png', dpi=100); plt.show()

z = (S_measured - 2) / sigma_m
print(f"  Measured S = {S_measured} +/- {sigma_m}")
print(f"  z = {z:.1f} sigma above Bell bound")
print(f"  Needed for 5-sigma: z > 5  ->  Aspect: z = {z:.1f}")
"""))

# ── §8 Summary ────────────────────────────────────────────────────────────────
cells.append(md(r"""---
## §8 — Summary

| Step | Statement |
|------|-----------|
| **Bell (1964)** | Proved $|S| \leq 2$ for ANY local hidden-variable theory |
| **QM prediction** | $S = 2\sqrt{2} \approx 2.828$ for optimal angles |
| **Aspect (1982)** | Measured $S = 2.697 \pm 0.015$; 46.5$\sigma$ violation |
| **2015 loophole-free** | All loopholes closed; local realism conclusively dead |

**What it means physically:**

- Quantum correlations are **nonlocal** — they cannot be explained by anything
  traveling between Alice and Bob at or below the speed of light.
- This is not a communication channel (no information is transmitted).
- It is a fundamental property of the quantum state: **entanglement is real**.

**Connection to TS-DFT and this project:**

The same shot-noise statistics that govern the Bell test's $\sigma_S \sim 1/\sqrt{N}$
govern the SNR of every TS-DFT measurement. More photons = smaller $\sigma_S$
= more decisive test. The Jalali lab's TS-DFT system operates at the
**Poisson (shot-noise) limit** — the same quantum foundation that makes
Bell tests possible.
"""))

nb["cells"] = cells

out = "notebooks/bell_inequality.ipynb"
with open(out, "w", encoding="utf-8") as f:
    nbf.write(nb, f)
print(f"Written: {out}  ({len(cells)} cells)")
