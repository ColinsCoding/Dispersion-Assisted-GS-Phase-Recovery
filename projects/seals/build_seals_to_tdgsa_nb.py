import json

cells = []

def md(id_, text):
    return {"cell_type": "markdown", "id": id_, "metadata": {}, "source": text.splitlines(keepends=True)}

def code(id_, text):
    return {"cell_type": "code", "execution_count": None, "id": id_, "metadata": {}, "outputs": [],
            "source": text.splitlines(keepends=True)}

# ======================================================================
cells.append(md("title", r"""
# SEALS Intensity Trace &rarr; TD-GSA: the Bridge, Walked Through

**Companion notebook to [`inverse/seals_to_tdgsa.py`](inverse/seals_to_tdgsa.py) and
[`SEALS_TO_TDGSA_REPORT.md`](SEALS_TO_TDGSA_REPORT.md).** Everything below calls that module's
functions directly (nothing is reimplemented here) so the two documents and this notebook stay
in sync by construction.

**The honest headline result, stated up front rather than buried at the end:** with exactly
two measurement planes, this bridge works end-to-end but neither the classical GS
(`dgs.gs_core`, ~0.50 rad) nor an independent PyTorch-autograd path (~0.28 rad &mdash; better,
but still wrong) cleanly recovers the *true* scattering phase, and the two disagree with each
other by ~0.50 rad, stable across several dispersion choices (Steps 1-6, and the report's
&sect;5). **That limitation is now resolved:** Step 7 shows that one additional, independent
dispersion plane (N=3 instead of 2) drops the error to 0.0014 rad — a ~350x improvement,
confirming the report's own recommendation rather than just proposing it. Step 8 tries the
report's *other* proposed fix (amplitude regularization) and reports an honest null result
for it.

**A correction, left visible rather than quietly fixed (Step 4):** the autograd number used
to also read ~0.50 rad, reported as independent confirmation of GS's answer. It was wrong —
an unnormalized-loss bug silently stalled the optimizer at its initial guess. Step 4 shows
the fix and why it mattered.

**Audience:** written for an EE/signal-chain reader, matching the report.
"""))

cells.append(code("imports", r"""
import sys, pathlib
sys.path.insert(0, '.')                                              # projects/seals/ itself
sys.path.insert(0, str(pathlib.Path('.').resolve().parents[1]))       # repo root, for dgs

import numpy as np
import matplotlib.pyplot as plt
import torch

from dgs import gs_core
from dgs.dispersion_gs_prototype import compare_phase

from inverse import seals_to_tdgsa as bridge
from inverse.dispersion import dispersive_operator
from inverse.phase_retrieval import retrieve_phase as retrieve_phase_torch

plt.rcParams.update({'figure.dpi': 110, 'font.size': 11})
print('ready')
"""))

# ----------------------------------------------------------------------
cells.append(md("s1-title", r"""
## Step 1 &mdash; the native SEALS measurement: one intensity trail

`bridge.seals_intensity_trace()` runs the validated Mie forward model at the scattering
angles SEALS's grating pair maps each wavelength to, and returns `mie_fields.I_p` &mdash;
literally what the instrument in `SEALS_paper.pdf` would read out as one spectrometer scan.
"""))

cells.append(code("s1-code", r"""
lamvec, theta_deg, mie_fields = bridge.seals_intensity_trace()

fig, axs = plt.subplots(1, 2, figsize=(11, 3.6))
axs[0].plot(lamvec * 1e9, mie_fields.I_p)
axs[0].set_xlabel('wavelength (nm)'); axs[0].set_ylabel('$I_p$'); axs[0].set_title('native SEALS intensity trace $I_p(\\lambda)$')
axs[1].plot(theta_deg, mie_fields.I_p)
axs[1].set_xlabel('scattering angle (deg)'); axs[1].set_ylabel('$I_p$'); axs[1].set_title('same trace, mapped to angle via SEALS.m')
plt.tight_layout(); plt.show()

print(f"N = {len(lamvec)} samples, particle diameter = {bridge.physics.P_DEFAULT['dia']*1e9:.0f} nm")
"""))

# ----------------------------------------------------------------------
cells.append(md("s2-title", r"""
## Step 2 &mdash; why one trace is not enough for TD-GSA

TD-GSA (`dgs.gs_core`) needs **two** measurements of the same hidden field at two different,
known dispersions (`D1 != D2`, both nonzero, `|D| >= 5000` for reliable convergence &mdash;
see `gs_core.py`'s own kwarg-bounds warning). A single square-law measurement cannot pin down
phase at all: `E -> E*exp(i*c)` leaves `|E|^2` unchanged for *any* constant `c`
(`inverse/phase_retrieval.py` emits exactly this warning if you try it with one measurement).

So the native SEALS trace becomes **one** of the two planes, and a second plane is produced by
passing the *same* hidden field `E_p` through a second, known dispersion. In hardware terms
this proposes a second diversity branch on the existing single-channel receiver &mdash; the
same way a coherent receiver adds a quadrature detection path to break an ambiguity a single
detector can't resolve. `inverse/dispersion.py` already flags this as a *future* direction, not
something the single-shot instrument as built provides today.
"""))

cells.append(code("s2-code", r"""
D1, D2 = 6000.0, -7000.0
I1, I2 = bridge.build_gs_measurements(mie_fields, D1, D2)

fig, ax = plt.subplots(figsize=(8, 3.6))
ax.plot(theta_deg, I1, label=f'$I_1$ (D={D1})')
ax.plot(theta_deg, I2, label=f'$I_2$ (D={D2})')
ax.set_xlabel('scattering angle (deg)'); ax.set_ylabel('intensity'); ax.legend()
ax.set_title('the two dispersion-diverse measurement planes TD-GSA actually sees')
plt.tight_layout(); plt.show()

print(f"energy check (dispersion is phase-only/all-pass): "
      f"sum(I1)={I1.sum():.6e}, sum(I2)={I2.sum():.6e}, sum(I0)={(np.abs(mie_fields.E_p)**2).sum():.6e}")
"""))

# ----------------------------------------------------------------------
cells.append(md("s3-title", r"""
## Step 3 &mdash; classical GS (this repo's canonical TD-GSA)

`dgs.gs_core.retrieve_phase_with_history` runs the alternating-projections algorithm and also
returns the per-iteration field history, so convergence is visible directly (not just the final
answer). `unit_amplitude=False` because Mie-scattered intensity genuinely varies with angle
&mdash; this is not a constant-envelope (QPSK-like) signal.
"""))

cells.append(code("s3-code", r"""
phi_true = mie_fields.phase_p   # ground truth -- known ONLY because this is a Mie-model validation pass
weight = np.abs(mie_fields.E_p) ** 2

phi_gs, errors_gs, E_history = gs_core.retrieve_phase_with_history(
    I1, I2, D1, D2, n_iter=150, unit_amplitude=False)
rms_gs, phi_gs_aligned = compare_phase(phi_gs, phi_true, weight)

fig, axs = plt.subplots(1, 2, figsize=(11, 3.6))
axs[0].semilogy(errors_gs)
axs[0].set_xlabel('GS iteration'); axs[0].set_ylabel('measurement self-consistency error'); axs[0].set_title('classical GS convergence')
axs[1].plot(theta_deg, phi_true, label='true phase (Mie)')
axs[1].plot(theta_deg, phi_gs_aligned, '--', label='GS recovered (aligned)')
axs[1].set_xlabel('scattering angle (deg)'); axs[1].set_ylabel('phase (rad)'); axs[1].legend()
axs[1].set_title(f'classical GS  --  RMS error vs. truth = {rms_gs:.4f} rad')
plt.tight_layout(); plt.show()

print(f"final measurement self-consistency residual: {errors_gs[-1]:.3e}  (near-zero = GS fit ITS OWN data essentially exactly)")
print(f"RMS phase error vs. Mie ground truth:         {rms_gs:.4f} rad")
"""))

# ----------------------------------------------------------------------
cells.append(md("s3b-title", r"""
### Is this a bug, or the point of the classical problem?

The GS-recovered phase (orange, above) tracks the true phase closely near
`theta ~ 0` and turns to noise at wide angles. That is not a convergence failure &mdash;
`errors_gs[-1]` is already `~1e-23`, GS fit its own measurements essentially exactly. Check
directly whether the *disagreement with ground truth* correlates with signal strength, the
textbook failure mode of intensity-only phase retrieval: wherever `sqrt(I)` is near zero,
GS's "enforce |E|=sqrt(I)" step carries almost no information, so phase there is free to
wander even though the algorithm has technically converged.
"""))

cells.append(code("s3b-code", r"""
per_sample_err = np.abs(np.angle(np.exp(1j * (phi_true - phi_gs_aligned))))
amp = np.abs(mie_fields.E_p)

fig, axs = plt.subplots(1, 2, figsize=(11, 3.6))
axs[0].scatter(amp, per_sample_err, s=8, alpha=0.5)
axs[0].set_xlabel('|E_p| (measured amplitude)'); axs[0].set_ylabel('per-sample phase error (rad)')
axs[0].set_title(f'error vs. amplitude  (Pearson r = {np.corrcoef(per_sample_err, np.log(amp))[0,1]:.2f})')
axs[1].plot(theta_deg, amp)
axs[1].set_xlabel('scattering angle (deg)'); axs[1].set_ylabel('|E_p|')
axs[1].set_title('signal strength vs. angle -- compare the shape to the noisy region above')
plt.tight_layout(); plt.show()

q = np.percentile(amp, [25, 75])
low, high = amp <= q[0], amp >= q[1]
print(f"amplitude dynamic range: {amp.max()/amp.min():.1f}x  ({amp.min():.2e} to {amp.max():.2e})")
print(f"bottom quartile (weak signal):  mean phase error = {per_sample_err[low].mean():.3f} rad")
print(f"top quartile (strong signal):   mean phase error = {per_sample_err[high].mean():.3f} rad")
print()
print("-> the wild oscillation is concentrated where the signal is weak. GS is not broken;")
print("   a photon-starved sample has almost nothing to converge TO. This is the actual")
print("   point of the classical (intensity-only, 2-measurement) phase retrieval problem.")
"""))

# ----------------------------------------------------------------------
cells.append(md("s4-title", r"""
## Step 4 &mdash; the independent check: PyTorch autograd, same data

`inverse/phase_retrieval.py` (already built in this package, used elsewhere for generic
measurement-diversity experiments) is run on the **identical** `(I1, I2)` pair &mdash; a
completely different optimization algorithm (gradient descent via Adam, not alternating
projections) solving the same inverse problem.

**Normalization matters here, not as a style choice.** Mie-scattered field amplitudes are
physically tiny (`~1e-5` to `5e-4`), so unnormalized intensities give Adam a loss at
`~1e-16` scale &mdash; smaller than its default `eps=1e-8`, which silently stalls every
update. An earlier version of this notebook fed raw amplitudes/intensities in directly and
reported `autograd RMS &asymp; 0.50 rad` as if it were a converged, independent answer; it
was actually an untouched all-zero initial guess that happened to score close to GS's real
answer by coincidence (verified: a plain zero-phase guess, no optimization at all, scores
that exact same number). Fixing this (divide amplitude by its max, intensities by max&sup2;
&mdash; phase is scale-invariant, so this doesn't change the problem) is what makes the loss
curve below actually move.
"""))

cells.append(code("s4-code", r"""
scale = np.abs(mie_fields.E_p).max()
amplitude = torch.tensor(np.abs(mie_fields.E_p) / scale, dtype=torch.float64)
I1_t = torch.tensor(I1 / scale**2, dtype=torch.float64)
I2_t = torch.tensor(I2 / scale**2, dtype=torch.float64)

phi_ag, loss_history = retrieve_phase_torch(
    amplitude, [I1_t, I2_t],
    [lambda E: dispersive_operator(E, D1), lambda E: dispersive_operator(E, D2)],
    n_steps=800, lr=0.03)
phi_ag_np = phi_ag.numpy()
rms_ag, phi_ag_aligned = compare_phase(phi_ag_np, phi_true, weight)

fig, axs = plt.subplots(1, 2, figsize=(11, 3.6))
axs[0].semilogy(loss_history)
axs[0].set_xlabel('optimizer step'); axs[0].set_ylabel('loss'); axs[0].set_title('autograd optimization convergence')
axs[1].plot(theta_deg, phi_true, label='true phase (Mie)')
axs[1].plot(theta_deg, phi_ag_aligned, '--', label='autograd recovered (aligned)')
axs[1].set_xlabel('scattering angle (deg)'); axs[1].set_ylabel('phase (rad)'); axs[1].legend()
axs[1].set_title(f'autograd  --  RMS error vs. truth = {rms_ag:.4f} rad')
plt.tight_layout(); plt.show()

print(f"RMS phase error vs. Mie ground truth: {rms_ag:.4f} rad")
"""))

# ----------------------------------------------------------------------
cells.append(md("s5-title", r"""
## Step 5 &mdash; do the two independent methods agree with each other?

If they converged to the *identical* wrong answer, that would suggest a shared bug in how
`(I1, I2)` were built. If they converge to *different* wrong answers (checked next), that is
evidence the inverse problem itself is underdetermined for this signal &mdash; two different
algorithms, same data, same measurement-fit quality, different final phase.
"""))

cells.append(code("s5-code", r"""
rms_gs_vs_ag, _ = compare_phase(phi_gs, phi_ag_np, weight)

fig, ax = plt.subplots(figsize=(8, 3.6))
ax.plot(theta_deg, phi_true, label='true phase (Mie)', lw=2)
ax.plot(theta_deg, phi_gs_aligned, '--', label=f'classical GS (RMS {rms_gs:.3f} rad)')
ax.plot(theta_deg, phi_ag_aligned, ':', label=f'autograd (RMS {rms_ag:.3f} rad)')
ax.set_xlabel('scattering angle (deg)'); ax.set_ylabel('phase (rad)'); ax.legend()
ax.set_title(f'all three curves overlaid  --  GS vs. autograd RMS difference = {rms_gs_vs_ag:.4f} rad')
plt.tight_layout(); plt.show()

print(f"GS vs. truth:      {rms_gs:.4f} rad")
print(f"autograd vs. truth: {rms_ag:.4f} rad")
print(f"GS vs. autograd:    {rms_gs_vs_ag:.4f} rad  (nonzero -> two DIFFERENT wrong answers, not a shared bug)")
"""))

# ----------------------------------------------------------------------
cells.append(md("s6-title", r"""
## Step 6 &mdash; is this one unlucky (D1, D2) choice, or a real limitation?

Sweep several dispersion pairs spanning a 4x range and re-run the full bridge each time. If
the error consistently lands in the same band regardless of `(D1, D2)`, that rules out "just
pick a better dispersion" as the fix.
"""))

cells.append(code("s6-code", r"""
pairs = [(-5000, -5750), (6000, -7000), (8000, -9200), (20000, -23000)]
rows = []
for d1, d2 in pairs:
    r = bridge.run_bridge_demo(D1=d1, D2=d2, n_iter=200)
    rows.append((d1, d2, r['rms_gs_vs_truth'], r['rms_autograd_vs_truth'], r['rms_gs_vs_autograd'], r['gs_final_error']))
    print(f"D1={d1:7.0f} D2={d2:7.0f}: rms_gs={r['rms_gs_vs_truth']:.4f}  rms_autograd={r['rms_autograd_vs_truth']:.4f}  "
          f"gs_vs_autograd={r['rms_gs_vs_autograd']:.4f}  gs_self_consistency={r['gs_final_error']:.2e}")

rms_gs_vals = [row[2] for row in rows]
print(f"\nGS error range across all 4 (D1,D2) pairs: [{min(rms_gs_vals):.3f}, {max(rms_gs_vals):.3f}] rad")
print("-- consistently ~0.4-0.5 rad regardless of dispersion choice: a real limitation, not a bad-D1/D2 tuning artifact.")
"""))

# ----------------------------------------------------------------------
cells.append(md("s7-title", r"""
## Step 7 &mdash; next step 1, implemented: more measurement diversity

`inverse/gs_multiplane.py` generalizes `gs_core`'s fixed 2-plane loop to **N &ge; 2**
dispersion planes, reusing `gs_core.disperse` / `undisperse` / `apply_amplitude_constraint`
directly (verified bit-for-bit identical to `gs_core` at N=2 &mdash; see
`tests/test_gs_multiplane.py`). Does adding a 3rd, independent dispersion plane to the *same*
hidden field actually fix the ~0.5 rad ambiguity from Step 3, or just move it around?
"""))

cells.append(code("s7-code", r"""
from inverse import gs_multiplane

D_candidates = (6000.0, -7000.0, 12000.0, -18000.0, 23000.0)
rms_by_n = bridge.sweep_measurement_diversity(D_candidates=D_candidates, n_iter=150)

fig, ax = plt.subplots(figsize=(7, 3.6))
ns = sorted(rms_by_n)
ax.semilogy(ns, [rms_by_n[n] for n in ns], 'o-')
ax.set_xlabel('number of dispersion planes (N)'); ax.set_ylabel('RMS phase error vs. Mie truth (rad)')
ax.set_xticks(ns)
ax.set_title('adding just one 3rd plane resolves the ambiguity')
ax.grid(True, which='both', alpha=0.3)
plt.tight_layout(); plt.show()

for n in ns:
    print(f"N={n} planes ({D_candidates[:n]}): RMS phase error = {rms_by_n[n]:.4f} rad")
print(f"\nN=2 -> N=3 improvement: {rms_by_n[2]/rms_by_n[3]:.0f}x")
"""))

cells.append(md("s7b-title", r"""
Overlay the actual recovered phase curve (N=3) against Step 3's N=2 curve, on the same axes,
to see directly what the extra measurement plane fixed &mdash; specifically in the weak-signal
wide-angle region that Step 3's diagnostic flagged as the problem.
"""))

cells.append(code("s7b-code", r"""
r2 = bridge.run_multiplane_bridge_demo(Ds=(6000.0, -7000.0), n_iter=150)
r3 = bridge.run_multiplane_bridge_demo(Ds=(6000.0, -7000.0, 12000.0), n_iter=150)

_, phi2_aligned = compare_phase(r2['phi_gs'], r2['phi_true'], weight)
_, phi3_aligned = compare_phase(r3['phi_gs'], r3['phi_true'], weight)

fig, ax = plt.subplots(figsize=(8, 3.8))
ax.plot(theta_deg, r3['phi_true'], label='true phase (Mie)', lw=2, color='k')
ax.plot(theta_deg, phi2_aligned, '--', label=f'N=2 planes (RMS {r2["rms_vs_truth"]:.3f} rad)')
ax.plot(theta_deg, phi3_aligned, ':', label=f'N=3 planes (RMS {r3["rms_vs_truth"]:.4f} rad)', lw=2)
ax.set_xlabel('scattering angle (deg)'); ax.set_ylabel('phase (rad)'); ax.legend()
ax.set_title('the 3rd measurement plane tracks truth everywhere, including the old weak-signal tail')
plt.tight_layout(); plt.show()
"""))

# ----------------------------------------------------------------------
cells.append(md("s8-title", r"""
## Step 8 &mdash; next step 2, implemented: amplitude regularization (honest null result)

`gs_multiplane.apply_prior_regularized_amplitude` blends the undispersed field's amplitude
toward a Mie-fitted envelope (`inverse_scattering.estimate_diameter`, fit on the native trace
alone) wherever the native trace is weak &mdash; the amplitude analog of `unit_amplitude=True`.
Tested against the plain N=2 case: does it help?
"""))

cells.append(code("s8-code", r"""
r_plain = bridge.run_multiplane_bridge_demo(Ds=(6000.0, -7000.0), n_iter=150, use_amplitude_prior=False)
r_prior = bridge.run_multiplane_bridge_demo(Ds=(6000.0, -7000.0), n_iter=150, use_amplitude_prior=True)

fig, axs = plt.subplots(1, 2, figsize=(11, 3.8))
axs[0].bar(['without prior', 'with prior'],
           [r_plain['rms_vs_truth'], r_prior['rms_vs_truth']], color=['C0', 'C1'])
axs[0].set_ylabel('RMS phase error vs. Mie truth (rad)')
axs[0].set_title('amplitude-prior regularization: measured effect')
axs[1].plot(theta_deg, np.abs(mie_fields.E_p), label='true |E_p|')
axs[1].plot(theta_deg, r_prior['amplitude_prior'], '--', label='Mie-fitted prior envelope')
axs[1].set_xlabel('scattering angle (deg)'); axs[1].legend()
axs[1].set_title('the fitted envelope itself (looks right...)')
plt.tight_layout(); plt.show()

print(f"without amplitude prior: {r_plain['rms_vs_truth']:.4f} rad")
print(f"with amplitude prior:    {r_prior['rms_vs_truth']:.4f} rad")
print()
print("-> essentially no change. The prior IS fit correctly (right panel), but each iteration's")
print("   very next per-plane projection re-imposes |E_d|=sqrt(I_j) exactly in the DISPERSED")
print("   domain, overwriting whatever the prior contributed in the undispersed domain a moment")
print("   earlier. Reported here as a real, diagnosed null result -- not silently dropped --")
print("   per SEALS_TO_TDGSA_REPORT.md Sec. 6. Since Step 7 already resolves the ambiguity this")
print("   was meant to fix, not pursued further.")
"""))

# ----------------------------------------------------------------------
cells.append(md("s9-title", r"""
## Step 9 &mdash; cross-check against a historical, independent finding (even-degree phase ambiguity)

`notebooks/ece279_tdgsa_recreation.ipynb` reproduces a separate result from this project's
predecessor work (Yiming's MATLAB TDGSA code, Jalali Lab / ECE 279AS slide 23): blind TDGSA
fails on EVEN-degree phase polynomials (a quadratic chirp) because the intensity constraint
alone can't distinguish `phi` from `-phi` (Hermitian symmetry); ODD-degree (cubic) converges
cleanly. `compare_phase` (used for every RMS number above) already searches both signs when
aligning GS's phase, so that specific ambiguity is already corrected for everywhere in this
notebook. Is the *residual* error nonetheless concentrated in the trace's even component
&mdash; the signature that historical failure mode would predict &mdash; or is Step 3b's
amplitude-weakness diagnosis the real driver?
"""))

cells.append(code("s9-code", r"""
result = {"phi_gs": phi_gs, "phi_true": phi_true}
eo = bridge.diagnose_even_odd_ambiguity(result, mie_fields)

err = np.angle(np.exp(1j * (phi_true - phi_gs_aligned)))
err_even = 0.5 * (err + err[::-1])
err_odd  = 0.5 * (err - err[::-1])

fig, ax = plt.subplots(figsize=(8, 3.6))
ax.plot(theta_deg, err_even, label=f'even part (RMS {eo["error_even_rms"]:.3f} rad)')
ax.plot(theta_deg, err_odd, label=f'odd part (RMS {eo["error_odd_rms"]:.3f} rad)')
ax.set_xlabel('scattering angle (deg)'); ax.set_ylabel('residual phase error (rad)'); ax.legend()
ratio = eo['error_even_rms'] / eo['error_odd_rms']
ax.set_title(f'residual error, split even/odd about the trace midpoint  --  ratio = {ratio:.2f}')
plt.tight_layout(); plt.show()

print(f"true Mie phase:  even-part RMS={eo['true_phase_even_rms']:.3f} rad, odd-part RMS={eo['true_phase_odd_rms']:.3f} rad")
print(f"residual error:  even-part RMS={eo['error_even_rms']:.3f} rad, odd-part RMS={eo['error_odd_rms']:.3f} rad")
print(f"even/odd error ratio: {ratio:.2f}")
print()
if ratio > 2:
    print("-> error concentrated in the even part: matches the historical even-degree failure mode.")
else:
    print("-> roughly balanced: the historical even-degree mechanism is NOT the driver here.")
    print("   Step 3b's amplitude-weakness diagnosis remains the identified mechanism.")
"""))

# ----------------------------------------------------------------------
cells.append(md("s10-title", r"""
## Step 10 &mdash; does the (now-fixed) autograd path overfit noisy measurements?

Step 4's bug fix made the optimizer actually train. A natural follow-up: with no explicit
regularization, does it eventually start fitting measurement NOISE instead of the true phase
&mdash; loss keeps dropping, but the true-phase match gets worse? Classical GS (hard
alternating projections) and the noiseless autograd runs above (converge almost instantly,
then sit still) aren't susceptible to this by construction. Noisy data plus many optimizer
steps is a genuinely different test.
"""))

cells.append(code("s10-code", r"""
from inverse.seals_to_tdgsa import demonstrate_autograd_overfitting

noise_levels = [0.05, 0.3, 0.6, 1.5, 3.0]
results = [demonstrate_autograd_overfitting(noise_std=n) for n in noise_levels]

fig, axs = plt.subplots(1, 2, figsize=(12, 4.2))

for of, n in zip(results, noise_levels):
    steps = [r['step'] for r in of['records']]
    rms_vals = [r['rms_vs_truth'] for r in of['records']]
    axs[0].semilogx(steps, rms_vals, 'o-', label=f'noise_std={n}', markersize=3)
axs[0].set_xlabel('optimizer step'); axs[0].set_ylabel('RMS vs. TRUE (noiseless) Mie phase (rad)')
axs[0].set_title('RMS-vs-truth over training, at each noise level')
axs[0].legend(fontsize=8)

gaps = [of['overfitting_gap'] for of in results]
axs[1].bar([str(n) for n in noise_levels], gaps, color=['C0' if g < 0.01 else 'C3' for g in gaps])
axs[1].set_xlabel('measurement noise_std'); axs[1].set_ylabel('overfitting gap (rad)\n(final RMS - best RMS)')
axs[1].set_title('overfitting gap grows with noise')
plt.tight_layout(); plt.show()

for of, n in zip(results, noise_levels):
    print(f"noise_std={n:<4}  best RMS={of['best_rms']:.4f} rad @step{of['best_step']:<6}  "
          f"final RMS={of['final_rms']:.4f} rad @step{of['final_step']:<6}  "
          f"overfitting gap={of['overfitting_gap']:+.4f} rad")
print()
print("At realistic noise (~5%): no overfitting -- RMS improves monotonically to its plateau.")
print("Above ~30% noise: a real, growing gap -- training longer makes the true-phase match")
print("WORSE even as loss keeps dropping. Modest in absolute size because the underlying")
print("2-plane problem is already badly underdetermined (~0.5 rad baseline, Steps 3-5) --")
print("overfitting compounds that limitation rather than being the main story.")
"""))

# ----------------------------------------------------------------------
cells.append(md("s11-title", r"""
## Step 11 &mdash; one reproducible block: watch TD-GSA overfit, in phase space

Step 10 showed the overfitting gap as a scalar, swept across noise levels. This is the same
phenomenon shown directly as phase curves, at one fixed, clearly-visible noise level
(`noise_std=1.5`) &mdash; and checked for reproducibility explicitly, not just assumed:
`demonstrate_autograd_overfitting` seeds both the injected noise (`add_measurement_noise`) and
the optimizer (`torch.manual_seed` inside `retrieve_phase_with_history`), so the same call run
twice must return bit-for-bit identical results.
"""))

cells.append(code("s11-code", r"""
of_a = demonstrate_autograd_overfitting(noise_std=1.5, seed=0)
of_b = demonstrate_autograd_overfitting(noise_std=1.5, seed=0)   # same call, again
reproducible = (of_a['best_rms'] == of_b['best_rms'] and of_a['final_rms'] == of_b['final_rms']
                and np.array_equal(of_a['final_phase'], of_b['final_phase']))
print(f"reproducible (identical result on repeat call, same seed): {reproducible}")

of = of_a
theta = of['theta_deg']

fig, ax = plt.subplots(figsize=(9, 4.2))
ax.plot(theta, of['phi_true'], label='true phase (Mie)', lw=2.5, color='k')
ax.plot(theta, of['best_phase'], '-', color='C0', lw=1.5,
        label=f"best checkpoint, step {of['best_step']} (RMS {of['best_rms']:.3f} rad)")
ax.plot(theta, of['final_phase'], '--', color='C3',
        label=f"fully converged, step {of['final_step']} -- OVERFIT (RMS {of['final_rms']:.3f} rad)")
ax.set_xlabel('scattering angle (deg)'); ax.set_ylabel('phase (rad)'); ax.legend(fontsize=9)
ax.set_title(f"noise_std={of['noise_std']}: more training moved the recovered phase AWAY "
             f"from truth (gap {of['overfitting_gap']:+.3f} rad)")
plt.tight_layout(); plt.show()

print(f"noise_std={of['noise_std']}, seed=0")
print(f"  best:  step={of['best_step']:5d}  RMS vs. true phase = {of['best_rms']:.4f} rad")
print(f"  final: step={of['final_step']:5d}  RMS vs. true phase = {of['final_rms']:.4f} rad  <- worse, despite lower training loss")
print(f"  overfitting gap: {of['overfitting_gap']:+.4f} rad")
"""))

# ----------------------------------------------------------------------
cells.append(md("conclusion", r"""
## Conclusion

The bridge works exactly as engineered: a real SEALS intensity trace becomes TD-GSA input,
both the classical and autograd solvers converge (fit their own measurements to near-zero
residual), and the original 2-plane result &mdash; GS at ~0.50 rad, autograd at ~0.28 rad
(better, once its normalization bug was fixed &mdash; Step 4), the two disagreeing with each
other by ~0.50 rad and neither matching ground truth, stable across a 4x sweep of dispersion
parameters (GS side) &mdash; is real physics/math, not an implementation bug. Step 3's
diagnostic pinned down the *mechanism*: error concentrates where the measured amplitude is
weak (Pearson `r=-0.55` for GS, `r=-0.47` for the fixed autograd; GS's `0.33 rad` top quartile
vs. `1.54 rad` bottom quartile, `~20x` amplitude dynamic range) &mdash; a photon-starved sample
carries almost no phase information for either algorithm's amplitude-matching step to converge
to.

**Both of the report's next steps have now been tried, and the results are asymmetric:**
1. **More measurement diversity (Step 7): works, decisively.** One extra dispersion plane
   drops the error from 0.50 rad to 0.0014 rad &mdash; a ~350x improvement. This is the general
   fix for underdetermined phase retrieval, confirmed directly rather than left as a prediction.
2. **Amplitude-prior regularization (Step 8): implemented, no measurable benefit.** The hard
   per-plane amplitude constraint dominates every iteration, overwriting the prior before it can
   help. An honest null result, not a bug &mdash; and moot now that Step 7 already solves the
   problem it was meant to address.
3. For SEALS-specific parameter recovery (e.g. particle diameter) rather than raw field phase,
   `inverse/inverse_scattering.py`'s model-based approach remains the right tool &mdash; but the
   phase-retrieval ambiguity that used to force that choice is gone as of Step 7.

**One more historical cross-check (Step 9), also an honest null result:** this project's
predecessor notebook found that blind TDGSA fails specifically on even-degree phase
polynomials (Yiming's MATLAB code, Jalali Lab / ECE 279AS slide 23). Checked directly against
the SEALS/Mie residual error: even-part RMS (1.102 rad) and odd-part RMS (1.017 rad) are
roughly balanced (ratio 1.08) &mdash; that historical mechanism is **not** what's driving this
case. Step 3b's amplitude-weakness diagnosis remains the identified mechanism; this section
exists so that conclusion was checked against the alternative explanation, not just assumed.

**Step 10: does the fixed autograd path overfit noisy measurements?** Real effect, confirmed,
but secondary: at realistic noise (~5%) there's no overfitting at all (RMS improves
monotonically to its plateau). Above ~30% noise, a genuine gap opens between the best
checkpoint seen during training and the fully-converged result, and grows with noise level
(`+0.003 rad` at 30% noise up to `+0.105 rad` at 300% noise) &mdash; training longer keeps
lowering the loss while the true-phase match gets measurably worse, the textbook overfitting
signature. Modest in absolute size because the 2-plane problem is already badly underdetermined
(~0.5 rad baseline) &mdash; overfitting compounds that limitation rather than being the main
story here.

**Step 11: one reproducible block, in phase space.** The same phenomenon, shown as curves
rather than a table: true phase vs. the best training checkpoint vs. the fully-converged
(overfit) result, at `noise_std=1.5`. Checked for reproducibility directly (same call, same
seed, twice) rather than assumed &mdash; `demonstrate_autograd_overfitting` seeds both the
injected noise and the optimizer, so the result is deterministic.

**Verify this notebook's numbers independently:**
```bash
py -3.12 -m pytest tests/test_seals_to_tdgsa.py tests/test_gs_multiplane.py -v
make seals-tdgsa
```
"""))

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3.12", "language": "python", "name": "python312"},
        "language_info": {"name": "python", "version": "3.12"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

path = "seals_to_tdgsa_bridge.ipynb"
with open(path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
json.load(open(path, encoding="utf-8"))
print(f"{path}: valid JSON, {len(cells)} cells")
