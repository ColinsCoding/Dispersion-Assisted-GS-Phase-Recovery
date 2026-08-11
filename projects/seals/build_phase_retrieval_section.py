"""Appends the 'Phase Retrieval and Inverse Scattering' section to seals_stable.ipynb.
Run once; re-running regenerates the appended section from scratch (idempotent
w.r.t. the base notebook, which this script re-reads fresh each time)."""
import json

BASE_PATH = "seals_stable.ipynb"
OUT_PATH = "seals_stable.ipynb"
N_BASE_CELLS = 23   # seals_stable.ipynb's original cell count -- everything after this index is ours

nb = json.load(open(BASE_PATH, encoding="utf-8"))
nb["cells"] = nb["cells"][:N_BASE_CELLS]   # idempotent: drop any previously-appended section

# --- pre-existing bug fix, unrelated to this task's new content ---
# Cell 0 of the original notebook is the seals_stable.py module docstring, but it was
# stored as a CODE cell with a stray leading "#" before the opening triple-quote (and
# mangled non-ASCII characters), so the notebook has never actually been executable
# top-to-bottom -- it fails with SyntaxError on the very first cell. This is required to
# fix in order to run (and therefore validate) either the original cells or the new
# section below; the fix only changes this cell's type/formatting, not any numerical code.
MODULE_DOCSTRING_MD = r"""
# seals_stable.py -- Spectrally Encoded Angular Light Scattering (SEALS)

Numerically stable Python port of SEALS.m / mie-2.m / rayleighdebye.m, plus OAM
angular-momentum decomposition, 3D/4D spectral maps, and poker binary-hand
statistics as a Bayesian card-inference demo.

**Fixes vs. Attempt2.ipynb:**
- SS2 `SEALS()`: denominator was `tan(inner)**2`, should be `tan(inner)*tan(a)`
- SS3 `rayleighdebye()`: `P_theta` -> NaN at theta=0; fixed with Taylor guard
- SS4 `mie()`: `E_theta`/`E_phi` were real arrays (silent imag drop); recurrence
  `range(3,nmax)` skipped `p[2]` (`range(2,nmax)` correct); 500xnmax `print()`
  calls removed; full vectorization of angular loop
- SS5 New: Lorenz-Mie partial-wave angular-momentum spectrum `|a_n|^2`, `|b_n|^2`
- SS6 New: 3D scattering (theta, phi) surface via spherical harmonic expansion
- SS7 New: 4D spectral-angular SEALS map (lambda x theta heatmap)
- SS8 New: Spectrally encoded OAM -- LG(p,l) mode decomposition; l -> OAM channel
- SS9 New: Poker binary-hand statistics (P(rank|hole+community) via Monte Carlo)
- SS10 New: Glossy card BRDF (Phong specular on felt-flat card surface)

**Parameter reference (default):** dia=9940 nm, npar=1.39, nmed=1.00,
d=909.09 nm (1100 lines/mm), D=65 mm, a=0.9023 rad, dcorr=-0.42 mm, P=5.8 mm,
NA=0.70, lambda=1580-1600 nm (telecom C-band), mangle=20 deg.
"""
nb["cells"][0] = {"cell_type": "markdown", "id": "module-docstring", "metadata": {},
                   "source": MODULE_DOCSTRING_MD.splitlines(keepends=True)}

def md(id_, text):
    return {"cell_type": "markdown", "id": id_, "metadata": {}, "source": text.splitlines(keepends=True)}

def code(id_, text):
    return {"cell_type": "code", "execution_count": None, "id": id_, "metadata": {}, "outputs": [],
            "source": text.splitlines(keepends=True)}

new_cells = []

new_cells.append(md("pr-title", r"""
---
# Phase Retrieval and Inverse Scattering

**New research extension**, built on top of the validated SEALS/Mie/RDG model above --
not part of the original MATLAB implementation. Implementation lives in
[`inverse/`](inverse/) as reusable modules; this section demonstrates and connects them.
"""))

new_cells.append(code("pr-imports", r"""
import sys
sys.path.insert(0, '.')   # so 'inverse' resolves relative to this notebook's own directory
import numpy as np
import torch
import matplotlib.pyplot as plt

from inverse import measurement, dispersion, phase_retrieval as pr, inverse_scattering as inv

torch.set_default_dtype(torch.float64)
plt.rcParams.update({'figure.dpi': 110, 'font.size': 11})
print('inverse/ package loaded')
"""))

new_cells.append(md("pr-1-detector", r"""
## 1. What the detector measures

A real detector -- and every intensity plot already produced above -- measures
$I=|E|^2$, never $E$ itself. `inverse.measurement.intensity_measurement` makes this an
explicit function call rather than something implicit in how a plot is drawn.
"""))

new_cells.append(code("pr-1-demo", r"""
torch.manual_seed(0)
E_demo = torch.randn(20, dtype=torch.complex128)
I_demo = measurement.intensity_measurement(E_demo)
print('I >= 0 everywhere:', bool(torch.all(I_demo >= 0)))

# global phase invariance -- E and E*exp(i*const) are different fields with IDENTICAL intensity
const_phase = torch.tensor(1.7)
E_shifted = E_demo * torch.exp(1j * const_phase)
max_diff = (I_demo - measurement.intensity_measurement(E_shifted)).abs().max().item()
print(f'max |I(E) - I(E*exp(i*1.7))| = {max_diff:.3e}  (should be ~0)')
"""))

new_cells.append(md("pr-2-why", r"""
## 2. Why intensity loses phase

$|Ae^{i\phi}|^2=A^2$ for *any* $\phi$ -- the phase multiplies $E$ by a unit-magnitude
factor that squaring-then-magnitude discards completely. This is why every phase-retrieval
method below needs *something else*: a known reference, a second measurement through a
known transform, or a constrained physical model -- never raw intensity alone.
"""))

new_cells.append(md("pr-3-mie-phase", r"""
## 3. Existing Mie phase

`mie()` above already computes complex far fields and their phase -- nothing new is
added here, just exposed explicitly via `measurement.mie_complex_fields`, which
reconstructs `E_p`, `E_s` from the validated `I_p`, `I_s`, `T_p`, `T_s` outputs
(`E = sqrt(I) * exp(i*phase)`, an exact algebraic identity, not new physics).
Rayleigh-Debye-Gans has no phase in the original code or here -- `rayleigh_debye()`
returns intensity only, and this section does not add one.
"""))

new_cells.append(code("pr-3-demo", r"""
# reusing 'p' as already defined earlier in this notebook (p = P_DEFAULT.copy(), with
# 'lamvec'/'lambda0' added) -- NOT reassigning it here, to avoid clobbering those keys.
lam0 = p['lambda0']
theta_rad = np.deg2rad(theta_seals + p['mangle'])
theta_rad = np.clip(np.abs(theta_rad), 1e-4, np.pi - 1e-4)

fields = measurement.mie_complex_fields(p['npar'], p['nmed'], p['dia'], lam0, theta_rad, p['r'])
print(f'E_p: complex, shape {fields.E_p.shape}, |E_p| range [{np.abs(fields.E_p).min():.3e}, {np.abs(fields.E_p).max():.3e}]')
print(f'phase_p range: [{np.degrees(fields.phase_p).min():.1f}, {np.degrees(fields.phase_p).max():.1f}] deg')

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(np.degrees(theta_rad), np.degrees(fields.phase_p), label='phase_p')
ax.plot(np.degrees(theta_rad), np.degrees(fields.phase_s), '--', label='phase_s')
ax.set(xlabel='scattering angle (deg)', ylabel='phase (deg)', title='Existing Mie phase (forward calculation)')
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
plt.tight_layout(); plt.show()
"""))

new_cells.append(md("pr-4-inverse-scattering", r"""
## 4. Model-based inverse scattering

Given a *synthetic measured* intensity spectrum generated from the validated Mie model at
a known diameter, with noise added, recover that diameter -- without knowing it in
advance. `inverse_scattering.estimate_diameter` uses a bounded, derivative-free search
(`scipy.optimize.minimize_scalar`) directly against the validated NumPy/SciPy Mie model,
**not autograd**: `mie()` calls `scipy.special.spherical_jn/yn` on plain floats, so it is
not a PyTorch computational graph, and this module does not fake a gradient through it.

The loss is on **log** intensity, because SEALS intensities span tens of dB -- an ordinary
squared-error loss on linear intensity would be dominated by the forward-scattering peak
and effectively ignore the weaker side lobes, which carry most of the size information.
"""))

new_cells.append(code("pr-4-demo", r"""
true_diameter = p['dia']
I_meas = inv.synthesize_measurement(true_diameter, p['npar'], p['nmed'], lam0, theta_rad, p['r'],
                                     noise_std=0.05, seed=42)

bounds = (true_diameter * 0.9, true_diameter * 1.1)
result = inv.estimate_diameter(I_meas, p['npar'], p['nmed'], lam0, theta_rad, p['r'], bounds)

rel_err = abs(result.diameter - true_diameter) / true_diameter
print(f'true diameter      = {true_diameter*1e6:.4f} um')
print(f'recovered diameter = {result.diameter*1e6:.4f} um  ({rel_err*100:.2f}% error, {result.n_evals} evals)')

# loss landscape (requested explicitly): a 1D scan around the search bounds
dia_scan = np.linspace(bounds[0], bounds[1], 150)
loss_scan = [inv.log_intensity_loss(d, p['npar'], p['nmed'], lam0, theta_rad, p['r'], I_meas) for d in dia_scan]

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(dia_scan * 1e6, loss_scan)
ax.axvline(true_diameter * 1e6, color='black', ls='--', label='true diameter')
ax.axvline(result.diameter * 1e6, color='tab:red', ls=':', label='recovered diameter')
ax.set(xlabel='diameter (um)', ylabel='log-intensity loss', title='Loss landscape (why derivative-free: highly oscillatory)')
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
plt.tight_layout(); plt.show()
"""))

new_cells.append(md("pr-4b-phase-connection", r"""
### Connecting inverse scattering to phase (step 6)

Because the Mie forward model predicts a complex field, estimating the particle parameters
from intensity measurements permits reconstruction of the phase predicted by that physical
model. This is more constrained than recovering an arbitrary unknown complex field: no
separate phase optimization happens here at all -- `result.predicted_fields.phase_p` is
just `phase_p` from the (validated) Mie model, evaluated at the *fitted* diameter. Compared
directly against Mie's *true* phase (Section 3) modulo the unavoidable global-phase offset.
"""))

new_cells.append(code("pr-4b-demo", r"""
phase_true_mie = torch.tensor(fields.phase_p, dtype=torch.float64)
phase_fitted_mie = torch.tensor(result.predicted_fields.phase_p, dtype=torch.float64)

global_offset_mie = torch.angle(torch.exp(1j * (phase_true_mie - phase_fitted_mie)).mean())
err_mie = pr.wrapped_phase_error(phase_fitted_mie + global_offset_mie, phase_true_mie)
print(f'mean |phase error| between true-diameter and fitted-diameter Mie phase '
      f'(global-phase corrected): {np.degrees(err_mie.abs().mean().item()):.2f} deg')
print(f'(a {rel_err*100:.2f}% diameter error is expected to leave a small but nonzero '
      f'phase mismatch -- this is NOT a phase-retrieval result, it is a consequence of the '
      f'{rel_err*100:.2f}% parameter error already reported above.)')
"""))

new_cells.append(md("pr-5-generic", r"""
## 5. Generic phase-retrieval example

Section 4 fit a *physical parameter* and read phase off the fitted model. This section is
the different, less-constrained problem: recover an *arbitrary* phase profile from
intensity-only measurements, with no physical model beyond "the amplitude is known."

**This uses a synthetic chirped-Gaussian pulse in time, not the Mie angular field directly.**
An earlier version of this section tried applying the dispersion operator directly to
`fields.E_p` (indexed by *scattering angle*) -- but `dispersive_operator` models
group-velocity dispersion, a physical process in *time*, via an FFT over its input axis.
Applying it to an angle-indexed array runs the same math but does not correspond to any real
optical propagation, and produced a degenerate result (a flat-phase field matched both
"dispersed" measurements to numerical-noise-level loss, `~1e-22`, while remaining nowhere
near the true phase) -- a good illustration of exactly the kind of unjustified-diversity
mistake this task's instructions warn against. Using a genuine time-domain synthetic pulse
here instead keeps the physics honest; Section 4b above is where the real Mie phase
connection lives.
"""))

new_cells.append(code("pr-5-demo", r"""
N_t = 256
t_grid = torch.linspace(-8, 8, N_t, dtype=torch.float64)
amp_true = torch.exp(-t_grid ** 2 / (2 * 2.0 ** 2))
phase_true = 0.5 * 0.3 * t_grid ** 2
E_true = amp_true * torch.exp(1j * phase_true)

D1, D2 = 0.6, -1.4
op1 = lambda E: dispersion.dispersive_operator(E, D1)
op2 = lambda E: dispersion.dispersive_operator(E, D2)
I1 = measurement.intensity_measurement(op1(E_true))
I2 = measurement.intensity_measurement(op2(E_true))

phase_est, loss_hist = pr.retrieve_phase(amp_true, [I1, I2], [op1, op2], n_steps=400, lr=0.05)

raw_err = pr.wrapped_phase_error(phase_est, phase_true)
global_offset = torch.angle(torch.exp(1j * (phase_true - phase_est)).mean())
err_after_global_fix = pr.wrapped_phase_error(phase_est + global_offset, phase_true)

print(f'final loss: {loss_hist[-1]:.3e}')
print(f'mean |error|, raw:                {np.degrees(raw_err.abs().mean().item()):.2f} deg')
print(f'mean |error|, after removing the best-fit global phase offset: '
      f'{np.degrees(err_after_global_fix.abs().mean().item()):.2f} deg')
"""))

new_cells.append(code("pr-5-plot", r"""
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
t_np = t_grid.numpy()
axes[0].semilogy(loss_hist)
axes[0].set(xlabel='iteration', ylabel='loss', title='Phase retrieval: convergence')
axes[0].grid(True, alpha=0.3)

axes[1].plot(t_np, np.degrees(phase_true.numpy()), color='black', lw=2, label='true phase')
axes[1].plot(t_np, np.degrees((phase_est + global_offset).numpy()), '--', color='tab:red',
             label='recovered (global-phase corrected)')
axes[1].set(xlabel='t', ylabel='phase (rad)', title='True vs. recovered phase')
axes[1].legend(fontsize=8); axes[1].grid(True, alpha=0.3)

axes[2].plot(t_np, np.degrees(err_after_global_fix.numpy()))
axes[2].set(xlabel='t', ylabel='wrapped error (deg)', title='Residual error after global-phase correction')
axes[2].grid(True, alpha=0.3)
plt.tight_layout(); plt.show()
"""))

new_cells.append(md("pr-6-diversity", r"""
## 6. Measurement diversity

Does the second dispersion measurement actually help, on this same synthetic field?
Compared directly rather than assumed -- one measurement (`D1` only, explicitly flagged as
underdetermined by `retrieve_phase`) versus two (`D1` and `D2` together).
"""))

new_cells.append(code("pr-6-demo", r"""
import warnings
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter('always')
    phase_single, loss_single = pr.retrieve_phase(amp_true, [I1], [op1], n_steps=400, lr=0.05)
print('warning raised for single measurement:', any('UNDERDETERMINED' in str(w.message) for w in caught))

offset_single = torch.angle(torch.exp(1j * (phase_true - phase_single)).mean())
err_single = pr.wrapped_phase_error(phase_single + offset_single, phase_true)
err_two = err_after_global_fix   # from Section 5

print(f'mean |error| (global-phase corrected), single dispersion: {np.degrees(err_single.abs().mean().item()):.2f} deg')
print(f'mean |error| (global-phase corrected), two dispersions:   {np.degrees(err_two.abs().mean().item()):.2f} deg')
"""))

new_cells.append(md("pr-7-limitations", r"""
## 7. Limitations

- **Global phase ambiguity** is unavoidable for intensity-only measurements (Section 1) --
  every comparison above corrects for it explicitly before reporting error; without that
  correction the raw error is not meaningful.
- **This is a gradient-descent optimizer, not the temporal Gerchberg-Saxton algorithm** in
  `dgs/gs_core.py`. It uses the same dispersion transfer function and the same two-measurement
  diversity idea, but a different (simpler) optimization procedure -- results are not
  claimed to match `gs_core.retrieve_phase`'s convergence behavior.
- **Model-based inversion (Section 4) is more constrained than generic retrieval (Section 5)**
  precisely because it searches over one physical scalar (diameter) instead of an entire
  phase profile -- the loss landscape plot in Section 4 shows *why* even that 1-D problem is
  non-convex (Mie resonances), and Section 5's problem has far more free parameters.
  Do not conflate the two: `estimate_diameter` never claims to solve generic phase retrieval,
  and `retrieve_phase` never claims to uniquely recover an arbitrary phase from one measurement.
- **Noise and initialization** were fixed at the values used above (5% multiplicative noise,
  seed=42 for Section 4; flat-phase start for Section 5/6); different choices change the
  specific numbers printed but not the qualitative distinctions this section is making.
- **A "known" transform must actually apply to the axis being measured.** `dispersive_operator`
  is only physically meaningful along a genuine time axis; applying it to angle-indexed data
  (as an earlier draft of Section 5 did) is not "more diversity," it is a different, unphysical
  problem that happened to admit a degenerate near-zero-loss solution far from the truth. This
  is exactly why Section 6 checks a *specific number* rather than assuming diversity helps.
"""))

nb["cells"].extend(new_cells)
# pin a kernel that actually has scipy+torch (this file needs both, unlike seals_intro.ipynb);
# only touches kernelspec metadata, not cell content or execution results.
nb["metadata"]["kernelspec"] = {"display_name": "Python (.venv Spring2026)", "language": "python", "name": "spring2026"}

with open(OUT_PATH, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
json.load(open(OUT_PATH, encoding="utf-8"))
print(f"appended {len(new_cells)} cells; seals_stable.ipynb now has {len(nb['cells'])} total")
