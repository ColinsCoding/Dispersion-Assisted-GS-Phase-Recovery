"""
generate_journal_club_figures.py -- 18 publication-style figures summarizing
the full SEALS -> TD-GSA investigation (SEALS_TO_TDGSA_REPORT.md Sec. 1-11),
for presentation to a physical-science audience (journal club).

PURELY A PRESENTATION SCRIPT. No new analysis -- every number and array here
comes from the already-tested functions in seals_to_tdgsa.py, gs_multiplane.py,
and noise_robustness.py (see tests/test_seals_to_tdgsa.py,
tests/test_gs_multiplane.py, tests/test_noise_robustness.py for the underlying
verification). This script only handles curation, ordering, and formatting.

Run: py -3.12 generate_journal_club_figures.py
Output: journal_club_figures/fig01_....png ... fig18_....png (300 dpi)
"""
import sys
import pathlib

sys.path.insert(0, '.')
sys.path.insert(0, str(pathlib.Path('.').resolve().parents[1]))

import numpy as np
import matplotlib.pyplot as plt
import torch

from dgs import gs_core
from dgs.dispersion_gs_prototype import compare_phase

from inverse import seals_to_tdgsa as bridge
from inverse import gs_multiplane
from inverse import noise_robustness
from inverse.dispersion import dispersive_operator
from inverse.phase_retrieval import retrieve_phase as retrieve_phase_torch

# ── journal-quality style ────────────────────────────────────────────────────
plt.rcParams.update({
    'figure.dpi': 150, 'savefig.dpi': 300,
    'font.family': 'sans-serif', 'font.size': 12,
    'axes.labelsize': 12, 'axes.titlesize': 12, 'legend.fontsize': 10,
    'xtick.labelsize': 10, 'ytick.labelsize': 10,
    'axes.grid': True, 'grid.alpha': 0.25,
    'figure.constrained_layout.use': True,
})
TRUE_C, GS_C, AG_C, N3_C, BAD_C = 'k', 'C0', 'C1', 'C2', 'C3'

OUTDIR = pathlib.Path('journal_club_figures')
OUTDIR.mkdir(exist_ok=True)


def savefig(fig, name):
    path = OUTDIR / name
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print(f"  wrote {path}")


def display_pair(phi_est_aligned, phi_true):
    """For plotting only: phase is only physically defined mod 2*pi, so a
    raw line plot of two independently-wrapped arctan2 outputs shows sharp,
    meaningless jumps wherever either one crosses the -pi/pi branch cut,
    even when the two curves actually agree closely. Build the recovered
    curve for display as (unwrapped true phase) + (the real, already
    correctly-wrapped signed error) instead, so the only jumps that can
    appear are genuine large errors, never branch-cut artifacts. Does not
    change any RMS number already computed elsewhere -- display only."""
    phi_true_unwrapped = np.unwrap(phi_true)
    err = np.angle(np.exp(1j * (phi_est_aligned - phi_true)))
    return phi_true_unwrapped + err, phi_true_unwrapped


# ── shared data (computed once, reused across figures) ──────────────────────
print("Computing shared results (reuses tested functions, no new analysis)...")
lamvec, theta_deg, mie_fields = bridge.seals_intensity_trace()
phi_true = mie_fields.phase_p
weight = np.abs(mie_fields.E_p) ** 2
amp_true = np.abs(mie_fields.E_p)
D1, D2 = 6000.0, -7000.0

r2plane = bridge.run_bridge_demo(D1=D1, D2=D2, n_iter=150)
diag = bridge.diagnose_amplitude_dependence(r2plane, mie_fields)
eo = bridge.diagnose_even_odd_ambiguity(r2plane, mie_fields)

I1, I2 = bridge.build_gs_measurements(mie_fields, D1, D2)
_, _, E_history = gs_core.retrieve_phase_with_history(I1, I2, D1, D2, n_iter=150, unit_amplitude=False)
gs_errors = [float(np.sqrt(np.mean((np.abs(gs_core.disperse(E, D2)) ** 2 - I2) ** 2))) for E in E_history]

r3plane = bridge.run_multiplane_bridge_demo(Ds=(D1, D2, 12000.0), n_iter=150)
rms_by_n = bridge.sweep_measurement_diversity(n_iter=150)

r_prior_off = bridge.run_multiplane_bridge_demo(Ds=(D1, D2), n_iter=150, use_amplitude_prior=False)
r_prior_on = bridge.run_multiplane_bridge_demo(Ds=(D1, D2), n_iter=150, use_amplitude_prior=True)

of_sweep = {n: bridge.demonstrate_autograd_overfitting(noise_std=n) for n in [0.05, 0.3, 0.6, 1.5, 3.0]}
of_single = bridge.demonstrate_autograd_overfitting(noise_std=1.5, seed=0)

noise_sweep = noise_robustness.sweep_noise_robustness(noise_levels=(0.0, 0.05, 0.15, 0.3, 0.6, 1.5))

print("Done. Writing figures...")

# ── Fig 1: native SEALS trace vs wavelength ─────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(lamvec * 1e9, mie_fields.I_p, color=GS_C)
ax.set_xlabel('Wavelength (nm)'); ax.set_ylabel(r'Intensity $I_p$ (a.u.)')
ax.set_title(f'Native SEALS intensity trace (N={len(lamvec)} samples)')
savefig(fig, 'fig01_seals_intensity_vs_wavelength.png')

# ── Fig 2: same trace vs scattering angle ───────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(theta_deg, mie_fields.I_p, color=GS_C)
ax.set_xlabel(r'Scattering angle $\theta$ (deg)'); ax.set_ylabel(r'Intensity $I_p$ (a.u.)')
ax.set_title('Same trace, mapped to scattering angle via SEALS grating pair')
savefig(fig, 'fig02_seals_intensity_vs_angle.png')

# ── Fig 3: true Mie phase (ground truth) ────────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(theta_deg, np.unwrap(phi_true), color=TRUE_C, lw=2)
ax.set_xlabel(r'Scattering angle $\theta$ (deg)'); ax.set_ylabel(r'Phase $\phi(\theta)$ (rad)')
ax.set_title('Ground-truth Mie scattering phase (known only for validation)')
savefig(fig, 'fig03_true_mie_phase.png')

# ── Fig 4: 2-plane classical GS convergence ─────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 4))
ax.semilogy(gs_errors, color=GS_C)
ax.set_xlabel('GS iteration'); ax.set_ylabel('Measurement self-consistency residual')
ax.set_title('2-plane classical GS: converges to $\\sim 10^{-23}$ (fits its own data exactly)')
savefig(fig, 'fig04_2plane_gs_convergence.png')

# ── Fig 5: 2-plane GS recovered phase vs truth ──────────────────────────────
_, phi_gs_aligned_2p = compare_phase(r2plane['phi_gs'], phi_true, weight)
phi_gs_disp, phi_true_disp = display_pair(phi_gs_aligned_2p, phi_true)
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(theta_deg, phi_true_disp, color=TRUE_C, lw=2, label='True phase (Mie)')
ax.plot(theta_deg, phi_gs_disp, '--', color=GS_C,
        label=f"GS recovered (RMS {r2plane['rms_gs_vs_truth']:.3f} rad)")
ax.set_xlabel(r'Scattering angle $\theta$ (deg)'); ax.set_ylabel('Phase (rad)'); ax.legend()
ax.set_title('2-plane GS: converges, but NOT to the true phase')
savefig(fig, 'fig05_2plane_gs_vs_truth.png')

# ── Fig 6: per-sample phase error vs amplitude (mechanism) ──────────────────
fig, ax = plt.subplots(figsize=(6, 4))
ax.scatter(diag['amplitude'], np.abs(diag['per_sample_error']), s=10, alpha=0.5, color=GS_C)
ax.set_xlabel(r'$|E_p|$ (measured amplitude)'); ax.set_ylabel('Per-sample phase error (rad)')
ax.set_title(f"Error concentrates at low amplitude (Pearson $r$={diag['pearson_r_abs_err_vs_log_amplitude']:.2f}, "
             f"$p$={diag['pearson_p_value']:.1e})")
savefig(fig, 'fig06_error_vs_amplitude_mechanism.png')

# ── Fig 7: signal amplitude vs angle (context for Fig 6) ────────────────────
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(theta_deg, diag['amplitude'], color=GS_C)
ax.set_xlabel(r'Scattering angle $\theta$ (deg)'); ax.set_ylabel(r'$|E_p|$')
ax.set_title(f"Signal strength vs. angle ({diag['amplitude_dynamic_range']:.0f}x dynamic range)")
savefig(fig, 'fig07_amplitude_vs_angle.png')

# ── Fig 8: autograd optimization loss curve (properly normalized) ───────────
scale = amp_true.max()
amp_n = torch.tensor(amp_true / scale, dtype=torch.float64)
I1_n = torch.tensor(I1 / scale ** 2, dtype=torch.float64)
I2_n = torch.tensor(I2 / scale ** 2, dtype=torch.float64)
_, loss_hist = retrieve_phase_torch(amp_n, [I1_n, I2_n],
    [lambda E: dispersive_operator(E, D1), lambda E: dispersive_operator(E, D2)], n_steps=800, lr=0.03)
fig, ax = plt.subplots(figsize=(6, 4))
ax.semilogy(loss_hist, color=AG_C)
ax.set_xlabel('Optimizer step (Adam)'); ax.set_ylabel('Normalized MSE loss')
ax.set_title('Independent autograd path: genuine gradient-descent convergence')
savefig(fig, 'fig08_autograd_convergence.png')

# ── Fig 9: GS vs autograd vs truth, three-way overlay ────────────────────────
_, phi_ag_aligned_2p = compare_phase(r2plane['phi_autograd'], phi_true, weight)
phi_ag_disp, _ = display_pair(phi_ag_aligned_2p, phi_true)
fig, ax = plt.subplots(figsize=(6.5, 4))
ax.plot(theta_deg, phi_true_disp, color=TRUE_C, lw=2, label='True phase (Mie)')
ax.plot(theta_deg, phi_gs_disp, '--', color=GS_C, label=f"Classical GS (RMS {r2plane['rms_gs_vs_truth']:.3f} rad)")
ax.plot(theta_deg, phi_ag_disp, ':', color=AG_C, label=f"Autograd (RMS {r2plane['rms_autograd_vs_truth']:.3f} rad)")
ax.set_xlabel(r'Scattering angle $\theta$ (deg)'); ax.set_ylabel('Phase (rad)'); ax.legend(fontsize=9)
ax.set_title(f"Two independent algorithms, two DIFFERENT wrong answers "
             f"({r2plane['rms_gs_vs_autograd']:.2f} rad apart)")
savefig(fig, 'fig09_gs_vs_autograd_vs_truth.png')

# ── Fig 10: dispersion-pair stability sweep ─────────────────────────────────
pairs = [(-5000, -5750), (6000, -7000), (8000, -9200), (20000, -23000)]
pair_rms = [bridge.run_bridge_demo(D1=d1, D2=d2, n_iter=200)['rms_gs_vs_truth'] for d1, d2 in pairs]
fig, ax = plt.subplots(figsize=(6, 4))
ax.bar([f"{d1}\n{d2}" for d1, d2 in pairs], pair_rms, color=GS_C)
ax.set_ylabel('GS RMS phase error vs. truth (rad)'); ax.set_xlabel(r'$(D_1, D_2)$ (normalized)')
ax.set_title('2-plane failure is stable across dispersion choice, not one bad pick')
savefig(fig, 'fig10_dispersion_pair_stability.png')

# ── Fig 11: RMS vs number of measurement planes -- the headline fix ─────────
fig, ax = plt.subplots(figsize=(6, 4))
ns = sorted(rms_by_n)
ax.semilogy(ns, [rms_by_n[n] for n in ns], 'o-', color=N3_C, markersize=7)
ax.set_xticks(ns); ax.set_xlabel('Number of dispersion planes, $N$')
ax.set_ylabel('RMS phase error vs. true phase (rad)')
ax.set_title('Adding one 3rd measurement plane resolves the ambiguity')
savefig(fig, 'fig11_rms_vs_n_planes.png')

# ── Fig 12: phase overlay, N=2 vs N=3 ───────────────────────────────────────
_, phi2_aligned = compare_phase(bridge.run_multiplane_bridge_demo(Ds=(D1, D2))['phi_gs'], phi_true, weight)
_, phi3_aligned = compare_phase(r3plane['phi_gs'], phi_true, weight)
phi2_disp, _ = display_pair(phi2_aligned, phi_true)
phi3_disp, _ = display_pair(phi3_aligned, phi_true)
fig, ax = plt.subplots(figsize=(6.5, 4))
ax.plot(theta_deg, phi_true_disp, color=TRUE_C, lw=2, label='True phase (Mie)')
ax.plot(theta_deg, phi2_disp, '--', color=BAD_C, label=f"N=2 planes (RMS {r2plane['rms_gs_vs_truth']:.3f} rad)")
ax.plot(theta_deg, phi3_disp, '-', color=N3_C, lw=1.5, label=f"N=3 planes (RMS {r3plane['rms_vs_truth']:.4f} rad)")
ax.set_xlabel(r'Scattering angle $\theta$ (deg)'); ax.set_ylabel('Phase (rad)'); ax.legend(fontsize=9)
ax.set_title('The 3rd plane tracks truth everywhere, including the old weak-signal tail')
savefig(fig, 'fig12_n2_vs_n3_phase_overlay.png')

# ── Fig 13: amplitude-prior regularization -- honest null result ────────────
fig, ax = plt.subplots(figsize=(5, 4))
ax.bar(['without prior', 'with prior'], [r_prior_off['rms_vs_truth'], r_prior_on['rms_vs_truth']],
       color=[GS_C, BAD_C])
ax.set_ylabel('RMS phase error vs. truth (rad)')
ax.set_title('Amplitude-prior regularization: no measurable benefit')
savefig(fig, 'fig13_amplitude_prior_null_result.png')

# ── Fig 14: even/odd residual decomposition (historical cross-check) ────────
err = np.angle(np.exp(1j * (phi_true - phi2_aligned)))
err_even = 0.5 * (err + err[::-1]); err_odd = 0.5 * (err - err[::-1])
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(theta_deg, err_even, color=GS_C, label=f"Even part (RMS {eo['error_even_rms']:.3f} rad)")
ax.plot(theta_deg, err_odd, color=AG_C, label=f"Odd part (RMS {eo['error_odd_rms']:.3f} rad)")
ax.set_xlabel(r'Scattering angle $\theta$ (deg)'); ax.set_ylabel('Residual phase error (rad)'); ax.legend(fontsize=9)
ax.set_title('Historical even-degree-phase ambiguity: ruled out (ratio 1.08, balanced)')
savefig(fig, 'fig14_even_odd_cross_check.png')

# ── Fig 15: autograd overfitting -- RMS vs training step, multiple noise levels ──
fig, ax = plt.subplots(figsize=(6.5, 4))
for n, of in of_sweep.items():
    steps = [r['step'] for r in of['records']]
    rms_vals = [r['rms_vs_truth'] for r in of['records']]
    ax.semilogx(steps, rms_vals, 'o-', markersize=3, label=f'noise={n*100:.0f}%')
ax.set_xlabel('Optimizer step'); ax.set_ylabel('RMS phase error vs. true phase (rad)')
ax.set_title('Autograd path: RMS-vs-truth over training, at each noise level')
ax.legend(fontsize=8)
savefig(fig, 'fig15_overfitting_rms_vs_step.png')

# ── Fig 16: overfitting gap vs noise level ──────────────────────────────────
noise_vals = list(of_sweep.keys())
gaps = [of_sweep[n]['overfitting_gap'] for n in noise_vals]
fig, ax = plt.subplots(figsize=(5.5, 4))
ax.bar([f"{n*100:.0f}%" for n in noise_vals], gaps, color=[GS_C if g < 0.01 else BAD_C for g in gaps])
ax.set_xlabel('Measurement noise (multiplicative std.)'); ax.set_ylabel('Overfitting gap (rad)\n(final RMS $-$ best RMS)')
ax.set_title('Real overfitting, growing with noise -- but a secondary effect')
savefig(fig, 'fig16_overfitting_gap_vs_noise.png')

# ── Fig 17: reproducible single-case overfit, in phase space ────────────────
of_best_disp, of_true_disp = display_pair(of_single['best_phase'], of_single['phi_true'])
of_final_disp, _ = display_pair(of_single['final_phase'], of_single['phi_true'])
fig, ax = plt.subplots(figsize=(6.5, 4))
ax.plot(of_single['theta_deg'], of_true_disp, color=TRUE_C, lw=2, label='True phase (Mie)')
ax.plot(of_single['theta_deg'], of_best_disp, '-', color=GS_C,
        label=f"Best checkpoint, step {of_single['best_step']} (RMS {of_single['best_rms']:.3f} rad)")
ax.plot(of_single['theta_deg'], of_final_disp, '--', color=BAD_C,
        label=f"Fully converged, step {of_single['final_step']} -- OVERFIT (RMS {of_single['final_rms']:.3f} rad)")
ax.set_xlabel(r'Scattering angle $\theta$ (deg)'); ax.set_ylabel('Phase (rad)'); ax.legend(fontsize=8)
ax.set_title(f"noise=150%: more training moved the fit AWAY from truth "
             f"(gap {of_single['overfitting_gap']:+.3f} rad)")
savefig(fig, 'fig17_reproducible_overfit_example.png')

# ── Fig 18: 3-plane accuracy vs measurement noise ───────────────────────────
fig, ax = plt.subplots(figsize=(6, 4))
noise_pct = [r['noise_std'] * 100 for r in noise_sweep]
rms_noise = [r['rms_vs_truth'] for r in noise_sweep]
ax.semilogy(noise_pct, rms_noise, 'o-', color=N3_C, markersize=6)
ax.axhline(0.5044, color=BAD_C, ls='--', lw=1, label='2-plane structural floor (0.50 rad)')
ax.set_xlabel('Measurement noise (%, multiplicative std.)'); ax.set_ylabel('RMS phase error vs. truth (rad)')
ax.set_title('3-plane fix: noise re-introduces error, but stays well below the 2-plane floor')
ax.legend(fontsize=9)
savefig(fig, 'fig18_noise_robustness.png')

print(f"\nAll 18 figures written to {OUTDIR.resolve()}")
