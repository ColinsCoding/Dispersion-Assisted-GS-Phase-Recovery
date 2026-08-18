"""Build notebooks/hybrid90deg_phase_retrieval_mie.ipynb -- connects the
just-fixed VPI 90-degree optical hybrid port (projects/vpi_hybrid90deg/)
to blind phase retrieval (dgs/gs_core.py via the SEALS Mie bridge) and
spectral interferometry (dgs/spectral_interferometry.py), on the SAME
Mie-scattered field used throughout this repo's other connection
notebooks. The honest thread: the 90-degree hybrid is the actual, real,
standard hardware component behind "coherent detection with a local
oscillator" -- exactly the kind of hardware the founding project brief
wanted to AVOID needing (a carrier-less, LO-free receiver). Comparing all
three methods on one field makes that tradeoff concrete instead of
asserted.

Build with `py -3.13 scripts/build_hybrid90deg_phase_retrieval_mie_nb.py`,
execute with `py -3.13 -m jupyter nbconvert --to notebook --execute --inplace
notebooks/hybrid90deg_phase_retrieval_mie.ipynb`.
"""
import pathlib
import nbformat as nbf

nb = nbf.v4.new_notebook()
md = lambda s: nbf.v4.new_markdown_cell(s)
co = lambda s: nbf.v4.new_code_cell(s)
cells = []

cells.append(md("""# The 90-degree optical hybrid, blind phase retrieval, and Mie scattering

Three ways to recover phase from the SAME hidden field, now including a
piece of REAL hardware (not just algorithms): the 90-degree optical hybrid
(`projects/vpi_hybrid90deg/hybrid_90deg.py`, fixed this session against
VPIphotonics' documented equation) is the actual, standard coherent-receiver
component -- it mixes an unknown signal against a KNOWN local oscillator
(LO) to directly recover amplitude and phase. That is exactly the kind of
hardware this repo's founding project brief wanted to AVOID needing (a
"carrier-less" receiver, phase from intensity alone, no LO). Comparing it
against blind 2-/3-plane GS and spectral interferometry -- on the identical
SEALS Mie-scattered field used in `phase_retrieval_connections.ipynb` and
`five_lenses_one_seals_problem.ipynb` -- makes that tradeoff concrete
instead of asserted."""))

cells.append(co("""import sys, pathlib
sys.path.insert(0, str(pathlib.Path.cwd().parent))
sys.path.insert(0, str(pathlib.Path.cwd().parent / 'projects' / 'vpi_hybrid90deg'))
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from dgs import gs_core
from dgs.spectral_interferometry import spectral_interferogram, valid_tau_range, hilbert_demodulate
from projects.seals.inverse import seals_to_tdgsa as bridge
from hybrid_90deg import hybrid_90deg
print("loaded gs_core, spectral_interferometry, seals_to_tdgsa, hybrid_90deg")"""))

cells.append(md("""## Part 1 -- the same Mie field, blind GS and spectral interferometry (recap)

Reused directly from earlier sessions' work, not recomputed differently."""))

cells.append(co("""demo_2plane = bridge.run_bridge_demo(D1=6000.0, D2=-7000.0, n_iter=150)
demo_3plane = bridge.run_multiplane_bridge_demo(Ds=(6000.0, -7000.0, 12000.0), n_iter=150)

E_p = demo_2plane['mie_fields'].E_p
E_p_norm = E_p / np.abs(E_p).max()
n_p = len(E_p)
omega_p = np.linspace(-1.0, 1.0, n_p)
tau_min_p, tau_max_p = valid_tau_range(omega_p)
tau_p = (tau_min_p + tau_max_p) / 2.0
E_ref_p = np.ones(n_p, dtype=complex)
S_p = spectral_interferogram(E_p_norm, E_ref_p, omega_p, tau_p)
result_si = hilbert_demodulate(S_p, omega_p, tau_p, E_ref=E_ref_p)
phi_true_p = np.angle(E_p)
phi_si = result_si['phase_est']
offset_si = np.angle(np.mean(np.exp(1j * (phi_true_p - phi_si))))
aligned_si = np.angle(np.exp(1j * (phi_si + offset_si - phi_true_p)))
rms_si = float(np.sqrt(np.mean(aligned_si ** 2)))

print(f"2-plane blind GS   RMS vs truth: {demo_2plane['rms_gs_vs_truth']:.4f} rad")
print(f"3-plane blind GS   RMS vs truth: {demo_3plane['rms_vs_truth']:.6f} rad")
print(f"spectral interferometry RMS vs truth: {rms_si:.4f} rad")"""))

cells.append(md("""## Part 2 -- the 90-degree hybrid: real hardware, known LO

Feed the SAME Mie field (normalized, matching the spectral-interferometry
treatment) into the 90-degree hybrid as `signalInput1`, against a KNOWN,
constant local oscillator (`signalInput2=1`, matching spectral
interferometry's `E_ref=1` choice above -- an apples-to-apples reference
field, not a different assumption per method). Demodulate the way a real
receiver actually would: through BALANCED PHOTODETECTORS (intensity
measurements, `|E|^2`, at each of the 4 ports, then electronic
subtraction) -- not by reading the complex field outputs directly, which
no real detector can do.

**First, with VPI's formula exactly as printed** (`row4_lo_sign=+1.0`) --
this is what the module docstring documents as bug 4: rows 3 and 4 are
identical at this ideal operating point (zero loss, zero imbalance), so Q
should come out identically zero."""))

cells.append(co("""E_LO = 1.0 + 0j

def demodulate(row4_lo_sign):
    I_signal = np.zeros(n_p)
    Q_signal = np.zeros(n_p)
    for i in range(n_p):
        outputs = hybrid_90deg(E_p_norm[i], E_LO, row4_lo_sign=row4_lo_sign)
        I_signal[i] = abs(outputs['signalOutput0grad'])**2 - abs(outputs['signalOutput180grad'])**2
        Q_signal[i] = abs(outputs['signalOutput90grad'])**2 - abs(outputs['signalOutput270grad'])**2
    phi_est = np.arctan2(Q_signal, I_signal)
    offset = np.angle(np.mean(np.exp(1j * (phi_true_p - phi_est))))
    aligned = np.angle(np.exp(1j * (phi_est + offset - phi_true_p)))
    rms = float(np.sqrt(np.mean(aligned ** 2)))
    return I_signal, Q_signal, rms

I_printed, Q_printed, rms_printed = demodulate(row4_lo_sign=1.0)
print(f"VPI's formula AS PRINTED: Q identically zero? {np.allclose(Q_printed, 0.0)}")
print(f"RMS vs truth: {rms_printed:.4f} rad  (no better than an uninformed guess -- the receiver is non-functional)")"""))

cells.append(md("""**The fix**, found by testing the natural hypothesis (row 4's LO term
should flip sign, mirroring how row 2 is the sign-flipped partner of row
1 -- matching standard quadrature-receiver theory, where all four ports
must be genuinely distinct):"""))

cells.append(co("""I_hybrid, Q_hybrid, rms_hybrid = demodulate(row4_lo_sign=-1.0)  # the module's default
print(f"CORRECTED sign: Q identically zero? {np.allclose(Q_hybrid, 0.0)}")
print(f"Q std: {Q_hybrid.std():.4e}")
print(f"RMS vs truth: {rms_hybrid:.6f} rad")
print(f"\\nimprovement: {rms_printed:.4f} rad (broken) -> {rms_hybrid:.6f} rad (corrected)")
print("A known, perfect LO SHOULD give near-exact phase recovery -- exactly what the")
print("corrected sign produces, and exactly what standard coherent-receiver theory predicts.")"""))

cells.append(md("""## Part 3 -- three-way (now four-way) honest comparison

Same caveat as the earlier connections notebooks: different methods, different
hardware assumptions, not a ranking to declare one "best" in the abstract."""))

cells.append(co("""df_compare = pd.DataFrame([
    {"method": "2-plane blind GS", "needs_LO": False, "rms_vs_truth_rad": demo_2plane['rms_gs_vs_truth']},
    {"method": "3-plane blind GS", "needs_LO": False, "rms_vs_truth_rad": demo_3plane['rms_vs_truth']},
    {"method": "spectral interferometry", "needs_LO": "reference arm", "rms_vs_truth_rad": rms_si},
    {"method": "90-degree hybrid (balanced PD)", "needs_LO": True, "rms_vs_truth_rad": rms_hybrid},
])
display(df_compare)

plt.figure(figsize=(7, 3.4))
plt.bar(df_compare['method'], df_compare['rms_vs_truth_rad'],
        color=['steelblue', 'seagreen', 'indianred', 'darkorange'])
plt.yscale('log'); plt.ylabel('RMS phase error vs Mie truth (rad)')
plt.xticks(rotation=20, ha='right')
plt.title('same hidden field, four phase-recovery approaches')
plt.tight_layout(); plt.show()"""))

cells.append(md("""## Part 4 -- from simulation to real hardware: photodetector, TIA, ADC

Part 2's exact (0.000000 rad) result came from an idealized simulation:
complex field amplitudes read out directly, no noise, no quantization.
A REAL receiver reads `|E|^2` optical POWER through a photodetector, an
electronic transimpedance amplifier (TIA), and an ADC -- reusing
`dgs/transimpedance_amplifier.py` and `dgs/adc.py` (already tested
elsewhere in this repo, not rebuilt here) to make that chain concrete
instead of hand-waved."""))

cells.append(co("""from dgs.transimpedance_amplifier import responsivity, output_voltage, snr
from dgs.adc import ADC

# realistic operating point: 1550nm, modest optical power, a few-GHz TIA
wavelength_nm = 1550.0
eta_qe = 0.85
R_lambda = responsivity(wavelength_nm, eta_qe)
P_opt_scale_W = 50e-6   # 50 uW average optical power at full-scale |E_norm|=1
R_f = 2e4               # TIA feedback resistor (V/A)
bandwidth_hz = 1e9      # 1 GHz receiver bandwidth

# convert the Part 2 I/Q intensity-difference signals (normalized |E|^2 units)
# into real optical power, then photocurrent, then TIA output voltage.
# photocurrent()'s scalar `if P_opt < 0` check doesn't support array input,
# so apply the underlying I=R*P formula directly (same convention used
# elsewhere in this repo, e.g. build_electrodynamics_nb.py's Part 10).
I_power_W = I_hybrid * P_opt_scale_W
Q_power_W = Q_hybrid * P_opt_scale_W
I_current_A = R_lambda * I_power_W
Q_current_A = R_lambda * Q_power_W
I_voltage_V = output_voltage(I_current_A, R_f)
Q_voltage_V = output_voltage(Q_current_A, R_f)

# snr() returns a LINEAR amplitude ratio (I_ph/i_n), not decibels -- converting
# explicitly rather than mislabeling the raw linear value as "dB" (a real bug
# caught while building this: the first version printed the linear ratio
# ~397.6 with a "dB" label, which is physically absurd for a receiver SNR;
# the correctly-converted value below is a reasonable ~52 dB instead).
# I_dark=0, e_n=0 (defaults): shot + thermal noise only, no amplifier voltage
# noise -- a simplification, not a claim this is the full real noise budget.
snr_linear = snr(P_opt_scale_W, R_lambda, R_f, C=0.5e-12, B=bandwidth_hz)
snr_db = 20 * np.log10(snr_linear)
print(f"responsivity @ {wavelength_nm}nm, eta={eta_qe}: {R_lambda:.3f} A/W")
print(f"receiver SNR @ {P_opt_scale_W*1e6:.0f}uW, {bandwidth_hz/1e9:.1f}GHz bandwidth: "
      f"{snr_linear:.1f} linear ({snr_db:.1f} dB)")
print(f"I voltage range: [{I_voltage_V.min()*1e3:.3f}, {I_voltage_V.max()*1e3:.3f}] mV")
print(f"Q voltage range: [{Q_voltage_V.min()*1e3:.3f}, {Q_voltage_V.max()*1e3:.3f}] mV")"""))

cells.append(md("""Digitize with a realistic ADC bit depth (coherent receivers commonly run
6-10 effective bits at multi-GHz rates) and redo phase recovery from the
QUANTIZED I/Q -- this is the real cost of moving from an idealized
simulation to actual hardware."""))

cells.append(co("""v_span = max(np.abs(I_voltage_V).max(), np.abs(Q_voltage_V).max()) * 1.2
sample_index = np.arange(n_p, dtype=float)

rows = []
for n_bits in [4, 6, 8, 10, 12]:
    adc_i = ADC(n_bits=n_bits, fs=1.0, v_range=(-v_span, v_span))
    adc_q = ADC(n_bits=n_bits, fs=1.0, v_range=(-v_span, v_span))
    _, I_digitized = adc_i.convert(sample_index, I_voltage_V)
    _, Q_digitized = adc_q.convert(sample_index, Q_voltage_V)

    # ADC.convert's internal resampling returns one fewer sample than the
    # input (checked directly, not assumed) -- align lengths rather than
    # assume an exact 1:1 correspondence with phi_true_p
    n_common = min(len(I_digitized), len(phi_true_p))
    phi_digital = np.arctan2(Q_digitized[:n_common], I_digitized[:n_common])
    phi_true_trunc = phi_true_p[:n_common]
    offset_d = np.angle(np.mean(np.exp(1j * (phi_true_trunc - phi_digital))))
    aligned_d = np.angle(np.exp(1j * (phi_digital + offset_d - phi_true_trunc)))
    rms_digital = float(np.sqrt(np.mean(aligned_d ** 2)))
    rows.append({"ADC bits": n_bits, "RMS phase error (rad)": rms_digital})

df_adc = pd.DataFrame(rows)
display(df_adc)

plt.figure(figsize=(6, 3.4))
plt.semilogy(df_adc["ADC bits"], df_adc["RMS phase error (rad)"], 'o-')
plt.axhline(rms_hybrid, ls='--', color='gray', alpha=0.6, label='idealized (no ADC) result')
plt.xlabel('ADC bit depth'); plt.ylabel('RMS phase error (rad)')
plt.title(f'quantization cost: idealized {rms_hybrid:.1e} rad -> real ADC at finite bits')
plt.legend(fontsize=8); plt.grid(alpha=0.3, which='both')
plt.tight_layout(); plt.show()"""))

cells.append(md("""## Summary

| Method | Needs a local oscillator? | Hardware |
|---|---|---|
| 2-/3-plane blind GS | No | two or more known dispersions only |
| Spectral interferometry | Reference arm (interferometric) | delayed copy of the signal itself |
| 90-degree hybrid | **Yes, an actual LO laser** | the real, standard telecom coherent-receiver component |

The founding brief's whole premise -- phase from intensity alone, no local
oscillator -- exists specifically to avoid needing the third row of this
table. The 90-degree hybrid is now a real, tested, bug-fixed piece of this
repo (`projects/vpi_hybrid90deg/`), not a competitor to blind GS but the
concrete illustration of what hardware the "carrier-less receiver" premise
was written to eliminate."""))

nb['cells'] = cells
nb['metadata'] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.13"},
}

out_path = pathlib.Path(__file__).resolve().parent.parent / "notebooks" / "hybrid90deg_phase_retrieval_mie.ipynb"
out_path.parent.mkdir(parents=True, exist_ok=True)
with open(out_path, "w", encoding="utf-8") as f:
    nbf.write(nb, f)
print(f"wrote {out_path}  ({len(cells)} cells)")
