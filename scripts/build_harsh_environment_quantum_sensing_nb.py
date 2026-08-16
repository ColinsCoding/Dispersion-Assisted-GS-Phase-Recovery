"""Build notebooks/harsh_environment_quantum_sensing.ipynb

Three real environmental stressors (temperature, ionizing radiation,
mechanical vibration) applied to dgs.optical_loops's EXISTING
ring-resonator model, each checked against the resonance's own linewidth
-- the actual criterion for whether a sensor still works. The radiation
section shows a genuine size-scale contrast: a compact microring survives
a dose that would cripple dgs.optical_loops's recirculating fiber loop,
purely from path length, using both of that module's functions unmodified.

Research-partner notebook template: thermal drift -> radiation (microring
vs. fiber loop) -> vibration -> combined summary -> engineering
interpretation -> research discussion -> possible experiments -> future
improvements.

Engine: dgs/harsh_environment_quantum_sensing.py (numpy only, built on
dgs/optical_loops.py).
"""
import json, pathlib

cells = []

def md(src): cells.append({"cell_type":"markdown","metadata":{},"source":[s+"\n" for s in src.splitlines()]})
def code(src): cells.append({"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":[s+"\n" for s in src.splitlines()]})

# ── Title ─────────────────────────────────────────────────────────────────────
md("""# Harsh-Environment Quantum Sensing: Stressing the Ring Resonator

`dgs/optical_loops.py`'s ring resonator model (finesse, resonance width,
critical coupling) is one of this session's three kept, verified modules.
This notebook stresses it three physically different ways -- temperature,
ionizing radiation, and mechanical vibration -- and checks each effect
against the ring's OWN resonance linewidth, the actual criterion for
whether a sensor still functions, not just "is there some effect."

The radiation section produces a genuine, non-obvious finding: a compact
microring (round-trip path ~60 microns) barely notices a radiation dose
that would force `dgs.optical_loops`'s recirculating fiber loop (5 km
path) to need hundreds of extra dB of amplifier gain just to stay
lossless -- the SAME loss-per-length physics, wildly different outcomes,
purely from path length. Engine:
`dgs/harsh_environment_quantum_sensing.py`.
""")

code("""%matplotlib inline
import sys, pathlib
sys.path.insert(0, str(pathlib.Path('..').resolve()))

import numpy as np
import matplotlib.pyplot as plt

from dgs import optical_loops as ol
from dgs import harsh_environment_quantum_sensing as hes

print('Setup complete.')
""")

# ── 1. Thermal drift ──────────────────────────────────────────────────────────
md("""## 1. Thermal Drift vs. Resonance Linewidth

The thermo-optic effect shifts a ring's resonance wavelength; checked
here against the ring's own FWHM (derived from
`dgs.optical_loops.ring_FWHM_phase`).
""")

code("""for dT in (0.5, 1.0, 3.0, 5.0, 10.0):
    check = hes.verify_thermal_detuning_vs_linewidth(dT)
    print(f\"dT={dT:>5.1f}K: shift={check['delta_lambda_nm']:.4f}nm, \"
          f\"FWHM={check['fwhm_lambda_nm']:.4f}nm, fraction={check['fraction_of_linewidth']:.2f}, \"
          f\"exceeds: {check['exceeds_linewidth']}\")
""")

code("""dT_range = np.linspace(0, 15, 100)
fractions = [hes.verify_thermal_detuning_vs_linewidth(dT)['fraction_of_linewidth'] for dT in dT_range]

fig, ax = plt.subplots(figsize=(8, 4.5))
ax.plot(dT_range, fractions, color='firebrick')
ax.axhline(1.0, color='black', ls='--', label='resonance FWHM (detuned beyond this = failure)')
ax.set_xlabel('temperature swing (K)'); ax.set_ylabel('shift / FWHM')
ax.set_title('Thermal detuning crosses the resonance linewidth within a few degrees')
ax.legend()
plt.tight_layout()
plt.savefig('harsh_env_thermal_drift.png', dpi=100, bbox_inches='tight')
plt.show()
""")

# ── 2. Radiation: the size-scale contrast ────────────────────────────────────
md("""## 2. Radiation: Microring vs. Recirculating Fiber Loop

Both reuse `dgs.optical_loops` functions directly (`ring_finesse`,
`loop_threshold_gain_dB`) -- only the physical path length differs.
""")

code("""print('Microring (path length ~63 um):')
doses_ring = [0, 10, 50, 100, 500]
finesses = []
for dose in doses_ring:
    r = hes.microring_finesse_under_radiation(dose)
    finesses.append(r['finesse'])
    print(f\"  dose={dose:>4}krad: finesse={r['finesse']:.4f} (change: {r['finesse_relative_change']:.2e})\")

print('\\nRecirculating fiber loop (path length 5 km):')
doses_loop = [0, 1, 5, 10, 20]
gains = []
for dose in doses_loop:
    r = hes.recirculating_loop_threshold_gain_under_radiation(dose)
    gains.append(r['threshold_gain_dB'])
    print(f\"  dose={dose:>4}krad: threshold_gain={r['threshold_gain_dB']:>7.2f}dB \"
          f\"(+{r['additional_gain_needed_dB']:.2f}dB needed)\")
""")

code("""fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
axes[0].plot(doses_ring, finesses, 'o-', color='steelblue')
axes[0].set_xlabel('radiation dose (krad)'); axes[0].set_ylabel('finesse')
axes[0].set_title('Microring: finesse barely moves')
axes[0].set_ylim(min(finesses)-0.5, max(finesses)+0.5)

axes[1].plot(doses_loop, gains, 'o-', color='firebrick')
axes[1].set_xlabel('radiation dose (krad)'); axes[1].set_ylabel('required threshold gain (dB)')
axes[1].set_title('Recirculating fiber loop: gain requirement explodes')

plt.tight_layout()
plt.savefig('harsh_env_radiation_contrast.png', dpi=100, bbox_inches='tight')
plt.show()
""")

# ── 3. Vibration ──────────────────────────────────────────────────────────────
md("""## 3. Vibration-Induced Phase Noise vs. Resonance Linewidth

Mechanical vibration modulates the effective optical path length,
checked the same way as the thermal case.
""")

code("""for dL_nm in (0.1, 1.0, 5.0, 10.0, 20.0):
    check = hes.verify_vibration_jitter_vs_linewidth(dL_nm * 1e-9)
    print(f\"dL={dL_nm:>5.1f}nm: jitter={check['phase_jitter_rad']:.4f}rad, \"
          f\"FWHM={check['fwhm_phase_rad']:.4f}rad, fraction={check['fraction_of_linewidth']:.3f}, \"
          f\"exceeds: {check['exceeds_linewidth']}\")
""")

code("""dL_range = np.linspace(0, 50, 100)
vib_fractions = [hes.verify_vibration_jitter_vs_linewidth(dL*1e-9)['fraction_of_linewidth'] for dL in dL_range]

fig, ax = plt.subplots(figsize=(8, 4.5))
ax.plot(dL_range, vib_fractions, color='darkorange')
ax.axhline(1.0, color='black', ls='--', label='resonance FWHM')
ax.set_xlabel('vibration displacement amplitude (nm)'); ax.set_ylabel('phase jitter / FWHM')
ax.set_title('Vibration-induced phase jitter vs. resonance linewidth')
ax.legend()
plt.tight_layout()
plt.savefig('harsh_env_vibration.png', dpi=100, bbox_inches='tight')
plt.show()
""")

# ── 4. Engineering interpretation ────────────────────────────────────────────
md("""## 4. Engineering Interpretation

- Section 1's finding -- a few degrees of temperature swing already
  detunes the resonance by more than its own linewidth -- is the real
  reason thermal stabilization (TECs, athermal waveguide design) is a
  major practical concern in silicon photonics, not a minor footnote.
- Section 2 is the module's central result: the SAME loss-per-length
  physics gives opposite practical outcomes purely from path length. This
  is a genuine, quantified argument for why compact integrated photonics
  (not long fiber links) tends to be preferred for radiation environments
  -- not an assertion, a number (a microring's finesse changes by
  $10^{-4}$ relative at a dose that would need +250 dB of extra gain from
  the fiber loop).
- Section 3's vibration effect is comparatively mild at the illustrative
  displacement amplitudes used here -- worth noting as a genuine
  DIFFERENCE from Sections 1-2, not glossing over that not every stressor
  is equally severe for this particular device.
""")

# ── 5. Research discussion ───────────────────────────────────────────────────
md("""## 5. Research Discussion

- All three stressors here reduce to the same underlying operation --
  "how much does an extra path-length or loss perturbation compare to the
  ring's resonance width" -- worth factoring into one shared utility
  function rather than three parallel `verify_*_vs_linewidth` functions,
  if a fourth stressor (e.g. humidity-induced index change) is added
  later.
- `dgs.contour_integration_residues`' Kramers-Kronig work (also one of
  this session's kept modules) relates a resonator's real and imaginary
  response -- radiation-induced loss (Section 2) changes the imaginary
  part of the effective index directly; a follow-up could compute
  whether/how that shows up in the real part too via causality.
- The ILLUSTRATIVE radiation and vibration parameters (module docstring)
  should be replaced with cited values from actual radiation-hardened
  photonics literature before this analysis informs any real hardware
  decision -- flagged here, not resolved.
""")

# ── 6. Possible experiments ───────────────────────────────────────────────────
md("""## 6. Possible Experiments

1. Find the microring radius at which radiation-induced finesse
   degradation becomes comparable to the recirculating loop's -- is there
   a practical crossover length scale, or does the contrast hold across
   any physically reasonable microring size?
2. Combine two stressors at once (e.g. thermal drift AND vibration
   simultaneously) and check whether their combined detuning exceeds the
   linewidth at a LOWER threshold than either alone -- a more realistic
   harsh-environment scenario than one stressor in isolation.
3. Sweep the ring's own design parameters (`t`, `a`) to find whether a
   lower-finesse (wider-linewidth) design trades sensing precision for
   meaningfully better environmental robustness.
""")

# ── 7. Future improvements ───────────────────────────────────────────────────
md("""## 7. Future Improvements

- Real radiation-induced attenuation partially anneals over time (a
  well-documented effect in space photonics) -- this module treats dose
  as a permanent, static degradation, not a time-dependent process.
- The thermal and vibration models both assume a single dominant
  first-order effect (thermo-optic coefficient, simple path-length
  modulation); real devices have additional second-order effects
  (thermal expansion changing the physical ring radius, not just the
  index) not captured here.
""")

# ── Write notebook ────────────────────────────────────────────────────────────
nb = {
    "nbformat": 4, "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.13.0"},
    },
    "cells": cells,
}
out = pathlib.Path("notebooks/harsh_environment_quantum_sensing.ipynb")
out.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print(f"Written: {out}  ({len(cells)} cells)")
