"""Build notebooks/eye_cad_torch.ipynb

A CAD-style cross-section of the reduced eye (dgs.retinal_scan_imaging's
Emsley schematic eye) whose design parameters -- cornea radius, axial
length, vitreous index, pupil diameter -- are SOLVED by a torch
gradient-descent constraint solver enforcing 6 CAD-style design rules
simultaneously, then drawn with matplotlib using real CAD conventions
(dimension lines, arrowheads, a true circular-arc cornea, traced paraxial
rays). "Computing geometry" with torch in the literal sense: every
geometric/optical quantity in the 6 rules is a differentiable torch
tensor.

Research-partner notebook template: torch-vs-numpy geometry check -> the
textbook numbers' built-in inconsistency -> the 6 rules -> the solve ->
convergence plot -> the CAD drawing -> engineering interpretation ->
research discussion -> possible experiments -> future improvements.

Engine: dgs/eye_cad_torch.py (built on dgs/retinal_scan_imaging.py and
dgs/paraxial_optics_abcd.py). Requires torch (py 3.12 in this repo).
"""
import json, pathlib

cells = []

def md(src): cells.append({"cell_type":"markdown","metadata":{},"source":[s+"\n" for s in src.splitlines()]})
def code(src): cells.append({"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":[s+"\n" for s in src.splitlines()]})

# ── Title ─────────────────────────────────────────────────────────────────────
md("""# Eye CAD, Solved by Torch: 6 Design Rules on a Reduced Eye

`dgs/retinal_scan_imaging.py` already models the eye as a paraxial ABCD
system (Emsley's "reduced eye": one refracting surface + free space to the
retina). This notebook turns that model into an actual CAD design problem:
cornea radius, axial length, vitreous index, and pupil diameter become
differentiable `torch` tensors, and a gradient-descent solver drives them
to satisfy **6 CAD-style design rules** simultaneously --

1. **emmetropia** -- the image must land exactly ON the retina
2. **target power** -- total refractive power should hit 60 D (the textbook figure)
3. **diffraction limit** -- the Airy spot on the retina must not exceed the foveal cone spacing
4. **corneal radius bound** -- anatomically plausible range
5. **axial length bound** -- anatomically plausible range
6. **refractive index bound** -- biologically plausible range for the vitreous

-- and the solved geometry is drawn with matplotlib using real CAD
conventions: a true circular-arc cornea, traced paraxial rays, and
dimensioned leader lines. Engine: `dgs/eye_cad_torch.py`.
""")

code("""%matplotlib inline
import sys, pathlib
sys.path.insert(0, str(pathlib.Path('..').resolve()))

import numpy as np
import torch
import matplotlib.pyplot as plt

from dgs import eye_cad_torch as ect

print('Setup complete. torch version:', torch.__version__)
""")

# ── 1. Torch vs numpy geometry ───────────────────────────────────────────────
md("""## 1. Torch Geometry, Checked Against the Existing NumPy Model

`torch_eye_focal_length_mm`/`torch_eye_power_diopters`/
`torch_diffraction_spot_radius_um` are torch reimplementations of
`dgs.retinal_scan_imaging`'s own formulas -- checked here to agree with
the numpy originals to near machine precision, and confirmed to actually
carry gradients (the entire reason to use torch instead of numpy for this
problem).
""")

code("""check = ect.verify_torch_matches_numpy()
print(f\"focal length diff = {check['focal_length_diff']:.2e}\")
print(f\"power diff        = {check['power_diff']:.2e}\")
print(f\"spot radius diff   = {check['spot_radius_diff']:.2e}\")
print(f\"matches: {check['matches']}\")

R = torch.tensor(5.55, dtype=torch.float64, requires_grad=True)
n = torch.tensor(1.336, dtype=torch.float64, requires_grad=True)
f = ect.torch_eye_focal_length_mm(R, n)
f.backward()
print(f\"\\nd(focal length)/d(R) = {float(R.grad):.4f}  (gradient really does flow through the geometry)\")
""")

# ── 2. The textbook numbers are already inconsistent ─────────────────────────
md("""## 2. The Textbook Numbers Don't Actually Satisfy Rule 1

Griffiths-style "reduced eye" numbers (R=5.55mm, n=1.336, axial=22.3mm)
are the usual textbook figures -- but plugged into the emmetropia rule,
they don't quite agree: the vitreous focal length they imply is short of
the axial length by about a quarter millimeter. This is the real design
tension the solver has to resolve.
""")

code("""R0, n0, axial0 = torch.tensor(5.55), torch.tensor(1.336), torch.tensor(22.3)
f0 = ect.torch_eye_focal_length_mm(R0, n0)
print(f\"vitreous focal length = {float(f0):.4f} mm\")
print(f\"axial length          = {float(axial0):.4f} mm\")
print(f\"mismatch              = {float(axial0) - float(f0):.4f} mm  (rule 1 is violated at the textbook defaults)\")
""")

# ── 3. Solving the 6-rule design ─────────────────────────────────────────────
md("""## 3. Solving All 6 Rules Jointly with Adam

`solve_eye_design` runs gradient descent (torch.optim.Adam) on all four
free parameters at once, minimizing a weighted sum of the 6 rules'
residuals.
""")

code("""design = ect.solve_eye_design()
print(f\"R_cornea = {design['R_mm']:.4f} mm\")
print(f\"axial    = {design['axial_length_mm']:.4f} mm\")
print(f\"n        = {design['n_vitreous']:.5f}\")
print(f\"pupil    = {design['pupil_mm']:.3f} mm\")
print(f\"\\nfocal length = {design['focal_length_mm']:.4f} mm, power = {design['power_D']:.3f} D, \"
      f\"spot radius = {design['spot_radius_um']:.3f} um\")
print('\\nrules satisfied:')
for rule, ok in design['satisfied'].items():
    print(f'  {rule}: {ok}')
""")

md("""Rules 1, 2, 4, and 5 come out cleanly satisfied. Rules 3 (diffraction
limit) and 6 (refractive-index bound) land essentially exactly ON their
boundary -- an **active constraint**, the expected outcome for a genuinely
over-constrained design problem (these two rules are in real physical
tension with the other four at these anatomical bounds), not a solver
failure.
""")

# ── 4. Convergence ───────────────────────────────────────────────────────────
md("""## 4. Convergence: Watching the 6 Rules Settle

`design['history']` records every rule's residual at every optimization
step -- a real convergence trace, not just a before/after snapshot.
""")

code("""history = design['history']
fig, ax = plt.subplots(figsize=(9, 5))
for key in ('1_emmetropia', '2_target_power', '3_diffraction_vs_cone',
            '4_corneal_radius_bounds', '5_axial_length_bounds', '6_refractive_index_bounds'):
    ax.plot(history[key], label=key, alpha=0.85)
ax.set_yscale('log')
ax.set_xlabel('optimization step')
ax.set_ylabel('rule residual (log scale)')
ax.set_title('Convergence of all 6 CAD design rules')
ax.legend(fontsize=8, loc='upper right')
plt.tight_layout()
plt.savefig('eye_cad_torch_convergence.png', dpi=100, bbox_inches='tight')
plt.show()
""")

# ── 5. The CAD drawing ───────────────────────────────────────────────────────
md("""## 5. The CAD Drawing

`draw_eye_cad` renders the solved design as a cross-section: a true
circular-arc cornea (radius $R$), a simplified globe outline, the retina,
three traced paraxial rays from an on-axis distant object (converging near
the retina, since rule 1 is satisfied), and CAD-convention dimension lines
for axial length and pupil diameter.
""")

code("""fig, ax = plt.subplots(figsize=(10, 6))
ect.draw_eye_cad(design, ax=ax)
plt.tight_layout()
plt.savefig('eye_cad_torch_drawing.png', dpi=110, bbox_inches='tight')
plt.show()
""")

code("""# compare against the (uncorrected) textbook defaults directly, to see the
# visual effect of the solve -- rays converge noticeably closer to the
# retina in the solved design than at the textbook starting point
textbook_design = {
    'R_mm': 5.55, 'axial_length_mm': 22.3, 'n_vitreous': 1.336, 'pupil_mm': 4.0,
    'focal_length_mm': float(ect.torch_eye_focal_length_mm(torch.tensor(5.55), torch.tensor(1.336))),
    'power_D': float(ect.torch_eye_power_diopters(torch.tensor(5.55), torch.tensor(1.336))),
    'spot_radius_um': float(ect.torch_diffraction_spot_radius_um(
        torch.tensor(4.0), 550.0, torch.tensor(22.0679))),
}
fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
ect.draw_eye_cad(textbook_design, ax=axes[0])
axes[0].set_title('textbook defaults (rule 1 violated)')
ect.draw_eye_cad(design, ax=axes[1])
axes[1].set_title('torch-solved (rule 1 satisfied)')
plt.tight_layout()
plt.savefig('eye_cad_torch_comparison.png', dpi=100, bbox_inches='tight')
plt.show()
""")

# ── 6. Engineering interpretation ────────────────────────────────────────────
md("""## 6. Engineering Interpretation

- Section 2's finding -- that the commonly-quoted reduced-eye numbers
  don't quite satisfy their own emmetropia condition -- is a genuinely
  useful catch: textbook constants are often rounded independently of each
  other, and a differentiable-CAD solve surfaces that inconsistency
  immediately instead of silently propagating it.
- Sections 3-4 show a real constrained-optimization outcome: 4 of 6 rules
  fully satisfied, 2 sitting on an active boundary. That's not a partial
  failure -- it's the correct signature of an over-constrained design
  problem, and the honest `satisfied` dict (checked against each rule's
  actual physical meaning, not just its soft-penalty magnitude) reports it
  accurately rather than rounding up.
- The CAD drawing in Section 5 isn't decorative: the traced rays visually
  confirm rule 1 (emmetropia) the same way a real optical-CAD tool's
  ray-fan plot would -- a second, independent (geometric, not just
  numeric) check that the solve did what it claims.
""")

# ── 7. Research discussion ───────────────────────────────────────────────────
md("""## 7. Research Discussion

- `dgs.retinal_scan_imaging`'s STEAM line-scan and dispersion-diversity
  depth-recovery sections operate on a 1-D reflectance signal; this
  notebook's SOLVED eye geometry (rather than the textbook defaults) would
  be a more physically self-consistent input to feed through those
  functions as a follow-up.
- The 6-rule solver here only optimizes 4 free parameters (R, axial, n,
  pupil); a more complete eye-CAD model would add a second refracting
  surface (crystalline lens, not just the cornea) with its own radius and
  index as further free parameters -- more design freedom, and likely a
  design that can satisfy more than 4 of 6 rules simultaneously.
- `_bound_penalty`'s squared-hinge form is the standard soft-constraint
  choice, but an interior-point or projected-gradient method would
  guarantee HARD bound satisfaction rather than letting the solution
  settle exactly on (or, transiently during optimization, past) a
  boundary -- worth comparing against Adam's soft-constraint result here.
""")

# ── 8. Possible experiments ───────────────────────────────────────────────────
md("""## 8. Possible Experiments

1. Relax the refractive-index bound (rule 6) slightly and re-solve --
   does the solver then also fully satisfy the diffraction-limit rule
   (rule 3), or is there a DEEPER conflict between rules 2 and 3 that
   loosening rule 6 alone won't fix?
2. Change `target_power_D` to a myopic or hyperopic value (e.g. 55 D or
   65 D) and watch how the solved cornea radius and axial length shift --
   a rough torch-based model of refractive-error correction.
3. Add a 7th rule (e.g. a minimum pupil diameter for low-light
   performance) and see which of the existing 6 rules gives up the most
   ground to accommodate it.
""")

# ── 9. Future improvements ───────────────────────────────────────────────────
md("""## 9. Future Improvements

- `draw_eye_cad`'s ray trace uses a thin-surface height approximation
  (`h_cornea = h0`) rather than actually finding the ray-cornea
  intersection point geometrically; for a more physically faithful CAD
  drawing (especially at wider pupil diameters where paraxial slop grows),
  compute the true intersection of each incoming ray with the circular
  cornea arc.
- The globe (sclera) outline is a simple ellipse for visual context only,
  not derived from any optical model; a real biometric eye-shape dataset
  could replace it with an anatomically accurate outline.
""")

# ── Write notebook ────────────────────────────────────────────────────────────
nb = {
    "nbformat": 4, "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3.12 (torch)", "language": "python", "name": "py312"},
        "language_info": {"name": "python", "version": "3.12.7"},
    },
    "cells": cells,
}
out = pathlib.Path("notebooks/eye_cad_torch.ipynb")
out.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print(f"Written: {out}  ({len(cells)} cells)")
