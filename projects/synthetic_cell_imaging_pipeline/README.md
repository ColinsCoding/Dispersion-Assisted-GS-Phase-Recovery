# Synthetic-Cell Computational Imaging Pipeline

An original, MATLAB-primary educational research project connecting field theory,
coordinate-aware calculus, linear-algebra inverse problems, chemical kinetics, control
theory, and Boolean decision logic into one coherent computational-imaging pipeline:

```
sample/cell -> optical field -> propagation -> detector -> sampled waveform/image
            -> matrix forward model -> reconstruction -> Boolean classification -> feedback
```

## Disclaimer: textbook physics vs. public patent concepts vs. our simulation

Every technique used in this project is standard, public-domain mathematics taught in
undergraduate physics/engineering courses:

1. **Textbook physics/math** — electrostatics (`E=-grad(V)`, `div`, `curl`, Laplacian),
   separation of variables producing Bessel's equation, even/odd Fourier symmetry,
   linear-algebra inverse problems (SVD, condition number, Tikhonov regularization),
   first-order reaction kinetics, and PID control theory. All of this is in standard
   textbooks (Griffiths' *Introduction to Electrodynamics*, any linear-algebra or
   signals-and-systems text, any controls text) and is not owned by anyone.
2. **Public patent-adjacent concepts** — the general *shape* of this pipeline (a blur-
   matrix forward model `y=Hx+n`, regularized reconstruction, automated Boolean
   pass/fail gating, PID-based autofocus/alignment) resembles the general family of
   ideas found across many companies' patented computational-imaging and automated-
   inspection *implementations*. This project does not reproduce, reference, or claim
   correspondence with any specific patent, company, or instrument. Every equation here
   is independently derived from the public-domain math above.
3. **Our educational simulation** — the specific synthetic cell object (a membrane ring +
   off-center nucleus + organelle dots), the specific blur kernel, the specific kinetics
   scenario, and the specific PID/Boolean-gating example are original constructions
   built for this project, not real measurement data and not copied from any textbook
   or patent's worked example.

## Part map

| Part | Topic | MATLAB file | Notebook section |
|---|---|---|---|
| 1 | Field theory (`E=-grad V`, `div`, `curl`, Laplacian) | `matlab/part1_field_theory.m` | ✓ |
| 2 | Coordinates & Bessel's equation | `matlab/part2_bessel_coordinates.m` | ✓ |
| 3 | Even/odd symmetry & Fourier transforms | `matlab/part3_even_odd_symmetry.m` | ✓ |
| 4 | Cell-scale forward model `y=Hx+n` | `matlab/part4_forward_model.m` | ✓ |
| 5 | Matrix analysis (rank/SVD/cond/pinv, Tikhonov) | `matlab/part5_matrix_analysis.m` | ✓ |
| 6 | Chemical kinetics + optical measurement fit | `matlab/part6_kinetics_fit.m` | ✓ |
| 7 | Subsystem input/output/gain/loss/units | `matlab/part7_gain_table.m` | — |
| 8 | Boolean decision logic | `matlab/part8_boolean_decision.m` | — |
| 9 | PID feedback (autofocus/alignment) | `matlab/part9_pid_feedback.m` | — |
| 10 | Manufacturing analogy (tolerances) | this README, below | — |
| 11 | MATLAB data pipeline (functions, not one script) | `matlab/part11_pipeline/*.m` | — |
| 12 | PyTorch autograd vs. finite differences | (Python-only) | ✓ |
| 13 | Laser safety (high level) | this README, below | — |
| 14 | Final integrated project | `matlab/part14_integrated_project.m` | — |

Parts 10 and 13 are discussion-only per the assignment (no lab-hardware automation).
Part 12 is Python/PyTorch-only, run against the exact CSV Part 6's MATLAB script exports,
so the classical (MATLAB), classical (SciPy), and autograd (PyTorch) fits are a genuine
cross-language, cross-method check on identical data — all three landed on `k=0.3473`
against a true value of `0.35` (0.76% error) when last verified.

## Run it

**MATLAB** (each part is independently runnable; verified against MATLAB R2025b):
```matlab
cd matlab
part1_field_theory();          % ... through part9_pid_feedback()
cd part11_pipeline; run_pipeline(); cd ..
part14_integrated_project();   % runs everything together
```
or non-interactively: `matlab -batch "part14_integrated_project()"`.

**Python/Jupyter**:
```bash
py -3.13 scripts/build_notebook.py     # regenerates the notebook from source
py -3.13 -m jupyter nbconvert --to notebook --execute --inplace notebooks/synthetic_cell_imaging_pipeline.ipynb
```

**Tests**:
```bash
py -3.13 tests/test_python_equivalents.py
```

## Part 10 — Manufacturing analogy (tolerances, not vague comparison)

The SAME `y=Hx+n` / SVD-conditioning / Boolean-gating / PID-feedback mathematics used
for the synthetic-cell pipeline applies, with different physical `H` and different
tolerance budgets, to several manufacturing and metrology contexts:

| Application | What plays the role of `x` | What plays the role of `H` | Typical tolerance budget |
|---|---|---|---|
| Optical alignment | beam position/angle | optomechanical transfer matrix | µm-scale lateral, µrad-scale angular |
| Camera manufacturing | lens/sensor stack alignment | optical PSF / MTF response | focus shift < ~1/4 wave (Rayleigh-style) |
| Semiconductor inspection | die pattern | imaging-system blur + noise | critical-dimension tolerance, often < 10% of feature size |
| Laser material processing | beam focus position | thermal/optical propagation matrix | focus depth-of-field, often tens of µm |
| Robotic positioning | end-effector pose | kinematic/control transfer function | repeatability spec, often tens of µm to mm depending on the arm |

In every row, a **condition number** analysis (Part 5) tells you which error directions
get amplified by measurement noise, a **Boolean gate** (Part 8) turns a continuous
tolerance check into an accept/reject decision, and a **PID loop** (Part 9) is the
standard way to close the loop and drive the system back within tolerance — the same
three ideas, different physical `H` each time.

## Part 13 — Laser safety (high level)

This project performs no physical laser measurements; Part 6's "optical measurement" is
entirely synthetic. If this pipeline's ideas are ever applied to a real laser-based
instrument, standard institutional practice applies:

- **Training requirements** — laser safety training (often laser-class-specific) is
  required before operating any Class 3B/4 system, per your institution's Laser Safety
  Officer (LSO) program.
- **Controlled access** — laser labs should be access-controlled (interlocked doors,
  posted signage) so untrained personnel cannot enter an active beam path.
- **Eyewear** — appropriate laser safety eyewear (matched to the specific wavelength(s)
  and power in use) must be selected by qualified personnel (the LSO), not guessed at.
- **Beam containment** — beams should be enclosed or terminated in a beam dump wherever
  practical; open-beam paths should be minimized.
- **Beam height** — keep beam paths well below or above eye level where people stand or
  sit, per your institution's SOP.
- **Follow institutional SOPs** — every item above is a *category* of requirement; the
  actual procedure is whatever your institution's Laser Safety Officer and written SOPs
  specify for the specific laser class and application in use.

This project does not provide, and will not provide, instructions for defeating
interlocks, operating without required PPE, or any other bypass of these controls. Do
not assume ionizing radiation, high-power operation, or nonlinear-optical effects are
safe to explore outside a properly supervised laboratory setting.

## Files

```
README.md, equations.md, problems.md, solutions.md
matlab/part1_field_theory.m ... part9_pid_feedback.m, part14_integrated_project.m
matlab/part11_pipeline/{acquire,preprocess,calibrate,reconstruct,classify,store_checkpoint,run_pipeline}.m
notebooks/synthetic_cell_imaging_pipeline.ipynb
scripts/build_notebook.py
tests/test_python_equivalents.py
```
