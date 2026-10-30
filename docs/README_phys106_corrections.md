# PHYS 106 Midterm 1: corrections in SymPy and Lean

A study guide for correcting the handwritten work on PHYS 106 (Modern Physics, Dr. Jensen,
Fall 2026) Midterm 1. Source: the four-page CamScanner PDF dated 10-28-26. The scan has the
student's work and the grader's red-ink notes, but **not** the printed problem text, graphs, or
answer choices. Everything that depends on those is marked `MISSING_SOURCE_*`.

## Files

| File | What it is |
|---|---|
| [`notebooks/phys106_midterm1_corrections_sympy_lean.ipynb`](../notebooks/phys106_midterm1_corrections_sympy_lean.ipynb) | The study notebook (10 sections + checklist), executed with outputs saved |
| [`lean/phys106_midterm1_proofs.lean`](../lean/phys106_midterm1_proofs.lean) | 25 Lean 4 / Mathlib theorems: the algebra and arithmetic behind each correction |
| [`tools/build_phys106_nb.py`](../tools/build_phys106_nb.py) | Regenerates the notebook from source (edit this, not the .ipynb) |

## Run

Python 3 with SymPy, NumPy, Pandas, Matplotlib (tested: Python 3.12, SymPy 1.14, pandas 2.3):

```bash
py -3.12 tools/build_phys106_nb.py
```
```bash
py -3.12 -m jupyter nbconvert --to notebook --execute --inplace notebooks/phys106_midterm1_corrections_sympy_lean.ipynb
```

The Lean check is optional and separate from the Python. It needs a Lake project that requires
Mathlib (default `~/lean/NavierStokesAndEuler`, or set `PHYS106_LEAN_PROJECT`). Run it from that
project:

```bash
lake env lean /path/to/lean/phys106_midterm1_proofs.lean
```

If `lake` or Mathlib is missing, the notebook prints that and labels every theorem `UNVERIFIED`.
To skip the check, set `RUN_LEAN = False` in §8.

**Last verified 2026-10-28:** `lake env lean` exit 0, no errors or warnings (Lean v4.34.0-rc2 +
Mathlib), both standalone and from the notebook's §8 cell. The notebook's 19 numerical checks
all pass.

## Corrections at a glance

| Part | Exam answer | Correction | Status |
|---|---|---|---|
| Q1.2 | 268.328 km | L = L₀/γ = **360 km** (square dropped in 1 − β²) | corrected |
| Q1.3 | "s = 0, simultaneous" | Δt′ = −γvΔx/c² = **−2.67 ms**, Mt. E first. s² = −(600 km)² is spacelike and invariant, so it can't decide simultaneity | corrected |
| Q1.4 | −0.588 m/s | u = −0.4c ⇒ u′ = **−10/11 c ≈ −0.909c** | corrected |
| Q2.1 | 1.41×10⁶ m/s | **1.416×10⁶ m/s** | assumes V_s = 5.7 V (grader questioned this) |
| Q2.2 | −9.05×10⁻¹⁹ | h·λ was used where h·f belongs; φ = hc/λ − eV_s = **1.27 eV** | same assumption |
| Q2.3 | dN/dt·(1.18×10⁻⁴⁰) | P = R·hc/λ, with R read from the graph | blocked: graph missing |
| Q3.3 | blank | qE = qvB ⇒ v = E/B = **6.67×10⁶ m/s** | assumes velocity selector |
| Q3.4 | n = 3 (hydrogen formula) | a 10 eV photon has λ = hc/E = **124 nm**; the plural "wavelengths" need the exam's energy diagram | partly blocked |

## What each kind of claim means

* **Tutoring correction:** an explanation of the physics and of where the work went wrong.
* **Numerical model:** values from exact SI constants with units tracked by SymPy. Checked for
  consistency, which doesn't confirm the inputs were read correctly from the exam.
* **Formally verified theorem:** a Lean statement that compiled. It proves algebra about the
  model (isolations with their nonzero hypotheses, residuals, exact arithmetic). It does not
  prove the physics.

## Still needed from the original exam

The full Q1 prompt, Q1.1 and Q3.1 choices, the Q2.1 setup (is 5.7 V the stopping potential?),
the Q2.3 graph, the entire Q2.4 prompt, the Q3.2 figure, the Q3.3 prompt, the Q3.4 energy-level
diagram, and per-part point values. Notebook §10 lists why each one matters.
