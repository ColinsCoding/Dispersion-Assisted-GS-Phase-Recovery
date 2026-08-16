"""
projects/seals/morphology -- particle-sizing / morphology-discrimination
research extension on top of the validated SEALS forward model.

SEPARATE RESEARCH QUESTION from ../inverse/ (phase retrieval): this package
does not try to recover the hidden phase of the field. It asks whether
particle DIAMETER (and, later, other morphology parameters) can be inferred
from the SHAPE of the intensity-vs-angle scattering profile -- a classical
Mie/SEALS particle-sizing question, closer to ../inverse/inverse_scattering.py
in spirit (model-based, not blind).

SCOPE AND HONESTY CONSTRAINTS (carried through every module here):
  - All traces come from the Mie forward model already validated elsewhere
    in this repo (_seals_physics.py, cross-checked against the original
    MATLAB mie-2.m) -- SIMULATED, not real instrument data.
  - Polystyrene beads (the only thing actually validated, incl. against the
    SEALS paper's own 7.32um/9.94um measurements) are NOT biologically
    equivalent to yeast, normal cells, or cancer cells. Any yeast/cell/
    cancer discussion in this package is explicitly hypothetical future
    work, never presented as a validated result.
  - Never claim optical classification equals clinical diagnosis.

Modules:
  bead_comparison -- Part 1+2: two-bead reproduction (7.32um vs 9.94um) and
                      a diameter sweep, with simple feature extraction
                      (lobe count/spacing, peak/integrated intensity,
                      centroid, variance)
"""
