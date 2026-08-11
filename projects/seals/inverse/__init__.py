"""
projects/seals/inverse -- phase-retrieval and inverse-scattering extensions on
top of the validated SEALS forward model (../seals_stable.py).

These are research extensions, not part of the original SEALS MATLAB
implementation. See ../README.md, "Phase retrieval / inverse scattering".

Modules:
  _seals_physics     -- side-effect-free copy of the validated Mie/SEALS/RDG
                         functions (see that file's docstring for why)
  measurement         -- the |E|^2 detector model; reconstructs Mie's complex
                          fields E_p, E_s from the validated I_p/I_s/T_p/T_s
  dispersion          -- a PyTorch dispersive operator H_D(omega), matching
                          dgs.gs_core.disperse's convention (extension, not
                          part of the original SEALS paper)
  phase_retrieval     -- generic multi-measurement phase retrieval via
                          PyTorch autograd
  inverse_scattering  -- SEALS-specific: recover particle diameter from a
                          synthetic intensity spectrum via a derivative-free
                          search against the validated (non-differentiable)
                          Mie model, then read off the model-predicted phase
"""
