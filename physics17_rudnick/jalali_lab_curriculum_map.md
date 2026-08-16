# Mapping Physics 17 + the current prep threads to Jalali-lab positioning

Your message bundled a lot together: Lagrangian statics, Jalali-lab modern
physics, "electrical networking," Maxwell's equations, modern-physics-level QM,
the two Griffiths books, elementary calculus, ODE/PDE, microcontroller, MATLAB,
classical computers vs. quantum information science and AI, and RF/dielectric.
This maps each thread against what's *actually already built* in this repo
(so you can see the real gaps) and against the Physics 17 reader's chapters,
then prioritizes.

The through-line stays what it's been since [[project_jalali_ucla]]: Prof.
Jalali's photonic time-stretch / STEAM camera *is* the forward model
`H(f)=exp(jπDf²)` this repo already does phase retrieval for, and his course
**EC ENGR 279AS** (RF/microwave transmitter design — electromagnetics, microwave
circuits, photonics/optoelectronics) is the direct on-ramp back into that lab.

## 1. What's already built (don't re-derive these)

| Thread | Already covered by |
|---|---|
| Maxwell's equations | [`griffiths/electrodynamics.py`](../griffiths/electrodynamics.py), [`griffiths/fields.py`](../griffiths/fields.py), [`griffiths/potentials.py`](../griffiths/potentials.py), [`griffiths/radiation.py`](../griffiths/radiation.py), [`dgs/maxwell_discrete_symmetries.py`](../dgs/maxwell_discrete_symmetries.py), [`dgs/maxwell_solid_state_continuity.py`](../dgs/maxwell_solid_state_continuity.py), [`dgs/pde_em.py`](../dgs/pde_em.py) |
| Modern-physics-level QM | [`griffiths/quantum.py`](../griffiths/quantum.py), [`griffiths/atomic.py`](../griffiths/atomic.py), [`griffiths/modern.py`](../griffiths/modern.py), [`dgs/modern_physics.py`](../dgs/modern_physics.py), [`dgs/jalali_modern_physics.py`](../dgs/jalali_modern_physics.py), [`dgs/quantum_oscillator.py`](../dgs/quantum_oscillator.py), [`dgs/quantum_operators.py`](../dgs/quantum_operators.py) |
| Two Griffiths (E&M + QM) | Both are literally separate sections of [`griffiths/`](../griffiths/) — E&M: electrostatics/magnetostatics/dielectrics/magnetic_matter/radiation; QM: quantum.py/atomic.py. `griffiths_prep.ipynb` and `griffiths_ch*.ipynb` in `notebooks/` already work problem sets from both |
| ODE/PDE | [`dgs/pde_separation.py`](../dgs/pde_separation.py), [`dgs/pde_em.py`](../dgs/pde_em.py), [`dgs/numerical_methods.py`](../dgs/numerical_methods.py) (finite-diff, Taylor), [`dgs/heat_equation_fourier.py`](../dgs/heat_equation_fourier.py) |
| RF/dielectric | [`dgs/rf_microwave.py`](../dgs/rf_microwave.py), [`dgs/rf_physics.py`](../dgs/rf_physics.py), [`dgs/allpass_dispersion_analog.py`](../dgs/allpass_dispersion_analog.py), [`griffiths/dielectrics.py`](../griffiths/dielectrics.py), and now [`dgs/microplastic/physics.py`](../dgs/microplastic/physics.py) (complex refractive index / dielectric response of a lossy medium — month 1 of the microplastic-sensing sub-project, done 2026-08-10) |
| Quantum info science vs. classical computers/AI | [`dgs/quantum_information.py`](../dgs/quantum_information.py), [`dgs/quantum_bridge.py`](../dgs/quantum_bridge.py), [`dgs/quantum_internet_link_budget.py`](../dgs/quantum_internet_link_budget.py), [`dgs/analog_computing_universality.py`](../dgs/analog_computing_universality.py) (the classical/analog-computing side of that comparison) |
| MATLAB | [`projects/seals/matlab/`](../projects/seals/matlab/) (SEALS.m, mie-2.m, rayleighdebye.m — Mie scattering, directly reusable for month 4 of the microplastic project) and [`projects/optical_hybrid_90deg/matlab/`](../projects/optical_hybrid_90deg/matlab/) |
| Elementary calculus | [`notebooks/calculus_for_college.ipynb`](../notebooks/calculus_for_college.ipynb), [`notebooks/griffiths_prep.ipynb`](../notebooks/griffiths_prep.ipynb) §1–§7 |
| Lagrangian mechanics | [`dgs/lagrangian.py`](../dgs/lagrangian.py) |

**Takeaway:** most of what the fragmented message was reaching for already
exists. The gap isn't breadth, it's a couple of specific holes below.

## 2. Real gaps

- **Microcontroller, for real.** `embedded/` has bare C (`fir_lowpass.c`,
  `solar_mppt.c`) but nothing targeting an actual MCU toolchain (no
  Arduino/STM32/ESP32 build, no register-level or HAL code). If "microcontroller"
  in your message meant *credential-building hardware work* (e.g. for an ADC
  front-end controller relevant to the real I1/I2 bench measurement in
  [[user_background]]), this is the honest gap — everything else here is
  simulation.
- **Physics 17 chapters 2, 3, 5, 6, 7** (Fluids, Elasticity, Sound Waves,
  Thermodynamics, Kinetic Theory) — see [`README.md`](README.md). Chapters 1 and
  4 (Oscillations, Waves) are already well covered by `dgs/vibration_modes.py`,
  `dgs/pierce_oscillator.py`, `dgs/eigen_modes.py`, `dgs/dispersive_fourier.py`
  so aren't urgent to re-derive.
- **"Electrical networking"** from your message is ambiguous — could mean (a)
  literal EE network theory (two-port networks, S-parameters, impedance
  matching — relevant to RF/microwave and not yet in `dgs/`), or (b)
  professional networking to get back into contact with the lab. Worth
  clarifying which you meant; the curriculum answer differs a lot from the
  outreach answer.

## 3. Priority order for Jalali-lab positioning specifically

1. **Keep the real I1/I2 bench measurement moving** ([[user_background]],
   [[feedback_gs_convergence]]) — it's the actual capstone and the strongest
   single artifact for reopening a lab conversation, stronger than any new
   notebook.
2. **RF/microwave two-port network theory** (S-parameters, impedance matching) —
   the one clean gap that maps directly onto EC ENGR 279AS's stated syllabus
   and isn't covered anywhere in `dgs/` yet.
3. **Physics 17 kinetic theory + thermodynamics chapters** — genuinely useful
   background for photodetector noise modeling (`dgs/snr.py`'s Johnson-Nyquist
   noise assumes thermodynamic equilibrium arguments this reader derives from
   scratch) and gives a from-first-principles connection between classical
   statistical mechanics and the shot-noise floor already used throughout the
   phase-recovery work.
4. **Physics 17 sound waves / elasticity chapters** — lower priority; mostly
   acoustics-specific, less directly load-bearing for the photonics/RF thread
   unless you want it for its own sake.
5. **Real microcontroller work** — only prioritize this if the goal is a
   physical ADC/detector-front-end build for the bench measurement; otherwise
   it's a parallel credential rather than something the lab reconnection
   depends on.

## 4. Open question

"How to turn this undergrad researcher position into a [...] electrical
networking lookup" is the one part of your message I couldn't confidently
parse — tell me whether you meant EE network theory (I'll add it to `dgs/` next
to `rf_microwave.py`) or literal networking/outreach strategy (different kind of
help entirely), and I'll act on it.
